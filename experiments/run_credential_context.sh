#!/usr/bin/env bash
# Run inside a GPU.ai CUDA instance after copying the prepared data/ directory.
# Instance billing/termination is managed separately by the submitting operator.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v cc >/dev/null; then
  echo 'The CUDA PyTorch/Triton environment requires a C compiler (Ubuntu: build-essential and python3-dev).' >&2
  exit 1
fi
python3 -m venv --system-site-packages .context-venv
PYTHON=.context-venv/bin/python
"$PYTHON" -m pip install -r requirements-context.txt
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$PYTHON" -m unittest discover -v
mkdir -p context-results
"$PYTHON" -m sensitive_context.train_lora \
  --train context-data/train.jsonl --replay context-data/replay.jsonl \
  --dev context-data/dev.jsonl --out context-results/lora-r16 \
  --epochs 2 --rank 16 --learning-rate 0.00005 --batch-size 1 --accumulation 16

# Checkpoint selection uses development completion loss. The paired evaluations
# use the same GPU, corrected model configuration and constrained decoder.
CONTEXT_PYTHON="$PYTHON" bash experiments/evaluate_credential_context.sh
"$PYTHON" -m pip freeze > context-results/environment.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > context-results/COMPLETED
