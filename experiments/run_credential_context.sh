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

# The checkpoint is selected only by dev completion loss. Test results are read
# after training/selection. Baseline and candidate use the same GPU and decoder.
for SPLIT in dev test; do
  "$PYTHON" -m sensitive_context.evaluate --data "context-data/$SPLIT.jsonl" \
    --out "context-results/v2-$SPLIT.json" \
    --exclude context-data/train.jsonl --exclude context-data/replay.jsonl
  "$PYTHON" -m sensitive_context.evaluate --data "context-data/$SPLIT.jsonl" \
    --adapter context-results/lora-r16/adapter --out "context-results/candidate-$SPLIT.json" \
    --exclude context-data/train.jsonl --exclude context-data/replay.jsonl
done
if [[ -f context-data/legacy-eval.jsonl ]]; then
  "$PYTHON" -m sensitive_context.evaluate --data context-data/legacy-eval.jsonl --legacy --limit 300 \
    --out context-results/v2-legacy.json --exclude context-data/train.jsonl --exclude context-data/replay.jsonl
  "$PYTHON" -m sensitive_context.evaluate --data context-data/legacy-eval.jsonl --legacy --limit 300 \
    --adapter context-results/lora-r16/adapter --out context-results/candidate-legacy.json \
    --exclude context-data/train.jsonl --exclude context-data/replay.jsonl
fi
"$PYTHON" -m pip freeze > context-results/environment.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > context-results/COMPLETED
