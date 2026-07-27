#!/bin/bash
# Bootstrap + train + generate, designed to run detached on the Vast instance.
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf

echo "=== pip install ==="
pip install --quiet --upgrade -r /workspace/requirements-train.txt

echo "=== training ==="
python /workspace/train_sft.py 2>&1

echo "=== generation ==="
python /workspace/gen_preds.py 2>&1

echo "ALL_DONE"
