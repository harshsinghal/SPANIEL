#!/usr/bin/env bash
# Run one model's dev, concept and PII accuracy checks concurrently.
# Used on a separate evaluation worker after a runtime-repair training rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
CONTEXT_PYTHON=${CONTEXT_PYTHON:-.context-venv/bin/python}
phase=${1:?Pass baseline or candidate}
case "$phase" in
  baseline) prefix=v2; adapter_args=() ;;
  candidate) prefix=candidate; adapter_args=(--adapter context-results/lora-r16/adapter) ;;
  *) echo 'Pass baseline or candidate' >&2; exit 2 ;;
esac
mkdir -p context-results
case "${CONTEXT_EVAL_SUITE:-all}" in
  all) splits=(dev test legacy) ;;
  concept) splits=(dev test) ;;
  legacy) splits=(legacy) ;;
  *) echo 'CONTEXT_EVAL_SUITE must be all, concept or legacy' >&2; exit 2 ;;
esac
pids=()
for split in "${splits[@]}"; do
  data=$split; legacy_args=()
  if [[ "$split" == legacy ]]; then data=legacy-eval; legacy_args=(--legacy --limit 300); fi
  "$CONTEXT_PYTHON" -m sensitive_context.evaluate \
    --data "context-data/$data.jsonl" --out "context-results/$prefix-$split.json" \
    --exclude context-data/train.jsonl --exclude context-data/replay.jsonl \
    "${adapter_args[@]}" "${legacy_args[@]}" \
    > "context-results/$prefix-$split.log" 2>&1 &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
if [[ "$result" == 0 ]]; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "context-results/${phase^^}_COMPLETED"
fi
exit "$result"
