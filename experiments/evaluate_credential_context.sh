#!/usr/bin/env bash
# Accuracy evaluations only. Concurrent execution is not a latency benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
CONTEXT_PYTHON=${CONTEXT_PYTHON:-.context-venv/bin/python}
evaluate_one() {
  local name=$1 split=$2 adapter=$3 legacy=$4
  local args=(--data "context-data/$split.jsonl" --out "context-results/$name.json"
    --exclude context-data/train.jsonl --exclude context-data/replay.jsonl)
  if [[ "$adapter" == yes ]]; then args+=(--adapter context-results/lora-r16/adapter); fi
  if [[ "$legacy" == yes ]]; then args+=(--legacy --limit 300); fi
  "$CONTEXT_PYTHON" -m sensitive_context.evaluate "${args[@]}" > "context-results/$name.log" 2>&1
}
if [[ ${CONTEXT_EVAL_JOBS:-1} == 3 ]]; then
  (evaluate_one v2-dev dev no no; evaluate_one v2-legacy legacy-eval no yes) & context_pid_a=$!
  (evaluate_one candidate-dev dev yes no; evaluate_one candidate-legacy legacy-eval yes yes) & context_pid_b=$!
  (evaluate_one v2-test test no no; evaluate_one candidate-test test yes no) & context_pid_c=$!
  context_exit=0
  for context_pid in "$context_pid_a" "$context_pid_b" "$context_pid_c"; do
    wait "$context_pid" || context_exit=1
  done
  exit "$context_exit"
elif [[ ${CONTEXT_EVAL_JOBS:-1} == 1 ]]; then
  evaluate_one v2-dev dev no no
  evaluate_one candidate-dev dev yes no
  evaluate_one v2-test test no no
  evaluate_one candidate-test test yes no
  evaluate_one v2-legacy legacy-eval no yes
  evaluate_one candidate-legacy legacy-eval yes yes
else
  echo 'CONTEXT_EVAL_JOBS must be 1 or 3' >&2
  exit 1
fi
