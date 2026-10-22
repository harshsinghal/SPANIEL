# GLiNER2.5 baseline for Spaniel's contextual sensitivity policy

**Harsh Singhal · 21 October 2026.** This is an initial off-the-shelf comparison, not an equally trained architecture benchmark or a claim about the best achievable GLiNER performance.

The checkpoint is [`fastino/gliner2.5-base-v1`](https://huggingface.co/fastino/gliner2.5-base-v1), revision `54785d51df8d86a0d8f2eb212fec15871aad2c4d`. Its [official library](https://github.com/fastino-ai/GLiNER2) provides native boundary extraction with source offsets and optional entity descriptions. This experiment uses that local interface, not a hosted inference API.

## Configuration fixed before prediction

The [protocol](../experiments/gliner-context-baseline-protocol.json) was committed before the first prediction on the frozen concept test. Both configurations use threshold 0.5 and flat overlap handling:

1. **Labels:** the same requested entity names as the Spaniel test.
2. **Policy descriptions:** those names plus the already frozen policy descriptions, without rewriting them based on results.

Both are reported; the better configuration is not selected as the sole baseline. There is no threshold optimization, GLiNER fine-tuning or test-driven label editing. The candidate Spaniel training run continues unchanged and selects its checkpoint using development completion loss only.

All 159 concept-test documents are processed in full. Preflight checks the encoded input against the checkpoint limit: maximum 1,021 tokens for label-only requests and 1,082 with descriptions. There is no chunking or truncation. Returned offsets must identify nonempty original source text. Invalid predictions would be inspection errors, never successful no-match results.

GLiNER runs on the Mac Mini CPU in float32 with four threads; Spaniel's official baseline/candidate comparison runs on the same GPU.ai RTX 4090 in bf16. **This is not a latency or throughput comparison across architectures.** Versions and hashes are in the reports and [environment record](../reports/gliner-environment.txt).

## Frozen concept test

The test has 159 authored synthetic cases from 17 families: 81 sensitive and 78 benign. Of these, 75 are credential cases (39 sensitive, 36 benign) and 84 compensation cases (42 each). Substitutions and long variants are correlated; they are not independent human annotations.

| Configuration and scope | Sensitive documents detected | Benign documents falsely blocked | Exact typed span F1 | Inspection errors |
|---|---:|---:|---:|---:|
| Labels — all concepts | 70/81 (86.4%) | 54/78 (69.2%) | 0.421 | 0 |
| Descriptions — all concepts | 72/81 (88.9%) | 67/78 (85.9%) | 0.415 | 0 |
| Labels — credentials | 28/39 (71.8%) | 24/36 (66.7%) | 0.510 | 0 |
| Descriptions — credentials | 30/39 (76.9%) | 29/36 (80.6%) | 0.500 | 0 |
| Labels — compensation | 42/42 (100%) | 30/42 (71.4%) | 0.362 | 0 |
| Descriptions — compensation | 42/42 (100%) | 38/42 (90.5%) | 0.357 | 0 |

High compensation recall here coexists with frequent blocking of public information. The policy requires discrimination about disclosure context, which is different from identifying every salary amount or credential-shaped value. These numbers do not measure general NER quality or establish that an encoder cannot learn the policy. The next architecture comparison should adapt both models using the same training data and select thresholds only on development data.

The [label-only report](../reports/credential-context/gliner-labels-test.json) and [description report](../reports/credential-context/gliner-policy-test.json) include per-family outcomes, exact offsets, dataset/model/code hashes and configuration. Source text and tagged outputs are omitted from public reports; the deterministic scenario generator reconstructs the synthetic inputs.

## Conventional PII regression set

On the original 300-document PII evaluation with label-only requests, GLiNER2.5 obtains **0.715 exact typed span F1**, precision 0.812 and recall 0.638. All 300 documents contain expected entities, so this set cannot measure benign false-block rate. There are no inspection errors. This uses the original source targets, not the separately adjudicated historical labels.

These are additional task-specific measurements, not a replacement for the model author's benchmarks. See the [row-level regression report](../reports/credential-context/gliner-labels-legacy.json).

## Reproduce

Create a separate Python 3.10+ environment and install [requirements-gliner-context.txt](../requirements-gliner-context.txt). Generate the frozen data using the [milestone instructions](milestones/2026-10-21-credential-context.md), then run:

```bash
python -m sensitive_context.evaluate_gliner \
  --data context-data/test.jsonl --schema labels \
  --out context-results/gliner-labels-test.json
python -m sensitive_context.evaluate_gliner \
  --data context-data/test.jsonl --schema policy-descriptions \
  --out context-results/gliner-policy-test.json
python -m sensitive_context.evaluate_gliner \
  --data context-data/legacy-eval.jsonl --schema labels --legacy \
  --out context-results/gliner-labels-legacy.json
```

The evaluator pins the checkpoint revision and validates the source offsets. It uses the same decision and exact-span scoring functions as Spaniel. Reproducing the legacy set requires the original recorded evaluation artifact.
