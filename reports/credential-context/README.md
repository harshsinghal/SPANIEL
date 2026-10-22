# Credential and context experiment: evidence guide

These files support the [October milestone](../../docs/milestones/2026-10-21-credential-context.md). They are separate from the historical v2 PII evaluation. The public reports omit document text, tagged predictions and exception bodies; case IDs, source hashes and character offsets identify the evidence. The generated concept cases can be reconstructed with the pinned generator.

## Which files answer which questions?

| Files | Evidence |
|---|---|
| `v2-dev.json`, `candidate-dev.json` | 100 authored development cases; appropriate for diagnosis, not an untouched accuracy claim |
| `candidate-dev-mps.json` | Additional development observation on Apple MPS with the byte-identical selected adapter; not substituted into the matched GPU comparison |
| `v2-test.json`, `candidate-test.json`, `comparison.json` | 159 reserved cases from 17 scenario families, with matched baseline/candidate decoding and frozen screening rules |
| `v2-legacy.json`, `candidate-legacy.json` | The same 300 original PII evaluation documents, checking retention of the original source annotations |
| `gliner-labels-test.json`, `gliner-policy-test.json` | Off-the-shelf GLiNER2.5 on the same concept cases, at threshold 0.5; both predeclared schema modes are reported |
| `gliner-labels-legacy.json` | GLiNER2.5 on the same 300 original PII evaluation documents |
| `run_manifest.json`, `training_history.json` | GPU.ai training recipe, data hashes, completion-only supervision, epoch losses and selected checkpoint |
| `evaluation-execution.json`, `environment.txt` | Evaluation scheduling and dependency environment; concurrent jobs make elapsed times unsuitable as latency benchmarks |
| `retrieval-existing-smoke.json`, `retrieval-context-smoke.json` | Actual local file-reader and policy-gate checks on Apple MPS; illustrative integration cases, not the frozen GPU benchmark |
| `ambiguity-review.json` | Six observations with insufficient context and no binary gold; no accuracy score or calibrated abstention claim |
| `report-hashes.json` | Checksums for the six exported Spaniel evaluations and the comparison |

The first run's training and model-evaluation artifacts are quarantined under `invalid-rope-run/` and explicitly marked invalid. Its loader used the wrong RoPE setting. They are diagnostic records only, and its adapter was not publicly released. Corrected artifacts will occupy the paths above after the rerun. GLiNER reports are unaffected. See the [runtime investigation](../../docs/runtime-compatibility.md).

The original experiment protocol is a frozen pre-run record; the separate [runtime-repair addendum](../../experiments/credential-context-runtime-repair.json) explains the restart without changing data or screening thresholds.

## Metric definitions

- **Sensitive-document recall:** fraction of sensitive documents with at least one validated requested finding. An inspection error is not credited as detection. Exact span scoring separately checks whether the finding was correct.
- **Benign false-block rate:** fraction of benign documents with a validated requested finding. This excludes inspection failures.
- **Benign total withholding:** false blocks plus inspection failures on benign documents. The gate withholds in either case; this is the total interruption burden in the experiment.
- **Exact typed span F1:** micro F1 over source start, end and requested label. Wrong boundaries count as a false positive and a false negative. Inspection failures contribute missed gold spans, not clean passes.
- **Correct contrast pairs:** both the sensitive and benign members receive the correct document decision. An inspection failure does not make a pair correct.
- **Production validation:** always false for this experiment, regardless of the numeric screening outcome.

The 300 legacy documents all contain requested PII, so they cannot measure benign false-block rates. Their original source annotations are not the separately adjudicated historical v2 labels used for the published 0.918 score. Do not compare these scores as if they used identical gold.

The concept cases use exact canonical labels one at a time. Substitutions and long variants from the same family are correlated, and no independent human adjudication has been completed. GLiNER is off the shelf while the candidate has task-specific training; the comparison does not isolate architecture quality. See the [GLiNER protocol](../../experiments/gliner-context-baseline-protocol.json) and [capability record](../../CAPABILITIES.md).

## Reconstructing evidence

```bash
python -m sensitive_context.build --out context-data
python -m sensitive_context.report --results context-results \
  --out reports/credential-context
python -m sensitive_context.plot_results --reports reports/credential-context \
  --out docs/coverage/context-comparison.png
```

The export requires completed, aligned evaluation files; it cannot convert partial output into a completed benchmark. Raw run outputs remain local. Exact legacy replay and regression reproduction requires the recorded original source artifacts; the available three-source rebuild is a different training artifact. Source terms and revisions are recorded in [training data and attribution](../../docs/training-data-and-evaluation.md).
