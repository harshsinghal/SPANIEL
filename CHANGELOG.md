# SPANIEL progress record

This file records artifacts and evidence, including negative results. Historical model scores remain tied to their original datasets and revisions.

## 2026-10-21 — Credential and compensation experiment

- Defined actual credentials, benign lookalikes, private compensation, public ranges and unresolved context separately.
- Added 5,248 synthetic training rows, 100 development rows, 159 frozen test rows and six ambiguity-review cases across disjoint scenario families.
- Added informal/transcription-like text, repeated values in different contexts, public versus private keys, and controlled long-document variants.
- Validated the historical mixture; rejected 64 incompatible targets from replay eligibility and sampled 1,536 public-source training rows.
- Added explicit completion-loss masking, no-truncation preflight, a LoRA trainer, a common constrained evaluator and a GPU.ai run recipe.
- Training and candidate performance are pending. No new checkpoint quality is claimed by these preparation results.

## 2026-10-16 — Coverage audit

- Audited 389,521 training rows and 385,562 distinct retained texts.
- Embedded the corpus, reviewed clusters, compared normalized/raw views and checked proposed gaps with scenario retrieval.
- Identified repetition, benign-counterexample scarcity, long-context gaps and missing concept supervision.
- Recorded four misses, four false positives and one partial boundary in 26 adaptive development probes.

## 2026-10-14 — Governed retrieval

- Added a Python tool wrapper that withholds retrieved text on policy matches or inspection failure.
- Added exact source/tag checks, literal-markup ambiguity handling and bounded local file reads.
- Distinguished deterministic enforcement from imperfect model detection.

## 2026-10-13 — Public provenance and governance documentation

- Expanded README and model-card documentation for local AI governance, training data, licenses, source revisions and benchmark limitations.

## July 2026 — v1 and v2 research

- Published constrained span extraction and full-parameter SFT experiments.
- V2 increased unfamiliar-label-name F1 from 0.747 to 0.864 on the recorded evaluation.
- Recorded annotation audits and negative thinking-mode results in the [research journal](README.md#the-journey-as-blog-entries).
