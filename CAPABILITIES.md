# SPANIEL capability and evidence record

This record distinguishes an extraction interface from demonstrated capability. Supplying a label does not establish that the model recognizes it reliably. Last updated 21 October 2026.

| Area | Evidence/status | Appropriate interpretation |
|---|---|---|
| Conventional PII in the v2 task | Published synthetic 300-document evaluation; 0.918 adjudicated exact typed span F1 | Measured under the published guidelines; validate on your deployment |
| Credential discrimination | Context v0.1: 33/39 sensitive documents detected, 3/36 benign false blocks, 0.889 span F1 | Improved on the synthetic test, but fails the ≥95% recall and ≤5% false-block screens; experimental |
| Private compensation versus public pay information | Context v0.1: 37/42 sensitive documents detected, 6/42 benign false blocks, 0.851 span F1 | Experimental; public-pay errors and spoken corrections remain failures |
| Informal/noisy transcript text | 24 synthetic cases: 9/12 sensitive detected, 1/12 benign falsely blocked | Small simulated-text slice; no real ASR/audio validation |
| Long retrieved context | 60 controlled dilution cases: 23/30 sensitive detected, 2/30 benign falsely blocked | Five correlated scenario families, ≤936 words; no enterprise-document reliability claim |
| Exact source preservation | Enforced by completed copy-or-tag decoding plus validation | Independent of whether sensitive entities were correctly identified |
| Fail-closed retrieval | Tested application behavior on matches and inspection failures | Learned false negatives can still release sensitive content |
| Linked/discontinuous evidence | Companion data representation only | Current decoder emits contiguous spans; model-generated evidence groups are not implemented |
| Legal strategy, layoffs, reviews and general business confidentiality | No validated concept-specific training/evaluation release | Research targets; do not present arbitrary-label requests as supported protection |
| Multilingual inputs | No validated multilingual release; audit found source language metadata errors | English is the intended scope, not proof every source row is English |
| GLiNER/encoder comparisons | [GLiNER2.5 off-the-shelf baseline](docs/gliner-context-baseline.md), same frozen concept cases and source-offset scoring | Initial baseline; no equal-data architecture comparison or speed claim |

Context v0.1 is [available as a separate adapter](https://huggingface.co/Harsh/spaniel-credential-context-lora-v0.1) and does not replace v2. On the 300-document original-source PII regression, exact typed span F1 is **0.937 for v2 and 0.924 for the adapter** (-1.26 percentage points). Inspection errors are 1 and 1, respectively. Every legacy document contains requested entities, so this set cannot measure benign false blocks. It uses original source annotations, not the separately adjudicated historical labels behind v2's 0.918 score.

“Experimental” means a specific testable hypothesis and artifact exist. “Best effort” means the interface may return findings, but reliability for that scope has not been established. Neither label grants permission to send data: the application owns the policy and handles uncertainty.

[Current milestone](docs/milestones/2026-10-21-credential-context.md) · [Coverage audit](docs/coverage-audit.md) · [Published evaluation](docs/training-data-and-evaluation.md)
