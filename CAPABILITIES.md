# SPANIEL capability and evidence record

This record distinguishes an extraction interface from demonstrated capability. Supplying a label does not establish that the model recognizes it reliably. Last updated 21 October 2026.

| Area | Evidence/status | Appropriate interpretation |
|---|---|---|
| Conventional PII in the v2 task | Published synthetic 300-document evaluation; 0.918 adjudicated exact typed span F1 | Measured under the published guidelines; validate on your deployment |
| Credential discrimination | Focus of the current training milestone; v2 audit shows misses and false positives | Best effort until candidate results are recorded; no disclosure-prevention guarantee |
| Private compensation versus public pay information | New explicit policy, authored contrasts and separate test families | Experimental concept, not a capability established by v2's PII score |
| Informal/noisy transcript text | Simulated training and evaluation cases prepared | Text experiment; no real ASR/audio validation |
| Long retrieved context | Controlled dilution cases and position slices prepared | Long-document reliability remains to be measured |
| Exact source preservation | Enforced by completed copy-or-tag decoding plus validation | Independent of whether sensitive entities were correctly identified |
| Fail-closed retrieval | Tested application behavior on matches and inspection failures | Learned false negatives can still release sensitive content |
| Linked/discontinuous evidence | Companion data representation only | Current decoder emits contiguous spans; model-generated evidence groups are not implemented |
| Legal strategy, layoffs, reviews and general business confidentiality | No validated concept-specific training/evaluation release | Research targets; do not present arbitrary-label requests as supported protection |
| Multilingual inputs | No validated multilingual release; audit found source language metadata errors | English is the intended scope, not proof every source row is English |
| GLiNER/encoder comparisons | [GLiNER2.5 off-the-shelf baseline](docs/gliner-context-baseline.md), same frozen concept cases and source-offset scoring | Initial baseline; no equal-data architecture comparison or speed claim |

“Experimental” means a specific testable hypothesis and artifact exist. “Best effort” means the interface may return findings, but reliability for that scope has not been established. Neither label grants permission to send data: the application owns the policy and handles uncertainty.

[Current milestone](docs/milestones/2026-10-21-credential-context.md) · [Coverage audit](docs/coverage-audit.md) · [Published evaluation](docs/training-data-and-evaluation.md)
