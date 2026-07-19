# v2: buying back the generalization gap

*2026-07-19 — four sources, ratified guidelines, and an 11.7-point jump on unseen entity names*

## The mixture

v2 rebuilt the training set around everything the audits taught us —
389,521 examples from four sources:

| Source | Share | Role |
|---|---|---|
| gravitee-io/pii-detection-dataset | 44.6% | format diversity (HTML, JSON, logs) |
| nvidia/Nemotron-PII | 25.7% | clean PII documents, date family auto-corrected |
| ai4privacy/pii-masking-openpii-1.5m (English slice) | 28.2% | independent generator, 33-label schema |
| Synthetic register docs (batch-generated) | 1.5% (upsampled ×3) | JSON logs, checkbox forms, prose registers |

Mechanisms added over v1:

- **Wide alias sampling**: ~22 surface forms per label (up from 2–4), the
  direct counter to the measured 19.7-point unseen-name gap.
- **Guideline conditioning**: 30% of examples carry short per-label rules in
  the request ("date: a calendar date with no time component"), with targets
  relabeled to obey the stated rule — annotation convention converted from
  silent stance to explicit instruction.
- **Deterministic gold repair**: every `date_time` span lacking a clock time
  relabeled to `date` (a rule, not a judgment — no LLM needed).
- **Containment-family negatives**: requests engineered to land between
  ontology levels (country requested, only states present) where abstention
  must be learned.
- **Family-aware subsets**: when `date_of_birth` is dropped from a request
  that keeps `date`, birthdate spans become `date` — a birthdate is still a
  date.
- The third dataset's schema was *not* unified with the others: each example
  carries its own label vocabulary, and the conditioning disambiguates.
  Cross-source synonym collisions are training signal, not noise.

Training: Qwen3-0.6B, full fine-tune, one epoch, 12,142 steps on an H100 NVL
(~11 hours). Every 500-step checkpoint pushed to the Hugging Face hub, which
makes the run instance-proof and — as a bonus — preserves the entire training
trajectory as git history, so per-checkpoint capability curves can be
reconstructed retroactively on cheap hardware.

## Results, on three axes

One inference procedure, three measuring sticks:

| Axis | v1-full | v2 | Δ |
|---|---|---|---|
| Original gold (noisy) | 0.944 | 0.930 | −1.4 |
| Corrected gold (adjudicated) | 0.924 | 0.918 | ~flat |
| **Novel entity names** | 0.747 | **0.864** | **+11.7** |

Copy drift under constrained decoding: 0.0% on both eval runs.

The reading, axis by axis:

- **Original gold going down is the design working.** v2 was deliberately
  trained to disagree with that gold's noise; the date-family repairs alone
  contradict thousands of its annotations. A model that improved on this axis
  would be re-learning the errors.
- **Corrected gold flat is the honest mixed result.** We removed one noise
  family deterministically but also diluted the eval's home distribution from
  37% to 26% of the mixture, and the fuzzy noise (occupation boundaries)
  remains in training. True accuracy held; it didn't climb.
- **The novel-name axis is where the mixture design aimed, and where it
  landed.** The unseen-name gap shrank from 19.7 points to 6.6. Free-form
  prompting — the product goal — moved from "degraded mode" to "nearly
  native." One stubborn floor remains: paraphrases outside any alias set's
  semantic reach ("mail routing code" for postcode) still score zero, which
  no amount of alias widening fixes; definition-in-prompt conditioning is the
  candidate remedy.

## The scoreboard, start to finish

| Milestone | Strict F1 (original gold) | Novel names |
|---|---|---|
| 120B zero-shot | 0.580 | — |
| 0.6B SFT, 50k examples | 0.873 | — |
| + constrained decoding | 0.898 | 0.747 |
| 0.6B SFT, 273k examples | 0.944 | — |
| **v2: 389k, 4 sources, guidelines, wide aliases** | 0.930 | **0.864** |

Total compute spend across every experiment in this series — five training
runs, three model sizes, all evaluations, all failures included: roughly $85
of rented GPU time and $7.50 of batch-mode frontier labor. The scoreboard's
most important column changed meaning along the way, which may be the realest
lesson of all: the work of evaluation *is* the work.
