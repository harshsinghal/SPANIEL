<p align="center">
  <img src="docs/spaniel_pic.png" width="340" alt="SPANIEL — SPAN Identification from Everyday Language">
</p>

<h1 align="center">SPANIEL</h1>

<p align="center">
  <strong>SPAN Identification from Everyday Language</strong><br>
  <em>A small retriever that fetches any span you name.</em>
</p>

<p align="center">
  <a href="https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2">Model</a> ·
  <a href="#try-it-the-demo-app">Demo</a> ·
  <a href="#results">Results</a> ·
  <a href="#the-journey-as-blog-entries">Blog series</a>
</p>

---

SPANIEL is a 0.6B-parameter PII extraction model that accepts **free-form
entity type names**, runs entirely on your own hardware, and is structurally
incapable of altering the text it annotates.

Give it a document and a list of types — including types it has never seen,
like `patient mrn` or `severance amount` — and it returns the document
byte-identical, with matching spans wrapped in XML tags:

```text
Entity types:
- person name
- patient mrn

Text:
Patient Brian Weaver (MRN BX-40912) called about his appointment...
```

```text
Patient <person name>Brian Weaver</person name> (MRN <patient mrn>BX-40912</patient mrn>) called about his appointment...
```

<p align="center">
  <img src="docs/spaniel-demo.gif" width="720" alt="SPANIEL demo: loading examples, tagging entities, copy-faithful output">
</p>

## Why this exists

Two reasons, one personal and one technical.

The personal one: [Harsh Singhal](https://www.linkedin.com/in/harshsinghal/)
wanted to scratch a long-standing itch — build a serious NER system on a
*small* language model, end to end: data curation, fine-tuning, evaluation
design, decoding, and serving, with every decision made deliberately rather
than inherited. SPANIEL is that itch, scratched in public. The
[journal entries](#the-journey-as-blog-entries) document every experiment,
including the failures.

The technical one: [GLiNER](https://github.com/urchade/GLiNER) showed that
open-type NER — *name any entity type at inference* — is possible with a
compact model, and it remains the accuracy-per-millisecond benchmark for
fixed-schema extraction. SPANIEL is inspired by that idea but takes the
generative route to chase two properties an encoder can't offer: **tagged
full-text output** (grounded, diffable, redaction-ready — with a constrained
decoder making copy drift impossible for completed generations), and a recipe that
**rides the small-LLM ecosystem** — every year's better 0.6B base model is a
one-line upgrade. The bet, now measured: a $2.50 fine-tune of a 0.6B model
beats a 120B model prompted zero-shot by 35 F1 points on this task.

## Try it: the demo app

One command, local-only — no data leaves your machine:

```bash
docker run -p 8377:8377 -v spaniel-models:/models ghcr.io/harshsinghal/spaniel
# open http://localhost:8377
```

- The model (~1.2 GB) is pulled from the Hugging Face hub on first start and
  cached in the named volume.
- The published image is CPU-only on every platform (a demo paragraph takes
  ~10–30 s). Run from source with a CUDA build of PyTorch for GPU inference.
- The UI ships with **15 preloaded examples** — medical forms, server logs,
  transcripts, invoices, and an encyclopedic-prose case that demonstrates the
  attribute-semantics stance — each with editable free-form entity types.
- Every successful response is generated under the constrained decoder. The
  server rejects an incomplete generation instead of returning a false
  **copy-faithful ✓** result.

<details>
<summary><strong>Running from source instead</strong></summary>

```bash
python3 -m venv .venv
.venv/bin/pip install torch -r requirements-demo.txt
PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2 \
PII_MODEL_REVISION=a58063469be8c7b929016d944d8d6d1545bc70a0 \
  .venv/bin/python -m uvicorn server:app --app-dir pii_tagger --port 8377
```

For CUDA, install the PyTorch wheel appropriate to the host before installing
`requirements-demo.txt`.

</details>

## Results

Strict span-level exact-match F1 · 300-document held-out eval · constrained decoding.

| Model | Trained names | Adjudicated gold | Unseen names |
|:--|:-:|:-:|:-:|
| gpt-oss-120b, zero-shot | 0.580 | — | — |
| v1 (0.6B, 273k examples) | 0.944 | 0.924 | 0.747 |
| **v2 (0.6B, 389k, 4 sources)** | 0.930 | 0.918 | **0.864** |

- Copy drift and malformed output under the constrained decoder: **0.0%** —
  incomplete generations are treated as failures, never as faithful output.
- The 120B number is an end-to-end tagged-regeneration score: 141/300 of its
  rows drifted or were malformed. Its conditional F1 on parseable rows was
  0.891, separating semantic extraction quality from output-contract reliability.
- The three columns are three measuring sticks over one inference procedure:
  the original (noisy) gold, an adjudicated and human-spot-checked gold, and
  requests phrased with entity names absent from all training aliases.
- v2's −1.4 on the first column is by design (it un-learned annotation
  noise); its +11.7 on the last column is the point.

## The journey, as blog entries

| # | Entry | One-line summary |
|:-:|:--|:--|
| 1 | [A $2.50 fine-tune that beats a 120B model](journal/2026-07-08-tagged-regeneration-recipe.md) | The tagged-regeneration format, label conditioning, alias sampling — and 0.6B tying 1.7B. |
| 2 | [What a sushi chain taught our PII model](journal/2026-07-09-attribute-vs-mention-semantics.md) | Attribute vs. mention semantics; the questions a dataset silently answers. With reading list. |
| 3 | [Making drift unrepresentable](journal/2026-07-12-constrained-decoding.md) | The copy-or-tag automaton, the vocabulary trie, and two constrained-decoding bugs that generalize. |
| 4 | [The $10 audit](journal/2026-07-18-ten-dollar-audit.md) | Frontier models in batch mode: gold was 9% wrong, the model had learned the noise, nano models can't judge. |
| 5 | [v2: buying back the generalization gap](journal/2026-07-19-v2-generalization.md) | Four sources, ratified guidelines, +11.7 on unseen entity names. |
| 6 | [Teaching a small model to think — four experiments, two sizes](journal/2026-07-21-thinking-negative-result.md) | Imitation, RL, a 3× bigger model, and a reasoning specialist. Size breaks the 0.6B wall; thinking still doesn't beat direct answering; and a math-reasoning specialist is *worse*, not better — reasoning skill doesn't transfer across domains. |

## Repository layout

| Path | What |
|:--|:--|
| `spans_to_xml.py` | Span → tagged-text conversion (the core representation) |
| `convert_pii_datasets.py` · `convert_ai4privacy.py` | Dataset converters |
| `build_sft_dataset.py` · `build_sft_dataset_v2.py` | Seeded, reproducible training-mixture builders |
| `label_aliases.json` · `label_aliases_v2.json` | Alias sets per label — the generalization mechanism |
| `guidelines_v2.md` | Canonical annotation rules, human-ratified |
| `pii_decode.py` | Constrained decoder: vocabulary trie + copy-or-tag automaton |
| `pii_eval.py` | Span-level scorer (strict/relaxed, per-label, operational failure rates) |
| `error_autopsy.py` | Disagreement classifier (boundary / miss / swap / spurious) |
| `run_constrained_eval.py` · `run_baseline.py` | Evaluation runners |
| `build_batch_jobs.py` | OpenAI batch-mode jobs (adjudication, audit, aliases, synthesis) |
| `pii_tagger/` | The local web app (FastAPI, constrained by default) |
| `vast_run/` | GPU training scripts (TRL SFT, env-configurable) |
| `tests/` | Fast regression tests for decoding, scoring, and reward invariants |
| `artifact_manifest.json` | Source revisions plus row counts and SHA-256 checksums |
| `journal/` | The blog entries above |

Large artifacts (converted datasets, training mixtures, model weights,
predictions) are intentionally not in the repo. Their expected hashes and
upstream revisions are recorded in `artifact_manifest.json`.

## Reproducing

```bash
# 0. Create an environment. Install the appropriate CUDA PyTorch wheel
# separately before the training requirements.
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt

# 1. Rebuild the original held-out eval set and the v2 mixture.
.venv/bin/python build_sft_dataset.py --outdir sft_data
.venv/bin/python convert_ai4privacy.py \
  --split train --out pii_xml/ai4privacy_en.jsonl
.venv/bin/python build_sft_dataset_v2.py --outdir sft_data_v2

# The published four-source mixture requires pii_xml/synth_docs.jsonl, the
# 1,966-document batch-generated artifact. Use --skip-synth for the fully
# public three-source mixture when that artifact is unavailable.

# 2. Verify byte identity against the published artifacts.
.venv/bin/python verify_artifacts.py

# 3. Train (any CUDA box; ~11 h on one H100 NVL).
.venv/bin/pip install -r requirements-train.txt
SFT_MODEL=Qwen/Qwen3-0.6B \
SFT_TRAIN=sft_data_v2/sft_train_v2.jsonl \
SFT_VAL=sft_data/sft_eval.jsonl \
SFT_RUN_DIR=runs/spaniel-v2 \
SFT_BS=16 SFT_ACCUM=2 \
  .venv/bin/python vast_run/train_sft.py

# 4. Evaluate with the constrained decoder.
.venv/bin/python run_constrained_eval.py \
  --model runs/spaniel-v2/final --n 300 --out preds.jsonl
.venv/bin/python pii_eval.py --pred preds.jsonl --n 300

# Fast local validation that does not download models or datasets.
make test
```

## Status & roadmap

Active. Next up:

- **Per-checkpoint capability curves** — the hub history preserves every
  500-step snapshot; a retroactive sweep charts when each capability emerged
  during training.
- **Size / architecture grid** — Granite-4.0-H-350M (hybrid Mamba-2) and
  Gemma-3-270M on the identical recipe: where is the capability floor, and do
  SSM hybrids hold up on constrained extraction?
- **From entities to topics** — extending the span vocabulary from PII to
  sensitive *discussions* (compensation, pricing, layoffs) in call
  transcripts, via reasoning-SFT + RL with the span scorer as verifiable
  reward.
