# SPANIEL 🐕

**SPAN Identification from Everyday Language** — a small retriever that
fetches any span you name.

A 0.6B-parameter PII extraction model that accepts
**free-form entity type names**, runs locally, and is structurally incapable
of altering the text it annotates.

Give it a document and a list of types — including types it has never seen,
like `patient mrn` or `insurance id` — and it returns the document
byte-identical with matching spans wrapped in XML tags:

```
Entity types:
- person name
- patient mrn
- insurance id

Text:
Patient Brian Weaver (MRN BX-40912) called about his appointment...

→ Patient <person name>Brian Weaver</person name> (MRN <patient mrn>BX-40912</patient mrn>) called...
```

**Repo**: [github.com/harshsinghal/SPANIEL](https://github.com/harshsinghal/SPANIEL) ·
**Models**: [qwen3-0.6b-pii-sft-v2](https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2)
(current) · earlier checkpoints: v1-full, v1-50k, and a 1.7B variant.

## Headline results

Strict span-level exact-match F1, 300-document held-out eval, constrained decoding:

| | Trained names | Adjudicated gold | **Unseen names** |
|---|---|---|---|
| gpt-oss-120b zero-shot | 0.580 | — | — |
| v1 (0.6B, 273k examples) | 0.944 | 0.924 | 0.747 |
| **v2 (0.6B, 389k, 4 sources)** | 0.930 | 0.918 | **0.864** |

Copy drift and malformed output under the constrained decoder: **0.0%** — not
measured low, but grammatically impossible.

## The journey, as blog entries

1. **[A $2.50 fine-tune that beats a 120B model](journal/2026-07-08-tagged-regeneration-recipe.md)**
   — the tagged-regeneration format, label conditioning, alias sampling, and
   the discovery that 0.6B ties 1.7B on this task.
2. **[What a sushi chain taught our PII model](journal/2026-07-09-attribute-vs-mention-semantics.md)**
   — attribute vs. mention semantics, the questions a dataset silently
   answers, and how fine-tuning mixtures should account for hidden semantic
   axes. Includes a seminal-papers reading list.
3. **[Making drift unrepresentable](journal/2026-07-12-constrained-decoding.md)**
   — the copy-or-tag automaton, the vocabulary trie, and two constrained-
   decoding bugs (pending tags, BPE merge closure) that generalize to any
   grammar-constrained system.
4. **[The $10 audit](journal/2026-07-18-ten-dollar-audit.md)**
   — frontier models in batch mode: our gold was 9% wrong, our model had
   learned the noise, nano models can't judge, and generalization costs
   19.7 points — measured, then bought back.
5. **[v2: buying back the generalization gap](journal/2026-07-19-v2-generalization.md)**
   — the four-source mixture, ratified guidelines, and +11.7 on unseen
   entity names.

## Repository layout

| Path | What |
|---|---|
| `spans_to_xml.py` | span → tagged-text conversion (the core representation) |
| `convert_pii_datasets.py`, `convert_ai4privacy.py` | dataset converters |
| `build_sft_dataset.py`, `build_sft_dataset_v2.py` | training-mixture builders (seeded, reproducible) |
| `label_aliases.json`, `label_aliases_v2.json` | alias sets per label — the generalization mechanism |
| `guidelines_v2.md` | canonical annotation rules, human-ratified |
| `pii_decode.py` | constrained decoder: vocabulary trie + copy-or-tag automaton |
| `pii_eval.py` | span-level scorer (strict/relaxed, per-label, operational failure rates) |
| `error_autopsy.py` | disagreement classifier (boundary / miss / swap / spurious) |
| `run_constrained_eval.py`, `run_baseline.py` | evaluation runners |
| `build_batch_jobs.py` | OpenAI batch-mode job builder (adjudication, audit, aliases, synthesis) |
| `pii_tagger/` | local web app: FastAPI + the model on Apple Silicon, constrained by default |
| `vast_run/` | GPU training scripts (TRL SFT, env-configurable) |
| `journal/` | the blog entries above |

Large artifacts (converted datasets, training mixtures, model weights,
predictions) are intentionally not in the repo; every one is reproducible
from the seeded builders or downloadable from the HF hub.

## Try it: the demo app

One command, runs entirely on your machine — no data leaves your device:

```bash
docker run -p 8377:8377 -v spaniel-models:/models ghcr.io/harshsinghal/spaniel
# open http://localhost:8377
```

The model (~1.2GB) is pulled from the Hugging Face hub on first start and
cached in the named volume. On Linux with an NVIDIA GPU add `--gpus all`;
on Mac/Windows it runs on CPU (a demo paragraph takes ~10-30s — the price of
local-only). The UI ships with **15 preloaded examples** — medical forms,
server logs, transcripts, invoices, an encyclopedic-prose case that
demonstrates the attribute-semantics stance — each with editable free-form
entity types. Every response is generated under the constrained decoder:
the "copy-faithful" badge is a guarantee, not a check.

Running from source instead:

```bash
cd pii_tagger && python -m venv .venv && .venv/bin/pip install torch transformers fastapi uvicorn huggingface_hub
PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2 .venv/bin/python -m uvicorn server:app --app-dir . --port 8377
```

## Reproducing

```bash
# convert sources and build the v2 mixture (seeded — byte-identical rebuilds)
python convert_ai4privacy.py --split train --out pii_xml/ai4privacy_en.jsonl
python build_sft_dataset_v2.py --outdir sft_data_v2

# train (any CUDA box; ~11h on one H100 NVL)
SFT_MODEL=Qwen/Qwen3-0.6B SFT_BS=16 SFT_ACCUM=2 python vast_run/train_sft.py

# evaluate with the constrained decoder
python run_constrained_eval.py --model <model_dir> --n 300 --out preds.jsonl
python pii_eval.py --pred preds.jsonl --n 300
```

## Status & roadmap

Active. Next: retroactive per-checkpoint capability curves (the hub history
preserves every 500-step snapshot), a size/architecture grid (Granite-350M
hybrid, Gemma-270M), and an extension from PII entities to sensitive-topic
spans (compensation/pricing/RIF discussions in call transcripts) via
reasoning-SFT + RL with the span scorer as verifiable reward.
