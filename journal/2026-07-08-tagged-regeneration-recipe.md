# A $2.50 fine-tune that beats a 120B model: the tagged-regeneration recipe

*2026-07-08 — the founding experiment of the SPANIEL project*

## The task, and why generative

The goal: **unconstrained PII extraction** — hand the model a document and a
free-form list of entity type names, get back the document with matching spans
wrapped in XML tags. Not a fixed taxonomy: the user should be able to type
`patient mrn` or `insurance id` and have it work.

A GLiNER-style encoder wins on latency-per-parameter for fixed schemas, but we
chose the generative route deliberately: the recipe rides three compounding
curves (better small base models, better post-training tooling, better
structured-output serving), and the tagged-text output format is the natural
shape for redaction — grounded, diffable against the source, and unambiguous
when a value appears multiple times.

## The format

Every training example is a chat triple:

- **system** — a fixed instruction: reproduce the text exactly, wrap matching
  entities in tags, use only the requested labels.
- **user** — `Entity types:` followed by the requested type names, then the
  document.
- **assistant** — the document, byte-identical, with `<label>entity</label>`
  tags inserted.

Two design decisions carry most of the value:

1. **The label set is an input, not a vocabulary.** Each example's request
   lists the types to extract; the model learns "find spans matching these
   descriptions," not a fixed head. Novel types at inference are just new
   lines in the request.
2. **Label names are sampled from alias sets.** `date_of_birth` appears in
   training as "date of birth", "dob", "birth date", "birthdate" — so the
   model cannot memorize request strings as opaque symbols and must route
   through meaning. This is the mechanism behind free-form generalization.

The request mix per example: 40% full source vocabulary, 40% labels present
plus 2–6 sampled negatives (teaching abstention), 20% strict subsets with the
dropped labels' tags removed (teaching "extract what is asked, not what you
recognize").

## Data

Two public datasets, converted from (text, char-span) format:
[gravitee-io/pii-detection-dataset](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset)
(~175k rows, 25 coarse labels, noisy) and
[nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII)
(200k rows, 55 fine labels, synthetic and clean). Junk labels (gravitee's
`FINANCIAL`, `TITLE`, `NRP`) were dropped after inspection. Aliases that
collide with literal tags in the source text (gravitee contains raw XML
documents) are resampled at build time so targets stay unambiguous.

## Results

Training: TRL SFT, full fine-tune, bf16, one epoch on a 50k subsample, single
rented GPU. Scoring: strict span-level exact-match F1, where malformed or
copy-drifted outputs forfeit all their entities (an output you can't parse is
an output that found nothing).

| Model | Strict F1 | Copy drift | Cost |
|---|---|---|---|
| gpt-oss-120b, zero-shot, same prompt | 0.580 | 45.3% | — |
| Qwen3-1.7B SFT | 0.872 | 6.7% | ~$2.50 |
| **Qwen3-0.6B SFT** | **0.873** | 4.7% | **~$1.80** |

Two findings worth the price of the experiment:

**The 120B model's failure is copy fidelity, not extraction.** On rows where
it held the copy contract it scored 0.891 — but it drifted on 45% of rows,
usually by absorbing in-text field names into its tags ("the user name:
lkernan" → `the <user name>lkernan</user name>`). The task was never
"find PII"; it was "find PII *while copying faithfully*," and that's what
fine-tuning buys.

**There is no capability cliff between 1.7B and 0.6B here.** The models tie
(0.873 vs 0.872), which means the task is bounded by data and format, not
parameters — and the 0.6B becomes the default: sub-1B, ~2.5× faster locally,
and cheap enough to ablate freely.

Later entries chart what happened when we stopped trusting the training
labels, made copy fidelity a mathematical guarantee, and measured what
"free-form" really costs.
