---
license: apache-2.0
base_model: Harsh/qwen3-0.6b-pii-sft-v2
base_model_relation: adapter
library_name: peft
language:
  - en
pipeline_tag: text-generation
datasets:
  - nvidia/Nemotron-PII
  - gravitee-io/pii-detection-dataset
  - ai4privacy/pii-masking-openpii-1.5m
tags:
  - spaniel
  - ai-governance
  - ai-security
  - sensitive-data-detection
  - credential-detection
  - pii-detection
  - named-entity-recognition
  - data-loss-prevention
  - local-inference
  - constrained-decoding
  - lora
  - qwen3
  - synthetic-data
  - experimental
---

# Spaniel Context v0.1 — Qwen3-0.6B credential and compensation LoRA

**Applied AI governance and AI security research by [Harsh Singhal](https://www.linkedin.com/in/harshsinghal/).**

This adapter investigates whether a 0.6B local language model can distinguish sensitive credentials from benign lookalikes and private compensation from public salary information. It continues supervised fine-tuning of [Spaniel v2](https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2), using source-grounded synthetic contrasts and validated PII replay. It is an experimental artifact, not a production disclosure-prevention guarantee.

**Experimental release: training and paired evaluation are complete. The adapter improves the authored synthetic test but fails its predeclared reliability screens. Spaniel v2 remains the default. Only the corrected selected weights are published.**

## Task and policy

| Label | Target | Exclusions under this experiment's policy |
|---|---|---|
| `password` | Actual or purported passwords, ordinary-word passphrases, revoked secrets | Unfilled placeholders, masked values, ordinary titles or phrases |
| `api key` | API secrets, bearer/session tokens, authentication values in URLs or JSON | Public identifiers, unfilled fields, token-format explanations |
| `private key` | Complete private-key blocks, including delimiters | Public verification keys and format descriptions |
| `private compensation` | Individual pay, private negotiations, nonpublic compensation budgets, including ranges | Public job advertisements, public market data, authorized public pay disclosures |

Context matters. The same amount can describe a public job advertisement or someone's private offer. A private conversation quoting a public advertisement does not, by itself, make that quoted range private compensation. A literal `CHANGEME` can be sensitive when someone actually uses it as their password.

These are application conventions, not legal determinations. The model does not check token validity, cryptographic validity, authorization or whether a value is a working credential. Missing context has no calibrated abstention output. An unchanged annotation is not evidence that ambiguous content is safe.

Public/private status in these synthetic cases is asserted in the text. The model cannot authenticate a document's claim that information is public or approved for disclosure. Applications must obtain access rights and disclosure permissions from trusted policy and metadata; the extractor does not supply them.

The frozen concept evaluation requests these exact labels one at a time. `private compensation` is not equivalent to a request for every `salary` mention. The new concept results do not establish synonym robustness or performance with many simultaneous labels; a separate retrieval smoke test requests all three credential labels together.

## Results and release status

The completed frozen test contains 159 authored synthetic documents from 17 scenario families: 81 sensitive and 78 benign. All documents are processed without truncation. Baseline and adapter use matched constrained decoding and runtime configuration on the same GPU.ai RTX 4090 within each suite.

| Scope | Model | Sensitive documents detected | Benign false blocks | Exact span F1 | Inspection errors |
|---|---|---:|---:|---:|---:|
| Credentials | Spaniel v2 | 24/39 (61.5%) | 7/36 (19.4%) | 0.579 | 0 |
| Credentials | Spaniel Context v0.1 | 33/39 (84.6%) | 3/36 (8.3%) | 0.889 | 0 |
| Compensation | Spaniel v2 | 10/42 (23.8%) | 6/42 (14.3%) | 0.167 | 0 |
| Compensation | Spaniel Context v0.1 | 37/42 (88.1%) | 6/42 (14.3%) | 0.851 | 0 |


Across all concept cases, including both frozen off-the-shelf GLiNER2.5 configurations:

| Model | Sensitive documents detected | Benign false blocks | Exact span F1 | Inspection errors |
|---|---:|---:|---:|---:|
| Spaniel v2 | 34/81 (42.0%) | 13/78 (16.7%) | 0.397 | 0 |
| Spaniel Context v0.1 | 70/81 (86.4%) | 9/78 (11.5%) | 0.869 | 0 |
| GLiNER2.5, labels | 70/81 (86.4%) | 54/78 (69.2%) | 0.421 | 0 |
| GLiNER2.5, policy descriptions | 72/81 (88.9%) | 67/78 (85.9%) | 0.415 | 0 |


![Context comparison](https://huggingface.co/Harsh/spaniel-credential-context-lora-v0.1/resolve/main/context-comparison.png)

GLiNER uses its native extractor at threshold 0.5 on CPU, without task fine-tuning. Both labels and policy descriptions were fixed before prediction. This is not an equal-data architecture comparison or a latency benchmark. Detailed reports, environment records and hashes are included in the repository's `evidence/` directory.

On the 300-document original-source PII regression, exact typed span F1 is **0.937 for v2 and 0.924 for the adapter** (-1.26 percentage points). Inspection errors are 1 and 1, respectively. Every legacy document contains requested entities, so this set cannot measure benign false blocks. It uses original source annotations, not the separately adjudicated historical labels behind v2's 0.918 score.

| Frozen screening requirement | Observed | Outcome |
|---|---:|---|
| Credential document recall ≥95% | 84.6% | Fail |
| Credential benign false blocks ≤5% | 8.3% | Fail |
| No candidate concept/legacy inspection errors | 1 | Fail |
| Legacy F1 decline ≤1 percentage point | -1.26 percentage points | Fail |


The candidate **fails all four predeclared screening requirements**. Credential recall and false blocks miss their targets; original-source PII span F1 declines from **0.937 to 0.924**, beyond the allowed one-point drop; and one legacy annotation is rejected in each model. It remains a separate experimental adapter; v2 stays the default. These are 159 authored synthetic cases across 17 scenario families, with correlated substitutions and long variants, no independent human adjudication, and a disclosed runtime-repair rerun. They establish progress on this test, not production readiness or a general architecture ranking.

The remaining failures are concrete. The adapter finds only 4/9 sensitive recovery-conversation variants, misses all three sensitive range-correction-call cases, falsely blocks all three authorized-public-pay cases, and blocks both benign README-variable cases. The separate application suite also misses an actual password equal to `CHANGEME` and falsely blocks a public job range. These failures prioritize new independently authored benign examples, indirect login language, corrected spoken ranges and trusted disclosure context.

The controlled long-context slice improves from 7/30 to 23/30 sensitive documents detected, with benign false blocks rising from 0/30 to 2/30; exact span F1 rises from 0.216 to 0.836. The 24-case transcript/noise slice improves from 6/12 to 9/12 sensitive documents detected, with false blocks falling from 3/12 to 1/12. These small synthetic slices do not validate enterprise documents or real ASR.

Correct decisions on both members of a sensitive/benign contrast pair rise from 25/78 to 59/78. For future training, these observed test families must be treated as development evidence and assessed against a newly reserved test; this release did not retrain on its failures.

Actual Apple MPS retrieval checks improved from 12/15 to 14/15 on existing examples but declined from 9/11 to 8/11 on new context examples. The latter newly blocks public job ranges and authorized pay, and releases an actual `CHANGEME` password. Six ambiguous cases are observations only, without binary gold or a calibrated review action. The adapter must not be treated as a reliable disclosure-prevention control on the basis of aggregate F1.

### Runtime-repair disclosure

The first training/evaluation run loaded a newer Qwen3 RoPE configuration incorrectly in an older runtime. That entire run was invalidated and its weights were never released publicly. Corrected training restarted from the original pinned base with the same data, hyperparameters, selection rule and screening thresholds. Some reserved-test predictions had already been observed under the invalid runtime; no training or labeling decisions were changed in response. This is a disclosed repair rerun, not an untouched multi-run benchmark. The [investigation](https://github.com/harshsinghal/SPANIEL/blob/main/docs/runtime-compatibility.md) records the verification of the actual positional-frequency tensor.

## Training provenance

| Property | Value |
|---|---|
| Base checkpoint | `Harsh/qwen3-0.6b-pii-sft-v2` |
| Base revision | `a58063469be8c7b929016d944d8d6d1545bc70a0` |
| Base weights SHA-256 | `3ce13010b5f0430e188be1261654cc41d7f05eafc6cb1ebffa6576c5e020dc6b` |
| Training provider | GPU.ai, RTX 4090, 24 GB |
| Method | Continued SFT with rank-16 LoRA, alpha 32, dropout 0.05 |
| Targets | Attention q/k/v/o and feed-forward gate/up/down projections |
| Trainable parameters | 10,092,544 |
| Recipe | Two epochs, learning rate 5e-5, effective batch 16, bf16, seed 20261021 |
| Sequence handling | 4,096-token limit; maximum observed 2,201; overflow rejected, no truncation |
| Loss | Assistant completion and chat ending only; all prompt tokens masked |
| Checkpoint selection | Epoch one, step 424; development completion loss 0.025829 versus 0.031610 at epoch two; no test-set selection |
| Adapter weights SHA-256 | `dfb60b5f56dea06094a73e9de7d8175d70ea9eb252ae8db0cb3ac3721077b504` |
| Adapter weights revision | `6cd01c4f621c1528f00c5f70f9f1c7855f0a434a` |

The v2 base was originally fully fine-tuned from Qwen3-0.6B. This experiment uses a LoRA adapter on that already trained model; it is not an adapter trained directly on the original Qwen checkpoint. The combined augmentation/replay experiment does not isolate either component or establish that LoRA is better than full fine-tuning.

## Data, coverage and attribution

The new deterministic generator produces 5,248 training examples, 100 development examples, 159 frozen test examples and six separate ambiguity-review cases. The 34 training, nine development and 17 test scenario families are disjoint. All substitutions and long variants of one family stay in one split. Values are fictional and non-operational; private-key bodies reproduce textual shapes, not valid keys. Template substitutions are correlated examples and have no independent human gold review.

The training mixture adds 1,536 validated original-training replay rows: 512 each from NVIDIA Nemotron-PII, Gravitee and Ai4Privacy. No examples from the unpublished fourth source are newly replayed, but the v2 base was already trained on that source. Stricter source/span validation rejects 64 historical targets from replay eligibility; the original corpus and v2 weights are unchanged. The original 300-document PII regression set is separate from training/replay.

Credit [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) (CC BY 4.0), [Gravitee PII Detection](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset) (Apache-2.0, with its upstream attributions), and [Ai4Privacy / Ai Suisse SA OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (CC BY 4.0; Copyright © 2026 Ai Suisse SA as stated in its card). SPANIEL converts source annotations into exact-source tagged completions, validates them, excludes incompatible targets and samples replay. Preserve source-specific attribution and notices; the model license does not replace dataset terms. These sources do not imply endorsement.

The [source-attribution record](https://github.com/harshsinghal/SPANIEL/blob/main/docs/training-data-and-evaluation.md#source-attribution-and-license-record) contains the exact source revisions, upstream creators, modifications and license links. [Data hashes](https://github.com/harshsinghal/SPANIEL/blob/main/reports/context-manifest.json), the [coverage audit](https://github.com/harshsinghal/SPANIEL/blob/main/docs/coverage-audit.md), [policy definitions](https://github.com/harshsinghal/SPANIEL/blob/main/reports/context-policy.json) and [experiment evidence](https://github.com/harshsinghal/SPANIEL/tree/main/reports/credential-context) trace the changes from corpus inspection to training and evaluation.

## Local inference and governed retrieval

Use the matching v2 base and this adapter with the SPANIEL source server. Install the repository's `requirements-adapter.txt` with a PyTorch build appropriate to your machine. Configure `PII_MODEL_ID`, `PII_MODEL_REVISION`, `PII_ADAPTER` and `PII_ADAPTER_REVISION` as documented in the milestone. The service merges the adapter in memory and keeps the same copy-or-tag constrained decoder.

The inference and experiment code is available at [source revision `506e083`](https://github.com/harshsinghal/SPANIEL/tree/506e0833087cb7861b1b42c71d455866ba92ac1f). This revision includes the adapter loader, constrained decoder, data generator and evaluators; later documentation commits record the completed results. The existing Docker image does not include the optional PEFT dependency. Use a source installation for this adapter.

```bash
git clone https://github.com/harshsinghal/SPANIEL.git
cd SPANIEL
git checkout 506e0833087cb7861b1b42c71d455866ba92ac1f
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-adapter.txt

export PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2
export PII_MODEL_REVISION=a58063469be8c7b929016d944d8d6d1545bc70a0
export PII_MODEL_CACHE="$PWD/.model-cache"
export PII_ADAPTER=Harsh/spaniel-credential-context-lora-v0.1
# Pin the selected weights; later card edits do not change this artifact.
export PII_ADAPTER_REVISION=6cd01c4f621c1528f00c5f70f9f1c7855f0a434a
python -m uvicorn server:app --app-dir pii_tagger --host 127.0.0.1 --port 8377
```

Use a CUDA-compatible PyTorch installation on an NVIDIA host. The recorded training/evaluation environment used PyTorch 2.13.0+cu126; the separate application checks used Apple MPS. Record `/api/model` and the adapter hash when reproducing an application result.

In a second terminal, run `python examples/context_retrieval/demo.py --report context-smoke.json` from the checkout with its environment activated. The demo returns a nonzero status when expectations are missed. Do not interpret a successful service startup as a passed detector evaluation.

The intended integration is a local retrieval tool: read the document inside the trusted tool, call SPANIEL, validate exact source recovery and nonempty requested spans, and return `content: null` on a policy match or inspection failure. Only the captured original text is returned when no match is found. The model's tagged output still contains sensitive values; it is not redacted text.

Completed constrained annotations preserve the source. This requires the project decoder plus completion validation; unconstrained text generation does not provide the guarantee. Correct structure does not imply correct detection, and a missed entity can still cause the retrieval tool to release sensitive content.

## Limits and next evidence

- English is the intended scope. No multilingual capability is established.
- Transcript cases are simulated text with colloquial wording and transcription-like noise. No audio, real ASR system or real transcript corpus is evaluated.
- Long-context cases use 250 or 900 words of synthetic public-information filler. The 60 long test variants come from five families; the maximum document length is 936 words. These are not representative enterprise documents.
- Benign scenarios are unevenly represented after deduplication. Independent benign examples and annotation review remain priorities.
- The format supports non-overlapping contiguous spans of at least two source characters. Linked/discontinuous evidence groups are a data sidecar, not a model output capability.
- Legal strategy, layoffs, performance reviews and broad business confidentiality remain unvalidated research targets. Supplying an arbitrary entity label does not confer a validated capability.
- No calibrated uncertainty or general adversarial-security claim is established. An initial GLiNER2.5 off-the-shelf baseline uses the same scenarios; it is not an equal-data architecture comparison or a speed benchmark.
- Local inference alone does not control an application's logs, traces, networking, alternate tools or authorization decisions.

## Author and research record

[Harsh Singhal on GitHub](https://github.com/harshsinghal) · [Hugging Face](https://huggingface.co/Harsh) · [SPANIEL repository](https://github.com/harshsinghal/SPANIEL).

Suggested citation: **Harsh Singhal. SPANIEL: Local PII Detection for AI Prompts — Credential and Context Experiment. 2026.** Include the adapter, base, code and dataset revisions used.
