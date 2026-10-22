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

**Draft: corrected training is in progress. The first run was invalidated by a RoPE configuration mismatch. Do not publish this draft or the invalid adapter as a completed-model release.**

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

Candidate results are pending. The original Spaniel v2 PII scores do not establish performance on these new concepts. Do not infer an accuracy gain from the existence of this adapter recipe.

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
| Checkpoint selection | Lowest development completion loss; no test-set checkpoint selection |

The v2 base was originally fully fine-tuned from Qwen3-0.6B. This experiment uses a LoRA adapter on that already trained model; it is not an adapter trained directly on the original Qwen checkpoint. The combined augmentation/replay experiment does not isolate either component or establish that LoRA is better than full fine-tuning.

## Data, coverage and attribution

The new deterministic generator produces 5,248 training examples, 100 development examples, 159 frozen test examples and six separate ambiguity-review cases. The 34 training, nine development and 17 test scenario families are disjoint. All substitutions and long variants of one family stay in one split. Values are fictional and non-operational; private-key bodies reproduce textual shapes, not valid keys. Template substitutions are correlated examples and have no independent human gold review.

The training mixture adds 1,536 validated original-training replay rows: 512 each from NVIDIA Nemotron-PII, Gravitee and Ai4Privacy. No examples from the unpublished fourth source are newly replayed, but the v2 base was already trained on that source. Stricter source/span validation rejects 64 historical targets from replay eligibility; the original corpus and v2 weights are unchanged. The original 300-document PII regression set is separate from training/replay.

Credit [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) (CC BY 4.0), [Gravitee PII Detection](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset) (Apache-2.0, with its upstream attributions), and [Ai4Privacy / Ai Suisse SA OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (CC BY 4.0; Copyright © 2026 Ai Suisse SA as stated in its card). SPANIEL converts source annotations into exact-source tagged completions, validates them, excludes incompatible targets and samples replay. Preserve source-specific attribution and notices; the model license does not replace dataset terms. These sources do not imply endorsement.

The [source-attribution record](https://github.com/harshsinghal/SPANIEL/blob/main/docs/training-data-and-evaluation.md#source-attribution-and-license-record) contains the exact source revisions, upstream creators, modifications and license links. [Data hashes](https://github.com/harshsinghal/SPANIEL/blob/main/reports/context-manifest.json), the [coverage audit](https://github.com/harshsinghal/SPANIEL/blob/main/docs/coverage-audit.md), [policy definitions](https://github.com/harshsinghal/SPANIEL/blob/main/reports/context-policy.json) and [experiment evidence](https://github.com/harshsinghal/SPANIEL/tree/main/reports/credential-context) trace the changes from corpus inspection to training and evaluation.

## Local inference and governed retrieval

Use the matching v2 base and this adapter with the SPANIEL source server. Install the repository's `requirements-adapter.txt` with a PyTorch build appropriate to your machine. Configure `PII_MODEL_ID`, `PII_MODEL_REVISION`, `PII_ADAPTER` and `PII_ADAPTER_REVISION` as documented in the milestone. The service merges the adapter in memory and keeps the same copy-or-tag constrained decoder.

The inference and experiment code is available at [source revision `eb46193b698b44eb0401b7f35b7ab1237dad703b`](https://github.com/harshsinghal/SPANIEL/tree/eb46193b698b44eb0401b7f35b7ab1237dad703b). This revision includes the adapter loader, constrained decoder, data generator and evaluators; later documentation commits record the completed results. The existing Docker image does not include the optional PEFT dependency. Use a source installation for this adapter.

```bash
git clone https://github.com/harshsinghal/SPANIEL.git
cd SPANIEL
git checkout eb46193b698b44eb0401b7f35b7ab1237dad703b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-adapter.txt

export PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2
export PII_MODEL_REVISION=a58063469be8c7b929016d944d8d6d1545bc70a0
export PII_MODEL_CACHE="$PWD/.model-cache"
export PII_ADAPTER=Harsh/spaniel-credential-context-lora-v0.1
# Pin the selected weights; later card edits do not change this artifact.
export PII_ADAPTER_REVISION=CORRECTED_ADAPTER_REVISION
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
