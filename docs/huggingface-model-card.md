---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: transformers
language:
  - en
pipeline_tag: text-generation
datasets:
  - nvidia/Nemotron-PII
  - gravitee-io/pii-detection-dataset
  - ai4privacy/pii-masking-openpii-1.5m
tags:
  - spaniel
  - pii-detection
  - named-entity-recognition
  - sensitive-data-detection
  - ai-governance
  - ai-security
  - data-loss-prevention
  - local-inference
  - constrained-decoding
  - synthetic-data
  - supervised-fine-tuning
  - privacy
---

# SPANIEL v2 — Local PII Detection for AI Prompts

**Created by [Harsh Singhal](https://www.linkedin.com/in/harshsinghal/) as applied research in AI governance, AI security, and fine-tuning small language models for local sensitive-data detection.**

SPANIEL is a 0.6B-parameter Qwen3 fine-tune for English PII extraction. It accepts natural-language entity types and marks matching spans in the source text. With the project's copy-or-tag constrained decoder, completed annotations preserve the source exactly.

The intended application is local inspection of text before it is shared with an AI service. The checkpoint provides annotations; an application must implement its own policy, redaction, blocking, or review actions. The original sensitive values remain present in tagged output.

**Runtime compatibility correction, October 2026:** the original checkpoint stores RoPE theta `1000000` in the newer `rope_parameters` format. Transformers 4.57.6 could silently use its older default `10000`. The current configuration now also supplies the equivalent legacy fields; **the model weights are unchanged**. When loading the older pinned revision, use the [corrected source loader](https://github.com/harshsinghal/SPANIEL/tree/eb46193b698b44eb0401b7f35b7ab1237dad703b). The [investigation](https://github.com/harshsinghal/SPANIEL/blob/eb46193b698b44eb0401b7f35b7ab1237dad703b/docs/runtime-compatibility.md) explains verification and why the first contextual fine-tuning run was invalidated and restarted. Reproducing a checkpoint requires its intended effective configuration as well as its weights.

## Research progression: credential and context discrimination

An October 2026 audit examined all 389,521 training rows and identified specific gaps in benign credential lookalikes, credential diversity, long context and confidentiality supervision. A follow-up experiment prepares 5,248 synthetic training examples, disjoint scenario-family development/test sets, 1,536 validated public-source replay examples and a continued LoRA fine-tune of this checkpoint.

The new task distinguishes private compensation from public salary ranges, including informal and noisy transcript text. This is a separate experimental concept; the PII results below do not validate it. The original v2 weights and their historical results remain unchanged. Candidate training/results are in progress; no accuracy gain is claimed yet.

The [current milestone and results work](https://github.com/harshsinghal/SPANIEL/pull/2) tracks the audit, data, policies, GPU.ai fine-tuning recipe, evaluation and capability limits. Credentials are the first area of deeper investment. Broader confidential-topic detection and real audio/ASR robustness remain research targets.

[GitHub repository](https://github.com/harshsinghal/SPANIEL) · [Constrained local demo](https://github.com/harshsinghal/SPANIEL#try-it-the-demo-app) · [Training data and attribution](https://github.com/harshsinghal/SPANIEL/blob/main/docs/training-data-and-evaluation.md) · [Research journal](https://github.com/harshsinghal/SPANIEL#the-journey-as-blog-entries)

## Model details

| Property | Value |
|---|---|
| Model ID | `Harsh/qwen3-0.6b-pii-sft-v2` |
| Author | Harsh Singhal |
| Base | `Qwen/Qwen3-0.6B` |
| Training | Full-parameter supervised fine-tuning (SFT), bf16 |
| Task | Natural-language-conditioned PII / entity extraction |
| Output | Source text with matching spans wrapped in requested tags |
| Language | English |
| Runtime interface | Causal text generation plus the project's constrained decoder |
| Reproducible inference revision | `a58063469be8c7b929016d944d8d6d1545bc70a0` |

The Hub pipeline remains `text-generation` because this checkpoint is a causal language model. It is not a drop-in `AutoModelForTokenClassification` checkpoint. Model-card updates do not change the pinned inference weights.

## Intended use in AI governance and security

Use SPANIEL as a local detection component when evaluating PII inspection in prompts, retrieved passages, documents, or tool outputs. These are intended integration scenarios, not separately validated benchmark domains.

A governance integration can combine findings with the recipient, data policy, and user authorization to decide what may leave the environment. This checkpoint itself does not enforce those decisions, redact text, detect prompt injection, or certify that a request is safe. Broader concepts such as confidential strategy, performance reviews, and layoff discussions are research goals rather than validated v2 capabilities.

Example input:

```text
Entity types:
- person name
- patient mrn

Text:
Patient Brian Weaver (MRN BX-40912) called yesterday.
```

Illustrative annotation:

```text
Patient <person name>Brian Weaver</person name> (MRN <patient mrn>BX-40912</patient mrn>) called yesterday.
```

## Try it in two minutes

The supported local server uses constrained decoding by default and rejects incomplete constrained output:

```bash
docker run --rm -p 127.0.0.1:8377:8377 \
  -v spaniel-models:/models ghcr.io/harshsinghal/spaniel
# Open http://127.0.0.1:8377
```

The model downloads on first startup and is cached. Submitted text is processed locally by this demo. The published image uses CPU-only PyTorch, including on Linux; a GPU flag alone will not accelerate it. For CUDA or Apple MPS, follow the [source installation](https://github.com/harshsinghal/SPANIEL#try-it-the-demo-app) with the appropriate PyTorch build.

![SPANIEL local PII detection demo](https://raw.githubusercontent.com/harshsinghal/SPANIEL/main/docs/spaniel-demo.gif)

## Usage

After starting the server, call its constrained endpoint. This Python example requires only the standard library:

```python
import json
from urllib.request import Request, urlopen

payload = {
    "text": "Reach Anita at anita@example.com.",
    "entities": ["person name", "email"],
    "constrained": True,
}
request = Request(
    "http://127.0.0.1:8377/api/tag",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)

if not result["constrained"] or not result["faithful"]:
    raise RuntimeError("No complete source-preserving annotation was returned")
print(result["tagged"])
```

`faithful` describes source preservation, not detection completeness. HTTP errors and timeouts require an explicit application decision; they must not be interpreted as a successful inspection with no findings.

For direct Transformers integration, use the repository's [CopyTagConstraint](https://github.com/harshsinghal/SPANIEL/blob/main/pii_decode.py), the [server's generation and completion validation](https://github.com/harshsinghal/SPANIEL/blob/main/pii_tagger/server.py), and `enable_thinking=False`. Unconstrained `model.generate()` does not provide the source-fidelity guarantee. Completion validation is required even when token masking is enabled.

## Results

Published July 2026 results: strict typed span exact-match F1 on 300 held-out NVIDIA Nemotron-PII test documents, under constrained decoding.

| Evaluation axis | SPANIEL v2 F1 |
|---|---:|
| Original source labels | 0.930 |
| Adjudicated labels, corrected and human-spot-checked | 0.918 |
| Requests with entity names absent from the training alias sets | 0.864 |

V1 scored 0.747 on unfamiliar names; v2's gain is 11.7 percentage points. Multiple interventions changed together, so this does not isolate alias training or any source's contribution. Unseen wording is different from an entirely new semantic concept.

The gpt-oss-120b zero-shot comparison scored 0.580 end to end on the tagged-regeneration task, with 141/300 outputs drifting or malformed. Its F1 conditional on parseable rows was 0.891. Interpret this as a task-and-output-contract comparison, not general model superiority.

The reported 0.0% copy drift applies to completed constrained output; incomplete output is a failure. It does not mean zero missed PII. See [evaluation methodology and scope](https://github.com/harshsinghal/SPANIEL/blob/main/docs/training-data-and-evaluation.md#evaluation-and-interpretation).

## Training

The recorded v2 run used Qwen3-0.6B, full-parameter SFT without LoRA, bf16, sequence length 3,072, effective batch size 32, one epoch, peak learning rate `1e-5`, cosine scheduling, and 3% warmup. Training took approximately 11 hours on an H100 NVL, with approximately two additional hours recorded for evaluation generation.

Training examples pair a request and source text with a tagged-text assistant response. The public trainer consumes message-format examples; verify loss-mask behavior for the installed TRL version when reproducing the run. The [training script](https://github.com/harshsinghal/SPANIEL/blob/main/vast_run/train_sft.py), [mixture builder](https://github.com/harshsinghal/SPANIEL/blob/main/build_sft_dataset_v2.py), and [research record](https://github.com/harshsinghal/SPANIEL/blob/main/journal/2026-07-19-v2-generalization.md) document the workflow.

## Datasets and how they were combined

**389,521 retained training examples**, with rounded shares:

| Source | Share | Role | Source license |
|---|---:|---|---|
| [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | 25.7% | Fine-grained synthetic PII/PHI examples; date-family label repair | CC BY 4.0 |
| [Gravitee PII Detection](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset) | 44.6% | Broad PII coverage and format diversity | Apache-2.0, with upstream attribution |
| [Ai4Privacy / Ai Suisse SA OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) | 28.2% | An additional synthetic corpus, English slice only | CC BY 4.0 |
| Project-generated synthetic documents | 1.5% after 3× sampling | 1,966 documents targeting JSON logs, checkbox forms, and prose | Not distributed in the repository |

Sources retain their label vocabularies in the request. Approximately 22 aliases per label vary the wording; absent-label and subset requests exercise selection and abstention. Optional definitions are included for labels with implemented rules. Span filtering and exact-text target construction keep examples compatible with the decoder.

The [source attribution record](https://github.com/harshsinghal/SPANIEL/blob/main/docs/training-data-and-evaluation.md#source-attribution-and-license-record) identifies exact dataset revisions, upstream creators, license links, and SPANIEL's modifications. Credit NVIDIA, Gravitee and its credited upstream contributors, and Ai4Privacy / Ai Suisse SA (Copyright © 2026 Ai Suisse SA as stated in its card). Preserve applicable attribution and notices when reusing source data. These sources do not imply endorsement.

The model's declared Apache-2.0 license does not replace dataset-specific terms. The synthetic fourth artifact is recorded by hash but is not supplied. A documented `--skip-synth` rebuild uses three sources and must not be described as the published four-source mixture or assumed to reproduce its scores.

## Limitations

- English intended scope; no multilingual performance claim. The corpus audit found non-English records carrying English source metadata, so filtering does not prove every training row is English.
- Evaluation uses synthetic PII data and a small held-out sample; validate on the intended domain.
- Current annotation rules target personal attributes. General world-fact mentions may deliberately remain untagged.
- Far paraphrases can fail; occupation and time-related labels were identified as weaker areas in the project reports.
- The decoder supports non-overlapping contiguous spans with a two-character minimum; nested and linked discontinuous evidence are outside this output format.
- No published calibration, adversarial-security, or end-to-end disclosure-prevention guarantee. Valid output can still contain incorrect or missed annotations.
- Local inference does not determine the privacy of an application's logs, networking, or subsequent requests.

## Author and citation

[Harsh Singhal](https://www.linkedin.com/in/harshsinghal/) · [GitHub](https://github.com/harshsinghal) · [Hugging Face](https://huggingface.co/Harsh)

Suggested citation: **Harsh Singhal. SPANIEL: Local PII Detection for AI Prompts. 2026.** Include the model and repository revision used. [Machine-readable citation](https://github.com/harshsinghal/SPANIEL/blob/main/CITATION.cff).
