# Milestone: credentials and contextual compensation

**Harsh Singhal · 21 October 2026 · Training experiment in progress; no new accuracy claim yet.**

This milestone asks whether a small local model can distinguish actual credential exposure from benign lookalikes, and private compensation from public salary information. It follows the [coverage audit](../coverage-audit.md). It is a continued LoRA fine-tune of the existing Spaniel v2 checkpoint, with the original checkpoint retained as the baseline.

```mermaid
flowchart LR
    A[Corpus audit and policy] --> B[Scenario families and source offsets]
    B --> C[Training data plus PII replay]
    B --> D[Development families]
    B --> K[Reserved test families]
    C --> E[GPU.ai LoRA training]
    D --> F[Select by development loss]
    E --> F
    F --> G[Concept test and PII regression]
    K --> G
    H[Unchanged Spaniel v2] --> G
    I[Off-the-shelf GLiNER2.5] --> G
    G --> J[Evidence, limits and release decision]
```

## Policy before labels

| Requested label | Sensitive examples | Benign examples |
|---|---|---|
| `password` | Actual or purported login passwords, ordinary words used as passphrases, revoked secrets | Unfilled templates, masked values, poem titles and ordinary phrases |
| `api key` | API secrets, bearer/session tokens and credentials in URLs or JSON | Public catalog IDs, token-format explanations and unfilled placeholders |
| `private key` | Complete private-key blocks, including delimiters and padding | Public verification keys and descriptions of key formats |
| `private compensation` | Individual pay, private negotiations and nonpublic compensation budgets, including ranges | Published job advertisements, public market surveys and authorized public pay disclosures |

The policy does not infer secrecy from token entropy or the word “salary.” The same range can be public in a job advertisement and private in an offer negotiation. A private conversation can merely quote a public advertisement; that alone does not make its quoted range private compensation. Conversely, a public survey can be followed by a confidential agreement about an individual's pay.

These are application conventions for this experiment. They do not establish legal confidentiality or a universally correct disclosure policy. Ambiguous cases are explicitly separated for review. Empty model output is not calibrated abstention.

The concept benchmark requests one of these exact labels at a time. In particular, `private compensation` is different from asking for every `salary` mention. Robustness to alternative names for these new policy concepts, or to many simultaneous labels, is not established by the single-label score. The separate retrieval smoke test exercises a three-label credential request.

## Dataset and splits

The reproducible generator is [sensitive_context/build.py](../../sensitive_context/build.py); authored scenario families are in [scenarios.py](../../sensitive_context/scenarios.py). Values are fictional and non-operational. Private-key bodies exercise the text shape, not cryptographic validity.

| Split | Rows | Scenario families | Role |
|---|---:|---:|---|
| New synthetic training | 5,248 | 34 | Sensitive/benign contrasts, formats, transcripts and long-context dilution |
| Development | 100 | 9 | Checkpoint selection and error investigation |
| Frozen synthetic test | 159 | 17 | Final candidate comparison after checkpoint selection |
| Ambiguity review | 6 | 6 | Missing context; never negative training or binary gold |
| Legacy training replay | 1,536 | Original training texts | 512 validated examples from each public source; no private-source replay |
| Legacy regression evaluation | 300 | Original evaluation texts | Check for loss of established PII extraction behavior |

Substitutions, positive/negative variants and long forms of one family stay in a single split. Exact duplicates are removed, and the training/replay texts are checked against evaluation texts. Template substitutions are correlated examples, not thousands of independent human-authored scenarios. The new labels have not received independent human adjudication.

The new training rows contain 3,221 sensitive and 2,027 benign documents. Categories are not uniformly balanced: deduplication collapses repeated benign templates more than varied sensitive values. For example, the revoked-credential category has 100 sensitive rows but only four distinct benign rows; access tokens have 200 and five. The [coverage counts](../../reports/context-coverage.json) expose this limitation. More independently authored benign scenarios remain valuable even if this candidate passes the small test. The separate legacy replay accounts for 22.6% of training rows.

Transcript cases include informal number expressions, speaker turns, delayed context, corrections, missing punctuation, spelling errors and line wrapping. **These are simulated text transcripts. No audio, real ASR system or real transcription corpus is evaluated.**

Long variants insert 250 or 900 words of generated public-information filler, with the relevant evidence at the beginning, middle or end. This measures controlled dilution; it is not equivalent to a diverse enterprise-document benchmark.

Annotations store exact source offsets. Companion context ranges describe surrounding evidence, including earlier messages; those ranges are not independent gold rationales. The current model still emits separate, non-overlapping contiguous spans and does not output linked evidence groups.

## Training and verification

The pinned base is `Harsh/qwen3-0.6b-pii-sft-v2` at `a58063469be8c7b929016d944d8d6d1545bc70a0`. The candidate uses rank-16 LoRA on attention and feed-forward projections, alpha 32, dropout 0.05, two epochs, learning rate `5e-5`, effective batch size 16, bf16 and a 4,096-token maximum.

The tokenizer preflight covers all **6,784 training rows** and the development set. Its maximum sequence is **2,201 tokens**, so nothing is truncated. Prompt tokens have loss mask `-100`; the assistant's tagged completion and chat ending receive loss. The original published training file is never edited.

Checkpoint selection uses development completion loss only. Baseline and candidate then use identical constrained decoding on the same GPU. The [protocol](../../experiments/credential-context-protocol.json) records the proposed screening thresholds and limitations before final test measurements.

The first experiment combines targeted augmentation and replay. It does not isolate either component, establish LoRA superiority over full fine-tuning, or compare against GLiNER/encoder architectures. Those require additional controlled runs.

## Reproduce

```bash
python -m sensitive_context.build --out context-data
python -m sensitive_context.replay \
  --legacy-train sft_data_v2/sft_train_v2.jsonl \
  --out context-data/replay.jsonl
```

The replay command validates the complete historical mixture, records rejected row indices/hashes, and samples only public sources. The exact replay hash depends on the exact original mixture; the public three-source rebuild remains a different artifact. To reproduce the 300-document regression, supply the first 300 rows of the recorded `sft_data/sft_eval.jsonl` as `context-data/legacy-eval.jsonl`.

On the prepared GPU.ai CUDA instance, run:

```bash
bash experiments/run_credential_context.sh
```

The runner assumes an available CUDA PyTorch installation, a C compiler and Python development headers (Ubuntu packages `build-essential` and `python3-dev`; needed by the image's Triton kernels). It creates a virtual environment, installs pinned experiment dependencies, runs regression tests, trains an adapter and writes separate baseline/candidate reports. No API keys or Hugging Face write credentials are uploaded to the training instance. Provisioning, artifact collection and instance termination are managed separately.

## What qualifies as progress

Report sensitive-document recall, benign false blocks, inspection failures, exact typed span F1 and correct contrast-pair decisions separately. A structurally valid tag is not necessarily correct. A parser failure must not become an all-clear result.

Credentials are the first area of deeper investment. Compensation is a deliberately narrower experimental concept. Legal strategy, layoffs, performance reviews and other confidentiality concepts remain unvalidated research targets. See the [capability record](../../CAPABILITIES.md) for the distinction between measured and best-effort behavior.
