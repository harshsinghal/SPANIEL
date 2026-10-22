# SPANIEL training data, attribution, and evaluation

Research by [Harsh Singhal](https://www.linkedin.com/in/harshsinghal/) · Documentation updated 21 October 2026

[Project overview](../README.md) · [Published model](https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2) · [Artifact manifest](../artifact_manifest.json)

This document explains the published SPANIEL v2 training recipe and its evidence. Dataset license statements below were checked against the exact revisions recorded in the manifest, as well as the current source cards. They describe source terms and provenance; this documentation update does not redistribute the datasets or establish a new license for their contents.

The [October coverage audit](coverage-audit.md) and [credential/context experiment](milestones/2026-10-21-credential-context.md) are separate from the historical v2 recipe below. The new experiment uses explicit completion-only masking, 5,248 newly generated contrasts and 1,536 validated public-source replay rows. Its own manifests, splits and eventual results must not be substituted into v2's historical benchmark tables.

## Why use a mixture?

The goal was to train a small local model to respond to natural-language PII requests across different text formats. A single dataset can teach a model its own label vocabulary, generation style, and annotation mistakes. The v2 recipe combines sources while keeping each source's requested labels explicit.

The published mixture contains **389,521 retained SFT examples**. Reported shares are rounded: Gravitee 44.6%, NVIDIA Nemotron-PII 25.7%, Ai4Privacy 28.2%, and project-generated documents 1.5%. These are training-example shares after preparation and sampling, not equal-sized source collections or counts of unique documents. See [the recorded v2 experiment](../journal/2026-07-19-v2-generalization.md).

### NVIDIA Nemotron-PII

Credit: **NVIDIA**, [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).

SPANIEL reads the pinned training parquet and converts its span annotations into tagged-text targets. Its test partition supplies the evaluation documents. The source describes synthetic persona-grounded records with fine-grained PII/PHI annotations.

SPANIEL modifies the labels in a documented way: a `date_time` annotation without a `HH:MM`-style clock time is relabeled `date`. When a request asks for dates while omitting the narrower date-of-birth label, a family-aware subset rule can treat the birthdate as a date. These rules express the project's annotation convention; they are not new annotations endorsed by NVIDIA.

### Gravitee PII Detection

Credit: **Gravitee**, [PII Detection dataset](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset).

SPANIEL uses the aggregate parquet for PII coverage and text-format diversity. It excludes the source labels `FINANCIAL`, `TITLE`, and `NRP` from requested targets, following the project's error analysis. Its adapter preserves the source vocabulary instead of forcing it into NVIDIA's taxonomy.

Gravitee's pinned card itself credits these upstream sources:

| Upstream source | License reported by Gravitee |
|---|---|
| [beki/privy](https://huggingface.co/datasets/beki/privy) | MIT |
| [Gretel PII Masking EN v1](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | Apache-2.0 |
| [Gretel Synthetic PII Finance Multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | Apache-2.0 |
| [urchade/synthetic-pii-ner-mistral-v1](https://huggingface.co/datasets/urchade/synthetic-pii-ner-mistral-v1) | Apache-2.0 |
| Gravitee's release-specific synthetic examples | Apache-2.0 |
| [Dataiku Kiji PII training data](https://huggingface.co/datasets/DataikuNLP/kiji-pii-training-data) | Apache-2.0 |

This table records upstream provenance as stated by the aggregate publisher. It is not an independent license audit of every underlying record. Keep upstream attribution and notices when reusing or redistributing data.

### Ai4Privacy OpenPII

Credit: **Ai4Privacy / Ai Suisse SA**, [OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m), Copyright © 2026 Ai Suisse SA as stated in the pinned card.

The converter selects English training records and validates that annotated values equal the original character slices. The recorded converted artifact has 163,696 rows; the mixture builder shuffles it deterministically and considers up to 110,000 rows before subsequent filtering. Non-English records are outside this release's training scope.

The source metadata uses `license: other` with `license_name: cc-by-4.0`; the card's terms explicitly identify CC BY 4.0. The table below reflects those stated terms rather than interpreting `other` as an absence of a license.

### Project-generated synthetic documents

The fourth component contains 1,966 batch-generated documents targeting formats identified during error analysis: JSON logs, checkbox forms, and prose. The builder makes three passes over these documents when constructing the mixture, sampling request forms along the way.

The manifest records the artifact's hash. Its contents are not distributed in this repository, and this page does not assign them a public reuse license. The public builder offers `--skip-synth` for a three-source variant. That variant is useful for running the workflow but is not a byte-identical reconstruction of the published four-source training set.

## Source attribution and license record

| Source | Exact revision used | License stated at that revision |
|---|---|---|
| NVIDIA Nemotron-PII | [b70ffaf5ff39e079776134c5bf4381f00a9fd1ed](https://huggingface.co/datasets/nvidia/Nemotron-PII/blob/b70ffaf5ff39e079776134c5bf4381f00a9fd1ed/README.md) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Gravitee PII Detection | [5b3d03212714c1c65fd07becba4aec92b8cab20f](https://huggingface.co/datasets/gravitee-io/pii-detection-dataset/blob/5b3d03212714c1c65fd07becba4aec92b8cab20f/README.md) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), with upstream credits above |
| Ai4Privacy / Ai Suisse SA OpenPII | [a785eb528e28be2693c3718a27e066970de5dadb](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m/blob/a785eb528e28be2693c3718a27e066970de5dadb/README.md) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

CC BY 4.0 permits sharing and adaptation, including commercial uses, subject to its conditions, including attribution, a license link, and identification of changes. Apache-2.0 also has notice and license obligations. Refer to the linked terms for the intended reuse. The modifications made by SPANIEL are described here and in its converters. Dataset authors do not endorse SPANIEL by virtue of this use.

Base model credit: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. Published SPANIEL inference is pinned to `a58063469be8c7b929016d944d8d6d1545bc70a0`. Later model-card commits can update documentation without changing those weights.

## Training-example design

[The v2 builder](../build_sft_dataset_v2.py) uses seed `20260719` and creates system/user/assistant message examples. The user supplies a type list and the text; the assistant target is the original text with inserted tags.

| Mechanism | Implementation and purpose |
|---|---|
| Alias sampling | Approximately 22 natural-language forms per label; use the sampled form in both the request and output tags |
| Full-vocabulary requests | A 40% branch requests the full source vocabulary; empty-positive cases also take this path |
| Absent-label requests | A 40% branch requests present labels with attempted sampling of 2–6 absent labels, biased toward related type families |
| Type subsets | A 20% branch narrows the requested types; a birthdate can become a generic date under the documented rule |
| Definition conditioning | A 30% sampling probability adds short rules when the requested labels have implemented definitions |
| Compatible targets | Remove spans shorter than two characters, reject incompatible overlaps, and avoid alias/tag collisions |

The probabilities describe builder branches, not audited final percentages after filtering. A source-preserving target does not by itself establish the semantic correctness of its annotations.

The recorded run used full-parameter SFT rather than LoRA, bf16, sequence length 3,072, one epoch, effective batch size 32, peak learning rate `1e-5`, cosine scheduling, and 3% warmup. Training took approximately 11 hours on an H100 NVL. See [the trainer](../vast_run/train_sft.py) and [the experiment record](../journal/2026-07-19-v2-generalization.md). The current trainer consumes message-format data; loss-mask behavior should be verified against the installed TRL version before claiming completion-only training.

## Evaluation and interpretation

The headline results use 300 held-out documents from NVIDIA Nemotron-PII and strict typed span exact-match F1. The three reported axes are original labels (0.930), corrected/adjudicated labels with human spot checks (0.918), and requests using unfamiliar entity names (0.864). They are distinct views of a small evaluation, not three independent domains.

The unfamiliar-name result improved from v1's 0.747 to v2's 0.864. V2 simultaneously changed the mixture, aliases, label rules, and request construction. A component ablation is needed to identify which intervention caused how much of the gain.

The prompted gpt-oss-120b baseline scored 0.580 end to end; 141/300 outputs drifted or were malformed, while F1 conditional on parseable outputs was 0.891. The comparison measures extraction plus adherence to the tagged-output contract, not general intelligence.

The evaluation is based on synthetic PII data. It does not establish enterprise-prompt performance, multilingual reliability, calibrated confidence, resistance to adversarial instructions, or complete prevention of disclosures. Source partitions are pinned, but comprehensive cross-source near-duplicate exclusion is not established by this documentation. Such checks belong in the next benchmark.

Annotation guidelines also matter: the current target is personal attributes, rather than every mention of a place or organization. The model can correctly follow that convention while missing an item expected under another policy.

## Rebuilding and auditing

Use [the README's commands](../README.md#reproducing) for conversion, mixture building, training, and the original-gold evaluation. The three-source variant must be labeled separately from the four-source published experiment. The corrected-gold and unfamiliar-name results require their own gold/request variants.

`artifact_manifest.json` records row counts and SHA-256 hashes for the English conversion, project-generated document artifact, original evaluation set, and four-source training set. `verify_artifacts.py` validates selected available files; it does not recreate a missing artifact or adjudicate labels.

For AI governance use, the next evaluation should join extraction quality to control effectiveness: sensitive content remaining after redaction, benign requests interrupted, failures handled, and useful downstream task performance. Those measurements will determine whether a local detector is suitable for a particular application policy.
