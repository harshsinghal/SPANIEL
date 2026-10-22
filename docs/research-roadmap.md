# Research milestones for local AI governance

**Harsh Singhal · SPANIEL.** Each milestone should leave a usable artifact and an evidence record, including failures. A source-preserving detector, an application enforcement rule and a governance policy are separate things to evaluate.

## 1. A governed retrieval boundary

**Available:** a Python tool that captures a document, runs local inspection, validates the returned spans, and returns null content on a policy match or inspection failure. Synthetic examples and enforcement tests demonstrate the contract.

The defensible claim is that a rejected result returns no document content through this tool. It does not cover another reader, a shell tool, logging before inspection, or missed detections. See [governed retrieval](governed-retrieval.md).

## 2. Contextual credential and compensation discrimination

**Completed experimental milestone:** the [coverage audit](coverage-audit.md), explicit policy, reproducible synthetic data, GPU.ai LoRA run, frozen family splits, original PII regression and an [off-the-shelf encoder baseline](gliner-context-baseline.md).

The [published adapter](https://huggingface.co/Harsh/spaniel-credential-context-lora-v0.1) improves overall synthetic exact span F1 from 0.397 to 0.869. It fails all four frozen screens: credential recall and false blocks, original PII retention (0.937 → 0.924 F1), and zero inspection errors. Application checks remain mixed. V2 stays the default. The [milestone record](milestones/2026-10-21-credential-context.md) records the outcome, including public-pay false alarms, missed indirect passwords and spoken range corrections. The next data iteration should address these failures while reserving new test families before training. An augmentation/replay ablation should also measure whether a stronger retention constraint improves the tradeoff.

## 3. An independently reviewed policy benchmark

**Proposed, not completed.** The current templates are useful development instruments, but their authors also defined the policy and labels. A stronger benchmark needs:

- Independent annotation and adjudication of disagreements, including genuinely unresolved disclosure context.
- Provenance and reuse terms for every source; real credentials and private records must not be published as benchmark examples.
- Splits by source, document and scenario family before augmentation, with paraphrases and near-duplicates kept together.
- More distinct benign scenarios, indirect references, mixed public/private facts, multiple entities, multilingual cases, and complete-document distractors.
- Separate requested-label paraphrase tests: a new concept score must not inherit v2's historical unfamiliar-label result.

Keep three annotation layers distinct: the literal entity span, the evidence for the sensitive concept, and the application's disclosure decision. Publicly available personal information may remain PII while being benign under a particular confidentiality policy. Missing context should be an explicit review category, not a negative example invented for scoring.

The deliverable is a versioned benchmark and annotation guide, with enough independent groups to support uncertainty estimates that account for correlated examples. Test data used to redesign training becomes development data; a new untouched test is then required.

## 4. Compare architectures under equal adaptation conditions

**Proposed, not completed.** The current GLiNER2.5 result is an off-the-shelf starting point. A fairer adaptation study should train and evaluate an encoder, a constrained decoder and selected hybrids on the same policy, data and split definitions.

| Candidate | Question | Additional measurement |
|---|---|---|
| GLiNER2.5 or another encoder extractor | Can policy-aware training retain fast span extraction while reducing benign false alerts? | Threshold selected on development data; source-offset and whole-document coverage checks |
| Spaniel LoRA versus full fine-tuning | Does full training improve the policy task enough to justify its cost and regression risk? | Multiple seeds, equal data, original PII retention, training memory and cost |
| Encoder proposals plus local contextual verifier | Can fast candidate generation reduce the amount of decoder work? | Proposal recall separately: a verifier cannot recover a span that never reaches it |
| Joint entity and sensitivity heads | Does learning the entity and its context separately improve policy transfer? | Relation/evidence accuracy, calibration and behavior on missing context |

Measure document decisions, exact spans and linked evidence separately. Compare latency only on matched hardware and runtime settings, including full input length, batch size, warm-up, end-to-end time, memory, and p50/p95 latency. If an operating threshold is selected, use development data and freeze it before the test. The current Spaniel API does not expose calibrated confidence or an abstention decision.

The deliverable is a reproducible comparison and ablations, not a claim that one architecture is best for every entity extraction task.

## 5. Transcript and semantic-evidence detection

**Proposed, not completed.** Move from simulated transcript text to appropriately licensed, consented audio/transcript material or controlled recordings, with separately recorded ASR outputs. Evaluate speaker attribution, delayed context, colloquial amounts, homophones, missing words and segmentation errors. Keep the recognized text unchanged when reporting spans; an inaudible value must not be reconstructed as if it appeared in the source.

For compensation, layoffs, legal strategy and performance reviews, annotate the actual evidence and its relation to the policy. A concept may depend on several disjoint passages. A future output can link a group of exact source spans to a concept and decision; it should not replace the evidence with a generated paraphrase. The current decoder's contiguous spans and data-side evidence groups do not yet implement this model capability.

The deliverable is a separate transcript/concept benchmark and, if supported by results, a separately versioned model. Conventional PII F1 cannot validate these broader claims.

## Release record for every milestone

Preserve the base and adapter revisions, data/code hashes, policy version, source terms, split logic, training configuration, selection rule, hardware, evaluation outputs, regressions and known unsupported cases. Summarize the evidence in the [capability record](../CAPABILITIES.md) and [change history](../CHANGELOG.md). Make each result reproducible before presenting it as progress in AI governance or AI security.
