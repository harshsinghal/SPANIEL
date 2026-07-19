# The $10 audit: what frontier models in batch mode revealed about our gold, our model, and our judges

*2026-07-18 — five batch jobs, four lessons, seven dollars and change*

## The setup

OpenAI's batch API prices frontier-model labor at roughly half interactive
rates, and nano-tier models at cents per million tokens. With a $10 budget we
ran a portfolio of four data-improvement jobs against the PII pipeline:

| Job | Scope | Model | Cost |
|---|---|---|---|
| A. Adjudicate the eval gold | 300 scored rows, 2,148 spans | gpt-5.6-sol | ~$2.10 |
| B. Audit training rows against label rules | 26,148 rows | gpt-5.4-nano | ~$2.90 |
| C. Generate label-name paraphrases | 77 labels × 20 | gpt-5.6-luna | ~$0.06 |
| D. Synthesize register-gap documents | 2,500 docs | gpt-5.4-nano/mini | ~$2.20 |

## Lesson 1: our gold was 9% wrong, and the model knew

The adjudicator flagged **8.8% of gold spans as not entities at all** — field
labels annotated as values (`Temporary Password`, `Policyholder Last Name` —
the literal form-field names), blank placeholders (`State of ________`),
document titles tagged as occupations, world-fact mentions tagged as personal
data. A stratified human review upheld the rulings.

Rescoring against corrected gold produced the most instructive number of the
project: our model's F1 *dropped* from 0.944 to 0.924 — because **precision
fell while recall rose**. The model, trained on this generator's data, had
faithfully learned its annotation noise; against clean gold, those learned
errors surface as false positives, while its recall on *real* entities was
excellent (0.950). The headline 0.944 was partly "agreement on shared noise."
Supervised fine-tuning transfers the question the dataset was answering — 
including the parts it answered wrong.

## Lesson 2: nano models generate; they do not judge

Job B asked a nano-tier model to audit training rows against subtle span
rules. It flagged **84% of rows** — including thousands of emails as
"violations" — against the sol-tier adjudicator's carefully-measured 9% noise
rate. The output was salvageable only where suggestions were *mechanically
verifiable* (a date_time without a clock time is a regex, not a judgment).
Priced lesson: cheap models for generation and filtering, expensive models for
judgment on small high-value slices, and deterministic rules wherever a rule
can be written at all.

## Lesson 3: generalization has a price, and now we know it

The same session measured what "free-form entity names" really costs: rerun
the eval with every request phrased in names the model never saw
("federal benefits number" for ssn, "mail routing code" for postcode).
Result: **0.747 vs 0.944** — a 19.7-point generalization gap, concentrated in
far paraphrases, with a distinctive signature: when the model connected an
unseen name it tagged with near-perfect precision; it just often failed to
make the connection. Phrase-level familiarity, not word-level — "inherited
family surname" fails even though "surname" is a trained alias.

## Lesson 4: measurement and medicine can arrive together

The same $10 bought the diagnosis *and* the treatment: Job A's corrected gold
became a permanent third evaluation axis; Job C's 1,466 new aliases became
v2's training signal against the generalization gap; Job D's synthetic
JSON-log and checkbox-form documents filled register gaps found by error
autopsy; and the adjudication rubric, human-ratified, became the project's
canonical annotation guidelines. The v2 training run built on all four is the
subject of the next entry.

Total spend: ~$7.50. The most expensive part of the exercise was noticing it
was possible.
