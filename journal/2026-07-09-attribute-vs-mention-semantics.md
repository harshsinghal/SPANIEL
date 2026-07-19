# What a sushi chain taught our PII model: attribute semantics, mention semantics, and the questions a dataset silently answers

*2026-07-09 — notes from the first real-world probe of the Qwen3-1.7B PII tagger*

## The setup

We fine-tuned Qwen3-1.7B on a tagged-regeneration task: given a document and a list of
requested entity types, reproduce the document exactly, wrapping matching spans in XML
tags. Training data came from two PII corpora (gravitee-io/pii-detection-dataset and
nvidia/Nemotron-PII), reformatted so that the label set is part of the prompt and label
names are sampled from natural-language alias sets — the design bet being that a model
conditioned on free-form type descriptions at training time will accept novel type
descriptions at inference time.

On the in-distribution eval this worked notably well: strict span-level F1 of 0.872,
against 0.580 for a 70×-larger model (gpt-oss-120b) prompted zero-shot with the same
format. Then we pasted in a paragraph about a sushi restaurant chain — Wikipedia-register
corporate prose — requested `company name, address, country name, location`, and got
three distinct failures:

1. **"United States" and "District of Columbia" were never tagged** — not in the combined
   request, and not even when `country name` was the *only* requested type.
2. **"California" was tagged as `country name`** when country was requested alone — a
   wrong-type assignment in the same sentence where the actual country name sat untagged.
3. **The output appended a period** after the final entity ("…subsidiary of
   `<company name>`Kura Sushi, Inc.`</company name>`**.**"), duplicating the sentence-final
   dot and violating the copy contract. The server's fidelity check caught it.

Every one of these looks like a bug. None of them is random. Each is the model giving a
correct answer to a question we didn't realize we had asked. Working out what those
questions were is the point of this note.

## Failure 1: the model doesn't do NER, and we never asked it to

### What NER has meant for thirty years

Named entity recognition has a specific lineage. The task was codified at MUC-6 in 1995,
industrialized by CoNLL-2003, and given its most-used modern form in OntoNotes. Across
all of these, the unit of annotation is the **mention**: every surface occurrence of a
name of a covered type gets a span, no matter what work that name is doing in the
sentence. The CoNLL and ACE annotation guidelines are explicit about exhaustiveness —
if "United States" appears three times, it is annotated three times, whether it names a
counterparty in a contract, a geographic backdrop in a news story, or half of a sports
team's name (ACE went further and annotated metonymy: "Washington announced…" is a GPE
mention used to refer to a government).

The model families built on this data inherited the assumption structurally. HMM and
CRF taggers, then BiLSTM-CRFs, then BERT-style token classifiers all frame the task as
sequence labeling over BIO tags: every token receives a decision, and the annotation's
exhaustiveness becomes exhaustive supervision. A CoNLL-trained CRF does not have an
opinion about whether "United States" is *interesting* — the guidelines already decided
that all mentions are, and the model's job is surface-form typing: *is this span the
name of a country?* Call this **mention semantics**: entityhood is a property of the
span's type, and context is consulted only to disambiguate type (is "Washington" a
person or a place?), never to decide whether typing applies at all.

### What PII data actually annotates

PII detection corpora look like NER datasets — spans, offsets, labels — but the
annotation question is different. A span is gold when it **discloses an attribute of a
data subject**: somebody's name, somebody's date of birth, the country field of
somebody's record. Nemotron-PII is synthetic, generated as documents *about people* —
application forms, intake reports, account letters — and its `country` spans are, with
overwhelming consistency, the value of a country slot in some person's address block.
A country mentioned as geography ("operates throughout the United States") essentially
never appears as gold, because the generator was never asked to produce that reading.

Call this **attribute semantics**: entityhood is a property of the span's *role in the
document's record structure*, and the type label names the slot, not the surface form.
The distinction is not new — the field has drawn it before, just under other names. MUC
itself had, alongside named entity recognition, a template-filling task where the goal
was slots of an event, not mentions in text. TAC-KBP slot filling asked for *the*
employer, *the* spouse, *the* country of residence of a query entity — role extraction,
where most mentions of valid-looking surface forms are correctly ignored. PII detection
is slot filling wearing NER's costume. The spans-and-labels format hides the fact that
the annotation answers "whose data is this?" rather than "what is this a name of?"

### The fine-tune transferred the question, not just the labels

SFT on this data taught the model p(tag | span, context, request) *under attribute
semantics*, because that is the only joint distribution the data exhibits. When our
request said `country name` and the text offered "throughout the United States," the
model correctly — by its training distribution — judged that no one's country attribute
was being disclosed, and moved on. The company names were tagged because `company_name`
in Nemotron *does* occur in prose-like contexts (letterheads, "your account with X"), so
its learned semantics is closer to mention-style.

The uncomfortable takeaway: **the label name in the prompt does not carry the task
semantics; the training distribution does.** We built alias variability so that the
model could accept any surface description of a type. It can. But what "country name"
*means* — mention or attribute — was fixed by the data, silently, and no phrasing of the
request can currently reach the other reading, because the other reading was never
exemplified. For a DLP product, attribute semantics is arguably the right default; the
"failure" on encyclopedic prose is the model refusing to do a task it was never taught.
But we didn't choose that stance. The data chose it, and we found out from a sushi chain.

## Failure 2: "California" as a country — the cost of never seeing true absence

The second failure is more damning than the first, because it is not principled
abstention — it is a wrong positive. Requested `country name` alone, the model skipped
"United States" *and* tagged "California."

Two forces conspire here. The first is a co-occurrence prior baked in by the training
mixture. Our request-set construction had three buckets: full source vocabulary,
present-labels-plus-sampled-negatives, and strict subsets. Negatives taught abstention
in the aggregate — the baseline numbers show high precision in-distribution, so
abstention was learned. But consider the conditional the model actually experienced:
*given that a geographic type was requested and the document contains geographic
material at all*, how often was the requested type truly absent? In Nemotron, city,
state, and country co-travel inside address blocks; a document with a state almost
always has a country nearby. Requests for `country` on state-bearing,
country-free text were rare. Under that prior, "a geo type was requested, geo spans
exist, therefore one of them is probably the referent" is a decent bet — in
distribution. Out of distribution it manifests as grabbing the nearest plausible span
of the right *family* and promoting it up the ontology.

The second force is architectural, and it separates this generation of extractors from
the last. A CRF or BERT token classifier trained with exhaustive BIO supervision spends
the vast majority of its gradient on O tokens; its default state is "not an entity,"
and emitting a span requires the features to overcome that mass. A generative tagger
has no O class. Emitting a tag is just another token continuation, locally scored, and
nothing in the loss makes absence the privileged default. The old sequence-labeling
framing got calibration-under-absence nearly for free; the generative framing has to
buy it explicitly, with data. This is the same asymmetry that shows up as hallucination
in seq2seq extraction generally: the decoder's fluency is indifferent to whether the
source licenses the content.

The fix is therefore also data-shaped, and it is specific: **hard negatives along
ontology containment edges.** Geographic types form a containment hierarchy
(city ⊂ county ⊂ state ⊂ country), and the model needs deliberate exposure to requests
that land *between* levels — `country` requested on text with only states and cities,
`city` requested on country-only text — with the target demonstrating full abstention.
Random negative sampling won't concentrate examples on these edges; they have to be
constructed. The same is true for any label family with internal structure:
`first name`/`last name`/`person name`, `date`/`date of birth`/`date time`, the id
cluster (`customer id`/`employee id`/`unique id`). We already treated the date family
as atomic when sampling subsets, which was a first, blunt acknowledgment of this
problem. Containment-aware negatives are the sharp version.

## Failure 3: one period, two jobs, and a contract with no enforcement

The duplicated period is the smallest failure and the most instructive about
architecture. The input ends "…a subsidiary of Kura Sushi, Inc." — a sentence-final
abbreviation, where a single `.` simultaneously terminates "Inc." and the sentence.
The model tagged the company as "Kura Sushi, Inc." (period inside the span, matching
how "Inc." appears mid-sentence) and then, continuing the sentence it was regenerating,
emitted the sentence-final period its language model expected to exist. Both decisions
are locally reasonable. Jointly they add a character, and the copy contract is broken.

This exact ambiguity is one of the oldest solved-ish problems in NLP. Sentence boundary
detection systems — Palmer and Hearst's SATZ in the 90s, the unsupervised Punkt
algorithm of Kiss and Strunk that still ships inside NLTK — exist almost entirely
because of abbreviation-final periods. Tokenizers in the Penn Treebank tradition had
special-case logic for it. Our model rediscovered the problem in a new costume: not as
a segmentation decision but as a generation decision, where the ambiguity surfaces as
two plausible continuations whose composition is invalid.

The deeper point is that **copy fidelity in an autoregressive model is a tendency, not
a property.** The decoder samples from a next-token distribution; 50,000 training
examples of exact copying make faithful continuations very probable, and our drift rate
of 6.7% (down from the zero-shot baseline's 45.3%) is that probability made visible.
It will never be zero by training alone. The previous seq2seq generation attacked this
architecturally — CopyNet and pointer-generator networks added explicit copy
distributions over source positions precisely because unconstrained decoders drift.
The modern equivalent for decoder-only models is constrained decoding: at each step the
generator is only permitted tokens that either (a) continue an exact copy of the source
from the current position or (b) open or close a tag from the requested set. Under that
automaton the period-duplication failure is not unlikely — it is unrepresentable. The
scorer's copy-fidelity check then stops being a metric and becomes a discharged proof
obligation.

## What other semantics are hiding

The three failures share a shape: an axis of meaning existed, the training data took a
stance on it, and we discovered the stance only when an input landed on the other side.
Mention-versus-attribute is one axis. Once you look for the shape, the annotation
literature of the last two decades reads as a catalog of others:

- **Assertion and factuality.** Clinical NLP spent years on this: NegEx and the ConText
  algorithm, the i2b2 assertion tasks — because "denies chest pain" contains a symptom
  mention that is asserted *absent*. Our analog: "customers may be asked for their SSN"
  contains an SSN *type* mention with no instance. Does a `ssn` request want it? Our
  data has an implicit answer; we don't know what it is yet.
- **Genericity and instantiation.** "Enter your password below" versus an actual
  password string. Attribute semantics probably handles this well (no value, no
  disclosure), but nobody verified it.
- **Subject scoping.** A patient record mentions the physician's name. Whose PII is in
  scope — the data subject's only, or any natural person's? GDPR and HIPAA answer this
  differently; our two corpora may too, and a mixture average of two stances is a
  coin-flip, not a policy.
- **Temporal validity.** The model tagged "Kula Sushi USA, Inc." — the *former* name.
  Under mention semantics, obviously right; under attribute semantics ("the company's
  name"), arguable. We got a stance for free and didn't order it.
- **Coreference.** "He works as a marine electrician" — is "he" a `person name` mention?
  Classic NER says no (named mentions only); a redaction product might need yes,
  because pronouns leak identity in context. No amount of label-name phrasing selects
  between these; only data can.
- **Boundary conventions.** Determiners in or out ("the District of Columbia"),
  honorifics in or out ("Dr. Weaver"), trailing punctuation in abbreviations — CoNLL,
  ACE, and OntoNotes each legislated these differently, and inter-corpus F1 penalties
  of several points are attributable to convention mismatch alone. Our two corpora
  vote differently on some of these, which is part of why boundary-tolerant "relaxed"
  F1 runs ~2 points above exact in every run we've scored.

Each axis is invisible in the label set, invisible in per-label F1, and perfectly
visible the moment a probe input separates the readings.

## What this means for building fine-tuning mixtures

The practice this pushes toward: treat a fine-tuning mixture as a **specification
document written in examples**, and treat unspecified axes as undefined behavior that
the dominant data source will define for you.

1. **Enumerate the axes before sourcing the data.** For an extraction task: mention vs
   attribute reading, assertion status, genericity, subject scope, temporal validity,
   coreference, boundary conventions, plus every ontology with containment structure in
   the label space. This list is a review artifact, like a threat model — it will be
   incomplete, and it is still the thing that makes gaps discussable.
2. **Make stances conditionable where the product needs both.** We already condition on
   the label set; the same mechanism extends to semantics selection. "any mention of a
   country" versus "the data subject's country of residence" can be different requests
   with different gold behavior in training. That converts a silent global stance into
   an instructable one — the entire alias-conditioning bet, applied one level up.
3. **Source per-axis, don't average.** Mixing an OntoNotes-register prose-NER conversion
   into the PII data (as a labeled-and-conditioned slice, per point 2) buys the mention
   reading without diluting the attribute reading. Mixing without conditioning would
   just interpolate the two stances into incoherence — the model would tag prose
   countries with some probability reflecting the mixture ratio, which is the worst of
   both.
4. **Engineer absence deliberately.** Calibration under true absence needs stratified
   negatives: per label family, per containment edge, at rates high enough to defeat
   co-occurrence priors. Synthetic data makes this cheap — absence is easy to
   manufacture — but only if the mixture design asks for it.
5. **Assume synthetic data is stance-pure.** Human-annotated corpora contain guideline
   drift and annotator disagreement, which act as accidental regularization across
   readings. A synthetic corpus executes its generator's ontology with total
   consistency. That makes the stances stronger and the undefined behavior more
   deterministic — better for debugging, worse for accidental coverage.
6. **Keep an out-of-distribution probe suite next to the eval.** Our 0.872 was computed
   on held-out data from the same generator that wrote the training set; it could not
   have surfaced any of these three failures even in principle. The sushi paragraph was
   a one-example OntoNotes-style probe and it found three. A permanent probe set —
   prose register, absence cases, assertion cases, boundary traps — belongs in the
   harness as a first-class artifact, scored per axis rather than per label.

The one-line version: **supervised fine-tuning transfers the question the dataset was
answering, not the question you meant to ask.** The label names are an interface; the
distribution is the implementation. Reading the implementation requires probing it with
inputs the training distribution would never generate — which is why the most useful
evaluation input this project has seen so far cost zero dollars and described a
conveyor-belt sushi restaurant.

---

## Further reading: the papers behind each claim

Grouped by the section of this note they deepen. The one-line annotations say what to
read each one *for*, in this project's terms.

### The NER task and its annotation ontology (Failure 1)

- **Grishman & Sundheim (1996), "Message Understanding Conference-6: A Brief History"**
  (COLING) — where "named entity" was codified as a task; readable, short, and useful
  for seeing how arbitrary some now-canonical decisions were at birth.
- **Tjong Kim Sang & De Meulder (2003), "Introduction to the CoNLL-2003 Shared Task"** —
  the dataset that defined mention-exhaustive NER for two decades of models; read the
  annotation section, not the results.
- **Doddington et al. (2004), "The Automatic Content Extraction (ACE) Program"** (LREC) —
  ACE's entity/mention distinction and metonymy handling is the most explicit statement
  of mention semantics as a designed choice rather than a default.
- **Hovy et al. (2006), "OntoNotes: The 90% Solution"** (NAACL) — the corpus behind most
  modern "general NER" claims; its GPE conventions are the stance our prose probe
  implicitly tested against.
- **Ratinov & Roth (2009), "Design Challenges and Misconceptions in Named Entity
  Recognition"** (CoNLL) — still the best single paper on boundary conventions, label
  granularity, and why NER numbers don't transfer across corpora; directly explains our
  exact-vs-relaxed F1 gap.
- **Ji & Grishman (2011), "Knowledge Base Population: Successful Approaches and
  Challenges"** (ACL) — the slot-filling task family; read to see how role extraction
  differs from mention detection in evaluation and error profile — PII detection's true
  lineage.

### The model lineage and what each generation got structurally (Failures 1–2)

- **Lafferty, McCallum & Pereira (2001), "Conditional Random Fields"** (ICML) — the
  sequence-labeling formalism; the O-class default and exhaustive per-token supervision
  discussed in Failure 2 are properties of this framing.
- **Collobert et al. (2011), "Natural Language Processing (Almost) from Scratch"**
  (JMLR) — the hinge between feature-engineered and neural sequence labeling.
- **Lample et al. (2016), "Neural Architectures for Named Entity Recognition"**
  (NAACL) — BiLSTM-CRF; the neural generation that kept the BIO contract intact.
- **Devlin et al. (2019), "BERT"** (NAACL) — token classification as fine-tuning; the
  last architecture where absence was the gradient-dominant default.
- **Li et al. (2020), "A Unified MRC Framework for Named Entity Recognition"** (ACL) —
  the pivot point: labels become *queries*, prefiguring our label-conditioning; also an
  early sighting of the over-firing problem when the query has no answer.
- **Zaratiana et al. (2024), "GLiNER: Generalist Model for Named Entity Recognition
  using Bidirectional Transformer"** (NAACL) — the bi-encoder alternative we chose
  against; read for how label verbalization is handled without generation.
- **Zhou et al. (2024), "UniversalNER: Targeted Distillation from Large Language Models
  for Open Named Entity Recognition"** (ICLR) — the closest published recipe to ours;
  compare their negative sampling scheme to our bucket mix.
- **Sainz et al. (2024), "GoLLIE: Annotation Guidelines Improve Zero-Shot Information
  Extraction"** (ICLR) — the strongest published version of "make the stance
  conditionable": annotation *guidelines* in the prompt select the semantics. This is
  the paper to read before designing v2's mention-vs-attribute conditioning.

### Sentence boundaries, tokenization, and the period (Failure 3)

- **Palmer & Hearst (1997), "Adaptive Multilingual Sentence Boundary Disambiguation"**
  (Computational Linguistics) — SATZ; the abbreviation-final period as a first-class
  research problem.
- **Kiss & Strunk (2006), "Unsupervised Multilingual Sentence Boundary Detection"**
  (Computational Linguistics) — Punkt, still inside NLTK; the collocational view of why
  "Inc." binds its period.

### Copy fidelity, faithfulness, and constrained generation (Failure 3)

- **Gu et al. (2016), "Incorporating Copying Mechanism in Sequence-to-Sequence
  Learning"** (ACL) — CopyNet; the architectural admission that decoders don't copy
  reliably on their own.
- **See, Liu & Manning (2017), "Get To The Point: Summarization with Pointer-Generator
  Networks"** (ACL) — the most-read copy-mechanism paper; its coverage loss is an
  ancestor of our copy-fidelity metric.
- **De Cao et al. (2021), "Autoregressive Entity Retrieval"** (ICLR) — GENRE;
  trie-constrained decoding forcing outputs into a valid set — the direct intellectual
  ancestor of the copy-or-tag automaton planned for v2.
- **Maynez et al. (2020), "On Faithfulness and Factuality in Abstractive
  Summarization"** (ACL) — the vocabulary for separating fluency from faithfulness;
  our drift metric is their intrinsic hallucination, operationalized exactly.
- **Willard & Louf (2023), "Efficient Guided Generation for Large Language Models"**
  (arXiv) — FSM-based constrained decoding made practical (Outlines); the
  implementation route for the copy grammar.

### Assertion, negation, and the semantics catalog

- **Chapman et al. (2001), "A Simple Algorithm for Identifying Negated Findings and
  Diseases in Discharge Summaries"** (J Biomed Inform) — NegEx; assertion status as a
  separate annotation axis, discovered the hard way by clinical NLP.
- **Harkema et al. (2009), "ConText"** (J Biomed Inform) — extends assertion to
  experiencer and temporality; "whose attribute is this?" as an algorithmic question —
  our subject-scoping axis, fifteen years early.
- **Uzuner et al. (2011), "2010 i2b2/VA Challenge on Concepts, Assertions, and
  Relations in Clinical Text"** (JAMIA) — a shared task where mention detection and
  assertion classification were deliberately separated; a model for scoring axes
  independently.

### Datasets as stances; probing as method

- **Gururangan et al. (2018), "Annotation Artifacts in Natural Language Inference
  Data"** (NAACL) — the canonical demonstration that datasets encode unintended
  regularities that models learn preferentially.
- **Geva et al. (2019), "Are We Modeling the Task or the Annotator?"** (EMNLP) — the
  sharpest short statement of this note's thesis: supervision transfers the data
  generator's decision function, not the task description.
- **Pavlick & Kwiatkowski (2019), "Inherent Disagreements in Human Textual Inferences"**
  (TACL) — label variation as signal about underdetermined task semantics rather than
  noise; the argument behind "synthetic data is stance-pure."
- **Plank (2022), "The 'Problem' of Human Label Variation"** (EMNLP) — survey and
  position piece connecting annotator disagreement to exactly the axis-enumeration
  practice proposed above.
- **Ribeiro et al. (2020), "Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList"** (ACL) — the methodological blueprint for the probe suite: capability-
  organized, minimal, adversarially curated test cases scored separately from benchmark
  F1.
- **Longpre et al. (2023), "The Flan Collection"** (ICML) — mixture design as a
  first-class research object: task balancing, prompt diversity, and ablations showing
  mixture composition dominates many architecture choices.

---

*Next actions tied to this note: convert a prose-NER corpus into the conditioned tagged
format as a mixture slice; build containment-edge negatives for the geo, person-name,
date, and id families; add constrained decoding to the inference path so copy fidelity
is structural; stand up the probe suite as a scored artifact alongside `pii_eval.py`.*
