# Teaching a small model to think — four experiments, two sizes, and a map of the limit

*2026-07-21 — a negative result, reported in full because negative results are data*

## The hypothesis

SPANIEL's one measured weakness is **far-paraphrase generalization**: ask for
an entity using wording semantically distant from anything in training —
"mail routing code" for a postal code — and the model often fails to make the
connection. On the held-out unseen-name evaluation it scores 0.86, with a
visible floor of paraphrases it never bridges.

The intuition: this is a *reasoning* failure, not a pattern-matching one. A
model that paused to reason — "a mail routing code is probably a postal
code; the digits after the city are the ZIP" — should bridge the gap. Qwen3
ships with a native thinking mode. So: teach SPANIEL to think, and turn it on
for hard requests.

This entry documents three cycles of trying, and the measurement that the
approach does not work at this model size. It didn't fail for want of care —
each cycle fixed the previous cycle's diagnosed flaw. It failed because the
underlying bet — that distilled reasoning transfers to a 0.6B model for this
task — is false, at least by the methods available to us.

## Cycle 0: the model can't think, and doesn't know it

First surprise: our fine-tuned model, asked to think, produces an **empty**
`<think></think>` block and answers as before. Our earlier training —
thousands of examples with no reasoning — didn't just skip the thinking
ability, it actively trained the habit of opening the block and immediately
closing it. The base model's reasoning content was gone. There was nothing to
toggle on; there was something to rebuild from scratch.

(Mechanism worth knowing: "thinking mode" is not a model feature. Qwen's chat
template, with `enable_thinking=False`, *pre-fills* an empty
`<think>\n\n</think>` into the prompt so the model skips the phase. With
`True`, it leaves the slot open. The switch is prompt formatting; the behavior
is trained.)

## Cycle 1: distill reasoning traces (943 examples)

We paid a larger model (gpt-5.4-mini, batch mode) to write reasoning traces
for our tagging tasks, and fine-tuned on 943 of them. Result: **half a
success.** The model now genuinely reasoned — and on the flagship failure case
it wrote, in its trace, *"a mail routing code is a postal code."* The semantic
hop happened.

Then it didn't tag the ZIP code. The trace reached the right conclusion and
the answer ignored it. A student writing the correct logic in the margin and
the wrong answer in the box. We named it the **reasoning–action gap**.

## Cycle 2: force the reasoning to commit (3,433 examples)

Diagnosis of cycle 1: traces reasoned freely and the answer didn't have to
honor them. Fix: every trace must **end with a commitment line** — "Tagging:
X as type; Y as type" — and we mechanically threw out any trace whose
commitment didn't match the answer that followed. We also demanded traces cite
surrounding-text evidence, generated in four structural styles for diversity,
and included a "solve it blind" variant (the writer never sees the answer;
kept only when its answer is verified correct).

On three probe questions, the reasoning–action gap **closed**: commitment now
equalled answer. But the full 300-question exam told a different story — the
model got *worse everywhere*, and a new failure appeared: it now confidently
reasoned its way to *rejecting* correct answers ("78704 is a postal address,
not a mail routing code, so it does not qualify"). Cause: our filters had kept
63% elimination-style traces, and the model over-learned candidate rejection.
Second lesson, more important: we had trained on thinking examples *only*, and
the model drifted away from the tagging skill it started with. **Thinking-only
training erodes the base skill.**

## Cycle 3: the mixed diet, and the verdict

Two fixes. First, rebalance the traces (cap the elimination style, add 701
verified blind-solved traces on hard paraphrase cases — the "make the
connection *and* act" behavior). Second, and decisive: **anchor the training
with 10,000 ordinary no-think examples** so the model rehearses its core skill
in the same run it learns to think. One model, two behaviors, both practiced;
the prompt formatting distinguishes them (empty-think prefill → answer
directly; blank slate → reason first).

We also taught the constrained decoder about thinking mode so copy fidelity
couldn't confound the measurement: reasoning flows free until `</think>`, then
the copy-or-tag automaton guarantees the answer.

Then the full exam — three difficulty tiers, both modes, constrained
decoding, pass bars fixed **before** running:

| Tier | thinking OFF | thinking ON |
|:--|:-:|:-:|
| Obvious wordings | **0.915** | 0.398 |
| Slightly-off wordings | **0.911** | 0.325 |
| Unusual wordings | **0.853** | 0.340 |

*(strict span-level exact-match F1, 300 documents per cell; copy drift 0% in all
OFF cells and ≤1% in all ON cells — the decoder held)*

**Bar 1 — don't damage the workhorse: passed cleanly.** Thinking-OFF matches
the previous model (0.93 / – / 0.86) tier for tier. The anchor mix completely
prevented the cycle-2 erosion. This is a real, reusable result: you can extend
this model with new material without wrecking its skill, provided you keep a
large anchor of the old material in the mix.

**Bar 2 — thinking must help: failed, decisively.** Thinking-ON collapsed to
~0.34 everywhere — roughly *half* the OFF score, not better. Precision cratered
to 0.26–0.31: when the model reasons first, it tags wildly. The reasoning is
fluent and worthless; the answer follows the noise the reasoning introduced.

## What we actually learned

**At 0.6B, distilled reasoning does not transfer for this task.** Three cycles
of increasingly careful trace curation moved the thinking-OFF behavior around
but never once made thinking-ON help — it hurt every time, on every tier. The
model can imitate the *form* of reasoning (fluent traces, correct commitments
in isolation) without gaining the *benefit* (better answers at scale). This
matches a known pattern — small models often can't usefully distill
chain-of-thought — and now there is a controlled, three-tier, confound-removed
measurement of it on a concrete task.

Two findings are worth carrying forward:

1. **The anchor-mix extension recipe** (cycle 3, bar 1) generalizes to *any*
   future capability — new datasets, new registers, new output formats. Before
   this, we didn't know how to add a skill without risking the core; now we do.
2. **Imitation is the wrong tool for reasoning here.** Imitation rewards
   *looking like you reasoned*. The remaining untried lever is reinforcement
   learning, which rewards *getting the answer right* and lets the model
   discover whatever internal process — including none — actually helps. That
   is a materially different mechanism, and it was worth spending a run to
   close the arc rather than argue about it.

## Cycle 4: reinforcement learning, and the reward–generalization gap

Everything above is imitation — we showed the model reasoning and asked it to
copy the style. RL discards the traces entirely: the model generates its own
attempts, each is scored, and it is pushed toward whatever it did on the
high-scoring ones. We used GRPO. The reward is our own span-F1 scorer (no LLM
judge needed — these are labeled documents, so the answer key is exact, and a
drifted or malformed answer scores zero automatically). Eight attempts per
prompt, 300 steps, starting from the cycle-3 model (which already has thinking
behavior to reshape and a protected core), on ~2,700 prompts including
far-paraphrase-rewritten training rows. The model was free to think or to emit
an empty think block and answer directly — the reward only grades the answer,
so any route that improves tagging gets reinforced.

**During training, the reward climbed** — mean 0.565 in the first half of the
run to 0.612 in the second. RL was, unmistakably, optimizing. This is the
thing imitation never produced: measurable self-improvement from outcome alone.

**On the held-out exam, nothing moved.** Same three tiers, both modes,
constrained decoding:

| Tier | c3 OFF | **RL OFF** | c3 ON | RL ON |
|:--|:-:|:-:|:-:|:-:|
| Obvious | 0.915 | 0.916 | 0.398 | 0.384 |
| Slightly-off | 0.911 | 0.909 | 0.325 | 0.318 |
| Unusual | 0.853 | 0.852 | 0.340 | ~0.24 |

Thinking-off is statistically identical to where it started. Thinking-on still
collapses. RL reproduced the imitation conclusion by a completely independent
method — and left behind a sharper finding than "it didn't work":

**The reward went up and the capability didn't.** The model learned to score
better on its *training prompts* without learning anything that transfers to
held-out documents. That gap — trainable reward, untransferable skill — is the
crisp diagnosis. It's not that the optimizer failed; it's that at 0.6B, on
this task, the thing RL can grip (fit the training distribution) and the thing
we want (a generalizing reasoning ability) are not the same thing, and
optimizing the first doesn't produce the second. A larger model, with more
capacity to hold a genuine reasoning procedure rather than memorize
per-prompt reward, might close that gap. This one doesn't.

At this point the honest conclusion was "the 0.6B cannot productively think
about this task, shown two independent ways." That would have been a clean
place to stop. It was also, it turned out, a premature one — because every
experiment so far shared one variable we had not moved: **the model was always
0.6B.** Two objections reopened the arc, and both produced findings.

## Cycle 5: is it the size? — Qwen3-1.7B

The objection: RL and SFT both failed, but every run started from a 0.6B whose
thinking, when we could produce it at all, was garbage (~0.35, tagging
wildly). Maybe the wall isn't "reasoning can't help this task" but "0.6B is
too small to hold a reasoning procedure." Reinforcement learning refines
existing behavior; it can't amplify a capability that was never there. So:
run the *exact* cycle-3 recipe — same thinking-mix data, same ratios, same
evaluation — on Qwen3-1.7B instead. Only the size changes.

The wall broke, cleanly:

| Tier (thinking-ON) | 0.6B | **1.7B** |
|:--|:-:|:-:|
| Obvious | 0.398 | **0.699** |
| Slightly-off | 0.325 | **0.647** |
| Unusual | 0.340 | **0.552** |

Thinking roughly *doubled* in quality purely from tripling the parameters. At
0.6B, thinking-on was incoherent; at 1.7B it is functional — it reasons, then
tags, with drift near zero. The three-cycle 0.6B negative result was not
"distilled reasoning can't work here." It was "0.6B is too small to hold it."
The objection was right.

**But functional is not the same as beneficial.** Even the good 1.7B
thinking-on (0.55–0.70) stays *below* the same model's thinking-off
(0.81–0.91) on every tier. The model can now reason coherently — and reasoning
still costs more than it gains. Thinking became a real capability that does not
earn its place over answering directly. (A caveat in its favor: the 1.7B was
fine-tuned from a v1-quality task base with narrower alias coverage, which
depresses its thinking-off numbers slightly; a v2-quality base would lift the
whole column. It would not close a 0.26 gap.)

## Cycle 6: is a *better* thinker better? — DeepSeek-R1-Distill-1.5B

The second objection was sharper. Qwen3's reasoning is a general-purpose
byproduct of its post-training. What if we start from a model whose *entire*
post-training was reasoning distillation — DeepSeek-R1-Distill-Qwen-1.5B, a
model built to think, at essentially the same size? If "better thinking →
better thinking-on," this is the strongest possible test.

A zero-shot probe tempered expectations first: R1-Distill, given our task with
the constrained decoder handling the output format, scored a mean reward of
0.062 — the same ballpark as an untrained base model. Its reasoning did not
transfer to our task for free. But the real question was whether it would
*after* the same SFT. We ran it: same mix (re-rendered for its template, with
tagging examples wrapped in a brief think since R1-Distill always thinks — it
has no thinking-off mode), same evaluation.

It was not better. It was far worse:

| Tier (thinking-ON) | Qwen3-1.7B | **R1-Distill-1.5B** |
|:--|:-:|:-:|
| Obvious | 0.699 | **0.184** |
| Slightly-off | 0.647 | **0.142** |
| Unusual | 0.552 | **0.121** |

Nearly four times worse, at comparable size, on identical training. The
specialist reasoner reasons at length in a math-and-code solving style that
does not fit "which spans in this document are personal information," and all
that off-target deliberation actively degrades its tagging. This is the most
counterintuitive result of the arc, and the most useful:

**Reasoning ability is not a general, transferable resource.** A model
distilled to reason brilliantly about mathematics reasons *worse* about PII
spans than a plain general model of the same size. "Better thinker" only helps
if it is a better thinker *for your domain*; a reasoning specialist trained on
the wrong domain is not a neutral prior, it is a harmful one.

## The whole chapter, honestly

Four training paradigms across two model sizes and two base models, and a
coherent picture:

- **0.6B cannot hold thinking for this task** — three imitation cycles and one
  RL run all fail; thinking-on is incoherent (~0.35).
- **1.7B can hold it** — thinking becomes functional (~0.55–0.70), a clean
  demonstration that the earlier wall was capacity, not method — **but it still
  does not beat answering directly.**
- **A same-size reasoning specialist is worse, not better** — domain-locked
  reasoning is a liability, not an asset.

The synthesis: what a niche reasoning task needs is *enough capacity plus
general, adaptable reasoning* — not raw reasoning skill imported from another
domain, and not a model too small to hold a procedure at all. And even when all
of that is present, thinking has to *earn* its place against direct answering,
which on this task — where most decisions are pattern-shaped and the hard ones
are semantic-lookup rather than multi-step — it does not.

The product model is unchanged through all of it: the non-thinking 0.6B (v2)
remains SPANIEL, unbeaten by any thinking variant at any size. What the arc
produced instead of a better model is a reusable safe-extension recipe, a
think-mode constrained decoder kept for future use, and four measurements that
together map the actual shape of a limit — which is worth more than another
incremental win would have been.

Total cost of being thorough about being wrong: roughly $60 of API and GPU
time across six experiments, and a genuinely clear answer to a question most
projects would have left as a hunch.
