# Teaching a 0.6B model to think, three times, and measuring that it didn't help

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

## The whole chapter, honestly

Two independent training paradigms — imitation (three cycles) and
reinforcement learning (one) — both say the same thing: **the 0.6B model
cannot productively think about this task.** Imitation gave a fluent-but-
useless reasoning style; RL gave a trainable-but-non-transferring reward. The
product model is unchanged: non-thinking v2 remains SPANIEL, exactly as good
as before this detour began.

What the detour produced instead of an improvement: a reusable safe-extension
recipe, a working think-mode constrained decoder (kept for any future larger
model), and two clean measurements of a real limit — one where the model
imitates the form of reasoning without the benefit, one where it optimizes a
reward without generalizing. Negative results, thoroughly earned.

Total cost of being wrong, carefully, twice: ~$40 of API and GPU time, and the
knowledge of exactly why — by two methods that agree.
