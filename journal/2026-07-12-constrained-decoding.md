# Making drift unrepresentable: a copy-or-tag automaton over a vocabulary trie

*2026-07-12 (measured 2026-07-18) — how the decoder became part of the model*

## Tendency vs. property

After SFT, our 0.6B model violated the copy contract on 4.7% of eval rows —
duplicated periods after sentence-final abbreviations, occasional truncations.
Fifty thousand examples of exact copying make faithful output very *probable*;
nothing makes it *certain*. An autoregressive decoder samples from a
next-token distribution, and copy fidelity trained-in is a tendency, not a
property.

The fix operates at decode time. At every generation step, mask the logits so
the model may only emit tokens that either (a) continue an exact
character-level copy of the source from the current position, or (b) open or
close a tag from the requested label set. EOS is only reachable once the
source is fully copied with no tag open. Under this automaton, drift and
malformed tags are not unlikely — they are **unrepresentable**. The intellectual
ancestry is GENRE's trie-constrained beam search and the Outlines/XGrammar
FSA compilers; ours differs in that the "grammar" is dynamic (it embeds the
request's document), so transitions are computed on demand.

## The trie

The engineering problem is a unit mismatch: rules defined over *characters*,
a model that emits *tokens* from a 151k-entry vocabulary. Testing every token
against the allowed continuation each step is billions of character
comparisons per document. The fix is a character trie over the decoded
vocabulary, built once (~0.5s): to find all tokens that legally continue a
copy, walk the *remaining source text* down the trie and collect token ids at
every node passed — each collected token is exactly a prefix of what must come
next. Cost per walk: bounded by the longest token, independent of vocabulary
size. A handful of walks per step (copy path + one per candidate tag) replaces
a 151k-item scan.

## Two bugs that generalize

Anyone building constrained decoding will meet these:

1. **Partially-typed tags.** `<` is a legal prefix of `<forename>`, so the
   model can emit it — and the state replay must recognize a trailing fragment
   as a *pending tag*, not an invalid state. Our first version killed
   generation mid-tag.
2. **BPE merges span rule boundaries.** The model never learned to open a
   mid-sentence tag with a bare `<` token — its training data tokenized
   " <" (space + bracket) as a single merged token. Constraining tags to open
   exactly at the copy position masked the model's *preferred* token at every
   tag site; greedy decoding silently took the best still-legal path — plain
   copying — and recall collapsed to 0.07 with *perfect* copy fidelity. The
   fix offers tag openings across a small lookahead of copy offsets so merged
   tokens stay legal. The lesson: **a constraint defined in character space
   must be closed under the tokenizer's merge table**, or you are not
   constraining the model, you are quietly steering it away from its trained
   behavior.

A later addition, informed by error analysis: a 2-character minimum entity
length (no gold entity in 15,212 spans is shorter), which mechanically
eliminated a class of stray-glyph false positives.

## Results

Same 0.6B model, same eval, decoder swapped:

| | Unconstrained | Constrained |
|---|---|---|
| Strict exact F1 | 0.873 | **0.898** |
| Recall | 0.830 | 0.890 |
| Copy drift | 4.7% | 0.3%* |
| Malformed tags | 0.3% | 0.0% |

*The single residual "drift" row is a max-token truncation, not an automaton
failure.*

The +2.5 F1 came almost entirely through recall: entities that drifted rows
used to forfeit are recovered because those rows can no longer drift. And the
constrained decode was slightly *faster* — tokens the model cannot emit are
tokens it never spends time considering.

Architecturally, the constraint is not a wrapper around the model. It lives in
the slot every serving stack already has — between the forward pass and the
sampler, beside temperature and top-p — and requires logit access, which means
owning the serving loop. That is a concrete, demonstrable advantage of running
your own weights: no API exposes a per-request dynamic grammar. The model and
the automaton are one system; the copy-fidelity check in our eval harness
stopped being a metric and became a discharged proof obligation.
