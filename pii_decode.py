#!/usr/bin/env python3
"""Copy-or-tag constrained decoding for the PII tagger.

At every generation step the model may only emit tokens whose text either
(a) continues an exact character-level copy of the source text from the
current position, or (b) opens/closes a tag from the requested name set at
a position between copied characters. Copy drift and malformed tags become
unrepresentable; EOS is only reachable when the source is fully copied and
no tag is open.

Usage:
    fn = CopyTagConstraint(tokenizer, text, names).prefix_allowed_tokens_fn()
    model.generate(..., prefix_allowed_tokens_fn=fn)

The vocabulary trie is built once per tokenizer and cached on the instance
of VocabTrie you pass around (build takes ~10s for a 150k vocab).
"""

import re


class VocabTrie:
    """Character trie over decoded token strings: node = {char: node}, ids under key 0."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.root = {}
        vocab_size = len(tokenizer)
        specials = set(tokenizer.all_special_ids)
        # batch-decode the whole vocab in chunks (fast in the Rust tokenizer)
        for tid in range(vocab_size):
            if tid in specials:
                continue
            s = tokenizer.decode([tid])
            if not s or "�" in s:   # partial UTF-8 byte tokens
                continue
            node = self.root
            for ch in s:
                node = node.setdefault(ch, {})
            node.setdefault(0, []).append(tid)

    def tokens_along(self, text, max_chars=64):
        """All token ids whose string is a non-empty prefix of `text`."""
        out, node = [], self.root
        for ch in text[:max_chars]:
            node = node.get(ch)
            if node is None:
                break
            ids = node.get(0)
            if ids:
                out.extend(ids)
        return out


class CopyTagConstraint:
    """Stateless per-step constraint; state is replayed from the generated text.

    Replaying (rather than tracking) state makes the function robust to the
    generate() implementation and costs O(len^2) chars per sequence, which is
    negligible at document scale.
    """

    def __init__(self, trie, source, names, eos_token_id):
        self.trie = trie
        self.source = source
        self.names = sorted(names, key=len, reverse=True)
        self.eos = eos_token_id
        self.tag_re = re.compile(
            "^<(/?)(" + "|".join(re.escape(n) for n in self.names) + ")>")

    def _tag_candidates(self, open_tag):
        """Complete tag strings valid in the current open/closed state."""
        if open_tag is not None:
            return [f"</{open_tag}>"]
        return [f"<{n}>" for n in self.names]

    MIN_ENTITY_CHARS = 2   # no 1-char entities exist in gold; forbidding them
                           # kills checkbox-glyph and stray-letter spurious tags

    def _state(self, generated):
        """Replay generated text -> (copy_pos, open_tag, open_pos, pending, valid).

        A partially-typed tag can only exist at the very end of the generated
        text; it is returned as `pending` so allowed() can complete it.
        """
        pos, open_tag, open_pos, i = 0, None, 0, 0
        while i < len(generated):
            rest = generated[i:]
            m = self.tag_re.match(rest)
            # prefer the tag reading when the source itself doesn't continue with it
            if m and not (pos < len(self.source)
                          and self.source[pos:].startswith(m.group(0))):
                closing, name = m.group(1) == "/", m.group(2)
                if closing:
                    if open_tag != name:
                        return pos, open_tag, open_pos, "", False
                    open_tag = None
                else:
                    if open_tag is not None:
                        return pos, open_tag, open_pos, "", False
                    open_tag = name
                    open_pos = pos
                i += len(m.group(0))
                continue
            if pos < len(self.source) and generated[i] == self.source[pos]:
                pos += 1
                i += 1
                continue
            # trailing partial tag: rest is a proper prefix of a valid tag
            if any(c.startswith(rest) and len(rest) < len(c)
                   for c in self._tag_candidates(open_tag)):
                return pos, open_tag, open_pos, rest, True
            return pos, open_tag, open_pos, "", False
        return pos, open_tag, open_pos, "", True

    def allowed(self, generated_text):
        pos, open_tag, open_pos, pending, ok = self._state(generated_text)
        if not ok:      # should not happen under constraint; fail open to EOS
            return [self.eos]
        remaining = self.source[pos:]
        allowed = []
        min_ent = self.MIN_ENTITY_CHARS
        if pending:
            # must finish the tag being typed, then copying resumes
            for c in self._tag_candidates(open_tag):
                if c.startswith(pending) and len(pending) < len(c):
                    opening = not c.startswith("</")
                    if opening and len(remaining) < min_ent:
                        continue
                    allowed.extend(
                        self.trie.tokens_along(c[len(pending):] + remaining))
            return list(set(allowed)) or [self.eos]
        # (a) continue copying
        if remaining:
            allowed.extend(self.trie.tokens_along(remaining))
        # (b) open a tag, or close the open one. BPE merges tag brackets into
        # the preceding text (" <", ": <", "08</"), so a single token may span
        # j copied characters plus the tag start — offer the tag at every
        # copy offset j within a small lookahead, not just at offset 0.
        lookahead = min(8, len(remaining))
        if open_tag is None:
            for n in self.names:
                tag = f"<{n}>"
                for j in range(lookahead):
                    if len(remaining) - j < min_ent:
                        break               # entity needs >= min_ent chars
                    allowed.extend(self.trie.tokens_along(
                        remaining[:j] + tag + remaining[j:]))
        else:
            close = f"</{open_tag}>"
            start_j = max(0, min_ent - (pos - open_pos))
            for j in range(start_j, lookahead + 1):
                allowed.extend(self.trie.tokens_along(
                    remaining[:j] + close + remaining[j:]))
        # (c) finish
        if not remaining and open_tag is None:
            allowed.append(self.eos)
        return list(set(allowed)) or [self.eos]

    def make_fn(self, tokenizer, prompt_len, think=False, think_budget=512):
        """prefix_allowed_tokens_fn for transformers generate().

        When think=True the model reasons freely inside a <think>...</think>
        block (all tokens allowed) before the copy-or-tag automaton engages on
        the answer. The block is force-closed if it exceeds think_budget tokens.
        The answer after </think> is fully constrained exactly as usual, so the
        copy guarantee is intact regardless of what the reasoning contained.
        """
        if not think:
            def fn(batch_id, input_ids):
                gen = tokenizer.decode(input_ids[prompt_len:],
                                       skip_special_tokens=True)
                return self.allowed(gen)
            return fn

        all_ids = list(range(len(tokenizer)))
        close_ids = tokenizer("</think>", add_special_tokens=False).input_ids

        def fn(batch_id, input_ids):
            gen_ids = input_ids[prompt_len:]
            gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
            close = gen.find("</think>")
            if close == -1:
                # still thinking: free generation, but force-close on budget
                if len(gen_ids) >= think_budget:
                    return close_ids
                return all_ids
            # answer phase: constrain only the text after the think block
            answer = gen[close + len("</think>"):].lstrip("\n")
            return self.allowed(answer)
        return fn
        return fn
