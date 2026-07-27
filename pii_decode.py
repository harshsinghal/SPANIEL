#!/usr/bin/env python3
"""Copy-or-tag constrained decoding for the PII tagger.

At every generation step the model may only emit tokens whose text either
(a) continues an exact character-level copy of the source text from the
current position, or (b) opens/closes a tag from the requested name set at
a position between copied characters. Copy drift and malformed tags become
unrepresentable; EOS is only reachable when the source is fully copied and
no tag is open.

Usage:
    trie = VocabTrie(tokenizer)
    constraint = CopyTagConstraint(trie, text, names, tokenizer.eos_token_id)
    fn = constraint.make_fn(tokenizer, prompt_len)
    model.generate(..., prefix_allowed_tokens_fn=fn)

Build one vocabulary trie per tokenizer and reuse it across requests.
"""

import re


def _gpt2_byte_decoder():
    """Inverse of the byte-to-Unicode alphabet used by byte-level BPEs."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value
            for value, codepoint in zip(byte_values, codepoints)}


_BYTE_DECODER = _gpt2_byte_decoder()


class VocabTrie:
    """Byte trie over token payloads; token ids are stored under ``None``.

    Matching bytes rather than independently decoded token strings matters for
    byte-fallback tokens. A token that carries only the first bytes of an emoji
    decodes to U+FFFD on its own, but is still a valid prefix of the source.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.root = {}
        self.token_bytes = {}
        vocab_size = len(tokenizer)
        specials = set(tokenizer.all_special_ids)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        byte_level = "ByteLevel" in repr(decoder)
        for tid in range(vocab_size):
            if tid in specials:
                continue
            payload = None
            if byte_level:
                raw = tokenizer.convert_ids_to_tokens(tid)
                if raw and all(ch in _BYTE_DECODER for ch in raw):
                    payload = bytes(_BYTE_DECODER[ch] for ch in raw)
            if payload is None:
                decoded = tokenizer.decode([tid])
                if decoded and "�" not in decoded:
                    payload = decoded.encode("utf-8")
            if not payload:
                continue
            self.token_bytes[tid] = payload
            node = self.root
            for byte in payload:
                node = node.setdefault(byte, {})
            node.setdefault(None, []).append(tid)

    def tokens_along(self, value, max_bytes=64):
        """All token ids whose payload is a non-empty prefix of ``value``."""
        if isinstance(value, str):
            value = value.encode("utf-8")
        out, node = [], self.root
        for byte in value[:max_bytes]:
            node = node.get(byte)
            if node is None:
                break
            ids = node.get(None)
            if ids:
                out.extend(ids)
        return out

    def bytes_for_ids(self, token_ids):
        """Return the exact byte payload for generated, non-special token ids."""
        out = bytearray()
        for tid in token_ids:
            payload = self.token_bytes.get(int(tid))
            if payload is None:
                if int(tid) in self.tokenizer.all_special_ids:
                    continue
                raise ValueError(f"token id {int(tid)} has no byte representation")
            out.extend(payload)
        return bytes(out)


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
        self.source_bytes = source.encode("utf-8")
        self.name_bytes = [name.encode("utf-8") for name in self.names]
        self.tag_re = re.compile(
            b"^<(/?)(" + b"|".join(re.escape(n) for n in self.name_bytes) + b")>")
        self.boundaries = [0]
        for char in source:
            self.boundaries.append(
                self.boundaries[-1] + len(char.encode("utf-8")))
        self.boundary_to_char = {
            byte_pos: char_pos
            for char_pos, byte_pos in enumerate(self.boundaries)
        }

    def _tag_candidates(self, open_tag):
        """Complete tag strings valid in the current open/closed state."""
        if open_tag is not None:
            return [b"</" + open_tag + b">"]
        return [b"<" + name + b">" for name in self.name_bytes]

    MIN_ENTITY_CHARS = 2   # no 1-char entities exist in gold; forbidding them
                           # kills checkbox-glyph and stray-letter spurious tags

    def _state(self, generated):
        """Replay generated bytes -> (copy_pos, open_tag, open_pos, pending, valid).

        A partially-typed tag can only exist at the very end of the generated
        text; it is returned as `pending` so allowed() can complete it.
        """
        if isinstance(generated, str):
            generated = generated.encode("utf-8")
        pos, open_tag, open_pos, i = 0, None, 0, 0
        while i < len(generated):
            rest = generated[i:]
            m = self.tag_re.match(rest)
            # prefer the tag reading when the source itself doesn't continue with it
            if m and not (pos < len(self.source_bytes)
                          and self.source_bytes[pos:].startswith(m.group(0))):
                closing, name = m.group(1) == b"/", m.group(2)
                if closing:
                    if open_tag != name:
                        return pos, open_tag, open_pos, b"", False
                    if (pos not in self.boundary_to_char
                            or self.boundary_to_char[pos]
                            - self.boundary_to_char[open_pos]
                            < self.MIN_ENTITY_CHARS):
                        return pos, open_tag, open_pos, b"", False
                    open_tag = None
                else:
                    if open_tag is not None:
                        return pos, open_tag, open_pos, b"", False
                    if pos not in self.boundary_to_char:
                        return pos, open_tag, open_pos, b"", False
                    open_tag = name
                    open_pos = pos
                i += len(m.group(0))
                continue
            if (pos < len(self.source_bytes)
                    and generated[i] == self.source_bytes[pos]):
                pos += 1
                i += 1
                continue
            # trailing partial tag: rest is a proper prefix of a valid tag
            if any(c.startswith(rest) and len(rest) < len(c)
                   for c in self._tag_candidates(open_tag)):
                return pos, open_tag, open_pos, rest, True
            return pos, open_tag, open_pos, b"", False
        return pos, open_tag, open_pos, b"", True

    def allowed(self, generated_text):
        """Allowed token ids after a generated string or exact byte prefix."""
        if isinstance(generated_text, str):
            generated_text = generated_text.encode("utf-8")
        pos, open_tag, open_pos, pending, ok = self._state(generated_text)
        if not ok:
            raise ValueError("generated prefix violated the copy-or-tag grammar")
        remaining = self.source_bytes[pos:]
        allowed = []
        min_ent = self.MIN_ENTITY_CHARS
        if pending:
            # must finish the tag being typed, then copying resumes
            for c in self._tag_candidates(open_tag):
                if c.startswith(pending) and len(pending) < len(c):
                    opening = not c.startswith(b"</")
                    remaining_chars = len(self.source) - self.boundary_to_char.get(
                        pos, len(self.source))
                    if opening and remaining_chars < min_ent:
                        continue
                    allowed.extend(
                        self.trie.tokens_along(c[len(pending):] + remaining))
            if not allowed:
                raise ValueError("no token can complete the pending tag")
            return list(set(allowed))
        # (a) continue copying
        if remaining:
            allowed.extend(self.trie.tokens_along(remaining))
        # (b) open a tag, or close the open one. BPE merges tag brackets into
        # the preceding text (" <", ": <", "08</"), so a single token may span
        # j copied characters plus the tag start — offer the tag at every
        # copy offset j within a small lookahead, not just at offset 0.
        boundary_index = self.boundary_to_char.get(pos)
        candidate_positions = []
        if boundary_index is None:
            candidate_positions = [
                byte_pos for byte_pos in self.boundaries
                if byte_pos > pos
            ][:8]
        else:
            candidate_positions = self.boundaries[
                boundary_index:boundary_index + 8]
        if open_tag is None:
            for name in self.name_bytes:
                tag = b"<" + name + b">"
                for candidate_pos in candidate_positions:
                    char_pos = self.boundary_to_char[candidate_pos]
                    if len(self.source) - char_pos < min_ent:
                        break
                    j = candidate_pos - pos
                    allowed.extend(self.trie.tokens_along(
                        remaining[:j] + tag + remaining[j:]))
        else:
            close = b"</" + open_tag + b">"
            open_char = self.boundary_to_char[open_pos]
            for candidate_pos in candidate_positions:
                if (self.boundary_to_char[candidate_pos] - open_char
                        < min_ent):
                    continue
                j = candidate_pos - pos
                allowed.extend(self.trie.tokens_along(
                    remaining[:j] + close + remaining[j:]))
        # (c) finish
        if not remaining and open_tag is None:
            allowed.append(self.eos)
        if not allowed:
            raise ValueError("no valid token transition for source prefix")
        return list(set(allowed))

    def is_complete(self, generated_text):
        """Whether ``generated_text`` is a complete, faithful tagged copy."""
        pos, open_tag, _, pending, ok = self._state(generated_text)
        return (
            ok
            and pos == len(self.source_bytes)
            and open_tag is None
            and not pending
        )

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
                gen = self.trie.bytes_for_ids(
                    input_ids[prompt_len:].tolist())
                return self.allowed(gen)
            return fn

        all_ids = [
            token_id for token_id in range(len(tokenizer))
            if token_id != self.eos
        ]
        close_bytes = b"</think>"

        def fn(batch_id, input_ids):
            gen_ids = input_ids[prompt_len:]
            gen = self.trie.bytes_for_ids(gen_ids.tolist())
            close = gen.find(b"</think>")
            if close == -1:
                # still thinking: free generation, but force-close on budget
                if len(gen_ids) >= think_budget:
                    pending = 0
                    for length in range(
                            1, min(len(gen), len(close_bytes) - 1) + 1):
                        if close_bytes.startswith(gen[-length:]):
                            pending = length
                    allowed = self.trie.tokens_along(
                        close_bytes[pending:])
                    if not allowed:
                        raise ValueError("no token can close the think block")
                    return allowed
                return all_ids
            # answer phase: constrain only the text after the think block
            answer = gen[close + len(b"</think>"):].lstrip(b"\n")
            return self.allowed(answer)
        return fn
