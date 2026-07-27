import unittest

from pii_decode import CopyTagConstraint, VocabTrie, _BYTE_DECODER


_BYTE_ENCODER = {value: char for char, value in _BYTE_DECODER.items()}


class _ByteLevelDecoder:
    def __repr__(self):
        return "ByteLevel(add_prefix_space=True)"


class _Backend:
    decoder = _ByteLevelDecoder()


class FakeByteTokenizer:
    """Small byte-level tokenizer with one emoji split across three tokens."""

    backend_tokenizer = _Backend()
    eos_token_id = 8
    all_special_ids = [eos_token_id]
    payloads = [
        b"\xf0\x9f",
        b"\xab",
        b"\xa0",
        b"A",
        b"B",
        b"<",
        b">",
        b"x",
        b"",
    ]

    def __len__(self):
        return len(self.payloads)

    def convert_ids_to_tokens(self, token_id):
        return "".join(_BYTE_ENCODER[byte]
                       for byte in self.payloads[token_id])

    def decode(self, token_ids):
        return b"".join(self.payloads[token_id] for token_id in token_ids) \
            .decode("utf-8", errors="replace")


class CopyTagConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeByteTokenizer()
        self.trie = VocabTrie(self.tokenizer)

    def test_byte_fallback_tokens_can_copy_an_emoji(self):
        constraint = CopyTagConstraint(
            self.trie, "🫠", ["person"], self.tokenizer.eos_token_id)

        self.assertIn(0, constraint.allowed(b""))
        self.assertIn(1, constraint.allowed(b"\xf0\x9f"))
        self.assertIn(2, constraint.allowed(b"\xf0\x9f\xab"))
        self.assertIn(
            self.tokenizer.eos_token_id,
            constraint.allowed("🫠"),
        )
        self.assertTrue(constraint.is_complete("🫠"))

    def test_invalid_generated_prefix_fails_loudly(self):
        constraint = CopyTagConstraint(
            self.trie, "AB", ["x"], self.tokenizer.eos_token_id)

        with self.assertRaises(ValueError):
            constraint.allowed("Z")

    def test_completion_validation_understands_literal_source_tags(self):
        constraint = CopyTagConstraint(
            self.trie, "Use <x> literally", ["x"],
            self.tokenizer.eos_token_id)

        self.assertTrue(constraint.is_complete("Use <x> literally"))
        self.assertFalse(constraint.is_complete("<x>A</x>"))


if __name__ == "__main__":
    unittest.main()
