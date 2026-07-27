import unittest

from pii_eval import (
    canonical,
    count_unrequested_tags,
    parse_tagged,
)
from rl_reward import score_one
from spans_to_xml import spans_to_xml


class SpanConversionTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_offsets(self):
        text = "Call Zoë 🫠 today"
        tagged = spans_to_xml(
            text,
            [{"start": 5, "end": 8, "label": "person name"}],
        )

        recovered, spans = parse_tagged(tagged, ["person name"])
        self.assertEqual(recovered, text)
        self.assertEqual(spans, [(5, 8, "person name")])

    def test_invalid_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            spans_to_xml(
                "short",
                [{"start": -1, "end": 2, "label": "x"}],
            )
        with self.assertRaises(ValueError):
            spans_to_xml(
                "short",
                [{"start": 2, "end": 2, "label": "x"}],
            )


class EvaluationTests(unittest.TestCase):
    def test_unrequested_tags_ignore_literal_source_markup(self):
        source = "<div>Hello</div>"
        output = "<div>Hello</div><secret>extra</secret>"

        self.assertEqual(
            count_unrequested_tags(output, source, ["person"]),
            2,
        )

    def test_v2_aliases_are_canonicalized(self):
        self.assertEqual(
            canonical("ai4privacy", "first name"),
            "ai4privacy:GIVENNAME",
        )
        self.assertEqual(
            canonical("nemotron", "account identifier"),
            "nemotron:account identifier",
        )


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.source = "Reach Anita at anita.k@corp.io."
        self.gold = (
            "Reach <person name>Anita</person name> at "
            "<email>anita.k@corp.io</email>."
        )
        self.names = ["person name", "email"]

    def test_perfect_output_gets_full_reward(self):
        self.assertEqual(
            score_one(self.gold, self.gold, self.names, self.source),
            1.0,
        )

    def test_unclosed_tag_forfeits_reward(self):
        malformed = (
            "Reach <person name>Anita</person name> at "
            "<email>anita.k@corp.io."
        )
        self.assertEqual(
            score_one(malformed, self.gold, self.names, self.source),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
