#!/usr/bin/env python3
"""Convert (text, spans) pairs into text with XML tags around the entities.

Input spans use the format from gravitee-io/pii-detection-dataset:
    [{"start": 78, "end": 88, "label": "DATE_TIME"}, ...]

Output for the example row:
    ... License Plate <US_LICENSE_PLATE>33-202773-98</US_LICENSE_PLATE>, dated <DATE_TIME>1993.08.01</DATE_TIME> ...

Usage:
    python spans_to_xml.py record.json    # single JSON object {"text": ..., "spans": [...]}
    python spans_to_xml.py data.jsonl     # one JSON object per line
    python spans_to_xml.py                # built-in demo example
"""

import json
import sys


def spans_to_xml(text: str, spans: list[dict]) -> str:
    """Return `text` with each span wrapped in <LABEL>...</LABEL> tags.

    Spans are inserted from the end of the text backwards so that earlier
    character offsets stay valid while tags are being added.
    Overlapping spans raise a ValueError since they can't be expressed
    as well-formed sibling tags.
    """
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for prev, cur in zip(ordered, ordered[1:]):
        if cur["start"] < prev["end"]:
            raise ValueError(f"Overlapping spans: {prev} and {cur}")

    result = text
    for span in reversed(ordered):
        start, end, label = span["start"], span["end"], span["label"]
        result = f"{result[:start]}<{label}>{result[start:end]}</{label}>{result[end:]}"
    return result


def convert_record(record: dict) -> str:
    return spans_to_xml(record["text"], record["spans"])


def main() -> None:
    if len(sys.argv) < 2:
        demo = {
            "text": (
                "Loading Plan for Vehicle 619JWTWPMAPHBTAFJ, License Plate "
                "33-202773-98, dated 1993.08.01. Prepared by Employee "
                "Gi-94938. Final delivery date: 1994.12.26."
            ),
            "spans": [
                {"start": 78, "end": 88, "label": "DATE_TIME"},
                {"start": 142, "end": 152, "label": "DATE_TIME"},
                {"start": 58, "end": 70, "label": "US_LICENSE_PLATE"},
            ],
        }
        print(convert_record(demo))
        return

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    print(convert_record(json.loads(line)))
        else:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
            for record in records:
                print(convert_record(record))


if __name__ == "__main__":
    main()
