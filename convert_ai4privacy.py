#!/usr/bin/env python3
"""Extract the English slice of ai4privacy/pii-masking-openpii-1.5m.

Filters language=='en', validates every span (value must equal the char
slice), and writes rows in our internal format:
    {"text": ..., "spans": [{"start","end","label"}], "uid": ...}
Also reports the label vocabulary, span-length stats (1-char entities are
incompatible with the decoder's MIN_ENTITY_CHARS=2), and validation
failures.

Usage:  python convert_ai4privacy.py [--split train] [--out pii_xml/ai4privacy_en.jsonl]
"""

import argparse
import json
from collections import Counter

import requests
from huggingface_hub import get_token

REPO = "ai4privacy/pii-masking-openpii-1.5m"
COLS = ["source_text", "privacy_mask", "language", "uid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="pii_xml/ai4privacy_en.jsonl")
    args = ap.parse_args()

    url = (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
           f"data/{args.split}.jsonl")
    print(f"streaming {url}")

    labels = Counter()
    span_len_1 = 0
    bad_offsets = 0
    kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        with requests.get(url, stream=True, timeout=120,
                          headers={"Authorization": f"Bearer {get_token()}"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                row = json.loads(line)
                if row["language"] != "en":
                    continue
                text = row["source_text"]
                spans = []
                ok = True
                for s in row["privacy_mask"]:
                    if text[s["start"]:s["end"]] != s["value"]:
                        bad_offsets += 1
                        ok = False
                        break
                    labels[s["label"]] += 1
                    if s["end"] - s["start"] == 1:
                        span_len_1 += 1
                    spans.append({"start": s["start"], "end": s["end"],
                                  "label": s["label"]})
                if not ok or not spans:
                    continue
                out.write(json.dumps({"text": text, "spans": spans,
                                      "uid": row["uid"]},
                                     ensure_ascii=False) + "\n")
                kept += 1
        print(f"  stream done: kept={kept}", flush=True)

    print(f"\nDONE kept={kept}  bad_offsets={bad_offsets}  one_char_spans={span_len_1}")
    print(f"labels ({len(labels)}):")
    for lab, n in labels.most_common():
        print(f"  {lab:<28}{n:>8,}")


if __name__ == "__main__":
    main()
