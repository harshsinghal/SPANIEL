#!/usr/bin/env python3
"""Classify span-level disagreements between a prediction file and gold.

For each requested canonical label, splits errors into:
  boundary   — pred overlaps gold, same label, wrong edges
  miss       — gold span with no same-label overlapping pred
  spurious   — pred span with no same-label overlapping gold
  label_swap — pred overlaps gold but with a different label

Usage:
    python error_autopsy.py --pred baseline/preds_full_constrained.jsonl \
        --labels occupation date_time --n 300 [--examples 8]
"""

import argparse
import json
from collections import Counter, defaultdict

from pii_eval import parse_example, parse_tagged, canonical


def spans_for(rows_gold, rows_pred, n):
    for gold_ex, pred in zip(rows_gold[:n], rows_pred[:n]):
        source = gold_ex.get("source", "nemotron")
        text, names = parse_example(gold_ex)
        _, gold_spans = parse_tagged(gold_ex["messages"][2]["content"], names)
        try:
            _, pred_spans = parse_tagged(pred["output"], names)
        except ValueError:
            continue
        gold_c = [(s, e, canonical(source, x)) for s, e, x in gold_spans]
        pred_c = [(s, e, canonical(source, x)) for s, e, x in pred_spans]
        yield text, gold_c, pred_c


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="sft_data/sft_eval.jsonl")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    want = {f"nemotron:{l}" for l in args.labels}
    rows_gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    rows_pred = [json.loads(l) for l in open(args.pred, encoding="utf-8")]

    counts = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))

    for text, gold_c, pred_c in spans_for(rows_gold, rows_pred, args.n):
        for g in gold_c:
            if g[2] not in want:
                continue
            exact = any(p == g for p in pred_c)
            if exact:
                counts[g[2]]["exact"] += 1
                continue
            same = [p for p in pred_c if p[2] == g[2] and overlap(p, g)]
            other = [p for p in pred_c if p[2] != g[2] and overlap(p, g)]
            if same:
                counts[g[2]]["boundary"] += 1
                examples[g[2]]["boundary"].append(
                    (text, g, same[0]))
            elif other:
                counts[g[2]]["label_swap"] += 1
                examples[g[2]]["label_swap"].append((text, g, other[0]))
            else:
                counts[g[2]]["miss"] += 1
                examples[g[2]]["miss"].append((text, g, None))
        for p in pred_c:
            if p[2] in want and not any(
                    g[2] == p[2] and overlap(p, g) for g in gold_c):
                counts[p[2]]["spurious"] += 1
                examples[p[2]]["spurious"].append((text, None, p))

    for label in sorted(counts):
        c = counts[label]
        total = sum(c.values())
        print(f"\n===== {label}  (gold-side events: {total}) =====")
        for kind in ("exact", "boundary", "miss", "label_swap", "spurious"):
            print(f"  {kind:<11}{c[kind]:>4}")
        for kind in ("boundary", "miss", "label_swap", "spurious"):
            exs = examples[label][kind][:args.examples]
            if not exs:
                continue
            print(f"\n  --- {kind} examples ---")
            for text, g, p in exs:
                if g and p:
                    s = min(g[0], p[0])
                    print(f"    gold={text[g[0]:g[1]]!r}  pred={text[p[0]:p[1]]!r}"
                          f"  ctx=...{text[max(0,s-30):s]!r}")
                elif g:
                    print(f"    gold={text[g[0]:g[1]]!r}"
                          f"  ctx=...{text[max(0,g[0]-45):g[1]+15]!r}")
                else:
                    print(f"    pred={text[p[0]:p[1]]!r}"
                          f"  ctx=...{text[max(0,p[0]-45):p[1]+15]!r}")


if __name__ == "__main__":
    main()
