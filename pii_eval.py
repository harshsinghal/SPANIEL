#!/usr/bin/env python3
"""Span-level scorer for tagged-text PII extraction.

Scores model outputs against a gold file produced by build_sft_dataset.py.
The model's output for each example is the input text regenerated with
<name>entity</name> tags, where names come from the example's request list.

Pipeline per example:
  1. Parse the tagged output with the request's names: recover (start, end,
     name) character spans in original-text space, or classify the row as
     malformed (bad tag structure) / copy drift (untagged text != input).
  2. Match predicted spans to gold spans:
       exact   — same start, end, and label
       relaxed — character overlap with same label (boundary-error diagnostic)
  3. Aggregate micro P/R/F1 overall, per canonical label, and per source.

Rows whose output can't be trusted (malformed / drift) count all their gold
spans as misses in the headline "strict" metrics; "parseable-only" metrics
are also reported. Operational failure rates are first-class outputs.

Usage:
    python pii_eval.py --pred preds.jsonl [--gold sft_data/sft_eval.jsonl]
    python pii_eval.py --self-test        # gold vs gold, must be all 1.0

Prediction file: JSONL aligned line-by-line with the gold file, each line
{"output": "<tagged text>"} (field name configurable via --pred-field).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_alias_to_slug():
    path = Path(__file__).with_name("label_aliases.json")
    aliases = json.loads(path.read_text())
    return {ds: {name: slug for slug, names in aliases[ds].items() for name in names}
            for ds in ("gravitee", "nemotron")}


ALIAS_TO_SLUG = load_alias_to_slug()


def parse_example(example):
    """Extract (text, request_names) from a gold chat example."""
    user = example["messages"][1]["content"]
    head, text = user.split("\n\nText:\n", 1)
    names = [line[2:] for line in head.splitlines()[1:]]
    return text, names


def parse_tagged(tagged, names):
    """Parse tagged text into (recovered_text, spans, unrequested_count).

    spans are (start, end, name) in recovered-text coordinates. Only tags for
    the requested names are treated as markup; anything else (including
    literal <...> in the source text) passes through as text. Raises
    ValueError on bad structure: nesting, stray close, unclosed tag.
    """
    token = re.compile(
        "<(/?)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ")>")
    out, spans = [], []
    pos, length = 0, 0
    open_name, open_at = None, None
    for m in token.finditer(tagged):
        chunk = tagged[pos:m.start()]
        out.append(chunk)
        length += len(chunk)
        closing, name = m.group(1) == "/", m.group(2)
        if not closing:
            if open_name is not None:
                raise ValueError(f"nested tag <{name}> inside <{open_name}>")
            open_name, open_at = name, length
        else:
            if open_name != name:
                raise ValueError(f"unmatched closing tag </{name}>")
            spans.append((open_at, length, name))
            open_name = None
        pos = m.end()
    if open_name is not None:
        raise ValueError(f"unclosed tag <{open_name}>")
    out.append(tagged[pos:])
    length += len(tagged) - pos
    return "".join(out), spans


def canonical(source, name):
    return f"{source}:{ALIAS_TO_SLUG.get(source, {}).get(name, name)}"


class Tally:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def prf(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f


def match_spans(gold, pred, relaxed=False):
    """Greedy 1:1 matching. Returns (tp, fp, fn) and per-label tallies."""
    unmatched_pred = list(pred)
    tp = 0
    matched_gold = set()
    for gi, g in enumerate(gold):
        for p in unmatched_pred:
            if g[2] != p[2]:
                continue
            hit = (g[:2] == p[:2]) if not relaxed else (p[0] < g[1] and g[0] < p[1])
            if hit:
                unmatched_pred.remove(p)
                matched_gold.add(gi)
                tp += 1
                break
    return tp, len(unmatched_pred), len(gold) - len(matched_gold), matched_gold, unmatched_pred


def score(gold_rows, pred_rows, pred_field):
    ops = {"rows": 0, "malformed": 0, "copy_drift": 0, "unrequested_tags": 0}
    strict = {"exact": Tally(), "relaxed": Tally()}
    parseable = {"exact": Tally(), "relaxed": Tally()}
    per_label = defaultdict(Tally)          # exact, strict
    per_source = defaultdict(lambda: {"exact": Tally(), "rows": 0})

    for gold_ex, pred in zip(gold_rows, pred_rows):
        ops["rows"] += 1
        source = gold_ex.get("source", "nemotron")
        text, names = parse_example(gold_ex)
        _, gold_spans = parse_tagged(gold_ex["messages"][2]["content"], names)
        gold_canon = [(s, e, canonical(source, n)) for s, e, n in gold_spans]
        per_source[source]["rows"] += 1

        output = pred[pred_field]
        # tags outside the request list: count occurrences of <word-ish> pairs
        # that our parser will treat as text but look like attempted tags
        row_bad = None
        try:
            recovered, pred_spans = parse_tagged(output, names)
        except ValueError:
            row_bad = "malformed"
        else:
            if recovered != text:
                row_bad = "copy_drift"

        if row_bad:
            ops[row_bad] += 1
            for tally in (strict["exact"], strict["relaxed"]):
                tally.fn += len(gold_canon)
            for _, _, c in gold_canon:
                per_label[c].fn += 1
            per_source[source]["exact"].fn += len(gold_canon)
            continue

        pred_canon = [(s, e, canonical(source, n)) for s, e, n in pred_spans]
        for mode, relaxed in (("exact", False), ("relaxed", True)):
            tp, fp, fn, _, _ = match_spans(gold_canon, pred_canon, relaxed)
            for tally in (strict[mode], parseable[mode]):
                tally.tp += tp
                tally.fp += fp
                tally.fn += fn
        tp, fp, fn, matched_gold, unmatched_pred = match_spans(gold_canon, pred_canon)
        per_source[source]["exact"].tp += tp
        per_source[source]["exact"].fp += fp
        per_source[source]["exact"].fn += fn
        for gi, (_, _, c) in enumerate(gold_canon):
            if gi in matched_gold:
                per_label[c].tp += 1
            else:
                per_label[c].fn += 1
        for _, _, c in unmatched_pred:
            per_label[c].fp += 1

    return ops, strict, parseable, per_label, per_source


def report(ops, strict, parseable, per_label, per_source):
    lines = []
    n = ops["rows"]
    lines.append(f"rows scored           {n}")
    lines.append(f"malformed output      {ops['malformed']} ({ops['malformed']/n:.1%})")
    lines.append(f"copy drift            {ops['copy_drift']} ({ops['copy_drift']/n:.1%})")
    lines.append("")
    for name, tallies in (("STRICT (headline)", strict), ("parseable rows only", parseable)):
        p, r, f = tallies["exact"].prf()
        rp, rr, rf = tallies["relaxed"].prf()
        lines.append(f"{name}")
        lines.append(f"  exact   P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
        lines.append(f"  relaxed P {rp:.3f}  R {rr:.3f}  F1 {rf:.3f}   (overlap, same label)")
    lines.append("")
    lines.append("per source (exact, strict):")
    for src, d in sorted(per_source.items()):
        p, r, f = d["exact"].prf()
        lines.append(f"  {src:<10} rows {d['rows']:>6}  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    lines.append("")
    lines.append(f"per label (exact, strict), worst first:")
    rows = [(c, t, t.prf()) for c, t in per_label.items() if t.tp + t.fn > 0]
    rows.sort(key=lambda x: x[2][2])
    for c, t, (p, r, f) in rows:
        lines.append(f"  {c:<45} support {t.tp + t.fn:>5}  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="sft_data/sft_eval.jsonl")
    ap.add_argument("--pred")
    ap.add_argument("--pred-field", default="output")
    ap.add_argument("--json", help="also write metrics to this JSON file")
    ap.add_argument("--n", type=int, help="score only the first N gold rows")
    ap.add_argument("--self-test", action="store_true",
                    help="score gold targets against themselves; must be all 1.0")
    args = ap.parse_args()

    gold_rows = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    if args.n:
        gold_rows = gold_rows[:args.n]
    if args.self_test:
        pred_rows = [{"output": g["messages"][2]["content"]} for g in gold_rows]
        pred_field = "output"
    else:
        if not args.pred:
            sys.exit("--pred required unless --self-test")
        pred_rows = [json.loads(l) for l in open(args.pred, encoding="utf-8")]
        pred_field = args.pred_field
        if len(pred_rows) != len(gold_rows):
            sys.exit(f"prediction rows ({len(pred_rows)}) != gold rows ({len(gold_rows)})")

    results = score(gold_rows, pred_rows, pred_field)
    print(report(*results))

    if args.json:
        ops, strict, parseable, per_label, per_source = results
        blob = {
            "ops": ops,
            "strict_exact": dict(zip("prf", strict["exact"].prf())),
            "strict_relaxed": dict(zip("prf", strict["relaxed"].prf())),
            "parseable_exact": dict(zip("prf", parseable["exact"].prf())),
            "per_label": {c: dict(zip("prf", t.prf()), support=t.tp + t.fn)
                          for c, t in per_label.items()},
        }
        Path(args.json).write_text(json.dumps(blob, indent=2))
        print(f"\nmetrics written to {args.json}")

    if args.self_test:
        _, strict_t, _, _, _ = results
        p, r, f = strict_t["exact"].prf()
        ok = (p, r, f) == (1.0, 1.0, 1.0) and results[0]["malformed"] == 0 \
             and results[0]["copy_drift"] == 0
        print("\nSELF-TEST:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
