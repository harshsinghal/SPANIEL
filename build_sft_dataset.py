#!/usr/bin/env python3
"""Build chat-format SFT data for unconstrained NER from the two PII datasets.

Each source row becomes one training example:
  system    fixed tagging instruction
  user      "Entity types:" + sampled natural-language label names + the text
  assistant the text regenerated with <name>entity</name> tags

Design (see sft_prompt_design artifact, as amended):
  - Label names are natural language (lowercase, no underscores). One alias is
    sampled per label per example from label_aliases.json; the sampled name is
    used verbatim in both the request list and the output tags.
  - Junk labels (gravitee FINANCIAL, TITLE, NRP) are stripped: their spans stay
    untagged and the labels are never requested.
  - Request-set buckets: 40% full source vocabulary, 40% labels present plus
    2-6 negatives, 20% strict subset (dropped labels vanish from request AND
    target). The nemotron date family {date, date_time, date_of_birth, time}
    is atomic for negatives and subset drops, so a generic date type is never
    requested while a more specific one present in the text is suppressed.
  - Rows with no (remaining) entities become "nothing matches" examples,
    capped at NEG_ROW_CAP of each source's output.

Usage:
    python build_sft_dataset.py [--limit N] [--outdir sft_data]

Outputs: sft_data/sft_train.jsonl   (gravitee train + nemotron train, shuffled)
         sft_data/sft_eval.jsonl    (nemotron test, capped at 2000 rows)
         plus a .stats.json next to each.
"""

import argparse
import ast
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from spans_to_xml import spans_to_xml

SEED = 20260708
NEG_ROW_CAP = 0.10          # max fraction of zero-entity rows per source
EVAL_CAP = 2000

SYSTEM_PROMPT = (
    "You tag entities in text. Reproduce the user's text exactly, wrapping each "
    "entity that matches a requested type in XML tags, like <label>entity</label>. "
    "Use only the requested labels. Tag every match. If nothing matches, reproduce "
    "the text unchanged. Never alter, add, or omit any other characters."
)

DATE_FAMILY = {"date", "date_time", "date_of_birth", "time"}

SOURCES = {
    "gravitee": ("gravitee-io/pii-detection-dataset", {"train": "data.parquet"}),
    "nemotron": ("nvidia/Nemotron-PII", {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }),
}

ALIASES = json.loads(Path(__file__).with_name("label_aliases.json").read_text())
REMOVED = {ds: set(labs) for ds, labs in ALIASES["_removed"].items()}
VOCAB = {ds: list(ALIASES[ds]) for ds in ("gravitee", "nemotron")}


def sample_names(rng, dataset, labels, text):
    """Pick one alias per label, unique within this request.

    An alias whose literal tag form (<alias> or </alias>) already occurs in the
    source text is avoided: the tagged target would be ambiguous to parse back
    (gravitee contains raw XML/HTML documents). Returns None if a label has no
    collision-free alias, and the caller skips the row.
    """
    names, taken = {}, set()
    for lab in labels:
        options = [a for a in ALIASES[dataset][lab]
                   if a not in taken
                   and f"<{a}>" not in text and f"</{a}>" not in text]
        if not options:
            return None
        names[lab] = rng.choice(options)
        taken.add(names[lab])
    return names


def build_request(rng, dataset, present):
    """Return (request_labels, target_labels) per the 40/40/20 bucket mix."""
    vocab = VOCAB[dataset]
    roll = rng.random()
    if roll < 0.40 or not present:                      # full vocabulary
        return list(vocab), list(present)
    if roll < 0.80:                                     # present + negatives
        pool = [l for l in vocab if l not in present]
        if dataset == "nemotron" and DATE_FAMILY & set(present):
            pool = [l for l in pool if l not in DATE_FAMILY]
        negs = rng.sample(pool, min(rng.randint(2, 6), len(pool)))
        req = list(present) + negs
        rng.shuffle(req)
        return req, list(present)
    # strict subset: drop 1..n-1 present labels; date family drops atomically
    droppable = list(present)
    fam = [l for l in droppable if l in DATE_FAMILY] if dataset == "nemotron" else []
    units = [l for l in droppable if l not in fam] + ([tuple(fam)] if fam else [])
    if len(units) < 2:
        return list(present), list(present)
    n_drop = rng.randint(1, len(units) - 1)
    dropped = set()
    for u in rng.sample(units, n_drop):
        dropped.update(u if isinstance(u, tuple) else (u,))
    kept = [l for l in present if l not in dropped]
    return kept, kept


def iter_rows(dataset, split):
    repo, files = SOURCES[dataset]
    path = hf_hub_download(repo, files[split], repo_type="dataset")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            spans = row["spans"]
            if isinstance(spans, str):
                spans = ast.literal_eval(spans)
            yield row["text"], spans


def build_split(dataset, split, rng, limit=None):
    examples, stats = [], {"rows": 0, "skipped_overlap": 0, "neg_rows": 0,
                           "neg_rows_dropped": 0, "buckets": {}}
    for text, spans in iter_rows(dataset, split):
        spans = [s for s in spans if s["label"] not in REMOVED.get(dataset, set())]
        present = sorted({s["label"] for s in spans})

        request, target = build_request(rng, dataset, present)
        if not target:
            # zero-entity example; cap later by subsampling
            stats["neg_rows"] += 1
            if stats["neg_rows"] > max(20, NEG_ROW_CAP * (len(examples) + 1)):
                stats["neg_rows_dropped"] += 1
                continue
        names = sample_names(rng, dataset, request, text)
        if names is None:
            stats["skipped_alias_collision"] = stats.get("skipped_alias_collision", 0) + 1
            continue
        kept_spans = [dict(s, label=names[s["label"]]) for s in spans
                      if s["label"] in target]
        try:
            tagged = spans_to_xml(text, kept_spans)
        except ValueError:
            stats["skipped_overlap"] += 1
            continue

        req_names = [names[l] for l in request]
        rng.shuffle(req_names)
        user = "Entity types:\n" + "\n".join(f"- {n}" for n in req_names) \
               + "\n\nText:\n" + text
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": tagged},
            ],
            "source": dataset, "split": split,
        })
        stats["rows"] += 1
        if limit and stats["rows"] >= limit:
            break
    return examples, stats


def write(path, examples, stats):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    Path(str(path) + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{path}: {len(examples)} examples  {stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="rows per source split")
    ap.add_argument("--outdir", default="sft_data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    train, train_stats = [], {}
    for ds in ("gravitee", "nemotron"):
        ex, st = build_split(ds, "train", rng, args.limit)
        train.extend(ex)
        train_stats[ds] = st
    rng.shuffle(train)
    write(outdir / "sft_train.jsonl", train, train_stats)

    eval_ex, eval_stats = build_split(
        "nemotron", "test", rng, args.limit or EVAL_CAP)
    write(outdir / "sft_eval.jsonl", eval_ex, {"nemotron": eval_stats})


if __name__ == "__main__":
    main()
