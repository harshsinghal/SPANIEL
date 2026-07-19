#!/usr/bin/env python3
"""Build the v2 SFT mixture.

Sources:
  gravitee   (HF parquet, cached)        ~173k rows
  nemotron   (HF parquet, cached)        ~100k rows, date-family auto-corrected
  ai4privacy (pii_xml/ai4privacy_en.jsonl) capped sample, 1-char spans dropped
  synth      (pii_xml/synth_docs.jsonl)  register-gap docs, upsampled x3

v2 additions over v1 (see guidelines_v2.md):
  - deterministic date-family relabeling: date_time without a clock time -> date
  - family-aware subsets: date_of_birth dropped while date requested ->
    its spans become date (a birthdate is still a date)
  - guideline conditioning: GUIDE_P of examples carry "- name: rule" lines
  - containment-family-biased negatives (geo/person/id/date clusters)
  - wide alias sampling from label_aliases_v2.json (~22 forms/label)
  - min entity length 2 everywhere (matches decoder constraint)

Usage: python build_sft_dataset_v2.py [--limit N] [--outdir sft_data_v2]
"""

import argparse
import ast
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from spans_to_xml import spans_to_xml

SEED = 20260719
GUIDE_P = 0.30
AI4P_CAP = 110_000
SYNTH_UPSAMPLE = 3
NEG_ROW_CAP = 0.08
TIME_RE = re.compile(r"\d{1,2}:\d{2}")

ALIASES = json.loads(Path("label_aliases_v2.json").read_text())
REMOVED = {ds: set(l) for ds, l in ALIASES.get("_removed", {}).items()}
VOCAB = {ds: [l for l in ALIASES[ds] if not l.startswith("_")]
         for ds in ("gravitee", "nemotron", "ai4privacy")}

RULES = {  # short per-label rules, rendered in guideline-conditioned prompts
    "date": "a calendar date with no time component",
    "date_time": "must contain both a date and a clock time",
    "time": "a clock time alone",
    "date_of_birth": "a date explicitly denoting someone's birth",
    "occupation": "the person's full job title; exclude department names, document titles, and bare credentials",
    "DATE": "a calendar date with no time component",
    "TIME": "a clock time alone",
}

FAMILIES = {
    "nemotron": [{"date", "date_time", "date_of_birth", "time"},
                 {"city", "state", "country", "county", "postcode"},
                 {"first_name", "last_name"},
                 {"customer_id", "employee_id", "unique_id", "device_identifier"}],
    "ai4privacy": [{"DATE", "TIME"},
                   {"CITY", "ZIPCODE", "STREET", "BUILDINGNUM"},
                   {"GIVENNAME", "SURNAME"}],
    "gravitee": [],
}

SYSTEM_PROMPT = (
    "You tag entities in text. Reproduce the user's text exactly, wrapping each "
    "entity that matches a requested type in XML tags, like <label>entity</label>. "
    "Use only the requested labels. Tag every match. If nothing matches, reproduce "
    "the text unchanged. Never alter, add, or omit any other characters."
)


def correct_nemotron(spans, text):
    for s in spans:
        if s["label"] == "date_time" and not TIME_RE.search(text[s["start"]:s["end"]]):
            s["label"] = "date"
    return spans


def iter_source(ds, limit):
    if ds == "gravitee":
        path = hf_hub_download("gravitee-io/pii-detection-dataset", "data.parquet",
                               repo_type="dataset")
        n = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                yield row["text"], [dict(s) for s in row["spans"]]
                n += 1
                if limit and n >= limit: return
    elif ds == "nemotron":
        path = hf_hub_download("nvidia/Nemotron-PII",
                               "data/train-00000-of-00001.parquet", repo_type="dataset")
        n = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                spans = [dict(s) for s in ast.literal_eval(row["spans"])]
                yield row["text"], correct_nemotron(spans, row["text"])
                n += 1
                if limit and n >= limit: return
    elif ds == "ai4privacy":
        rng = random.Random(SEED + 1)
        rows = open("pii_xml/ai4privacy_en.jsonl", encoding="utf-8").readlines()
        rng.shuffle(rows)
        cap = min(limit or AI4P_CAP, AI4P_CAP)
        for line in rows[:cap]:
            r = json.loads(line)
            yield r["text"], r["spans"]
    elif ds == "synth":
        rows = open("pii_xml/synth_docs.jsonl", encoding="utf-8").readlines()
        n = 0
        for _ in range(SYNTH_UPSAMPLE):
            for line in rows:
                r = json.loads(line)
                yield r["text"], [dict(s) for s in r["spans"]]
                n += 1
                if limit and n >= limit: return


def family_of(ds, label):
    for fam in FAMILIES.get(ds, []):
        if label in fam:
            return fam
    return None


def build_request(rng, ds, present):
    vocab = VOCAB[ds]
    roll = rng.random()
    if roll < 0.40 or not present:
        return list(vocab), list(present), {}
    if roll < 0.80:
        pool = [l for l in vocab if l not in present]
        # bias half the negatives toward containment families of present labels
        fam_pool = [l for l in pool
                    if any(l in (family_of(ds, p) or ()) for p in present)]
        k = min(rng.randint(2, 6), len(pool))
        negs = []
        for _ in range(k):
            src = fam_pool if fam_pool and rng.random() < 0.5 else pool
            c = rng.choice(src)
            if c not in negs: negs.append(c)
        req = list(present) + negs
        rng.shuffle(req)
        return req, list(present), {}
    # strict subset with family-aware relabeling (dob dropped, date kept)
    droppable = list(present)
    if len(droppable) < 2:
        return list(present), list(present), {}
    n_drop = rng.randint(1, len(droppable) - 1)
    dropped = set(rng.sample(droppable, n_drop))
    kept = [l for l in present if l not in dropped]
    relabel = {}
    if ds == "nemotron" and "date_of_birth" in dropped and "date" in kept:
        relabel["date_of_birth"] = "date"
    return kept, kept, relabel


def sample_names(rng, ds, labels, text):
    names, taken = {}, set()
    for lab in labels:
        opts = [a for a in ALIASES[ds][lab]
                if a not in taken and f"<{a}>" not in text and f"</{a}>" not in text]
        if not opts:
            return None
        names[lab] = rng.choice(opts)
        taken.add(names[lab])
    return names


def render_request(rng, ds, request, names):
    guided = rng.random() < GUIDE_P
    lines = []
    for l in request:
        rule = RULES.get(l)
        if guided and rule:
            lines.append(f"- {names[l]}: {rule}")
        else:
            lines.append(f"- {names[l]}")
    rng.shuffle(lines)
    return "Entity types:\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default="sft_data_v2")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    examples = []
    stats = {}
    for ds in ("gravitee", "nemotron", "ai4privacy", "synth"):
        st = {"rows": 0, "skip_overlap": 0, "skip_alias": 0, "neg_rows": 0,
              "neg_dropped": 0, "relabeled_subset": 0}
        vocab_ds = "nemotron" if ds == "synth" else ds
        for text, spans in iter_source(ds, args.limit):
            spans = [s for s in spans
                     if s["label"] in ALIASES[vocab_ds]
                     and s["label"] not in REMOVED.get(vocab_ds, set())
                     and s["end"] - s["start"] >= 2]
            present = sorted({s["label"] for s in spans})
            request, target, relabel = build_request(rng, vocab_ds, present)
            if not target:
                st["neg_rows"] += 1
                if st["neg_rows"] > max(20, NEG_ROW_CAP * (st["rows"] + 1)):
                    st["neg_dropped"] += 1
                    continue
            names = sample_names(rng, vocab_ds, request, text)
            if names is None:
                st["skip_alias"] += 1
                continue
            kept_spans = []
            for s in spans:
                lab = s["label"]
                if lab in relabel:
                    lab = relabel[lab]; st["relabeled_subset"] += 1
                if lab in target or lab in relabel.values() and lab in request:
                    if lab in names:
                        kept_spans.append(dict(s, label=names[lab]))
            try:
                tagged = spans_to_xml(text, kept_spans)
            except ValueError:
                st["skip_overlap"] += 1
                continue
            user = render_request(rng, vocab_ds, request, names) + "\n\nText:\n" + text
            examples.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": tagged}],
                "source": ds})
            st["rows"] += 1
        stats[ds] = st
        print(f"{ds}: {st}", flush=True)

    rng.shuffle(examples)
    path = outdir / "sft_train_v2.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    Path(str(path) + ".stats.json").write_text(json.dumps(stats, indent=1))
    print(f"TOTAL {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
