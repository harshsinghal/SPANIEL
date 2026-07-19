#!/usr/bin/env python3
"""Convert PII datasets to text with XML tags around entities, using spans_to_xml.

Supported datasets:
  gravitee  -> gravitee-io/pii-detection-dataset (train)
               spans are structured: [{"start", "end", "label"}]
  nemotron  -> nvidia/Nemotron-PII (train, test)
               spans are a Python-literal string:
               "[{'start': 3, 'end': 8, 'text': 'Jason', 'label': 'first_name'}]"

Output: JSONL, one record per row, with the entity-tagged text in `text_xml`
plus the source row's metadata columns. Labels are used verbatim as tag names.

Rows whose spans overlap, or whose offsets don't match the span's own `text`
field (nemotron), are skipped and counted.

Usage:
    python convert_pii_datasets.py gravitee train [--limit N]
    python convert_pii_datasets.py nemotron train [--limit N]
    python convert_pii_datasets.py nemotron test  [--limit N]

Outputs to ./pii_xml/<dataset>_<split>.jsonl
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from spans_to_xml import spans_to_xml

DATASETS = {
    "gravitee": {
        "repo": "gravitee-io/pii-detection-dataset",
        "files": {"train": "data.parquet"},
        "meta_cols": ["language", "source"],
    },
    "nemotron": {
        "repo": "nvidia/Nemotron-PII",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        "meta_cols": ["uid", "domain", "document_type", "document_format", "locale"],
    },
}


def parse_spans(raw) -> list[dict]:
    """Normalize a row's spans to a list of {start, end, label} dicts."""
    if isinstance(raw, str):  # nemotron: Python-literal string
        raw = ast.literal_eval(raw)
    return raw


def convert_row(text: str, spans: list[dict]) -> str:
    for span in spans:
        # Nemotron's span "text" field is sometimes case-normalized or a
        # non-string (e.g. age as int); the offsets are still correct, so
        # validate loosely.
        if "text" in span and (
            text[span["start"]:span["end"]].casefold()
            != str(span["text"]).casefold()
        ):
            raise ValueError(
                f"Offset mismatch: expected {span['text']!r}, "
                f"got {text[span['start']:span['end']]!r}"
            )
    return spans_to_xml(text, spans)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outdir", default="pii_xml")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    if args.split not in cfg["files"]:
        sys.exit(f"{args.dataset} has no split {args.split!r}; "
                 f"available: {list(cfg['files'])}")

    print(f"Downloading {cfg['repo']} :: {cfg['files'][args.split]} ...", flush=True)
    path = hf_hub_download(cfg["repo"], cfg["files"][args.split], repo_type="dataset")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    out_path = outdir / f"{args.dataset}_{args.split}.jsonl"

    written = skipped = 0
    skip_reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        parquet = pq.ParquetFile(path)
        done = False
        for batch in parquet.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                try:
                    tagged = convert_row(row["text"], parse_spans(row["spans"]))
                except ValueError as e:
                    skipped += 1
                    reason = str(e).split(":")[0]
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue
                record = {c: row[c] for c in cfg["meta_cols"]}
                record["text_xml"] = tagged
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    done = True
                    break
            if done:
                break

    print(f"Wrote {written} rows to {out_path} ({skipped} skipped)")
    for reason, count in sorted(skip_reasons.items()):
        print(f"  skipped for {reason}: {count}")


if __name__ == "__main__":
    main()
