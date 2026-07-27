#!/usr/bin/env python3
"""Verify local reproducible artifacts against the published manifest."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path):
    with path.open("rb") as source:
        return sum(1 for _ in source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        help="manifest paths to verify; defaults to every artifact that exists",
    )
    args = parser.parse_args()

    manifest = json.loads(
        (ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    selected = args.paths or [
        name for name in artifacts if (ROOT / name).exists()
    ]
    if not selected:
        raise SystemExit("no manifest artifacts found locally")

    failed = False
    for name in selected:
        if name not in artifacts:
            print(f"UNKNOWN  {name}")
            failed = True
            continue
        path = ROOT / name
        if not path.exists():
            print(f"MISSING  {name}")
            failed = True
            continue
        expected = artifacts[name]
        rows = line_count(path)
        digest = file_sha256(path)
        ok = rows == expected["rows"] and digest == expected["sha256"]
        print(f"{'OK' if ok else 'FAIL':<7} {name}  rows={rows}  sha256={digest}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
