"""Run one guarded read or an honest, small live-model smoke evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import Request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from spaniel_guard import LocalSpaniel, POLICIES, make_document_tool

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", nargs="?", help="Relative document path for a single read")
    parser.add_argument("--root", type=Path, default=HERE / "documents")
    parser.add_argument("--policy", choices=POLICIES, default="credential-exposure")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8377/api/tag")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--report", type=Path, help="Write suite metadata/results, never document text")
    args = parser.parse_args()
    if args.suite == bool(args.document) or (args.report and not args.suite):
        parser.error("choose a document or --suite; --report requires --suite")

    tagger = LocalSpaniel(args.endpoint, args.timeout)
    if args.document:
        result = make_document_tool(args.root, POLICIES[args.policy], tagger)(args.document)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "allowed" else 2

    try:
        request = Request(args.endpoint.removesuffix('/api/tag')+'/api/model')
        with tagger.opener.open(request, timeout=10) as response:
            metadata = json.loads(response.read(100_001))
    except Exception:
        metadata = {'configuration_unavailable': True}
    cases = json.loads((HERE / "cases.json").read_text())
    rows = []
    for case in cases:
        started = time.monotonic()
        result = make_document_tool(args.root, POLICIES[case["policy"]], tagger)(case["document"])
        row = {"id": case["id"], "category": case["category"], "policy": case["policy"],
               "expected": case["expected"], "actual": result["status"],
               "code": result["code"], "seconds": round(time.monotonic() - started, 3),
               "content_returned": result["content"] is not None}
        row["matched_expectation"] = row["expected"] == row["actual"]
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        "cases": len(rows),
        "matched_expectations": sum(r["matched_expectation"] for r in rows),
        "sensitive_released": sum(r["expected"] == "blocked" and r["actual"] == "allowed" for r in rows),
        "false_blocks": sum(r["expected"] == "allowed" and r["actual"] == "blocked" for r in rows),
        "unexpected_errors": sum(r["expected"] != "error" and r["actual"] == "error" for r in rows),
    }
    report = {"mode": "live-local-spaniel", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Synthetic document-level smoke cases, not a held-out benchmark or span F1. "
                      "Server configuration is declared metadata, not an attestation of weights or accuracy.",
              "model_configuration": metadata, "summary": summary, "results": rows}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)
    return 0 if all(r["matched_expectation"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
