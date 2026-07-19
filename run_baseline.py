#!/usr/bin/env python3
"""Zero-shot baseline: run eval examples through an OpenAI model, score with pii_eval.

Sends each example's exact system+user messages from the gold file (the same
prompt the fine-tune will train on) and collects the tagged-text output.

Usage:
    export OPENAI_API_KEY=...
    python run_baseline.py --model gpt-5-nano --n 300
    python pii_eval.py --gold sft_data/sft_eval.jsonl --pred baseline/<model>_preds.jsonl --n-gold 300

Predictions are line-aligned with the first N gold rows. API failures after
retries yield an empty output (scored as a total miss — honest accounting).
A .meta.json with token usage and failures is written alongside.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

RETRIES = 5
THINK_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*", re.MULTILINE)


def call_model(session, url, key_env, model, messages, max_out, reasoning_effort,
               temperature):
    payload = {"model": model, "messages": messages,
               "max_completion_tokens": max_out}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        payload["temperature"] = temperature
    delay = 2.0
    for attempt in range(RETRIES):
        try:
            r = session.post(url, json=payload, timeout=300,
                             headers={"Authorization": f"Bearer {os.environ[key_env]}"})
            if r.status_code == 200:
                d = r.json()
                content = d["choices"][0]["message"]["content"] or ""
                # reasoning models may prepend a think block; it is not part
                # of the tagged-text contract
                content = THINK_RE.sub("", content, count=1)
                return content, d.get("usage", {})
            if r.status_code == 400 and "max_completion_tokens" in r.text:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                continue
            if r.status_code == 400 and "reasoning_effort" in r.text:
                payload.pop("reasoning_effort", None)
                continue
            if r.status_code in (429, 500, 502, 503):
                time.sleep(delay)
                delay *= 2
                continue
            return None, {"error": f"{r.status_code}: {r.text[:300]}"}
        except requests.RequestException as e:
            time.sleep(delay)
            delay *= 2
    return None, {"error": "retries exhausted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--base-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gold", default="sft_data/sft_eval.jsonl")
    ap.add_argument("--outdir", default="baseline")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reasoning-effort", default="minimal",
                    help="'minimal'/'low'/... or '' to omit the parameter")
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    if args.key_env not in os.environ:
        sys.exit(f"{args.key_env} not set")

    rows = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    session = requests.Session()
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    failures = 0

    def one(row):
        messages = row["messages"][:2]  # system + user, drop gold assistant
        text_len = len(messages[1]["content"])
        max_out = min(16000, int(text_len / 2.5) + 1500)
        out, usage = call_model(session, args.base_url, args.key_env, args.model,
                                messages, max_out, args.reasoning_effort or None,
                                args.temperature)
        return out, usage

    print(f"{args.model}: {len(rows)} examples, {args.workers} workers ...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, rows))

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    pred_path = outdir / f"{args.model.replace('/', '_')}_preds.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for out, usage in results:
            if out is None:
                failures += 1
                f.write(json.dumps({"output": "", "error": usage.get("error", "?")}) + "\n")
            else:
                for k in usage_total:
                    usage_total[k] += usage.get(k, 0)
                f.write(json.dumps({"output": out}, ensure_ascii=False) + "\n")

    meta = {"model": args.model, "n": len(rows), "failures": failures,
            "seconds": round(time.time() - t0, 1), "usage": usage_total}
    Path(str(pred_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\npredictions: {pred_path}")
    print(f"score with: python pii_eval.py --gold <first {len(rows)} rows of gold> --pred {pred_path}")


if __name__ == "__main__":
    main()
