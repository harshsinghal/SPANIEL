#!/usr/bin/env python3
"""Generate constrained predictions for eval rows with a local model.

Usage:
    python run_constrained_eval.py --model models/qwen3-0.6b-pii-sft-v1-hf \
        --n 60 --out baseline/preds_0.6b_constrained.jsonl
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pii_decode import CopyTagConstraint, VocabTrie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b-pii-sft-v1-hf")
    ap.add_argument("--gold", default="sft_data/sft_eval.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    print("building vocab trie ...", flush=True)
    t0 = time.time()
    trie = VocabTrie(tokenizer)
    print(f"trie built in {time.time() - t0:.1f}s", flush=True)

    rows = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    t_start = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            user = row["messages"][1]["content"]
            head, text = user.split("\n\nText:\n", 1)
            names = [l[2:] for l in head.splitlines()[1:]]

            prompt = tokenizer.apply_chat_template(
                row["messages"][:2], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = enc.input_ids.shape[1]

            constraint = CopyTagConstraint(trie, text, names,
                                           tokenizer.eos_token_id)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=min(6144, prompt_len + 512),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=constraint.make_fn(
                        tokenizer, prompt_len),
                )
            output = tokenizer.decode(gen[0][prompt_len:],
                                      skip_special_tokens=True)
            out.write(json.dumps({"output": output}, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t_start)
                print(f"{i + 1}/{len(rows)}  ({rate:.2f} rows/s)", flush=True)
    print(f"DONE {len(rows)} rows in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
