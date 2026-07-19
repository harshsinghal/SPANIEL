#!/usr/bin/env python3
"""Generate tagged-text predictions for the 300 eval rows with the fine-tuned model."""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SFT_RUN_DIR", "/workspace/pii-sft-v1") + "/final"
EVAL = "/workspace/data/eval300.jsonl"
OUT = os.environ.get("SFT_PREDS", "/workspace/preds_sft_v1.jsonl")
BATCH = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype="bfloat16", device_map="cuda").eval()

rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
outputs = []
for i in range(0, len(rows), BATCH):
    batch = rows[i:i + BATCH]
    prompts = [tokenizer.apply_chat_template(
        r["messages"][:2], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)  # else Qwen3 prepends an empty <think> block
        for r in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    max_new = max(len(r["messages"][1]["content"]) for r in batch) // 2 + 512
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=min(max_new, 4096),
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    for j, seq in enumerate(gen):
        text = tokenizer.decode(seq[enc.input_ids.shape[1]:],
                                skip_special_tokens=True)
        outputs.append(text)
    print(f"{i + len(batch)}/{len(rows)}", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    for t in outputs:
        f.write(json.dumps({"output": t}, ensure_ascii=False) + "\n")
print("GENERATION_COMPLETE")
