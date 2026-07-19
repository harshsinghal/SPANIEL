#!/usr/bin/env python3
"""PII tagger web service: local inference with the fine-tuned Qwen3-1.7B.

Loads the SFT model once at startup (MPS on Apple Silicon), serves a single
page UI, and exposes POST /api/tag which builds the exact training-time
prompt and returns the tagged text.
"""

import re
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from pii_decode import CopyTagConstraint, VocabTrie

import os

MODEL_ID = os.environ.get("PII_MODEL_ID")          # HF repo id (docker path)
MODEL_DIR = os.environ.get(
    "PII_MODEL_DIR",
    str(Path(__file__).parent.parent / "models" / "qwen3-0.6b-pii-sft-v2-hf"))
if MODEL_ID:
    from huggingface_hub import snapshot_download
    MODEL_DIR = snapshot_download(MODEL_ID, local_dir=os.environ.get(
        "PII_MODEL_CACHE", "/models") + "/" + MODEL_ID.split("/")[-1])

SYSTEM_PROMPT = (
    "You tag entities in text. Reproduce the user's text exactly, wrapping each "
    "entity that matches a requested type in XML tags, like <label>entity</label>. "
    "Use only the requested labels. Tag every match. If nothing matches, reproduce "
    "the text unchanged. Never alter, add, or omit any other characters."
)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"loading {MODEL_DIR} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16).to(DEVICE).eval()
print("building vocab trie ...")
TRIE = VocabTrie(tokenizer)
print("model ready")

app = FastAPI(title="PII Tagger")


class TagRequest(BaseModel):
    text: str
    entities: list[str]
    constrained: bool = True


@app.post("/api/tag")
def tag(req: TagRequest):
    names = [n.strip().lower() for n in req.entities if n.strip()]
    # dedupe, keep order
    names = list(dict.fromkeys(names))
    if not names or not req.text.strip():
        return {"error": "need text and at least one entity type"}

    user = "Entity types:\n" + "\n".join(f"- {n}" for n in names) \
           + "\n\nText:\n" + req.text
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    max_new = min(6144, enc.input_ids.shape[1] + 256)
    gen_kwargs = {}
    if req.constrained:
        constraint = CopyTagConstraint(TRIE, req.text, names,
                                       tokenizer.eos_token_id)
        gen_kwargs["prefix_allowed_tokens_fn"] = constraint.make_fn(
            tokenizer, enc.input_ids.shape[1])
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id, **gen_kwargs)
    out = tokenizer.decode(gen[0][enc.input_ids.shape[1]:],
                           skip_special_tokens=True)
    out = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", out, count=1)
    elapsed = time.time() - t0

    # copy-fidelity check: output minus tags must equal the input
    stripped = re.sub(
        "<(/?)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ")>",
        "", out)
    return {
        "constrained": req.constrained,
        "tagged": out,
        "entities": names,
        "faithful": stripped == req.text,
        "seconds": round(elapsed, 2),
        "new_tokens": int(gen.shape[1] - enc.input_ids.shape[1]),
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"))
