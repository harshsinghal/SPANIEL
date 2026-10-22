#!/usr/bin/env python3
"""SPANIEL web service: local inference with the fine-tuned Qwen3-0.6B.

Loads the SFT model once at startup (MPS on Apple Silicon), serves a single
page UI, and exposes POST /api/tag which builds the exact training-time
prompt and returns the tagged text.
"""

import os
import re
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from pii_decode import CopyTagConstraint, VocabTrie
from tokenizer_compat import tokenizer_load_kwargs

MODEL_ID = os.environ.get("PII_MODEL_ID")          # HF repo id (docker path)
MODEL_REVISION = os.environ.get("PII_MODEL_REVISION")
MODEL_DIR = os.environ.get(
    "PII_MODEL_DIR",
    str(Path(__file__).parent.parent / "models" / "qwen3-0.6b-pii-sft-v2-hf"))
if MODEL_ID:
    from huggingface_hub import snapshot_download
    download_args = {}
    if MODEL_REVISION:
        download_args["revision"] = MODEL_REVISION
    MODEL_DIR = snapshot_download(
        MODEL_ID,
        local_dir=str(Path(os.environ.get("PII_MODEL_CACHE", "/models"))
                      / MODEL_ID.split("/")[-1]),
        **download_args,
    )

SYSTEM_PROMPT = (
    "You tag entities in text. Reproduce the user's text exactly, wrapping each "
    "entity that matches a requested type in XML tags, like <label>entity</label>. "
    "Use only the requested labels. Tag every match. If nothing matches, reproduce "
    "the text unchanged. Never alter, add, or omit any other characters."
)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"loading {MODEL_DIR} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR, **tokenizer_load_kwargs(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16).to(DEVICE).eval()
ADAPTER = os.environ.get("PII_ADAPTER")
ADAPTER_REVISION = os.environ.get("PII_ADAPTER_REVISION")
if ADAPTER:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("PII_ADAPTER requires requirements-adapter.txt") from exc
    adapter_kwargs = {"revision": ADAPTER_REVISION} if ADAPTER_REVISION else {}
    model = PeftModel.from_pretrained(model, ADAPTER, **adapter_kwargs) \
        .merge_and_unload().to(DEVICE).eval()
print("building vocab trie ...")
TRIE = VocabTrie(tokenizer)
print("model ready")

app = FastAPI(title="PII Tagger")
GENERATION_LOCK = threading.Lock()
MAX_TEXT_CHARS = int(os.environ.get("PII_MAX_TEXT_CHARS", "12000"))
MAX_ENTITIES = int(os.environ.get("PII_MAX_ENTITIES", "32"))
MAX_LABEL_CHARS = int(os.environ.get("PII_MAX_LABEL_CHARS", "64"))
MAX_NEW_TOKENS = int(os.environ.get("PII_MAX_NEW_TOKENS", "6144"))
ALLOW_UNCONSTRAINED = os.environ.get("PII_ALLOW_UNCONSTRAINED") == "1"


@app.get("/api/model")
def model_metadata():
    """Identify loaded artifacts without exposing local filesystem paths."""
    return {
        "base_model": MODEL_ID or "local-checkpoint",
        "base_revision": MODEL_REVISION,
        "adapter": ("local-adapter" if ADAPTER and Path(ADAPTER).is_dir() else ADAPTER),
        "adapter_revision": ADAPTER_REVISION,
        "source_preserving_decoder": True,
        "note": "Artifact identity does not establish detection accuracy; consult the model card.",
    }


class TagRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    entities: list[str] = Field(min_length=1, max_length=MAX_ENTITIES)
    constrained: bool = True


@app.post("/api/tag")
def tag(req: TagRequest):
    if not req.constrained and not ALLOW_UNCONSTRAINED:
        raise HTTPException(
            status_code=422,
            detail="unconstrained generation is disabled by this server",
        )
    names = [n.strip().lower() for n in req.entities if n.strip()]
    # dedupe, keep order
    names = list(dict.fromkeys(names))
    if not names or not req.text.strip():
        raise HTTPException(
            status_code=422,
            detail="need text and at least one entity type",
        )
    for name in names:
        if (len(name) > MAX_LABEL_CHARS
                or any(ch in name for ch in "<>/\r\n")
                or any(ord(ch) < 32 for ch in name)):
            raise HTTPException(
                status_code=422,
                detail=(
                    "entity types must be short, single-line names without "
                    "angle brackets or slashes"
                ),
            )

    user = "Entity types:\n" + "\n".join(f"- {n}" for n in names) \
           + "\n\nText:\n" + req.text
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]
    context_size = getattr(model.config, "max_position_embeddings", 32768)
    available = context_size - prompt_len
    source_tokens = len(tokenizer(
        req.text, add_special_tokens=False).input_ids)
    if available <= source_tokens:
        raise HTTPException(
            status_code=413,
            detail="input is too long for the model context window",
        )
    max_new = min(MAX_NEW_TOKENS, available, prompt_len + 512)
    gen_kwargs = {}
    constraint = None
    if req.constrained:
        constraint = CopyTagConstraint(TRIE, req.text, names,
                                       tokenizer.eos_token_id)
        gen_kwargs["prefix_allowed_tokens_fn"] = constraint.make_fn(
            tokenizer, prompt_len)
    t0 = time.time()
    with GENERATION_LOCK, torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id, **gen_kwargs)
    out = tokenizer.decode(gen[0][prompt_len:],
                           skip_special_tokens=True)
    out = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", out, count=1)
    elapsed = time.time() - t0

    if constraint is not None:
        faithful = constraint.is_complete(out)
        if not faithful:
            raise HTTPException(
                status_code=500,
                detail=(
                    "constrained generation stopped before producing a "
                    "complete faithful copy"
                ),
            )
    else:
        stripped = re.sub(
            "<(/?)(" + "|".join(
                re.escape(n) for n in sorted(names, key=len, reverse=True)
            ) + ")>",
            "",
            out,
        )
        faithful = stripped == req.text
    return {
        "constrained": req.constrained,
        "tagged": out,
        "entities": names,
        "faithful": faithful,
        "seconds": round(elapsed, 2),
        "new_tokens": int(gen.shape[1] - prompt_len),
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"))
