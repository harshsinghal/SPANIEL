#!/usr/bin/env python3
"""GRPO RL for SPANIEL: reward = span-F1 on the tagged answer.

Starts from the c3 checkpoint (thinking-capable + core-anchored), generates K
completions per prompt in thinking-on format, scores each with the span-F1
reward (0 for drift/malformed), and does group-relative policy optimization.
The model is free to think or emit an empty think block — the reward only
grades the final answer, so any internal route that improves tagging is
reinforced.

Env: GRPO_MODEL, GRPO_RUN_DIR, GRPO_HUB_ID, GRPO_STEPS, GRPO_K, GRPO_BS.
"""

import gzip
import json
import os
import sys

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, "/workspace")
from rl_reward import make_reward_fn

MODEL = os.environ.get("GRPO_MODEL", "Harsh/qwen3-0.6b-pii-think-c3")
RUN = os.environ.get("GRPO_RUN_DIR", "/workspace/grpo")
HUB = os.environ.get("GRPO_HUB_ID")
STEPS = int(os.environ.get("GRPO_STEPS", 300))
K = int(os.environ.get("GRPO_K", 8))
BS = int(os.environ.get("GRPO_BS", 8))


def load():
    rows = [json.loads(l) for l in
            gzip.open("/workspace/data/rl_prompts.jsonl.gz", "rt", encoding="utf-8")]
    return Dataset.from_list(rows)


def main():
    ds = load()
    print(f"RL prompts: {len(ds)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16")

    cfg = GRPOConfig(
        output_dir=RUN,
        num_generations=K,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=2,
        max_completion_length=640,
        temperature=0.9,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=STEPS,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        push_to_hub=bool(HUB),
        hub_model_id=HUB,
        hub_strategy="every_save",
        hub_private_repo=True,
        use_vllm=os.environ.get("GRPO_VLLM", "0") == "1",
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=make_reward_fn(),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(f"{RUN}/final")
    tok.save_pretrained(f"{RUN}/final")
    print("GRPO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
