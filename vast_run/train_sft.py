#!/usr/bin/env python3
"""SFT: Qwen3-1.7B on the PII tagged-regeneration task (v1 recipe proof).

Prompt-completion data; TRL masks prompt loss so training signal is the
tagged text only. Full fine-tune, bf16, single GPU.
"""

import gzip
import json
import os

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("SFT_MODEL", "Qwen/Qwen3-1.7B")
RUN = os.environ.get("SFT_RUN_DIR", "/workspace/pii-sft-v1")


def load_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return Dataset.from_list([json.loads(l) for l in f])


def main():
    train_ds = load_jsonl_gz("/workspace/data/train.jsonl.gz")
    val_ds = load_jsonl_gz("/workspace/data/val.jsonl.gz")
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="bfloat16", attn_implementation="sdpa")

    config = SFTConfig(
        output_dir=RUN,
        max_length=3072,
        packing=False,
        num_train_epochs=float(os.environ.get("SFT_EPOCHS", 1)),
        per_device_train_batch_size=int(os.environ.get("SFT_BS", 4)),
        gradient_accumulation_steps=int(os.environ.get("SFT_ACCUM", 8)),
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=8,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=4,
        push_to_hub=bool(os.environ.get("SFT_HUB_ID")),
        hub_model_id=os.environ.get("SFT_HUB_ID"),
        hub_strategy="every_save",
        hub_private_repo=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{RUN}/final")
    tokenizer.save_pretrained(f"{RUN}/final")
    print("TRAINING_COMPLETE")


if __name__ == "__main__":
    main()
