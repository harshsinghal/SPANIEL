#!/usr/bin/env python3
"""Full-parameter SFT on the SPANIEL tagged-regeneration task.

Prompt-completion data; TRL masks prompt loss so training signal is the
tagged text only. Paths and the base model are configured through environment
variables so the same script can reproduce v1 or v2.
"""

import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("SFT_MODEL", "Qwen/Qwen3-1.7B")
RUN = os.environ.get("SFT_RUN_DIR", "/workspace/pii-sft-v1")
TRAIN = os.environ.get("SFT_TRAIN", "/workspace/data/train.jsonl.gz")
VAL = os.environ.get("SFT_VAL", "/workspace/data/val.jsonl.gz")
KNOWN_MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
MODEL_REVISION = os.environ.get(
    "SFT_MODEL_REVISION", KNOWN_MODEL_REVISIONS.get(MODEL))
SEED = int(os.environ.get("SFT_SEED", "20260719"))


def load_jsonl(path):
    return load_dataset("json", data_files=path, split="train")


def main():
    train_ds = load_jsonl(TRAIN)
    val_ds = load_jsonl(VAL)
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    model_kwargs = {"revision": MODEL_REVISION} if MODEL_REVISION else {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype="bfloat16", attn_implementation="sdpa",
        **model_kwargs)

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
        seed=SEED,
        data_seed=SEED,
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
