"""Completion-only LoRA SFT of pinned SPANIEL v2; rejects silent truncation.

Run --dry-run for token/mask checks without loading weights or training.
The actual training command requires CUDA; use a GPU.ai instance for this run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess

from .schema import sha, validate

BASE_MODEL="Harsh/qwen3-0.6b-pii-sft-v2"
BASE_REVISION="a58063469be8c7b929016d944d8d6d1545bc70a0"


def encode(row, tokenizer, max_length):
    if row.get("split") not in ("train","dev") or row.get("decision")=="review_required":
        raise ValueError("Only train/dev examples with determinate labels belong in SFT")
    validate(row)
    chat=row["messages"]
    prompt=tokenizer.apply_chat_template(chat[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    prompt_ids=tokenizer(prompt,add_special_tokens=False)["input_ids"]
    ids=tokenizer(full,add_special_tokens=False)["input_ids"]
    if not full.startswith(prompt) or ids[:len(prompt_ids)]!=prompt_ids:
        raise ValueError("chat template/tokenizer changed the completion boundary")
    if len(ids)>max_length:raise OverflowError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; not truncated")
    if len(ids)<=len(prompt_ids):raise ValueError("empty completion")
    return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":[-100]*len(prompt_ids)+ids[len(prompt_ids):]}


def load_rows(paths, split):
    rows=[]
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row=json.loads(line)
            if row.get("split")!=split:raise ValueError(f"Expected {split} split in {path}")
            rows.append(row)
    hashes=[r["sha256"] for r in rows]
    if len(hashes)!=len(set(hashes)):raise ValueError("duplicate training/evaluation text")
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True);p.add_argument("--replay",type=Path)
    p.add_argument("--dev",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model",default=BASE_MODEL);p.add_argument("--revision",default=BASE_REVISION)
    p.add_argument("--max-length",type=int,default=4096);p.add_argument("--epochs",type=float,default=2)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accumulation",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=5e-5);p.add_argument("--rank",type=int,default=16)
    p.add_argument("--seed",type=int,default=20261021);p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    from transformers import AutoTokenizer
    from tokenizer_compat import tokenizer_load_kwargs
    load_args={} if Path(a.model).is_dir() else {"revision":a.revision}
    # The v2 tokenizer stores an empty special-token list; compatibility is local-only.
    if not Path(a.model).is_dir():
        from huggingface_hub import snapshot_download
        model_dir=snapshot_download(a.model,revision=a.revision,allow_patterns=["*token*","chat_template*","config.json"])
    else:model_dir=a.model
    tok=AutoTokenizer.from_pretrained(model_dir,**tokenizer_load_kwargs(model_dir))
    files=[a.train]+([a.replay] if a.replay else [])
    train=load_rows(files,"train");dev=load_rows([a.dev],"dev")
    if {r['family'] for r in train}&{r['family'] for r in dev}:raise ValueError("scenario-family split overlap")
    if {r['sha256'] for r in train}&{r['sha256'] for r in dev}:raise ValueError("train/dev duplicate text")
    # Reject, rather than silently dropping or truncating, any overlength example.
    tokenized_train=[encode(r,tok,a.max_length) for r in train]
    tokenized_dev=[encode(r,tok,a.max_length) for r in dev]
    manifest={"status":"preflight_passed","created_at":datetime.now(timezone.utc).isoformat(),
              "base_model":a.model,"base_revision":a.revision,"seed":a.seed,
              "parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              "input_sha256":{str(x):sha(x.read_text()) for x in files+[a.dev]},
              "train_rows":len(train),"dev_rows":len(dev),"maximum_sequence_tokens":max(len(r['input_ids']) for r in tokenized_train+tokenized_dev),
              "supervised_completion_tokens":sum(sum(x!=-100 for x in r['labels']) for r in tokenized_train),
              "mask":"All prompt tokens masked with -100; only assistant completion and chat ending supervised; no truncation or packing",
              "selection":f"Minimum development completion loss across {a.epochs:g} epochs; no test-set selection",
              "python":platform.python_version(),"training_script_sha256":sha(Path(__file__).read_text())}
    a.out.mkdir(parents=True,exist_ok=True)
    manifest_path=a.out/"run_manifest.json"
    if manifest_path.exists() and not a.dry_run:
        old=json.loads(manifest_path.read_text())
        if old['status'] not in ('preflight_passed','failed'):raise ValueError("run directory already contains a started/completed experiment")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)
    if a.dry_run:return
    import torch, transformers, peft
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():raise RuntimeError("This experiment must train on the authorized GPU.ai CUDA instance")
    if not torch.cuda.is_bf16_supported():raise RuntimeError("This recipe requires CUDA bf16 support")
    set_seed(a.seed);random.shuffle(tokenized_train)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="sdpa",**load_args)
    model.config.use_cache=False
    config=LoraConfig(task_type="CAUSAL_LM",r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.05,bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],revision=a.revision)
    model=get_peft_model(model,config)
    args=TrainingArguments(output_dir=str(a.out),num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.accumulation,learning_rate=a.learning_rate,lr_scheduler_type="cosine",warmup_ratio=0.03,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,load_best_model_at_end=True,
        metric_for_best_model="eval_loss",greater_is_better=False,per_device_eval_batch_size=1,
        logging_steps=10,report_to=[],dataloader_num_workers=0,seed=a.seed,data_seed=a.seed,optim="adamw_torch")
    trainer=Trainer(model=model,args=args,train_dataset=tokenized_train,eval_dataset=tokenized_dev,
                    data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),processing_class=tok)
    manifest.update(status="running",gpu=torch.cuda.get_device_name(0),gpu_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                    versions={"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__},
                    trainable_parameters=model.get_nb_trainable_parameters()[0])
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    try:
        result=trainer.train()
        trainer.save_model(str(a.out/"adapter"));tok.save_pretrained(a.out/"adapter")
        manifest.update(status="trained_not_evaluated",training_metrics=result.metrics,best_dev_loss=trainer.state.best_metric,
                        best_checkpoint=trainer.state.best_model_checkpoint,finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        manifest.update(status="failed",error_type=type(e).__name__)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        (a.out/"training_history.json").write_text(json.dumps(trainer.state.log_history,indent=2)+"\n")


if __name__=="__main__":main()
