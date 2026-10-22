"""Identical constrained evaluation for v2 and an optional candidate adapter.

Counts malformed/incomplete outputs as inspection errors, never clean passes.
Reports exact spans, sensitive-document recall, benign false blocks and pairs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from pii_eval import parse_tagged
from .schema import sha, validate
from .train_lora import BASE_MODEL, BASE_REVISION


def summarize(rows):
    classified=[r for r in rows if r["expected_decision"] in ("sensitive","benign")]
    positives=[r for r in classified if r["expected_decision"]=="sensitive"]
    negatives=[r for r in classified if r["expected_decision"]=="benign"]
    tp=fp=fn=0
    for r in classified:
        expected={tuple(s) for s in r["expected_spans"]};pred={tuple(s) for s in r.get("predicted_spans",[])}
        tp+=len(expected&pred);fp+=len(pred-expected);fn+=len(expected-pred)
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    groups=defaultdict(list)
    for r in classified:
        if r.get("pair_id"):groups[r["pair_id"]].append(r)
    pairs=[g for g in groups.values() if len(g)==2 and {x['expected_decision'] for x in g}=={'sensitive','benign'}]
    return dict(cases=len(rows),scored_cases=len(classified),sensitive_cases=len(positives),benign_cases=len(negatives),
        review_cases=len(rows)-len(classified),inspection_errors=sum(r.get('error') is not None for r in rows),
        document_true_positives=sum(r.get('detected') is True for r in positives),
        document_misses=sum(r.get('detected') is False for r in positives),
        sensitive_inspection_errors=sum(r.get('error') is not None for r in positives),
        benign_false_blocks=sum(r.get('detected') is True for r in negatives),
        benign_inspection_errors=sum(r.get('error') is not None for r in negatives),
        sensitive_document_recall=sum(r.get('detected') is True for r in positives)/len(positives) if positives else None,
        benign_false_block_rate=sum(r.get('detected') is True for r in negatives)/len(negatives) if negatives else None,
        benign_total_withhold_rate=sum(r.get('detected') is not False for r in negatives)/len(negatives) if negatives else None,
        exact_case_matches=sum(r.get('exact_match') is True for r in classified),
        span_tp=tp,span_fp=fp,span_fn=fn,span_precision=precision,span_recall=recall,
        exact_span_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        complete_contrast_pairs=len(pairs),correct_contrast_pairs=sum(all(r.get('decision_correct') is True for r in g) for g in pairs))


def load_examples(path, legacy=False, limit=0):
    rows=[]
    for i,line in enumerate(Path(path).read_text().splitlines()):
        row=json.loads(line)
        if legacy:
            from .replay import convert
            row=convert(row,i);row['split']='test';row['id']=f'legacy-eval-{i}'
        elif row.get('decision')!='review_required':validate(row)
        rows.append(row)
        if limit and len(rows)>=limit:break
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate evaluation ID')
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default=BASE_MODEL);ap.add_argument('--revision',default=BASE_REVISION);ap.add_argument('--adapter')
    ap.add_argument('--legacy',action='store_true');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--exclude',type=Path,action='append',default=[]);ap.add_argument('--max-new-tokens',type=int,default=8192)
    a=ap.parse_args();rows=load_examples(a.data,a.legacy,a.limit)
    excluded={json.loads(line)['sha256'] for path in a.exclude for line in path.read_text().splitlines()}
    if any(r['sha256'] in excluded for r in rows):raise ValueError('evaluation text overlaps supplied training data')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from pii_decode import VocabTrie,CopyTagConstraint
    from tokenizer_compat import tokenizer_load_kwargs
    kwargs={} if Path(a.model).is_dir() else {'revision':a.revision}
    if Path(a.model).is_dir():model_dir=a.model
    else:
        from huggingface_hub import snapshot_download
        model_dir=snapshot_download(a.model,revision=a.revision)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype=torch.bfloat16 if device!='cpu' else torch.float32
    tok=AutoTokenizer.from_pretrained(model_dir,**tokenizer_load_kwargs(model_dir))
    model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=dtype,attn_implementation='sdpa').to(device).eval()
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter).merge_and_unload().to(device).eval()
    trie=VocabTrie(tok);results=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    weights=Path(model_dir)/'model.safetensors'
    def file_hash(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    runtime=dict(created_at=datetime.now(timezone.utc).isoformat(),model=a.model,revision=a.revision,adapter=a.adapter,
                 device=device,dtype=str(dtype),data_sha256=sha(a.data.read_text()),
                 model_weights_sha256=file_hash(weights) if weights.exists() else None,
                 adapter_weights_sha256=file_hash(Path(a.adapter)/'adapter_model.safetensors') if a.adapter and Path(a.adapter).is_dir() else None,
                 evaluator_sha256=sha(Path(__file__).read_text()),decoding='greedy copy-or-tag constraint; full source and span validation')
    for i,row in enumerate(rows):
        t0=time.monotonic();text=row['text'];names=row['entities']
        expected=[(s['start'],s['end'],s['label']) for s in row['spans']]
        result=dict(id=row['id'],family=row.get('family'),pair_id=row.get('pair_id'),category=row.get('category','ambiguous'),
                    expected_decision=row['decision'],expected_spans=expected,sha256=row['sha256'])
        try:
            if any(tag in text for n in names for tag in (f'<{n}>',f'</{n}>')):raise ValueError('source markup ambiguity')
            from .schema import SYSTEM
            chat=row.get('messages',[{'role':'system','content':SYSTEM},{'role':'user','content':'Entity types:\n'+'\n'.join('- '+n for n in names)+'\n\nText:\n'+text}])[:2]
            prompt=tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,return_tensors='pt').to(device);plen=enc.input_ids.shape[1]
            source_tokens=len(tok(text,add_special_tokens=False)['input_ids'])
            available=model.config.max_position_embeddings-plen
            if available<=source_tokens:raise ValueError('unsupported input length')
            constraint=CopyTagConstraint(trie,text,names,tok.eos_token_id)
            with torch.inference_mode():
                output=model.generate(**enc,do_sample=False,pad_token_id=tok.eos_token_id,
                    max_new_tokens=min(a.max_new_tokens,available,max(512,2*source_tokens+256)),
                    prefix_allowed_tokens_fn=constraint.make_fn(tok,plen))
            tagged=tok.decode(output[0][plen:],skip_special_tokens=True)
            if not constraint.is_complete(tagged):raise ValueError('incomplete constrained output')
            recovered,predicted=parse_tagged(tagged,names)
            if recovered!=text or any(b-a<2 or not text[a:b].strip() for a,b,n in predicted):raise ValueError('invalid span/source recovery')
            result.update(predicted_spans=predicted,detected=bool(predicted),tagged=tagged,
                          exact_match=predicted==expected if row['decision']!='review_required' else None,
                          decision_correct=(bool(predicted)==bool(expected)) if row['decision']!='review_required' else None)
        except Exception as e:
            result.update(error=type(e).__name__+': '+str(e),detected=None,exact_match=False,decision_correct=False)
        result['seconds']=round(time.monotonic()-t0,3);results.append(result)
        a.out.write_text(json.dumps(dict(runtime=runtime,summary=summarize(results),results=results,status='running'),indent=2)+'\n')
        if (i+1)%5==0:print(f'{i+1}/{len(rows)} cases complete',flush=True)
    by_category={k:summarize([r for r in results if r['category']==k]) for k in sorted({r['category'] for r in results})}
    families={k:summarize([r for r in results if r['family']==k]) for k in sorted({r['family'] for r in results})}
    final=dict(runtime=runtime,summary=summarize(results),by_category=by_category,by_family=families,results=results,status='complete',
               scope='Authored synthetic scenario-family evaluation; substitutions and dilution variants are correlated, not independent benchmark samples. No real audio or human gold validation.')
    a.out.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final['summary'],indent=2),flush=True)


if __name__=='__main__':main()
