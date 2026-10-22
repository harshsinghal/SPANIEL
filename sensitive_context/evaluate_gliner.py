"""Pinned off-the-shelf GLiNER2.5 baseline; no threshold or test-set tuning.

Run in a separate environment. This compares a pretrained baseline with a
task-adapted Spaniel candidate, not equally fine-tuned model architectures.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .evaluate import load_examples, summarize
from .schema import POLICY, sha

MODEL = 'fastino/gliner2.5-base-v1'
REVISION = '54785d51df8d86a0d8f2eb212fec15871aad2c4d'


def source_spans(text, labels, result):
    entities = result.get('entities')
    if not isinstance(entities, dict) or set(entities)-set(labels):
        raise ValueError('unexpected entity schema')
    spans = []
    for label, matches in entities.items():
        for match in matches:
            start, end = match['start'], match['end']
            if (type(start) is not int or type(end) is not int or
                    not 0 <= start < end <= len(text) or
                    not text[start:end].strip() or text[start:end] != match['text']):
                raise ValueError('prediction does not identify a nonempty original source span')
            spans.append((start, end, label))
    return sorted(set(spans))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--schema', choices=['labels', 'policy-descriptions'], required=True)
    p.add_argument('--legacy', action='store_true')
    p.add_argument('--exclude', type=Path, action='append', default=[])
    a = p.parse_args()
    rows = load_examples(a.data, legacy=a.legacy)
    excluded = {json.loads(line)['sha256'] for path in a.exclude for line in path.read_text().splitlines()}
    if any(r['sha256'] in excluded for r in rows):
        raise ValueError('evaluation text overlaps supplied training data')
    import torch
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4)
    model_dir = snapshot_download(MODEL, revision=REVISION,
                                  allow_patterns=['*.json', '*.safetensors', '*.model', 'encoder_config/*'])
    model = AutoExtractor.from_pretrained(model_dir, map_location='cpu').float().eval()
    digest = hashlib.sha256()
    with (Path(model_dir)/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    weight_hash = digest.hexdigest()
    runtime = dict(created_at=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
                   device='cpu', dtype=str(next(model.parameters()).dtype), threads=4,
                   threshold=0.5, schema_mode=a.schema, overlap_policy='flat',
                   data_sha256=sha(a.data.read_text()), model_weights_sha256=weight_hash,
                   evaluator_sha256=sha(Path(__file__).read_text()),
                   versions={name: importlib.metadata.version(name) for name in ('torch','gliner2','transformers','peft')},
                   decoding='Native boundary extraction; exact source-offset validation; no truncation or chunking',
                   comparison_scope='Off-the-shelf checkpoint versus task-adapted Spaniel. Different runtimes/hardware: no speed comparison.')
    schemas = {}
    max_tokens = 0
    # Preflight all inputs before prediction. No silent max_len clipping.
    for row in rows:
        entities = row['entities'] if a.schema=='labels' else {name: POLICY.get(name, name) for name in row['entities']}
        schema = model.create_schema().entities(entities)
        prepared = model.processor.transform_for_inference(row['text'], schema, max_len=None)
        max_tokens = max(max_tokens, len(prepared.input_ids))
        if len(prepared.input_ids) > model.config.max_len:
            raise ValueError('input exceeds the checkpoint sequence limit; not truncated')
        schemas[row['id']] = entities
    runtime['maximum_encoded_tokens'] = max_tokens
    results = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        started = time.monotonic()
        expected = [(s['start'],s['end'],s['label']) for s in row['spans']]
        result = dict(id=row['id'],family=row.get('family'),pair_id=row.get('pair_id'),
                      category=row.get('category','ambiguous'),expected_decision=row['decision'],
                      expected_spans=expected,sha256=row['sha256'])
        try:
            with torch.inference_mode():
                extracted = model.extract_entities(row['text'], schemas[row['id']], threshold=0.5,
                    include_spans=True, include_confidence=True, max_len=None, overlap_policy='flat')
            predicted = source_spans(row['text'], row['entities'], extracted)
            result.update(predicted_spans=predicted,detected=bool(predicted),
                          exact_match=predicted==expected if row['decision']!='review_required' else None,
                          decision_correct=bool(predicted)==bool(expected) if row['decision']!='review_required' else None)
        except Exception as error:
            result.update(error=type(error).__name__+': '+str(error),detected=None,
                          exact_match=False,decision_correct=False)
        result['seconds'] = round(time.monotonic()-started,3)
        results.append(result)
        a.out.write_text(json.dumps(dict(runtime=runtime,summary=summarize(results),results=results,status='running'),indent=2)+'\n')
        if (i+1)%10==0:
            print(f'{i+1}/{len(rows)} cases complete',flush=True)
    final = dict(runtime=runtime,summary=summarize(results),results=results,status='complete',
                 by_category={k:summarize([r for r in results if r['category']==k]) for k in sorted({r['category'] for r in results})},
                 by_family={k:summarize([r for r in results if r['family']==k]) for k in sorted({r['family'] for r in results})},
                 scope='Same frozen synthetic scenarios, no independent human gold. Fixed 0.5 threshold, two predeclared schema modes reported separately. No test-set threshold selection, fine-tuning, or architecture-superiority claim.')
    a.out.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
