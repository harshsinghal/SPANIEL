"""Compare complete, aligned baseline/candidate reports without changing labels."""
import argparse
import json
from pathlib import Path

from .evaluate import summarize


def aligned(baseline,candidate):
    if baseline.get('status')!='complete' or candidate.get('status')!='complete':
        raise ValueError('both evaluations must be complete')
    for key in ('data_sha256','device','dtype','decoding','evaluator_sha256','model_weights_sha256'):
        if baseline['runtime'].get(key)!=candidate['runtime'].get(key):
            raise ValueError('evaluation configuration differs: '+key)
    a={r['id']:r for r in baseline['results']};b={r['id']:r for r in candidate['results']}
    if set(a)!=set(b) or len(a)!=len(baseline['results']) or len(b)!=len(candidate['results']):
        raise ValueError('evaluation IDs differ or repeat')
    for key in a:
        for field in ('sha256','expected_decision','expected_spans','category','family'):
            if a[key].get(field)!=b[key].get(field):raise ValueError('evaluation gold differs: '+key)
    return a,b


def compare(baseline,candidate,legacy_base,legacy_candidate):
    a,b=aligned(baseline,candidate);aligned(legacy_base,legacy_candidate)
    selectors={
        'all':lambda r:True,
        'credentials':lambda r:'compensation' not in r['category'],
        'compensation':lambda r:'compensation' in r['category'],
        'transcript_or_noise':lambda r:any(w in r['category'] for w in ('transcript','noisy')),
        'long_context':lambda r:r['category'].startswith('long_'),
        'long_beginning':lambda r:r['category'].startswith('long_') and r['id'].endswith('-0.0'),
        'long_middle':lambda r:r['category'].startswith('long_') and r['id'].endswith('-0.5'),
        'long_end':lambda r:r['category'].startswith('long_') and r['id'].endswith('-1.0'),
    }
    output={}
    metrics=('sensitive_document_recall','benign_false_block_rate','exact_span_f1')
    for name,predicate in selectors.items():
        before=summarize([r for r in a.values() if predicate(r)])
        after=summarize([r for r in b.values() if predicate(r)])
        delta={m:after[m]-before[m] if before[m] is not None and after[m] is not None else None for m in metrics}
        output[name]=dict(baseline=before,candidate=after,delta=delta)
    old=summarize(legacy_base['results']);new=summarize(legacy_candidate['results'])
    credential=output['credentials']['candidate']
    gates={
        'credential_recall_at_least_0.95':credential['sensitive_document_recall'] is not None and credential['sensitive_document_recall']>=0.95,
        'credential_false_blocks_at_most_0.05':credential['benign_false_block_rate'] is not None and credential['benign_false_block_rate']<=0.05,
        'no_candidate_inspection_errors':output['all']['candidate']['inspection_errors']==0 and new['inspection_errors']==0,
        'legacy_exact_f1_drop_at_most_0.01':new['exact_span_f1'] is not None and old['exact_span_f1'] is not None and new['exact_span_f1']>=old['exact_span_f1']-0.01,
    }
    families={name:dict(baseline=summarize([r for r in a.values() if r['family']==name]),
                        candidate=summarize([r for r in b.values() if r['family']==name]))
              for name in sorted({r['family'] for r in a.values()})}
    return dict(slices=output,families=families,legacy_regression=dict(baseline=old,candidate=new),screening=gates,
                meets_numeric_screening=all(gates.values()),production_validated=False,
                scope='Small authored synthetic scenario-family test plus legacy synthetic PII regression. Correlated substitutions and dilution variants; no independent human adjudication, real audio, or deployment evaluation.',
                artifacts={k:v['runtime'] for k,v in [('baseline',baseline),('candidate',candidate),('legacy_baseline',legacy_base),('legacy_candidate',legacy_candidate)]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    read=lambda name:json.loads((a.results/name).read_text())
    result=compare(read('v2-test.json'),read('candidate-test.json'),read('v2-legacy.json'),read('candidate-legacy.json'))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('screening','meets_numeric_screening','production_validated')},indent=2))


if __name__=='__main__':main()
