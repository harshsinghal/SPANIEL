"""Live application smoke cases for the experimental contextual adapter.

These illustrative cases are not the frozen evaluation set. An unresolved
context case is deliberately unscored: the model has no abstention contract.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import Request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from spaniel_guard import LocalSpaniel, Policy, make_document_tool

HERE = Path(__file__).resolve().parent
POLICIES = {
    'credentials': Policy('credential-context', 'experiment-1', ('password', 'api key', 'private key')),
    'compensation': Policy('private-compensation', 'experiment-1', ('private compensation',)),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', default='http://127.0.0.1:8377/api/tag')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    tagger = LocalSpaniel(a.endpoint)
    try:
        request = Request(a.endpoint.removesuffix('/api/tag')+'/api/model')
        with tagger.opener.open(request, timeout=10) as response:
            metadata = json.loads(response.read(100_001))
    except Exception:
        metadata = {'configuration_unavailable': True}
    cases = json.loads((HERE/'cases.json').read_text())
    rows = []
    for case in cases:
        started = time.monotonic()
        result = make_document_tool(HERE/'documents', POLICIES[case['policy']], tagger)(case['document'])
        row = dict(id=case['id'], policy=case['policy'], expected=case['expected'],
                   actual=result['status'], code=result['code'],
                   content_returned=result['content'] is not None,
                   seconds=round(time.monotonic()-started, 3))
        row['matched_expectation'] = result['status']==case['expected'] if case['expected'] is not None else None
        rows.append(row)
        print(json.dumps(row), flush=True)
    scored = [r for r in rows if r['expected'] is not None]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), model_configuration=metadata,
                  scope='Illustrative integration smoke test, not a benchmark. The credential policy requests three labels together; the frozen concept benchmark requests one at a time. Null expected outcomes are unscored ambiguity demonstrations.',
                  summary=dict(cases=len(rows), scored=len(scored),
                               matched=sum(r['matched_expectation'] is True for r in scored),
                               sensitive_released=sum(r['expected']=='blocked' and r['actual']=='allowed' for r in scored),
                               false_blocks=sum(r['expected']=='allowed' and r['actual']=='blocked' for r in scored),
                               unexpected_errors=sum(r['expected']!='error' and r['actual']=='error' for r in scored)),
                  results=rows)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(r['matched_expectation'] for r in scored) else 1


if __name__ == '__main__':
    raise SystemExit(main())
