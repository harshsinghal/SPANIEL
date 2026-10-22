"""Export evaluation evidence without republishing source text or predictions.

Raw reports remain local. Public row records contain decisions, offsets and
hashes so generated scenarios can be reconstructed and matched independently.
"""
import argparse
import json
from pathlib import Path

from .compare import compare
from .schema import sha


ROW_FIELDS = ('id', 'family', 'pair_id', 'category', 'expected_decision',
              'expected_spans', 'predicted_spans', 'sha256', 'detected',
              'exact_match', 'decision_correct', 'seconds')


def public_report(report):
    if report.get('status') != 'complete':
        raise ValueError('cannot publish an incomplete evaluation')
    out = {key: value for key, value in report.items() if key != 'results'}
    out['results'] = []
    for row in report['results']:
        public = {key: row[key] for key in ROW_FIELDS if key in row}
        if row.get('error') is not None:
            # A downstream exception can contain source excerpts or paths.
            public['error'] = 'inspection_failed'
        out['results'].append(public)
    out['export_note'] = 'Source text, tagged predictions and exception bodies omitted. Offsets are character offsets into the source identified by sha256.'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    names = ['v2-dev', 'candidate-dev', 'v2-test', 'candidate-test',
             'v2-legacy', 'candidate-legacy']
    reports = {name: json.loads((a.results/(name+'.json')).read_text()) for name in names}
    comparison = compare(reports['v2-test'], reports['candidate-test'],
                         reports['v2-legacy'], reports['candidate-legacy'])
    a.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (a.out/(name+'.json')).write_text(json.dumps(public_report(report), indent=2)+'\n')
    (a.out/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    manifest = {name+'.json': sha((a.out/(name+'.json')).read_text()) for name in names+['comparison']}
    (a.out/'report-hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'exported_reports': len(names), 'meets_numeric_screening': comparison['meets_numeric_screening']}))


if __name__ == '__main__':
    main()
