"""Render the public comparison figure from completed, source-aligned reports.

Optional dependency: matplotlib. No model inference or data generation occurs.
"""
import argparse
import json
from pathlib import Path

from .evaluate import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    files = ['v2-test', 'candidate-test', 'gliner-labels-test', 'gliner-policy-test']
    names = ['Spaniel v2', 'Spaniel Context v0.1', 'GLiNER2.5 · labels', 'GLiNER2.5 · descriptions']
    reports = [json.loads((a.reports/(name+'.json')).read_text()) for name in files]
    if any(r['status']!='complete' for r in reports):
        raise ValueError('all reports must be complete')
    reference = {r['id']:(r['sha256'],r['expected_spans'],r['expected_decision']) for r in reports[0]['results']}
    for report in reports[1:]:
        gold = {r['id']:(r['sha256'],r['expected_spans'],r['expected_decision']) for r in report['results']}
        if gold!=reference:
            raise ValueError('reports use different cases or gold')
    metrics = [('sensitive_document_recall','Sensitive documents detected\nHigher is better'),
               ('benign_total_withhold_rate','Benign documents withheld\nFalse blocks + inspection failures'),
               ('exact_span_f1','Exact typed span F1\nHigher is better')]
    colors = ['#8b97a6','#176b91','#5f8060','#9275a4']
    fig, axes = plt.subplots(2,3,figsize=(14,7.2))
    for row, (label, predicate) in enumerate([
        ('Credentials · 39 sensitive / 36 benign',lambda r:'compensation' not in r['category']),
        ('Compensation · 42 sensitive / 42 benign',lambda r:'compensation' in r['category'])]):
        summaries = [summarize([r for r in report['results'] if predicate(r)]) for report in reports]
        for col,(metric,title) in enumerate(metrics):
            ax=axes[row,col]
            values=[s[metric] for s in summaries]
            ax.barh(range(4),values,color=colors,height=.62)
            ax.invert_yaxis();ax.set_xlim(0,1.16)
            ax.set_yticks(range(4),names if col==0 else ['']*4,fontsize=10)
            ax.xaxis.set_major_locator(ticker.FixedLocator([0,.25,.5,.75,1]))
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(1) if col<2 else ticker.FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both',length=0,labelsize=9)
            ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
            for i,value in enumerate(values):
                text=f'{value:.1%}' if col<2 else f'{value:.3f}'
                ax.text(value+.025,i,text,va='center',fontsize=10)
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(title,fontsize=12,pad=20)
        axes[row,0].text(0,1.43 if row==0 else 1.12,label,transform=axes[row,0].transAxes,fontsize=11,fontweight='bold')
    fig.suptitle('Context-sensitive detection: recall, withholding and exact evidence',fontsize=17,x=.04,ha='left',y=.97)
    fig.text(.04,.055,'159 authored synthetic cases across 17 families; substitutions and long variants are correlated. No independent human gold.',fontsize=10,color='#444444')
    fig.text(.04,.025,'GLiNER is off the shelf; Spaniel Context is task-adapted. Different hardware: no speed or architecture-superiority claim. Lower benign withholding is better.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.19,right=.98,bottom=.17,top=.77,wspace=.13,hspace=.65)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
