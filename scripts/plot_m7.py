#!/usr/bin/env python3
"""Plot measured M6 summary values; matplotlib is required only here."""
import argparse
import csv
from pathlib import Path
import statistics

TASKS = ('task1', 'task1_dim', 'task2')
def read(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def plot(summary, out):
    import matplotlib.pyplot as plt
    table=read(summary/'main_table.csv'); curves=read(summary/'learning_curves.csv')
    cumulative=read(summary/'cumulative_cost.csv') if (summary/'cumulative_cost.csv').exists() else []
    out.mkdir(parents=True, exist_ok=True)
    fig, axes=plt.subplots(1,3,figsize=(15,5))
    for ax, task in zip(axes,TASKS):
        for r in table:
            if r['task'] != task or not r.get('gpu_hours_mean') or not r.get('accuracy_all_mean'): continue
            err=float(r['accuracy_all_std']) if r.get('accuracy_all_std') and r.get('protocol') != 'reference, single deterministic run' else None
            ax.errorbar(float(r['gpu_hours_mean']),float(r['accuracy_all_mean']),yerr=err,fmt='o',label=r['condition'])
        ax.set(xlabel='Mean measured total GPU-hours (GPU-h)',ylabel='Accuracy (all questions)',title=task)
        ax.legend(fontsize='small')
    fig.tight_layout(); fig.savefig(out/'cost_accuracy.png'); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,5))
    for ax, task in zip(axes,TASKS):
        for cond in ('siglip2','embeddinggemma2'):
            rows=[r for r in curves if r['condition']==cond and r.get(task+'_accuracy_all')]
            sizes=sorted({int(r['train_size']) for r in rows})
            for size in sizes:
                values=[float(r[task+'_accuracy_all']) for r in rows if int(r['train_size'])==size]
                ax.errorbar(size, statistics.mean(values), yerr=statistics.stdev(values) if len(values)>1 else None, fmt='o', label=cond if size==sizes[0] else None)
        ax.set(xlabel='Training pairs (count)', ylabel='Accuracy (all questions)',title=task); ax.legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png'); plt.close(fig)
    required={'condition','cumulative_evaluated_items','cumulative_gpu_hours'}
    if cumulative and required <= cumulative[0].keys():
        fig, ax=plt.subplots()
        for cond in sorted({r['condition'] for r in cumulative}):
            rows=[r for r in cumulative if r['condition']==cond and r['cumulative_evaluated_items'] and r['cumulative_gpu_hours']]
            if rows: ax.plot([int(r['cumulative_evaluated_items']) for r in rows],[float(r['cumulative_gpu_hours']) for r in rows],marker='.',label=cond)
        ax.set(xlabel='Cumulative evaluated items (count)',ylabel='Cumulative measured GPU-hours (GPU-h)',title='Recorded run sequence'); ax.legend(); fig.tight_layout(); fig.savefig(out/'cumulative_cost.png'); plt.close(fig)
    else: print('Cumulative cost table lacks measured points; figure skipped.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary-dir',type=Path,default=Path('results/m6/summary'))
    p.add_argument('--out-dir',type=Path,default=Path('results/m7/figures'))
    a=p.parse_args(); plot(a.summary_dir,a.out_dir)
if __name__=='__main__': main()
