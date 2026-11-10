#!/usr/bin/env python3
"""Find measured-cost candidates within 0.01 of each task's best accuracy."""
import argparse
import csv
from pathlib import Path

def reading(rows, task):
    rows = [r for r in rows if r['task'] == task and r.get('accuracy_all_mean')]
    if not rows: raise ValueError(f'No accuracy values for {task}')
    best = max(float(r['accuracy_all_mean']) for r in rows)
    eligible = [r for r in rows if best - float(r['accuracy_all_mean']) <= .0100000001]
    known = [r for r in eligible if r.get('gpu_hours_mean')]
    winner = min(known, key=lambda r: float(r['gpu_hours_mean'])) if known else None
    return dict(task=task, best_accuracy=best, condition=winner['condition'] if winner else '', accuracy_all_mean=winner['accuracy_all_mean'] if winner else '', gpu_hours_mean=winner['gpu_hours_mean'] if winner else '', eligible_missing_cost=';'.join(r['condition'] for r in eligible if not r.get('gpu_hours_mean')), status='incomplete_costs' if len(known) != len(eligible) else 'complete')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary-dir', type=Path, default=Path('results/m6/summary'))
    p.add_argument('--out', type=Path, default=Path('results/m7/cost_accuracy.csv'))
    a=p.parse_args()
    with (a.summary_dir / 'main_table.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    output=[reading(rows,t) for t in ('task1','task2')]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    for row in output: print(row)
if __name__ == '__main__': main()
