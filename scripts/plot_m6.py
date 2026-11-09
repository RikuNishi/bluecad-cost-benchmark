#!/usr/bin/env python3
"""Plot measured M6 summary CSVs (requires matplotlib on the VM)."""
import argparse
import csv
from pathlib import Path
import statistics


def read(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def plot(directory):
    import matplotlib.pyplot as plt
    table = read(directory / 'main_table.csv')
    curves = read(directory / 'learning_curves.csv')
    cumulative = read(directory / 'cumulative_cost.csv')
    protocol = 'All questions; 20,000 pairs; classifier thresholds tuned on validation'
    fig, ax = plt.subplots()
    for row in table:
        if row['task'] == 'task1' and row['gpu_hours_mean'] and row['accuracy_all_mean']:
            ax.errorbar(float(row['gpu_hours_mean']), float(row['accuracy_all_mean']), yerr=float(row['accuracy_all_std']), fmt='o', label=row['condition'])
    ax.set(xlabel='Mean measured total GPU-hours', ylabel='Task 1 accuracy (all questions)', title=protocol)
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'performance_cost.png'); plt.close(fig)
    fig, ax = plt.subplots()
    conditions = sorted({r['condition'] for r in curves})
    for condition in conditions:
        rows = [r for r in curves if r['condition'] == condition]
        sizes = sorted({int(r['train_size']) for r in rows})
        means, stds = [], []
        for size in sizes:
            vals = [float(r['task1_accuracy_all']) for r in rows if int(r['train_size']) == size]
            means.append(statistics.mean(vals)); stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        if len(sizes) > 1:
            line, = ax.plot(sizes, means, marker='o', label=condition)
            ax.fill_between(sizes, [m-s for m, s in zip(means, stds)], [m+s for m, s in zip(means, stds)], color=line.get_color(), alpha=.2)
        else:
            ax.errorbar(sizes, means, yerr=stds, fmt='o', label=condition)
    ax.set(xlabel='Training pairs', ylabel='Task 1 accuracy (all questions)', title=protocol)
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'learning_curves.png'); plt.close(fig)
    fig, ax = plt.subplots()
    for condition in sorted({r['condition'] for r in cumulative}):
        rows = [r for r in cumulative if r['condition'] == condition]
        ax.plot([int(r['cumulative_evaluated_items']) for r in rows], [float(r['cumulative_gpu_hours']) for r in rows], marker='.', label=condition)
    ax.set(xlabel='Cumulative evaluated questions', ylabel='Cumulative measured GPU-hours', title=protocol)
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'cumulative_cost.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary-dir', type=Path, default=Path('results/m6/summary'))
    plot(p.parse_args().summary_dir)

if __name__ == '__main__': main()
