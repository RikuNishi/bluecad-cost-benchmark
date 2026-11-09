#!/usr/bin/env python3
"""Assemble measured M6 tables. Missing or unattributed costs remain blank."""
import argparse
import csv
import json
from pathlib import Path
import statistics

CONDITIONS = ('siglip2', 'embeddinggemma2', 'embeddinggemma2_partial', 'qwen3vl8b_ft')
TASKS = ('task1', 'task1_dim', 'task2')
REFERENCE = 'qwen3vl8b_zs'


def metrics_path(root, condition, seed, task, size=20000):
    if size != 20000:
        return root / 'm6' / 'curves' / f'seed{seed}' / condition / f'size{size}' / f'{task}_metrics.json'
    directory = root / condition if seed == 0 or condition == REFERENCE else root / 'm6' / f'seed{seed}' / condition
    return directory / f'{task}_metrics.json'


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def attributed_records(root, ledger, mapping):
    """Join scoped successful-run IDs and manually verified legacy IDs to ledger."""
    lookup = {}
    for path in (root / 'm6' / 'costs').rglob('*.jsonl'):
        scope = path.relative_to(root / 'm6' / 'costs').with_suffix('').parts
        for record in read_jsonl(path):
            lookup[record['run_id']] = scope
    if mapping and mapping.exists():
        with mapping.open(newline='') as f:
            for row in csv.DictReader(f):
                lookup[row['run_id']] = tuple(row['scope'].split('/'))
    return [(r, lookup[r['run_id']]) for r in read_jsonl(ledger) if r['run_id'] in lookup]


def cost_for(records, condition, seed):
    scopes = [('seed' + str(seed), condition)]
    if condition in CONDITIONS[:2] and seed != 0:
        scopes.append(('shared', condition))
    selected = [r for r, scope in records if scope in scopes and r.get('gpu_hours') is not None]
    # Missing stages must not be silently represented as zero.
    kinds = {r['kind'] for r in selected}
    if condition == REFERENCE:
        required = {'qwen_zs_infer_' + task for task in TASKS}
    elif condition == 'qwen3vl8b_ft':
        required = {'qwen_ft_train'} | {'qwen_ft_infer_' + task for task in TASKS}
    elif condition == 'embeddinggemma2_partial':
        required = {'eg2_partial_train', 'extract', 'train', 'infer'}
    else:
        required = {'extract', 'train', 'infer'}
    if not required <= kinds:
        return None
    return sum(r['gpu_hours'] for r in selected)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, ledger, mapping=None):
    out = root / 'm6' / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    records = attributed_records(root, ledger, mapping)
    table, curves = [], []
    for condition in (*CONDITIONS, REFERENCE):
        seeds = [0] if condition == REFERENCE else [0, 1, 2]
        for task in TASKS:
            values = [json.loads(metrics_path(root, condition, s, task).read_text()) for s in seeds]
            row = {'condition': condition, 'task': task, 'protocol': 'reference, single deterministic run' if condition == REFERENCE else '3 seeds'}
            for key, source in [('accuracy_all', 'accuracy_all'), ('macro_f1_answered', 'macro_f1_answered'), ('answer_rate', 'answer_rate'), ('accuracy_answered', 'accuracy_answered')]:
                nums = [v[source] for v in values]
                for s, number in zip(seeds, nums): row[f'{key}_seed{s}'] = number
                if all(n is not None for n in nums):
                    row[f'{key}_mean'], row[f'{key}_std'] = mean_std(nums)
            costs = [cost_for(records, condition, s) for s in seeds]
            for s, cost in zip(seeds, costs): row[f'gpu_hours_seed{s}'] = cost
            if all(c is not None for c in costs): row['gpu_hours_mean'] = statistics.mean(costs)
            table.append(row)
    for condition in CONDITIONS:
        for seed in (0, 1, 2):
            for size in ((2000, 5000, 10000, 20000) if condition in CONDITIONS[:2] else (20000,)):
                row = dict(condition=condition, seed=seed, train_size=size)
                for task in TASKS:
                    row[task + '_accuracy_all'] = json.loads(metrics_path(root, condition, seed, task, size).read_text())['accuracy_all']
                curves.append(row)
    fields = ['condition', 'task', 'protocol']
    for key in ('accuracy_all', 'macro_f1_answered', 'answer_rate', 'accuracy_answered'):
        fields += [f'{key}_seed{s}' for s in range(3)] + [f'{key}_mean', f'{key}_std']
    fields += [f'gpu_hours_seed{s}' for s in range(3)] + ['gpu_hours_mean']
    write_csv(out / 'main_table.csv', table, fields)
    with (out / 'main_table.md').open('w') as f:
        f.write('| ' + ' | '.join(fields) + ' |\n|' + '|'.join(['---'] * len(fields)) + '|\n')
        for row in table: f.write('| ' + ' | '.join(str(row.get(k, '') if row.get(k) is not None else '') for k in fields) + ' |\n')
    write_csv(out / 'learning_curves.csv', curves, ['condition', 'seed', 'train_size'] + [t + '_accuracy_all' for t in TASKS])
    cumulative, totals = [], {}
    for record, scope in sorted(records, key=lambda x: x[0]['timestamp']):
        if scope[0] == 'shared': condition = scope[1]
        elif scope[0].startswith('seed') and len(scope) == 2: condition = scope[1]
        else: continue
        if record.get('gpu_hours') is None: continue
        hours, items = totals.get(condition, (0.0, 0))
        hours += record['gpu_hours']
        if 'infer' in record['kind']: items += record['items']
        totals[condition] = hours, items
        cumulative.append(dict(condition=condition, timestamp=record['timestamp'], run_id=record['run_id'], cumulative_gpu_hours=hours, cumulative_evaluated_items=items))
    write_csv(out / 'cumulative_cost.csv', cumulative, ['condition', 'timestamp', 'run_id', 'cumulative_gpu_hours', 'cumulative_evaluated_items'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    p.add_argument('--legacy-mapping', type=Path, default=Path('results/m6/legacy_cost_mapping.csv'))
    a = p.parse_args()
    summarize(a.results_dir, a.cost_log, a.legacy_mapping)

if __name__ == '__main__': main()
