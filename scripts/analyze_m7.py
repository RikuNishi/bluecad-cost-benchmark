#!/usr/bin/env python3
"""Analyze verified saved predictions; no model inference is performed."""
import argparse
from collections import Counter
import json
import hashlib
import statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.m7 import (CONDITIONS, TASKS, cell, correct, fixes, groups, overlap,
                             read_predictions, seeds, taxonomy, verify, write_csv)
from bluecad_cost.scoring import parse_prediction, score

def run(root, items_dir, out):
    items, cells = verify(root, items_dir)
    tax = []
    for cond in ('qwen3vl8b_ft', 'qwen3vl8b_zs'):
        for seed in seeds(cond):
            counts, letters = Counter(), Counter()
            for item in items['task2']:
                pred = cells['task2', cond, seed][item['id']]
                parsed = parse_prediction(pred['raw'], '2')
                if parsed != pred.get('pred'):
                    raise ValueError(f'{cond} seed{seed}: raw/pred mismatch for {item["id"]}')
                counts[taxonomy(pred['raw'], item['answer'])] += 1
                if parsed: letters[parsed] += 1
            row = dict(condition=cond, seed=seed, n_items=len(items['task2']))
            row.update({k: counts[k] for k in ('no_letter','letter_out_of_range','multiple_letters','parsed_but_wrong','parsed_correct')})
            row.update({'pred_' + k: letters[k] for k in 'ABCD'})
            tax.append(row)
    write_csv(out / 'qwen_task2_taxonomy.csv', list(tax[0]), tax)
    rows, breaks = [], []
    # A method fails in a seed when its prediction is wrong or unanswered; the
    # single zero-shot reference is reused, not treated as three independent runs.
    for task in TASKS:
        for seed in range(3):
            methods = {c: cells[task, c, 0 if c.endswith('_zs') else seed] for c in CONDITIONS}
            row = dict(task=task, seed=seed, n_items=len(items[task]))
            row.update(overlap(items[task], methods, task)); rows.append(row)
            for before, after in [('embeddinggemma2','embeddinggemma2_partial'), ('qwen3vl8b_zs','qwen3vl8b_ft')]:
                changes = fixes(items[task], methods[before], methods[after], task)
                for kind, ids in changes.items():
                    breaks.append(dict(analysis='comparison', task=task, seed=seed, condition=after, group=f'{before} to {after}: {kind}', n=len(ids), correct='', accuracy_all='', ids=json.dumps(ids)))
        for condition in CONDITIONS:
            for seed in seeds(condition):
                pred = cells[task, condition, seed]
                grouping = groups(items[task]) if task == 'task1_dim' else {}
                grouping = {'overall': {x['id']: 'all' for x in items[task]}, **grouping}
                for name, mapping in grouping.items():
                    for value in sorted(set(mapping.values())):
                        subset = [x for x in items[task] if mapping.get(x['id']) == value]
                        metrics = score(subset, list(pred.values()), '2' if task == 'task2' else '1')
                        ncorrect = sum(correct(x, pred[x['id']], task) for x in subset)
                        breaks.append(dict(analysis='group', task=task, seed=seed, condition=condition, group=f'{name}={value}', n=len(subset), correct=ncorrect, accuracy_all=metrics['accuracy_all'], accuracy_answered=metrics['accuracy_answered'], answer_rate=metrics['answer_rate'], ids=''))
    # Item-level seed-0 outlier: distinguish whether either/both other seeds fail.
    task='task1_dim'; cond='siglip2'
    buckets = {}
    for item in items[task]:
        flags = tuple(correct(item, cells[task,cond,s][item['id']], task) for s in range(3))
        if not flags[0]: buckets.setdefault('seed0_wrong_seed1_' + str(int(flags[1])) + '_seed2_' + str(int(flags[2])), []).append(item['id'])
    for name, ids in sorted(buckets.items()):
        breaks.append(dict(analysis='seed_outlier', task=task, seed=0, condition=cond, group=name, n=len(ids), correct='', accuracy_all='', ids=json.dumps(ids)))
    # Compare 10k/20k where predictions exist; do not infer missing cells.
    for cond in CONDITIONS[:2]:
        for seed in range(3):
            for task in TASKS:
                path = root / 'm6' / 'curves' / f'seed{seed}' / cond / 'size10000' / f'{task}_predictions.jsonl'
                if not path.exists():
                    breaks.append(dict(analysis='curve_missing', task=task, seed=seed, condition=cond, group=str(path), n='', correct='', accuracy_all='', ids='')); continue
                pred = read_predictions(path); indexed = {x['id']: x for x in pred}
                if len(pred) != len(items[task]) or set(indexed) != {x['id'] for x in items[task]}:
                    raise ValueError(f'{path}: invalid prediction coverage')
                for kind, ids in fixes(items[task], indexed, cells[task,cond,seed], task).items():
                    breaks.append(dict(analysis='curve_10k_to_20k', task=task, seed=seed, condition=cond, group=kind, n=len(ids), correct='', accuracy_all='', ids=json.dumps(ids)))
                if task == 'task2':
                    margins = {}
                    for item in items[task]:
                        scores = indexed[item['id']].get('scores')
                        if not isinstance(scores, list) or len(scores) != 4:
                            raise ValueError(f'{path}: missing four candidate scores')
                        ranked = sorted(scores, reverse=True)
                        margins[item['id']] = ranked[0] - ranked[1]
                    median = statistics.median(margins.values())
                    for label in ('low', 'high'):
                        subset = [x for x in items[task] if (margins[x['id']] <= median) == (label == 'low')]
                        for kind, ids in fixes(subset, indexed, cells[task,cond,seed], task).items():
                            breaks.append(dict(analysis='curve_margin', task=task, seed=seed, condition=cond, group=f'{label}_10k_margin_{kind}; median={median}', n=len(ids), correct='', accuracy_all='', ids=json.dumps(ids)))
    # Hash recorded training manifests and adapter configuration; no weights loaded.
    metadata=[]
    for seed in range(3):
        directory=cell(root, 'qwen3vl8b_ft', seed)
        manifest = (root.parent / 'manifests' / ('train_pairs_20000.jsonl' if seed == 0 else f'seeds/seed{seed}/train_pairs_20000.jsonl'))
        for kind, path in [('train_manifest', manifest), ('adapter_config', directory/'adapter'/'adapter_config.json'), ('run_config', directory/'task2_run_config.json')]:
            present=path.is_file()
            metadata.append(dict(seed=seed, artifact=kind, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest() if present else '', contents=path.read_text() if present and kind != 'train_manifest' else '', status='present' if present else 'missing'))
    write_csv(out/'qwen_training_artifacts.csv', ['seed','artifact','path','sha256','contents','status'], metadata)
    write_csv(out / 'error_overlap.csv', ['task','seed','n_items','all_failed','only_one_failed','none_failed'], rows)
    write_csv(out / 'breakdowns.csv', ['analysis','task','seed','condition','group','n','correct','accuracy_all','accuracy_answered','answer_rate','ids'], breaks)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--items-dir', type=Path, default=Path('data/official_items'))
    p.add_argument('--out-dir', type=Path, default=Path('results/m7'))
    a=p.parse_args(); run(a.results_dir,a.items_dir,a.out_dir)
if __name__ == '__main__': main()
