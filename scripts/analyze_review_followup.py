#!/usr/bin/env python3
"""Review follow-up diagnostics on saved predictions (CPU only, no inference).

Written after the 2026-11-11 Sol review of the benchmark comparison.

A) Letter-shift (off-by-one) analysis for Qwen Task 2. The benchmark paper
   (arXiv:2611.02433v2, Appendix C.5) reports its fine-tuned model answering
   "E" on 12.9% of items and the letter one past the ground truth on 54.1%.
   M7's taxonomy counted out-of-range letters but never measured the shift.
   This script extracts the A-E letter from each saved raw response and
   reports exact / one-past / one-before / shifted-back accuracy.

B) Overlap-stratified accuracy. The study's training pairs and the official
   evaluation items share referenced parts: an item's negatives/candidates
   can be models that also appear in the training or validation manifests.
   This script flags every item per seed and reports accuracy_all for the
   train-overlap, val-overlap and clean strata of every condition, so the
   size of any exposure effect can be read off directly.

Sanity: each cell's overall accuracy_all must reproduce the M6 EXPECTED
values (from bluecad_cost.m7) within rounding; mismatches are flagged.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.m7 import CONDITIONS, EXPECTED, TASKS, cell, correct, read_items, read_predictions, seeds, taxonomy
from bluecad_cost.scoring import parse_prediction

LETTERS = 'ABCDE'


def letter_ae(raw):
    """Best single A-E letter for a raw Task 2 response, or None.

    Precedence mirrors the study parser: an ANSWER:-tagged A-D letter wins;
    otherwise a lone standalone A-E letter (covers the "E" case the parser
    rejects); otherwise the last standalone A-E letter of several.
    """
    if not isinstance(raw, str):
        return None
    parsed = parse_prediction(raw, '2')
    if parsed:
        return parsed
    standalone = re.findall(r'\b[A-Z]\b', raw.upper())
    if len(standalone) == 1 and standalone[0] in LETTERS:
        return standalone[0]
    if len(standalone) > 1 and standalone[-1] in LETTERS:
        return standalone[-1]
    return None


def shift_analysis(items, predictions):
    """predictions: {item_id: row}. Returns (summary dict, confusion Counter)."""
    idx = {c: i for i, c in enumerate(LETTERS)}
    conf = Counter()
    tax = Counter()
    n_letter = exact = one_past = one_before = shifted = 0
    for item in items:
        row = predictions[item['id']]
        raw = row.get('raw')
        tax[taxonomy(raw, item['answer'])] += 1
        letter = letter_ae(raw)
        conf[(item['answer'], letter or 'NONE')] += 1
        if letter is None:
            continue
        n_letter += 1
        li, gi = idx[letter], idx[item['answer']]
        exact += li == gi
        one_past += li == gi + 1
        one_before += li == gi - 1
        shifted += li - 1 == gi
    n = len(items)
    summary = {
        'n_items': n, 'n_with_letter': n_letter,
        'exact_correct': exact, 'accuracy_all': exact / n,
        'accuracy_among_lettered': exact / n_letter if n_letter else '',
        'one_past': one_past, 'one_past_share_all': one_past / n,
        'one_past_share_lettered': one_past / n_letter if n_letter else '',
        'one_before': one_before, 'one_before_share_all': one_before / n,
        'shifted_back_correct': shifted, 'shifted_back_accuracy_all': shifted / n,
    }
    for k in ('no_letter', 'letter_out_of_range', 'multiple_letters', 'parsed_but_wrong', 'parsed_correct'):
        summary['tax_' + k] = tax[k]
    for c in LETTERS:
        summary['pred_' + c] = sum(v for (g, p), v in conf.items() if p == c)
    return summary, conf


def manifest_models(path):
    models = set()
    with path.open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                models.add(d['drawing_model'])
                models.add(d['render_model'])
    return models


def item_refs(item):
    return set(re.findall(r'sample_\d+', ' '.join(item['images'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='results root on the volume')
    p.add_argument('--repo', type=Path, required=True, help='repo root (manifests/)')
    p.add_argument('--items-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = {t: read_items(a.items_dir / f'{t}_items.json') for t in TASKS}

    # ---- A) letter shift, Qwen Task 2
    shift_rows, conf_rows = [], []
    for cond in ('qwen3vl8b_ft', 'qwen3vl8b_zs'):
        for seed in seeds(cond):
            preds = {r['id']: r for r in read_predictions(cell(a.root, cond, seed) / 'task2_predictions.jsonl')}
            summary, conf = shift_analysis(items['task2'], preds)
            shift_rows.append({'condition': cond, 'seed': seed, **summary})
            for (gt, pred), n in sorted(conf.items()):
                conf_rows.append({'condition': cond, 'seed': seed, 'gt': gt, 'pred_letter': pred, 'n': n})
    with (a.out / 'shift_task2.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(shift_rows[0])); w.writeheader(); w.writerows(shift_rows)
    with (a.out / 'shift_confusion.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(conf_rows[0])); w.writeheader(); w.writerows(conf_rows)
    for r in shift_rows:
        print(f"SHIFT {r['condition']} s{r['seed']}: exact={r['accuracy_all']:.3f} "
              f"one_past={r['one_past']} ({r['one_past_share_all']:.3f}) "
              f"shifted_back={r['shifted_back_accuracy_all']:.3f} tax={ {k[4:]: r[k] for k in r if k.startswith('tax_')} }")

    # ---- B) overlap strata
    val_models = manifest_models(a.repo / 'manifests' / 'val_pairs_2000.jsonl')
    train_models = {}
    for seed in range(3):
        path = (a.repo / 'manifests' / 'train_pairs_20000.jsonl') if seed == 0 else \
               (a.repo / 'manifests' / 'seeds' / f'seed{seed}' / 'train_pairs_20000.jsonl')
        if not path.exists():
            raise FileNotFoundError(f'missing train manifest for seed {seed}: {path}')
        train_models[seed] = manifest_models(path)
        print(f'MANIFEST seed{seed}: train models={len(train_models[seed])} val models={len(val_models)}')

    strata_rows = []
    for ti, task in enumerate(TASKS):
        refs = {it['id']: item_refs(it) for it in items[task]}
        for cond in CONDITIONS:
            for seed in seeds(cond):
                preds = {r['id']: r for r in read_predictions(cell(a.root, cond, seed) / f'{task}_predictions.jsonl')}
                buckets = {}
                for it in items[task]:
                    r = refs[it['id']]
                    stratum = 'train' if r & train_models[seed] else ('val' if r & val_models else 'clean')
                    b = buckets.setdefault(stratum, [0, 0])
                    b[0] += 1
                    b[1] += correct(it, preds[it['id']], task)
                tot_n = sum(v[0] for v in buckets.values()); tot_c = sum(v[1] for v in buckets.values())
                expected = EXPECTED[cond][ti][seed]
                check = 'ok' if abs(tot_c / tot_n - expected) <= 0.0016 else 'MISMATCH'
                for stratum in ('train', 'val', 'clean'):
                    n, c = buckets.get(stratum, [0, 0])
                    strata_rows.append({'task': task, 'condition': cond, 'seed': seed, 'stratum': stratum,
                                        'n': n, 'correct': c, 'accuracy_all': c / n if n else ''})
                strata_rows.append({'task': task, 'condition': cond, 'seed': seed, 'stratum': 'overall',
                                    'n': tot_n, 'correct': tot_c, 'accuracy_all': tot_c / tot_n,
                                    'expected': expected, 'check': check})
                print(f"STRATA {task} {cond} s{seed}: overall={tot_c / tot_n:.3f} (expected {expected}, {check}) "
                      + ' '.join(f"{s}={buckets.get(s, [0, 0])[1]}/{buckets.get(s, [0, 0])[0]}" for s in ('train', 'val', 'clean')))
    with (a.out / 'overlap_strata.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'condition', 'seed', 'stratum', 'n', 'correct', 'accuracy_all', 'expected', 'check'],
                           extrasaction='ignore')
        w.writeheader(); w.writerows(strata_rows)
    print('DONE')


if __name__ == '__main__':
    main()
