"""CPU-only verification and item-level analysis of saved M6 outputs."""
import csv
import json
import re
from collections import Counter
from pathlib import Path
from .scoring import parse_prediction, score

TASKS = ('task1', 'task1_dim', 'task2')
CONDITIONS = ('siglip2', 'embeddinggemma2', 'embeddinggemma2_partial', 'qwen3vl8b_ft', 'qwen3vl8b_zs')
EXPECTED = {
 'siglip2': ((.725,.728,.728),(.670,.707,.701),(.872,.885,.883)),
 'embeddinggemma2': ((.740,.731,.739),(.664,.673,.677),(.853,.872,.852)),
 'embeddinggemma2_partial': ((.763,.766,.769),(.760,.757,.760),(.954,.946,.951)),
 'qwen3vl8b_ft': ((.748,.748,.755),(.675,.719,.725),(.195,.189,.305)),
 'qwen3vl8b_zs': ((.653,),(.632,),(.334,)),
}

def seeds(condition):
    return (0,) if condition == 'qwen3vl8b_zs' else (0, 1, 2)

def cell(root, condition, seed):
    return root / condition if seed == 0 else root / 'm6' / f'seed{seed}' / condition

def read_items(path):
    return json.loads(path.read_text(encoding='utf8'))['items']

def read_predictions(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

def verify(root, items_dir, count=1000, expected=EXPECTED):
    """Fail on missing files, duplicate/incorrect IDs, or unreproduced metrics."""
    all_items = {t: read_items(items_dir / f'{t}_items.json') for t in TASKS}
    cells = {}
    for ti, task in enumerate(TASKS):
        items = all_items[task]
        ids = [x['id'] for x in items]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f'{task}: official item count or IDs invalid')
        for condition in CONDITIONS:
            for seed in seeds(condition):
                directory = cell(root, condition, seed)
                path = directory / f'{task}_predictions.jsonl'
                preds = read_predictions(path)
                pids = [p['id'] for p in preds]
                if len(pids) != count or len(set(pids)) != count or set(pids) != set(ids):
                    raise ValueError(f'{path}: prediction count or ID coverage mismatch')
                actual = score(items, preds, '2' if task == 'task2' else '1')
                saved = json.loads((directory / f'{task}_metrics.json').read_text())
                for key in ('n_items', 'n_answered', 'answer_rate', 'accuracy_all', 'accuracy_answered', 'macro_f1_answered', 'evaluated_ids', 'answered_ids'):
                    if actual[key] != saved[key]:
                        raise ValueError(f'{path}: saved {key} mismatch')
                if abs(actual['accuracy_all'] - expected[condition][ti][seed]) > 1e-9:
                    raise ValueError(f'{path}: M6 accuracy mismatch: {actual["accuracy_all"]}')
                cells[task, condition, seed] = {p['id']: p for p in preds}
    return all_items, cells

def taxonomy(raw, answer):
    parsed = parse_prediction(raw, '2')
    if parsed:
        return 'parsed_correct' if parsed == answer else 'parsed_but_wrong'
    # Standalone choice-like letters; the parser itself takes precedence.
    letters = re.findall(r'\b[A-Z]\b', raw.upper()) if isinstance(raw, str) else []
    if len(letters) > 1:
        return 'multiple_letters'
    if letters and letters[0] not in 'ABCD':
        return 'letter_out_of_range'
    return 'no_letter'

def correct(item, pred, task):
    value = pred.get('pred') if pred.get('pred') is not None else pred.get('raw')
    if pred.get('error') or str(value).upper() in ('ERROR', 'UNPARSEABLE'):
        return False
    return parse_prediction(value, '2' if task == 'task2' else '1') == item['answer']

def overlap(items, methods, task):
    counts = Counter(sum(not correct(item, predictions[item['id']], task) for predictions in methods.values()) for item in items)
    return {'all_failed': counts[len(methods)], 'only_one_failed': counts[1], 'none_failed': counts[0]}

def fixes(items, before, after, task):
    out = {'fixed': [], 'broken': [], 'both_wrong': [], 'both_correct': []}
    for item in items:
        a, b = correct(item, before[item['id']], task), correct(item, after[item['id']], task)
        out[{(False, True): 'fixed', (True, False): 'broken', (False, False): 'both_wrong', (True, True): 'both_correct'}[a,b]].append(item['id'])
    return out

def groups(items):
    """Only explicit official grouping fields, not image paths or inferred views."""
    result = {}
    for item in items:
        for key, val in item.items():
            if key == 'group' or key == 'tier':
                if isinstance(val, (str, int, float, bool)):
                    result.setdefault(key, {})[item['id']] = str(val)
        meta = item.get('meta')
        if isinstance(meta, dict):
            for key, val in meta.items():
                if isinstance(val, (str, int, float, bool)):
                    result.setdefault('meta.' + key, {})[item['id']] = str(val)
    return result

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
