#!/usr/bin/env python3
"""Create deterministic balanced train and validation pair manifests."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.itemmap import PREFIX

ROOT = Path(__file__).resolve().parents[1]


def indexed_ids(data):
    with (data / 'shard_index.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty shard index')
    key = next((k for k in ('model_id', 'model', 'id', 'sample_id') if k in rows[0]), None)
    if key is None:
        raise ValueError(f'Unknown shard index columns: {list(rows[0])}')
    return {row[key] for row in rows}


def build(data, split, count, seed):
    if count <= 0 or count % 2:
        raise ValueError('Pair count must be positive and even')
    ids = sorted(set((data / 'splits' / f'{split}.txt').read_text().split()) & indexed_ids(data))
    if len(ids) < 2:
        raise ValueError(f'{split} requires at least two indexed models')
    rng = random.Random(seed)
    rng.shuffle(ids)
    rows = []
    for i in range(count // 2):
        a = ids[i % len(ids)]
        for label in (1, 0):
            if label:
                b = a
            else:
                j = rng.randrange(len(ids) - 1)
                ai = ids.index(a)
                b = ids[j + (j >= ai)]
            view = rng.randrange(6)
            rows.append(dict(pair_id=f'{split}_{len(rows):05d}', drawing_model=a,
                drawing_path=f'{PREFIX}/drawings/{a}/normal/{a}_front.png', render_model=b,
                render_path=f'{PREFIX}/renders/{b}/{view:03d}.png', render_view=view,
                label=label, split=split))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-dir', type=Path, default=ROOT / PREFIX)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'manifests')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, count in (('train', 20000), ('val', 2000)):
        rows = build(args.data_dir, split, count, args.seed)
        missing = set()
        broken = set()
        for row in rows:
            for key in ('drawing_path', 'render_path'):
                path = row[key]
                relative = Path(path).relative_to(PREFIX)
                if not (args.data_dir / relative).is_file():
                    missing.add(path)
                    if (args.data_dir / relative.parts[0] / relative.parts[1]).is_dir():
                        broken.add(path)
        if broken:
            raise FileNotFoundError(f'{len(broken)} files missing from extracted model directories; example: {sorted(broken)[0]}')
        out = args.out_dir / f'{split}_pairs_{count}.jsonl'
        out.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        print(f'{out}: {count} pairs, {count//2} positive, {count//2} negative; {len(missing)} missing image files (extract/download their shards)')

if __name__ == '__main__':
    main()
