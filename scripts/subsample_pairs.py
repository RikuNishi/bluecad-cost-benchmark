#!/usr/bin/env python3
"""Make nested balanced subsets by shuffling intact positive/negative groups."""
import argparse
import json
from pathlib import Path
import random


def subsample(manifest, seed):
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if len(lines) != 20000:
        raise ValueError('Expected exactly 20,000 pairs')
    groups = []
    for i in range(0, len(lines), 2):
        a, b = map(json.loads, lines[i:i+2])
        if (a['label'], b['label']) != (1, 0) or a['drawing_model'] != b['drawing_model']:
            raise ValueError(f'Invalid pair group at row {i + 1}')
        groups.append(lines[i:i+2])
    random.Random(seed).shuffle(groups)
    for size in (2000, 5000, 10000):
        out = manifest.parent / f'train_pairs_{size}_seed{seed}.jsonl'
        out.write_text(''.join(line + '\n' for group in groups[:size // 2] for line in group))
        print(f'{out}: {size} pairs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    subsample(a.train_manifest, a.seed)

if __name__ == '__main__':
    main()
