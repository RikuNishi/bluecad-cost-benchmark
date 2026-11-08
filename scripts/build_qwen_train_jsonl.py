#!/usr/bin/env python3
"""Convert the shared manifests; unlike the authors' NEG_PER_POS=1, 70% hard-negative recipe, preserve the common pairs exactly."""
import argparse
import json
from pathlib import Path


def build(manifest, output, check_paths=True):
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open() as src, output.open('w') as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['label'] not in (0, 1):
                raise ValueError('Expected binary label')
            result = {'drawing_path': row['drawing_path'], 'render_path': row['render_path'], 'answer': 'Yes' if row['label'] else 'No'}
            if check_paths:
                for path in (result['drawing_path'], result['render_path']):
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
            dst.write(json.dumps(result) + '\n')
            count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', type=Path, default=Path('manifests/train_pairs_20000.jsonl'))
    p.add_argument('--val-manifest', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--out-dir', type=Path, default=Path('results/qwen3vl8b_ft'))
    p.add_argument('--skip-exists-check', action='store_true')
    args = p.parse_args()
    for name, manifest in [('train', args.train_manifest), ('val', args.val_manifest)]:
        print(f'{name}: {build(manifest, args.out_dir / (name + ".jsonl"), not args.skip_exists_check)} pairs')

if __name__ == '__main__':
    main()
