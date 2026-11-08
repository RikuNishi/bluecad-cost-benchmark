#!/usr/bin/env python3
"""Collect the unique images from pair manifests and official items."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.itemmap import load_items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--items-dir', type=Path, default=Path.home() / 'workspace/research/blueprint2cad-official/vlms/src/benchmark/items')
    p.add_argument('--manifest-dir', type=Path, default=Path('manifests'))
    p.add_argument('--out', type=Path, default=Path('manifests/images_all.jsonl'))
    args = p.parse_args()
    paths = set()
    for name in ('train_pairs_20000.jsonl', 'val_pairs_2000.jsonl'):
        for line in (args.manifest_dir / name).read_text().splitlines():
            row = json.loads(line)
            paths.update((row['drawing_path'], row['render_path']))
    for name, task in (('task1_items.json', '1'), ('task1_dim_items.json', '1'), ('task2_items.json', '2')):
        for item in load_items(args.items_dir / name, task):
            paths.add(item['drawing_path'])
            paths.update([item['render_path']] if task == '1' else item['render_paths'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps({'image_id': path, 'path': path}) + '\n' for path in sorted(paths)))
    print(f'{len(paths)} images: {args.out}')

if __name__ == '__main__':
    main()
