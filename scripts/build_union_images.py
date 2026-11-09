#!/usr/bin/env python3
"""Collect images from training manifests, canonical validation and official tasks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.itemmap import load_items


def build(pair_manifests, val, items_dir, out):
    paths = set()
    for manifest in [*pair_manifests, val]:
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                paths.update((row['drawing_path'], row['render_path']))
    for name, task in (('task1_items.json', '1'), ('task1_dim_items.json', '1'), ('task2_items.json', '2')):
        for item in load_items(items_dir / name, task):
            paths.add(item['drawing_path'])
            paths.update([item['render_path']] if task == '1' else item['render_paths'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps({'image_id': path, 'path': path}) + '\n' for path in sorted(paths)))
    print(f'{len(paths)} images: {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-manifest', type=Path, action='append', required=True)
    p.add_argument('--val-manifest', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--items-dir', type=Path, default=Path('data/official_items'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.pair_manifest, a.val_manifest, a.items_dir, a.out)

if __name__ == '__main__':
    main()
