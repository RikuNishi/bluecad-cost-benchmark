#!/usr/bin/env python3
"""Build seed 1 and 2 training manifests; retain only canonical validation."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def build_seeds(data_dir, manifest_dir):
    for seed in (1, 2):
        out = manifest_dir / 'seeds' / f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        val = out / 'val_pairs_2000.jsonl'
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name('build_pair_manifest.py')),
                            '--seed', str(seed), '--data-dir', str(data_dir), '--out-dir', str(out)], check=True)
        finally:
            val.unlink(missing_ok=True)
        train = out / 'train_pairs_20000.jsonl'
        content = train.read_bytes()
        print(f'{train}: {len(content.splitlines())} pairs; sha256={hashlib.sha256(content).hexdigest()}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/bluecad-30k'))
    p.add_argument('--manifest-dir', type=Path, default=Path('manifests'))
    a = p.parse_args()
    build_seeds(a.data_dir, a.manifest_dir)

if __name__ == '__main__':
    main()
