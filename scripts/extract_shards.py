#!/usr/bin/env python3
"""Extract selected shard archives, retaining already extracted files."""
import argparse
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('shards', nargs='+', help='Shard numbers (e.g. 0000), or all')
    p.add_argument('--data-dir', type=Path, default=Path('data/bluecad-30k'))
    args = p.parse_args()
    for kind in ('drawings', 'renders'):
        directory = args.data_dir / kind
        archives = sorted(directory.glob('*.tar*'))
        selected = archives if 'all' in args.shards else [a for a in archives if any(s in a.name for s in args.shards)]
        if not selected:
            raise FileNotFoundError(f'No matching {kind} shards in {directory}')
        for archive in selected:
            print(f'Extracting missing files from {archive}', flush=True)
            subprocess.run(['tar', '--extract', '--file', str(archive.resolve()), '--directory', str(directory), '--no-same-owner', '--skip-old-files'], check=True)

if __name__ == '__main__':
    main()
