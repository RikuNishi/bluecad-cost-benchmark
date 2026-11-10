#!/usr/bin/env python3
"""Verify all required M6 predictions before M7 analysis."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.m7 import verify

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--items-dir', type=Path, default=Path('data/official_items'))
    a = p.parse_args()
    _, cells = verify(a.results_dir, a.items_dir)
    print(f'Verified {len(cells)} cells; counts, IDs, saved metrics, and M6 accuracies match.')
if __name__ == '__main__': main()
