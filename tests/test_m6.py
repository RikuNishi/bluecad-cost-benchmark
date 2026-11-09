import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M6Tests(unittest.TestCase):
    def test_seed_cli_and_config(self):
        for name, default in [('train_embeddinggemma_partial', 0), ('train_qwen_lora', 13)]:
            path = ROOT / 'scripts' / (name + '.py')
            output = subprocess.check_output([sys.executable, str(path), '--help'], text=True)
            self.assertIn('--seed', output)
            source = path.read_text()
            self.assertIn(f"p.add_argument('--seed', type=int, default={default})", source)
            self.assertIn("'seed': args.seed" if default == 0 else 'seed=args.seed', source)

    def test_subsets(self):
        mod = load('subsample_pairs')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'train.jsonl'
            rows = [dict(pair_id=str(i), drawing_model=str(i // 2), label=1-i%2) for i in range(20000)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            mod.subsample(path, 2)
            sets = []
            for size in (2000, 5000, 10000):
                out = path.parent / f'train_pairs_{size}_seed2.jsonl'
                data = [json.loads(line) for line in out.read_text().splitlines()]
                self.assertEqual(sum(r['label'] for r in data), size // 2)
                self.assertTrue(all(data[i]['drawing_model'] == data[i+1]['drawing_model'] for i in range(0, size, 2)))
                sets.append({r['pair_id'] for r in data})
            self.assertTrue(sets[0] < sets[1] < sets[2] < {r['pair_id'] for r in rows})
            before = out.read_bytes(); mod.subsample(path, 2); self.assertEqual(before, out.read_bytes())

    def test_union(self):
        mod = load('build_union_images')
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); a = d / 'a'; b = d / 'b'; val = d / 'val'; out = d / 'out'
            for path, x, y in [(a, 'z', 'a'), (b, 'a', 'b'), (val, 'b', 'c')]:
                path.write_text(json.dumps(dict(drawing_path=x, render_path=y)) + '\n')
            def items(path, task):
                return [dict(drawing_path='d', render_path='e')] if task == '1' else [dict(drawing_path='d', render_paths=['f', 'a'])]
            with patch.object(mod, 'load_items', side_effect=items): mod.build([a, b], val, d, out)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(rows, [dict(image_id=x, path=x) for x in ['a', 'b', 'c', 'd', 'e', 'f', 'z']])

    def test_seed_manifests_remove_val(self):
        mod = load('build_seed_manifests')
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); canonical = d / 'val_pairs_2000.jsonl'; canonical.write_text('canonical')
            def fake(cmd, check):
                out = Path(cmd[cmd.index('--out-dir') + 1]); (out / 'val_pairs_2000.jsonl').write_text('bad')
                (out / 'train_pairs_20000.jsonl').write_text('{}\n')
            with patch.object(mod.subprocess, 'run', side_effect=fake): mod.build_seeds(d, d)
            self.assertEqual(canonical.read_text(), 'canonical')
            for s in (1, 2): self.assertFalse((d / 'seeds' / f'seed{s}' / 'val_pairs_2000.jsonl').exists())

    def test_summary(self):
        mod = load('summarize_m6')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for condition in (*mod.CONDITIONS, mod.REFERENCE):
                for seed in ([0] if condition == mod.REFERENCE else [0, 1, 2]):
                    for size in ((2000, 5000, 10000, 20000) if condition in mod.CONDITIONS[:2] else (20000,)):
                        for task in mod.TASKS:
                            path = mod.metrics_path(root, condition, seed, task, size); path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(dict(accuracy_all=.5 + seed*.1, macro_f1_answered=.4, answer_rate=1, accuracy_answered=.5)))
            records = [dict(run_id='a', kind='extract', gpu_hours=2, items=9, timestamp=1), dict(run_id='b', kind='infer', gpu_hours=1, items=10, timestamp=2), dict(run_id='c', kind='train', gpu_hours=.5, items=20, timestamp=3)]
            ledger = root / 'costs.jsonl'; ledger.write_text(''.join(json.dumps(r)+'\n' for r in records))
            mapping = root / 'map.csv'; mapping.write_text('run_id,scope\na,seed0/siglip2\nb,seed0/siglip2\nc,seed0/siglip2\n')
            mod.summarize(root, ledger, mapping)
            out = root / 'm6' / 'summary'
            with (out / 'main_table.csv').open() as f: rows = list(csv.DictReader(f))
            row = rows[0]; self.assertAlmostEqual(float(row['accuracy_all_mean']), .6); self.assertAlmostEqual(float(row['accuracy_all_std']), .1)
            self.assertAlmostEqual(float(row['accuracy_all_seed2']), .7)
            self.assertTrue(any(r['protocol'] == 'reference, single deterministic run' for r in rows))
            with (out / 'cumulative_cost.csv').open() as f: cumulative = list(csv.DictReader(f))
            self.assertEqual(float(cumulative[-1]['cumulative_gpu_hours']), 3.5)
            self.assertEqual(int(cumulative[-1]['cumulative_evaluated_items']), 10)

if __name__ == '__main__': unittest.main()
