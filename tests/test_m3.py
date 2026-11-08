import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bluecad_cost.itemmap import load_items, map_path
from bluecad_cost.head import best_threshold


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M3Tests(unittest.TestCase):
    def test_manifest(self):
        build = script('build_pair_manifest').build
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / 'splits').mkdir()
            (data / 'splits/train.txt').write_text('a\nb\nphantom\n')
            (data / 'shard_index.csv').write_text('model_id,shard\na,0\nb,0\n')
            rows = build(data, 'train', 20, 3)
            self.assertEqual(rows, build(data, 'train', 20, 3))
            self.assertEqual(sum(r['label'] for r in rows), 10)
            self.assertTrue(all((r['drawing_model'] == r['render_model']) == bool(r['label']) for r in rows))
            self.assertEqual({r['drawing_model'] for r in rows}, {'a', 'b'})

    def test_mapping(self):
        self.assertEqual(map_path('/x/png_out/g/a/dim/002.png'), 'data/bluecad-30k/drawings/a/dim/a_left_dim.png')
        self.assertEqual(map_path('/x/renders/g/a/005.png'), 'data/bluecad-30k/renders/a/005.png')

    def test_real_items(self):
        directory = Path.home() / 'workspace/research/blueprint2cad-official/vlms/src/benchmark/items'
        if not directory.is_dir():
            self.skipTest('Official item clone unavailable')
        for name, task in [('task1_items.json', '1'), ('task1_dim_items.json', '1'), ('task2_items.json', '2')]:
            original = json.loads((directory / name).read_text())['items']
            mapped = load_items(directory / name, task)
            self.assertEqual([i['id'] for i in mapped], [i['id'] for i in original])
            for a, b in zip(original, mapped):
                expected = [map_path(x) for x in a['images']]
                actual = [b['drawing_path']] + ([b['render_path']] if task == '1' else b['render_paths'])
                self.assertEqual(sorted(expected), sorted(actual))
                if task == '2':
                    self.assertEqual(b['render_paths'], [p for p in expected if '/renders/' in p])
                for path in actual:
                    model_dir = ROOT / Path(path).parent
                    if model_dir.is_dir():
                        self.assertTrue((ROOT / path).is_file(), path)

    def test_threshold_and_argmax(self):
        self.assertEqual(best_threshold([.1, .2, .8, .9], [0, 0, 1, 1]), .5)
        self.assertEqual(script('eval_frozen').argmax_letter([.2, .8, .8, .1]), 'B')

    def test_train_synthetic(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        from bluecad_cost.head import train, predict
        torch.manual_seed(3)
        d = torch.randn(100, 8)
        r = torch.cat((d[:50], -d[50:]))
        labels = torch.tensor([1] * 50 + [0] * 50)
        head, mean, scale = train(d, r, labels)
        accuracy = ((predict(head, mean, scale, d, r) >= .5) == labels.bool()).float().mean().item()
        self.assertGreater(accuracy, .9)

if __name__ == '__main__':
    unittest.main()
