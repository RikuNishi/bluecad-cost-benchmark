import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.partial import discover_blocks, pair_batch, parse_scope, select_scope


class Parameter:
    def __init__(self, n): self.n, self.requires_grad = n, True
    def numel(self): return self.n


class Module:
    def __init__(self, n=0, children=None):
        self.own = [Parameter(n)] if n else []
        self.children = children or {}
    def parameters(self):
        yield from self.own
        for child in self.children.values(): yield from child.parameters()
    def named_modules(self, prefix=''):
        yield prefix, self
        for name, child in self.children.items():
            yield from child.named_modules(f'{prefix}.{name}' if prefix else name)


class ModuleList(Module):
    def __init__(self, modules):
        self.modules = modules
        super().__init__(children={str(i): m for i, m in enumerate(modules)})
    def __len__(self): return len(self.modules)
    def __iter__(self): return iter(self.modules)


class PartialTests(unittest.TestCase):
    def test_specs(self):
        self.assertEqual(parse_scope('all'), (None, None))
        self.assertEqual(parse_scope('last_k:vision.layers:2'), ('vision.layers', 2))
        for spec in ('last_k:x:0', 'last_k:x:-1', 'last_k::2', 'last_k:x:1x', 'other'):
            with self.subTest(spec=spec), self.assertRaises(ValueError): parse_scope(spec)

    def test_scope(self):
        layers = ModuleList([Module(2), Module(3), Module(5)])
        projection = Module(7)
        other = Module(11)
        model = Module(children={'vision': Module(children={'layers': layers, 'projection': projection}), 'text': other})
        blocks = discover_blocks(model)
        self.assertEqual(list(blocks), ['vision.layers'])
        self.assertEqual(select_scope(model, 'last_k:vision.layers:2', blocks, {'vision.projection': projection}), 15)
        self.assertEqual([p.requires_grad for p in model.parameters()], [False, True, True, True, False])
        for spec in ('last_k:missing:1', 'last_k:vision.layers:4'):
            with self.assertRaises(ValueError): select_scope(model, spec, blocks, {})
        self.assertEqual(select_scope(model, 'all', blocks, {}), 28)

    def test_pair_batch(self):
        rows = [{'drawing_path': 'd1', 'render_path': 'r1', 'label': 1}, {'drawing_path': 'd2', 'render_path': 'r2', 'label': 0}]
        result = pair_batch(rows, {'d1': 'D1', 'r1': 'R1', 'd2': 'D2', 'r2': 'R2'})
        self.assertEqual(result, (['D1', 'D2'], ['R1', 'R2'], [1, 0]))
        self.assertEqual([len(x) for x in result], [2, 2, 2])

    def test_extractor_default(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/extract_features.py'
        spec = importlib.util.spec_from_file_location('extract_features', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = module.make_parser().parse_args(['--encoder', 'embeddinggemma2', '--images', 'images', '--out', 'out'])
        self.assertIsNone(args.model_path)
        self.assertEqual(module.MODELS[args.encoder], 'google/embeddinggemma-2')

if __name__ == '__main__': unittest.main()
