#!/usr/bin/env python3
"""Evaluate a trained frozen-feature head on unmodified official items."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.head import predict
from bluecad_cost.itemmap import load_items
from bluecad_cost.runwrap import record_run
from bluecad_cost.scoring import score


def argmax_letter(scores):
    if len(scores) != 4:
        raise ValueError('Task 2 requires four candidates')
    return 'ABCD'[max(range(4), key=lambda i: scores[i])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--items-dir', type=Path, default=Path.home() / 'workspace/research/blueprint2cad-official/vlms/src/benchmark/items')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    args = p.parse_args()
    import numpy as np
    import torch
    data = np.load(args.features, allow_pickle=False)
    artifact = torch.load(args.head, map_location='cpu', weights_only=True)
    if str(data['model_id']) != artifact['encoder_id'] or str(data['revision']) != artifact['revision']:
        raise ValueError('Feature encoder or revision does not match head')
    sets = [(name, task, load_items(args.items_dir / name, task)) for name, task in (('task1_items.json', '1'), ('task1_dim_items.json', '1'), ('task2_items.json', '2'))]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with record_run(args.cost_log, kind='infer', model_id=artifact['encoder_id'], items=sum(len(items) for _, _, items in sets), batch_size=1, precision='fp32', seed=artifact['seed'], notes='cached features; validation-tuned threshold'):
        features = torch.as_tensor(data['features'], device=device)
        index = {str(k): i for i, k in enumerate(data['image_ids'])}
        head = torch.nn.Linear(artifact['weight'].shape[1], 1).to(device)
        head.weight.data.copy_(artifact['weight'].to(device)); head.bias.data.copy_(artifact['bias'].to(device))
        mean, scale = artifact['mean'].to(device), artifact['scale'].to(device)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, task, items in sets:
            predictions, fixed = [], []
            for item in items:
                paths = [item['render_path']] if task == '1' else item['render_paths']
                d = features[index[item['drawing_path']]].expand(len(paths), -1)
                r = features[[index[path] for path in paths]]
                scores = predict(head, mean, scale, d, r).cpu().tolist()
                prediction = ('yes' if scores[0] >= artifact['threshold'] else 'no') if task == '1' else argmax_letter(scores)
                predictions.append({'id': item['id'], 'pred': prediction, 'scores': scores})
                if task == '1':
                    fixed.append({'id': item['id'], 'pred': 'yes' if scores[0] >= .5 else 'no'})
            stem = name.removesuffix('_items.json')
            (args.out_dir / f'{stem}_predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
            metrics = score(items, predictions, task)
            metrics['metadata'] = {'encoder_id': artifact['encoder_id'], 'revision': artifact['revision'], 'threshold': artifact['threshold'], 'threshold_policy': artifact['threshold_policy'], 'train_manifest_sha256': artifact['train_manifest_sha256'], 'val_manifest_sha256': artifact['val_manifest_sha256'], 'seed': artifact['seed']}
            if fixed:
                metrics['threshold_0.5'] = score(items, fixed, task)
            (args.out_dir / f'{stem}_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
            print(f'{stem}: accuracy_all={metrics["accuracy_all"]:.4f}')

if __name__ == '__main__':
    main()
