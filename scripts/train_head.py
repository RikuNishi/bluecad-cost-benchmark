#!/usr/bin/env python3
"""Train the shared logistic head and tune its threshold on validation pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.head import train, predict, best_threshold
from bluecad_cost.runwrap import record_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--train', type=Path, default=Path('manifests/train_pairs_20000.jsonl'))
    p.add_argument('--val', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    args = p.parse_args()
    import numpy as np
    import torch
    train_rows = [json.loads(x) for x in args.train.read_text().splitlines() if x.strip()]
    val_rows = [json.loads(x) for x in args.val.read_text().splitlines() if x.strip()]
    data = np.load(args.features, allow_pickle=False)
    model_id, revision = str(data['model_id']), str(data['revision'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with record_run(args.cost_log, kind='train', model_id=model_id, items=len(train_rows), seed=args.seed, batch_size=len(train_rows), precision='fp32', notes='full batch; threshold tuned on validation labels'):
        features = torch.as_tensor(data['features'], device=device)
        index = {str(k): i for i, k in enumerate(data['image_ids'])}
        def tensors(rows):
            return (features[[index[r['drawing_path']] for r in rows]], features[[index[r['render_path']] for r in rows]], torch.tensor([r['label'] for r in rows], device=device))
        d, r, y = tensors(train_rows)
        head, mean, scale = train(d, r, y, seed=args.seed)
        d, r, y = tensors(val_rows)
        scores = predict(head, mean, scale, d, r).cpu().tolist()
        labels = y.cpu().tolist()
        threshold = best_threshold(scores, labels)
        accuracy = lambda t: sum((s >= t) == bool(v) for s, v in zip(scores, labels)) / len(labels)
        artifact = {'weight': head.weight.detach().cpu(), 'bias': head.bias.detach().cpu(), 'mean': mean.cpu(), 'scale': scale.cpu(), 'encoder_id': model_id, 'revision': revision, 'train_manifest_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(), 'val_manifest_sha256': hashlib.sha256(args.val.read_bytes()).hexdigest(), 'seed': args.seed, 'threshold': threshold, 'threshold_policy': 'validation labels maximize accuracy', 'val_accuracy_tuned': accuracy(threshold), 'val_accuracy_0.5': accuracy(.5), 'epochs': 200, 'lr': 1e-3, 'weight_decay': 1e-4}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.out)
    print(f'Validation accuracy tuned={artifact["val_accuracy_tuned"]:.4f}, 0.5={artifact["val_accuracy_0.5"]:.4f}; {args.out}')

if __name__ == '__main__':
    main()
