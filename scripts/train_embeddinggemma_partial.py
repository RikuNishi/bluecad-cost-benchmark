#!/usr/bin/env python3
"""Partially update EmbeddingGemma 2; the joint head is diagnostic only."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bluecad_cost.partial import discover_blocks, pair_batch, select_scope
from bluecad_cost.head import best_threshold, pair_features
from bluecad_cost.runwrap import record_run
from extract_features import resolved_revision

MODEL = 'google/embeddinggemma-2'
OUT = Path('results/embeddinggemma2_partial')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def projections(model, blocks):
    # Only explicitly named image-side projection/pooling modules outside lists.
    return {name: module for name, module in model.named_modules()
            if name and any(x in name.lower() for x in ('vision', 'visual', 'image', 'multi_modal'))
            and any(x in name.lower().split('.')[-1] for x in ('project', 'pool'))
            and not any(name == key or name.startswith(key + '.') for key in blocks)}


def forward_images(model, paths, torch):
    from PIL import Image
    from sentence_transformers.util import batch_to_device
    images = [Image.open(path).convert('RGB') for path in paths]
    try:
        features = batch_to_device(model.tokenize(images), 'cuda')
        output = model(features)['sentence_embedding']
        return torch.nn.functional.normalize(output.float(), dim=-1)
    finally:
        for image in images:
            image.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('manifests/train_pairs_20000.jsonl'))
    p.add_argument('--val', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--images', type=Path, default=Path('manifests/images_all.jsonl'))
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    p.add_argument('--scope')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--probe', action='store_true')
    modes.add_argument('--check-consistency', type=int, metavar='N')
    args = p.parse_args()
    if not args.probe and args.check_consistency is None and not args.scope:
        p.error('--scope is required for training')
    if args.check_consistency is not None and args.check_consistency < 1:
        p.error('--check-consistency must be positive')
    import torch
    from sentence_transformers import SentenceTransformer
    from PIL import Image
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    torch.manual_seed(0)
    random.seed(0)
    model = SentenceTransformer(MODEL, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
    revision = resolved_revision(MODEL)
    if not revision:
        raise RuntimeError('Could not resolve model revision')
    blocks = discover_blocks(model)
    extra = projections(model, blocks)
    print('Top-level children:', [(n, sum(p.numel() for p in m.parameters())) for n, m in model.named_children()], flush=True)
    print('Image block lists:', [(n, len(m), sum(p.numel() for p in m.parameters())) for n, m in blocks.items()], flush=True)
    print('Image projections/pooling:', [(n, sum(p.numel() for p in m.parameters())) for n, m in extra.items()], flush=True)
    image_rows = rows(args.images)
    index = {r['image_id']: r['path'] for r in image_rows}
    if len(index) != len(image_rows):
        raise ValueError('Duplicate image IDs')
    if args.check_consistency is not None:
        selected = image_rows[:args.check_consistency]
        if len(selected) != args.check_consistency:
            raise ValueError('Not enough manifest images')
        model.eval()
        worst = 0.0
        with record_run(args.cost_log, kind='eg2_partial_consistency', model_id=MODEL, items=len(selected), batch_size=32, precision='bf16'):
            with torch.no_grad():
                for start in range(0, len(selected), 32):
                    paths = [r['path'] for r in selected[start:start+32]]
                    images = [Image.open(path).convert('RGB') for path in paths]
                    try:
                        encoded = model.encode(images, batch_size=32, convert_to_tensor=True)
                    finally:
                        for image in images: image.close()
                    actual = forward_images(model, paths, torch)
                    deviation = (1 - torch.nn.functional.cosine_similarity(encoded.float(), actual)).abs().max().item()
                    worst = max(worst, deviation)
        print(f'Max cosine deviation: {worst:.8g}', flush=True)
        if not math.isfinite(worst) or worst >= 1e-3:
            raise AssertionError('Training forward does not match extractor')
        return
    train = rows(args.train)
    val = rows(args.val) if not args.probe else []
    if len(train) < 96 and args.probe:
        raise ValueError('Probe requires at least 96 pairs')
    if not train or (not args.probe and not val):
        raise ValueError('Empty pair manifest')
    if args.probe:
        if not blocks:
            raise ValueError('No image-path block lists discovered')
        # Largest list by parameter count, deterministic ties by name.
        principal = max(blocks, key=lambda n: (sum(p.numel() for p in blocks[n].parameters()), n))
        if len(blocks[principal]) < 4:
            raise ValueError('Principal image block list has fewer than four blocks')
        scopes = [f'last_k:{principal}:{k}' for k in (1, 2, 4)] + ['all']
    else:
        scopes = [args.scope]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    kind = 'eg2_partial_probe' if args.probe else 'eg2_partial_train'
    with record_run(args.cost_log, kind=kind, model_id=MODEL, items=96 * len(scopes) if args.probe else len(train), seed=0, batch_size=32, precision='bf16', notes=f'revision={revision}; scaffold head diagnostic only'):
        for scope_index, scope in enumerate(scopes):
            if scope_index:
                del model, blocks, extra
                torch.cuda.empty_cache()
                model = SentenceTransformer(MODEL, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
                blocks = discover_blocks(model)
                extra = projections(model, blocks)
            torch.manual_seed(0)
            count = select_scope(model, scope, blocks, extra)
            print(f'Scope {scope}: trainable encoder parameters={count}', flush=True)
            head = torch.nn.Linear(768 * 4, 1, device='cuda')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=1e-5, weight_decay=0.0)
            order = list(range(len(train)))
            random.Random(0).shuffle(order)
            total = 3 if args.probe else math.ceil(len(order) / 32)
            warmup = math.ceil(total * .03)
            def lr_factor(step):
                if step < warmup: return step / max(1, warmup)
                return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(total):
                batch = [train[i] for i in order[step*32:(step+1)*32]]
                d, r, labels = pair_batch(batch, index)
                optimizer.zero_grad(set_to_none=True)
                emb = forward_images(model, d + r, torch)
                logits = head(pair_features(emb[:len(d)], emb[len(d):])).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(labels, device='cuda', dtype=torch.float32))
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (step + 1) % 20 == 0 or step + 1 == total:
                    print(f'{scope} step {step+1}/{total} loss={loss.item():.6f}', flush=True)
            torch.cuda.synchronize()
            results.append({'scope': scope, 'trainable_parameters': count, 'peak_vram_gb': torch.cuda.max_memory_allocated() / 1e9, 'seconds_per_step': (time.perf_counter()-started)/total})
            if not args.probe:
                model.eval(); head.eval(); scores = []; labels_all = []
                with torch.no_grad():
                    for start in range(0, len(val), 32):
                        d, r, labels = pair_batch(val[start:start+32], index)
                        emb = forward_images(model, d+r, torch)
                        scores.extend(torch.sigmoid(head(pair_features(emb[:len(d)], emb[len(d):]))).flatten().cpu().tolist())
                        labels_all.extend(labels)
                threshold = best_threshold(scores, labels_all)
                accuracy = lambda t: sum((s >= t) == bool(y) for s, y in zip(scores, labels_all)) / len(labels_all)
                diagnostic = {'threshold_0.5_accuracy': accuracy(.5), 'val_tuned_threshold': threshold, 'val_tuned_accuracy': accuracy(threshold), 'threshold_policy': 'validation labels maximize accuracy', 'diagnostic_only': True}
                print(f'Scaffold diagnostic: {diagnostic}', flush=True)
                encoder = args.out_dir / 'encoder'
                model.save(str(encoder))
                (encoder / 'm5_provenance.json').write_text(json.dumps({'model_id': MODEL, 'resolved_revision': revision}) + '\n')
                config = {'model_id': MODEL, 'resolved_revision': revision, 'scope': scope, 'trainable_parameters': count, 'trainable_projection_modules': list(extra), 'lr': 1e-5, 'weight_decay': 0.0, 'schedule': 'cosine', 'warmup_ratio': .03, 'batch_size': 32, 'epochs': 1, 'seed': 0, 'precision': 'bf16', 'scaffold_validation': diagnostic}
                (args.out_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
            del optimizer, scheduler, head
        if args.probe:
            (args.out_dir / 'probe.json').write_text(json.dumps({'principal_block_list': principal, 'results': results}, indent=2) + '\n')

if __name__ == '__main__':
    main()
