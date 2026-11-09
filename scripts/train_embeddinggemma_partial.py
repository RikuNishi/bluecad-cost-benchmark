#!/usr/bin/env python3
"""Partially update EmbeddingGemma 2; the joint head is diagnostic only.

Training uses FP32 master weights with BF16 autocast: pure-BF16 AdamW at
lr 1e-5 would round most updates away (BF16 ulp at typical weight scales
exceeds the per-step update), silently freezing the encoder. Final
features are extracted later in BF16 (as in M3) from the saved encoder.
Effective batch is 32 pairs via micro-batch x gradient accumulation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
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
MICRO_PAIRS = 8
ACCUM = 4


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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(features)['sentence_embedding']
        return torch.nn.functional.normalize(output.float(), dim=-1)
    finally:
        for image in images:
            image.close()


def load_model(torch):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device='cuda', model_kwargs={'torch_dtype': torch.float32})


def print_structure(model, blocks, extra):
    print('Top-level children:', [(n, sum(p.numel() for p in m.parameters())) for n, m in model.named_children()], flush=True)
    print('Image block lists:', [(n, len(m), sum(p.numel() for p in m.parameters())) for n, m in blocks.items()], flush=True)
    print('Image projections/pooling:', [(n, sum(p.numel() for p in m.parameters())) for n, m in extra.items()], flush=True)


def train_steps(model, head, optimizer, scheduler, pairs, index, torch, micro_pairs, accum, max_steps=None):
    """Run optimizer steps over (drawing, render, label) pairs; returns step count."""
    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)
    total = math.ceil(len(order) / (micro_pairs * accum))
    if max_steps is not None:
        total = min(total, max_steps)
    step = 0
    cursor = 0
    while step < total:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            batch = [pairs[i] for i in order[cursor:cursor + micro_pairs]]
            cursor += len(batch)
            if not batch:
                break
            d, r, labels = pair_batch(batch, index)
            emb = forward_images(model, d + r, torch)
            logits = head(pair_features(emb[:len(d)], emb[len(d):])).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(labels, device='cuda', dtype=torch.float32)) / accum
            loss.backward()
            loss_sum += loss.item()
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 20 == 0 or step == total:
            print(f'step {step}/{total} loss={loss_sum:.6f}', flush=True)
    return step


def make_optimizer(model, head, total_steps, torch):
    # Parameter groups: the encoder fine-tunes at 1e-5, but the fresh
    # scaffold head needs the common head's own learning rate (1e-3, as
    # in M3 head training). A single 1e-5 group left the head near its
    # init for hundreds of steps (loss pinned at ln 2) and starved the
    # encoder of gradient signal -- caught in the first full-run attempt.
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-5},
         {'params': list(head.parameters()), 'lr': 1e-3}],
        weight_decay=0.0)
    warmup = math.ceil(total_steps * .03)

    def lr_factor(step):
        if step < warmup:
            return step / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('manifests/train_pairs_20000.jsonl'))
    p.add_argument('--val', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--images', type=Path, default=Path('manifests/images_all.jsonl'))
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--scope')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--probe', action='store_true')
    modes.add_argument('--probe-one', metavar='SCOPE')
    modes.add_argument('--check-consistency', type=int, metavar='N')
    args = p.parse_args()
    if args.probe:
        return run_probe_driver(args)
    if not args.probe_one and args.check_consistency is None and not args.scope:
        p.error('--scope is required for training')
    if args.check_consistency is not None and args.check_consistency < 1:
        p.error('--check-consistency must be positive')
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = load_model(torch)
    revision = resolved_revision(MODEL)
    if not revision:
        raise RuntimeError('Could not resolve model revision')
    blocks = discover_blocks(model)
    extra = projections(model, blocks)
    print_structure(model, blocks, extra)
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
        with record_run(args.cost_log, kind='eg2_partial_consistency', model_id=MODEL, items=len(selected), batch_size=32, precision='bf16-autocast/fp32-master'):
            with torch.no_grad():
                for start in range(0, len(selected), 32):
                    paths = [r['path'] for r in selected[start:start + 32]]
                    images = [Image.open(path).convert('RGB') for path in paths]
                    try:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            encoded = model.encode(images, batch_size=32, convert_to_tensor=True)
                    finally:
                        for image in images:
                            image.close()
                    actual = forward_images(model, paths, torch)
                    deviation = (1 - torch.nn.functional.cosine_similarity(encoded.float(), actual)).abs().max().item()
                    worst = max(worst, deviation)
        print(f'Max cosine deviation: {worst:.8g}', flush=True)
        if not math.isfinite(worst) or worst >= 1e-3:
            raise AssertionError('Training forward does not match extractor')
        return
    train = rows(args.train)
    if not train:
        raise ValueError('Empty pair manifest')
    if args.probe_one:
        # One scope, measured in isolation; the parent probe driver parses
        # the final PROBE_RESULT line. Failures (e.g. CUDA OOM) exit nonzero
        # and are recorded by the driver instead of killing the whole probe.
        scope = args.probe_one
        count = select_scope(model, scope, blocks, extra)
        print(f'Scope {scope}: trainable encoder parameters={count}', flush=True)
        head = torch.nn.Linear(768 * 4, 1, device='cuda')
        total = 2
        optimizer, scheduler = make_optimizer(model, head, total, torch)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        train_steps(model, head, optimizer, scheduler, train, index, torch, MICRO_PAIRS, ACCUM, max_steps=total)
        torch.cuda.synchronize()
        result = {'scope': scope, 'trainable_parameters': count,
                  'peak_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
                  'seconds_per_step': (time.perf_counter() - started) / total}
        print('PROBE_RESULT ' + json.dumps(result), flush=True)
        return
    # Full training run.
    val = rows(args.val)
    if not val:
        raise ValueError('Empty validation manifest')
    scope = args.scope
    count = select_scope(model, scope, blocks, extra)
    print(f'Scope {scope}: trainable encoder parameters={count}', flush=True)
    head = torch.nn.Linear(768 * 4, 1, device='cuda')
    total = math.ceil(len(train) / (MICRO_PAIRS * ACCUM))
    optimizer, scheduler = make_optimizer(model, head, total, torch)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with record_run(args.cost_log, kind='eg2_partial_train', model_id=MODEL, items=len(train), seed=args.seed,
                    batch_size=MICRO_PAIRS * ACCUM, precision='bf16-autocast/fp32-master',
                    notes=f'revision={revision}; scope={scope}; micro_pairs={MICRO_PAIRS}; accum={ACCUM}; scaffold head diagnostic only'):
        model.train()
        train_steps(model, head, optimizer, scheduler, train, index, torch, MICRO_PAIRS, ACCUM)
        model.eval()
        head.eval()
        scores = []
        labels_all = []
        with torch.no_grad():
            for start in range(0, len(val), 32):
                d, r, labels = pair_batch(val[start:start + 32], index)
                emb = forward_images(model, d + r, torch)
                scores.extend(torch.sigmoid(head(pair_features(emb[:len(d)], emb[len(d):]))).flatten().cpu().tolist())
                labels_all.extend(labels)
        threshold = best_threshold(scores, labels_all)

        def accuracy(t):
            return sum((s >= t) == bool(y) for s, y in zip(scores, labels_all)) / len(labels_all)

        diagnostic = {'threshold_0.5_accuracy': accuracy(.5), 'val_tuned_threshold': threshold,
                      'val_tuned_accuracy': accuracy(threshold),
                      'threshold_policy': 'validation labels maximize accuracy', 'diagnostic_only': True}
        print(f'Scaffold diagnostic: {diagnostic}', flush=True)
        encoder = args.out_dir / 'encoder'
        model.save(str(encoder))
        (encoder / 'm5_provenance.json').write_text(json.dumps({'model_id': MODEL, 'resolved_revision': revision}) + '\n')
        config = {'model_id': MODEL, 'resolved_revision': revision, 'scope': scope,
                  'trainable_parameters': count, 'trainable_projection_modules': list(extra),
                  'encoder_lr': 1e-5, 'head_lr': 1e-3, 'weight_decay': 0.0, 'schedule': 'cosine', 'warmup_ratio': .03,
                  'batch_size_pairs': MICRO_PAIRS * ACCUM, 'micro_batch_pairs': MICRO_PAIRS,
                  'grad_accum': ACCUM, 'epochs': 1, 'seed': args.seed,
                  'precision': 'bf16-autocast/fp32-master', 'scaffold_validation': diagnostic}
        (args.out_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')


def run_probe_driver(args):
    """Probe each candidate scope in a subprocess; record failures, keep going."""
    import torch  # only for record_run's CUDA detection in this parent process
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = args.out_dir / 'probe.json'
    # Discover the principal block list once (cheap CPU-side metadata is not
    # available; reuse a short-lived subprocess printout instead).
    discovery = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--probe-one', '__discover__'],
        capture_output=True, text=True)
    principal = None
    for line in discovery.stdout.splitlines():
        if line.startswith('PRINCIPAL '):
            principal = line.split(' ', 1)[1].strip()
    if not principal:
        sys.stderr.write(discovery.stdout[-2000:] + discovery.stderr[-2000:])
        raise RuntimeError('Could not discover the principal image block list')
    scopes = [f'last_k:{principal}:{k}' for k in (1, 2, 4)] + ['all']
    results = []
    with record_run(args.cost_log, kind='eg2_partial_probe', model_id=MODEL,
                    items=sum(2 * MICRO_PAIRS * ACCUM for _ in scopes), batch_size=MICRO_PAIRS * ACCUM,
                    precision='bf16-autocast/fp32-master', notes=f'principal={principal}'):
        for scope in scopes:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--probe-one', scope],
                capture_output=True, text=True)
            entry = {'scope': scope, 'status': 'failed', 'error_tail': (proc.stderr or proc.stdout)[-500:]}
            for line in proc.stdout.splitlines():
                if line.startswith('PROBE_RESULT '):
                    entry = {'scope': scope, 'status': 'ok', **json.loads(line[len('PROBE_RESULT '):])}
            results.append(entry)
            probe_path.write_text(json.dumps(
                {'principal_block_list': principal, 'micro_batch_pairs': MICRO_PAIRS,
                 'grad_accum': ACCUM, 'results': results}, indent=2) + '\n')
            print(f'Probe {scope}: {entry["status"]}', flush=True)
    return 0


if __name__ == '__main__':
    # __discover__ is handled inside main via --probe-one with a sentinel:
    if '--probe-one' in sys.argv and '__discover__' in sys.argv:
        import torch as _torch
        _model = load_model(_torch)
        _blocks = discover_blocks(_model)
        if not _blocks:
            raise RuntimeError('No image-path block lists discovered')
        _principal = max(_blocks, key=lambda n: (sum(p.numel() for p in _blocks[n].parameters()), n))
        print_structure(_model, _blocks, projections(_model, _blocks))
        print('PRINCIPAL ' + _principal, flush=True)
        sys.exit(0)
    sys.exit(main())
