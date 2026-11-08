#!/usr/bin/env python3
"""Extract normalized frozen CUDA image embeddings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.runwrap import record_run

MODELS = {'siglip2': 'google/siglip2-base-patch16-224', 'embeddinggemma2': 'google/embeddinggemma-2'}


def resolved_revision(model_id):
    """Resolve the current HF revision (commit sha) for a model id.

    Do NOT read config._commit_hash: it is absent on some config classes
    (e.g. SiglipConfig raises AttributeError) and on SentenceTransformer
    wrappers, which is how the first pod run failed for both encoders.
    """
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder', choices=MODELS, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, help='Override the encoder config batch size')
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    args = p.parse_args()
    if args.batch_size is None:
        import re
        config = Path(__file__).resolve().parents[1] / 'configs' / f'frozen_{args.encoder}.yaml'
        args.batch_size = int(re.search(r'^  batch_size:\s*(\d+)', config.read_text(), re.M).group(1))
    if args.batch_size < 1:
        p.error('batch size must be positive')
    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    rows = [json.loads(line) for line in args.images.read_text().splitlines() if line.strip()]
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    model_id = MODELS[args.encoder]
    with record_run(args.cost_log, kind='extract', model_id=model_id, items=len(rows), batch_size=args.batch_size, precision='bf16') as run:
        if args.encoder == 'siglip2':
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_id, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
            model.eval()
        revision = resolved_revision(model_id)
        run.notes = f'resolved_revision={revision}'
        if not revision:
            raise RuntimeError('Loaded model did not expose its resolved revision')
        features = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            images = [Image.open(r['path']).convert('RGB') for r in batch]
            with torch.inference_mode():
                if args.encoder == 'siglip2':
                    inputs = processor(images=images, return_tensors='pt').to('cuda')
                    inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
                    emb = model.get_image_features(**inputs)
                    if not isinstance(emb, torch.Tensor):
                        emb = emb.pooler_output
                else:
                    emb = model.encode(images, batch_size=args.batch_size, convert_to_tensor=True)
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                if not torch.isfinite(emb).all() or (args.encoder == 'embeddinggemma2' and emb.shape[-1] != 768):
                    raise ValueError('Invalid embeddings')
                features.append(emb.cpu().numpy())
            for image in images:
                image.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, image_ids=np.array([r['image_id'] for r in rows]), features=np.concatenate(features), model_id=model_id, revision=revision)
    print(f'{len(rows)} images, revision {revision}: {args.out}')

if __name__ == '__main__':
    main()
