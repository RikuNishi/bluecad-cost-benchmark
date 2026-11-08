#!/usr/bin/env python3
"""Evaluate Qwen on the unchanged official benchmark items."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.itemmap import load_items, map_path
from bluecad_cost.runwrap import record_run
from bluecad_cost.scoring import parse_prediction, score

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
PROMPTS = {
    'task1_verification': ('You are shown a rendered view of a CAD model and a 2D engineering drawing. '
        'The two images may show the part from different viewpoints. Do they depict the same part? '
        'Answer with exactly one word: Yes or No.'),
    'task2_retrieval': ('The first image is a 2D engineering drawing of a CAD model. '
        'The following {n} images are rendered CAD models, labeled {labels}. '
        'Which rendered model does the drawing represent? Answer with exactly one letter.'),
}


def resolve_prompt(item):
    # Distributed items use task3_retrieval, while the released PROMPTS key is task2_retrieval.
    key = item['prompt_key']
    if key == 'task3_retrieval':
        key = 'task2_retrieval'
    return PROMPTS[key].format(**item.get('prompt_vars', {}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--tasks', default='task1,task1_dim,task2')
    p.add_argument('--limit', type=int)
    p.add_argument('--items-dir', type=Path, default=Path('data/official_items'))
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    args = p.parse_args()
    tasks = args.tasks.split(',')
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in ('task1', 'task1_dim', 'task2') for t in tasks):
        p.error('invalid or duplicate tasks')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    out = args.out_dir or Path('results/qwen3vl8b_ft' if args.adapter else 'results/qwen3vl8b_zs')
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision:
        raise RuntimeError('Could not resolve model revision')
    torch.manual_seed(13)
    # Include model load in the first task's measured run.
    model = processor = None
    for task in tasks:
        task_id = '2' if task == 'task2' else '1'
        items = load_items(args.items_dir / f'{task}_items.json', task_id)
        if args.limit is not None:
            items = items[:args.limit]
        config = dict(model_id=MODEL_ID, revision=revision, batch_size=1, precision='bf16', seed=13,
                      adapter=str(args.adapter) if args.adapter else None, max_new_tokens=2000, do_sample=False, limit=args.limit)
        (out / f'{task}_run_config.json').write_text(json.dumps(config, indent=2) + '\n')
        with record_run(args.cost_log, kind=f'qwen_zs_infer_{task}' if not args.adapter else f'qwen_ft_infer_{task}', model_id=MODEL_ID,
                        items=len(items), batch_size=1, precision='bf16', seed=13, notes=json.dumps(config)):
            if model is None:
                source = args.adapter if args.adapter and (args.adapter / 'preprocessor_config.json').exists() else MODEL_ID
                processor = AutoProcessor.from_pretrained(source, **({} if source == args.adapter else {'revision': revision}))
                model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=revision, torch_dtype=torch.bfloat16, device_map='auto')
                if args.adapter:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
                model.eval()
            predictions = []
            with (out / f'{task}_predictions.jsonl').open('w') as f:
                for item in items:
                    content = [{'type': 'image'} for _ in item['images']] + [{'type': 'text', 'text': resolve_prompt(item)}]
                    text = processor.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False)
                    images = []
                    try:
                        for path in item['images']:
                            with Image.open(map_path(path)) as image:
                                images.append(image.convert('RGB'))
                        inputs = processor(text=[text], images=images, return_tensors='pt').to(model.device)
                        with torch.no_grad():
                            generated = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
                        raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
                        row = {'id': item['id'], 'pred': parse_prediction(raw, task_id), 'raw': raw}
                    finally:
                        for image in images:
                            image.close()
                    predictions.append(row)
                    f.write(json.dumps(row) + '\n')
                    f.flush()
            metrics = score(items, predictions, task_id)
            metrics['metadata'] = config
            (out / f'{task}_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
            print(f'{task}: accuracy_all={metrics["accuracy_all"]:.4f}')

if __name__ == '__main__':
    main()
