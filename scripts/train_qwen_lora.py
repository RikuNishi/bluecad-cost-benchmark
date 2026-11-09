#!/usr/bin/env python3
"""Train the official-style Qwen adapter on the shared pair manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bluecad_cost.runwrap import record_run

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
SYSTEM_PROMPT = ('You are an expert mechanical engineer performing quality inspection '
                 'of 3D-printed upper-limb prosthetic components.')
QUESTION = ('You are shown two images of a mechanical part. The first image is a 2D '
            'engineering drawing. The second image is a shaded render of a 3D CAD model. '
            'The two images may show the part from different viewpoints. '
            'Do they depict the same part? Answer with exactly one word: Yes or No.')
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    'qkv', 'proj', 'linear_fc1', 'linear_fc2'], bias='none', task_type='CAUSAL_LM')
MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1024 * 28 * 28


def messages(ex, with_answer):
    result = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
              {'role': 'user', 'content': [{'type': 'image', 'image': ex['drawing_path']},
               {'type': 'image', 'image': ex['render_path']}, {'type': 'text', 'text': QUESTION}]}]
    if with_answer:
        result.append({'role': 'assistant', 'content': [{'type': 'text', 'text': ex['answer']}]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('results/qwen3vl8b_ft/train.jsonl'))
    p.add_argument('--val', type=Path, default=Path('results/qwen3vl8b_ft/val.jsonl'))
    p.add_argument('--train-manifest', type=Path, default=Path('manifests/train_pairs_20000.jsonl'))
    p.add_argument('--val-manifest', type=Path, default=Path('manifests/val_pairs_2000.jsonl'))
    p.add_argument('--out', type=Path, default=Path('results/qwen3vl8b_ft/adapter'))
    p.add_argument('--cost-log', type=Path, default=Path('results/costs.jsonl'))
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--load-4bit', action='store_true')
    p.add_argument('--max-samples', type=int)
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error('--max-samples must be positive')
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import inspect
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision:
        raise RuntimeError('Could not resolve model revision')
    def read(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        random.Random(args.seed).shuffle(rows)
        for row in rows:
            if row['answer'] not in ('Yes', 'No'):
                raise ValueError('Invalid answer')
            for key in ('drawing_path', 'render_path'):
                with Image.open(row[key]) as image:
                    image.verify()
        return rows
    train, val = read(args.train), read(args.val)
    if args.max_samples:
        train = train[:args.max_samples]
        val = val[:args.max_samples]
    if not train or not val:
        raise ValueError('Training and validation sets must be nonempty')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = dict(model_id=MODEL_ID, revision=revision, train_manifest_sha256=sha(args.train_manifest),
                  val_manifest_sha256=sha(args.val_manifest), train_jsonl_sha256=sha(args.train), val_jsonl_sha256=sha(args.val),
                  lora=LORA, load_4bit=args.load_4bit, quantization='nf4_double' if args.load_4bit else 'none',
                  gradient_checkpointing=True, learning_rate=1e-4, scheduler='cosine', warmup_ratio=0.03,
                  weight_decay=0.0, epochs=1, batch_size=1, grad_accum=16, logging_steps=10,
                  save_strategy='end', precision='bf16', seed=args.seed, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                  train_items=len(train), val_items=len(val), max_samples=args.max_samples)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out.parent / 'train_config.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.manual_seed(args.seed)
    with record_run(args.cost_log, kind='qwen_ft_train', model_id=MODEL_ID, items=len(train), batch_size=1,
                    precision='bf16' + ('/nf4' if args.load_4bit else ''), seed=args.seed, notes=json.dumps(config)):
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        kw = {}
        if args.load_4bit:
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['visual', 'lm_head'])
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=revision, dtype=torch.bfloat16, device_map='auto', **kw)
        model.config.use_cache = False
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(**LORA))
        model.print_trainable_parameters()
        adapted = {n.rsplit('.lora_A', 1)[0] for n, _ in model.named_modules() if n.endswith('lora_A')}
        vision = sum('visual' in n for n in adapted)
        print(f'[lora] {len(adapted)} modules: {vision} vision, {len(adapted)-vision} language')
        def collate(batch):
            assert len(batch) == 1
            ex = batch[0]
            full = processor.apply_chat_template(messages(ex, True), tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
            prompt = processor.apply_chat_template(messages(ex, False), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
            n = prompt['input_ids'].shape[1]
            if not torch.equal(full['input_ids'][0, :n], prompt['input_ids'][0]):
                raise ValueError('Prompt is not a prefix of the full sequence')
            full['labels'] = full['input_ids'].clone()
            full['labels'][:, :n] = -100
            return full
        # The brief requires end-only saving, unlike the official periodic evaluation/checkpoints.
        warmup = {'warmup_ratio': 0.03} if 'warmup_ratio' in inspect.signature(TrainingArguments.__init__).parameters else {'warmup_steps': 0.03}
        targs = TrainingArguments(output_dir=str(args.out), num_train_epochs=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=16, learning_rate=1e-4, lr_scheduler_type='cosine', **warmup,
            weight_decay=0.0, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            logging_steps=10, save_strategy='no', eval_strategy='no', remove_unused_columns=False, report_to='none', seed=args.seed)
        trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val, data_collator=collate)
        result = trainer.train()
        model.save_pretrained(args.out)
        processor.save_pretrained(args.out)
        summary = {'metrics': result.metrics, 'log_history': trainer.state.log_history, 'global_step': trainer.state.global_step,
                   'lora_vision_modules': vision, 'lora_language_modules': len(adapted)-vision}
        (args.out / 'trainer_state_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"[train] examples/sec: {result.metrics.get('train_samples_per_second')}")

if __name__ == '__main__':
    main()
