# M4 brief — Qwen3-VL 8B zero-shot inference + official-style fine-tune

Read `AGENTS.md` first and follow it. CPU-only work: write code + unit
tests; GPU runs happen later on RunPod. Reuse the M2/M3 machinery
(`src/bluecad_cost/runwrap.py` record_run, `scoring.py`,
`src/bluecad_cost/itemmap.py`, `manifests/`, `data/official_items/`).

The official implementation is cloned at
`~/workspace/research/blueprint2cad-official` (commit 3061b37) — read
`vlms/src/benchmark/prompts.py`, `vlms/src/benchmark/vlm.py`,
`vlms/src/finetune/qwen/config_qwen.py` and
`vlms/src/finetune/qwen/train_qwen.py` and mirror them as specified
below. Where the released code and this brief disagree, the brief wins
and the difference must be documented in code comments.

## Part 1 — zero-shot inference (`scripts/run_qwen_eval.py`)

- Model: `Qwen/Qwen3-VL-8B-Instruct`, BF16, `device_map="auto"`,
  loaded via `AutoModelForImageTextToText` + `AutoProcessor` (mirror
  `vlms.py`); record the resolved HF revision like M3 did
  (`HfApi().model_info(...).sha`).
- Inputs: the official item files via the existing item-path mapping
  (item `images` order preserved exactly). Prompt text = the item's
  `prompt_key` resolved through the official `PROMPTS` dict
  (task1_verification for Task 1 and Task 1-dim; the retrieval prompt
  with {n}/{labels} substituted for Task 2 — note items carry
  prompt_key 'task3_retrieval' while the dict key is
  'task2_retrieval'; map it and document).
- Generation: greedy (`do_sample=False`), `max_new_tokens=2000`
  (official VLM class default), batch=1 (milestone spec).
- Save raw generations to `results/qwen3vl8b_zs/<task>_predictions.jsonl`
  in the same prediction format the M2 scorer consumes (use
  scoring.parse_prediction on the raw text; keep the raw text in a
  `raw` field), then score with the M2 scorer into
  `<task>_metrics.json` (accuracy_all primary). Wrap the whole run in
  `record_run` (stage names: `qwen_zs_infer_<task>`), and write the
  standard run config JSON (model id, resolved revision, batch=1,
  precision bf16, seed).
- `--adapter <path>` option: same script evaluates a LoRA adapter
  (PeftModel on top of the base, like official vlm.py) into
  `results/qwen3vl8b_ft/...`. This is how the fine-tuned model is
  evaluated — through the SAME benchmark path as zero-shot.
- `--limit N` smoke option. `--tasks task1,task1_dim,task2` selector.

## Part 2 — training-data builder (`scripts/build_qwen_train_jsonl.py`)

Convert `manifests/train_pairs_20000.jsonl` (and
`val_pairs_2000.jsonl`) into the fine-tune JSONL format your training
script consumes: each line = {drawing_path, render_path, answer
("Yes"/"No")} using the HF data layout paths. The SAME 20,000 pairs
are used (comparability rule) — the authors' released data recipe
differs (NEG_PER_POS=1 with HARD_NEG_FRACTION=0.7); document that
deviation in the script docstring and in docs/M4_RUNBOOK.md.

## Part 3 — fine-tune (`scripts/train_qwen_lora.py`)

Mirror `config_qwen.py` + `train_qwen.py` exactly:
- Base `Qwen/Qwen3-VL-8B-Instruct`, `--load-4bit` NF4 + gradient
  checkpointing (the 24GB recipe), `enable_input_require_grads`.
- Example construction mirrors the official finetune pipeline:
  system message = config SYSTEM_PROMPT (verbatim, including its
  prosthetics wording — it is part of the official recipe; note it in
  the runbook), user message = [drawing image, render image,
  config QUESTION text] (drawing FIRST, render second — note this is
  the opposite order from benchmark Task 1), assistant answer =
  "Yes"/"No". Read train_qwen.py's dataset/collator to match its
  loss masking; if ambiguous, train on the answer tokens only and
  document the choice.
- LoRA: r=16, alpha=32, dropout=0.05, target_modules =
  [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv,proj,
  linear_fc1,linear_fc2] (covers LM + vision tower; log how many
  modules matched in each tower like the official script does).
- Optimizer/schedule: lr 1e-4, cosine, warmup_ratio 0.03,
  weight_decay 0.0, epochs 1, per_device_batch 1, grad_accum 16,
  logging_steps 10, save at end (adapter only), seed 13.
  Processor min_pixels=256*28*28, max_pixels=1024*28*28.
- Wrap training in record_run (stage `qwen_ft_train`) — note
  record_run measures wall time of the wrapped call; also log
  examples/sec from the trainer. Save adapter + trainer state summary
  to `results/qwen3vl8b_ft/adapter/`. Also write
  `results/qwen3vl8b_ft/train_config.json` (all hyperparameters +
  resolved base revision + pair-manifest hash).

## Part 4 — pod scripts with SELF-TERMINATION

`scripts/pod_m4a_run.sh` (zero-shot) and `scripts/pod_m4b_run.sh`
(fine-tune + eval), both `set -ex`, assuming: repo at
/workspace/bluecad-cost-benchmark, dataset already extracted under
data/bluecad-30k ON THE VOLUME, HF_HOME=/workspace/hf-cache,
HF_TOKEN in env.

AUTO-STOP (the point of this milestone's ops): each script installs a
trap on EXIT that, unless env M4_NO_AUTOSTOP=1, calls the RunPod API
to terminate its own pod:
  curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" -d '{"action":"terminate"}'
(RUNPOD_POD_ID is set by RunPod inside pods; RUNPOD_API_KEY is passed
as a pod env var at creation.) The trap must also run on failure, and
the script writes results + a `M4A_DONE` / `M4B_DONE` marker file
(with success/failure + tail of log) before exiting. Before the trap
fires, `sync` the filesystem.

Run A steps: pip install -q -e '.[gpu]' (idempotent) → run zero-shot
eval for task1, task1_dim, task2 → cat all metrics.
Run B steps: build train JSONL → train (output adapter) → run
run_qwen_eval.py --adapter on all three tasks → cat metrics.

## Part 5 — config + docs + tests

- `configs/qwen3vl8b.yaml`: fill in the real values (zero_shot +
  finetune sections mirroring the above, dataset manifest paths,
  output dirs). No null TODOs left for M4 fields.
- `docs/M4_RUNBOOK.md`: exact pod commands for Run A and Run B
  (including the env vars and the auto-stop explanation + how to
  disable it), expected durations (inference ~1–1.5h; training
  unknown — instruct reading steps/sec from the first logging output
  and estimating), smoke-test procedure with --limit.
- Unit tests (stdlib unittest, CPU): prompt resolution incl. the
  task3_retrieval→task2_retrieval mapping and Task 2 label
  substitution; train-JSONL builder on a tiny fake manifest (counts,
  50/50 preserved, paths exist-check skipped); LoRA config object
  matches config_qwen.py values; answer parsing reuse (a couple of
  raw-generation → label cases through scoring.parse_prediction).
- Update docs/MILESTONES.md M4 checkboxes only for the code-side
  items; leave the run items unchecked.

## Acceptance

- `python -m compileall src scripts` passes; full unittest suite
  passes (previous 20 + new).
- No changes to scoring.py semantics, manifests, or M3 scripts.
- Scripts must not hardcode any credential; RUNPOD_API_KEY and
  HF_TOKEN come from the environment only.
