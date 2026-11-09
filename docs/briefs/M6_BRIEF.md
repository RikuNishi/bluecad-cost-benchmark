# M6 brief — main comparison, 3 seeds (seeds 0, 1, 2)

Read `AGENTS.md` first and follow it. CPU-only work: write code + unit
tests; GPU runs happen later on RunPod. Reuse the M2–M5 machinery
(`scripts/build_pair_manifest.py`, `scripts/extract_features.py`,
`scripts/train_head.py`, `scripts/eval_frozen.py`,
`scripts/train_embeddinggemma_partial.py`, `scripts/train_qwen_lora.py`,
`scripts/build_qwen_train_jsonl.py`, `scripts/run_qwen_eval.py`,
`src/bluecad_cost/runwrap.py`).

Goal (research plan, weeks 9–12): the main comparison over 3 seeds with
uncertainty — per-model performance–cost tables and figures, learning
curves (accuracy vs number of training pairs), and cumulative-cost
curves. `configs/common.yaml` already fixes `seeds: [0, 1, 2]`.

## Seed semantics (fixed — do not improvise alternatives)

A "seed cell" s ∈ {0, 1, 2} is defined by:

- Train pair manifest: `scripts/build_pair_manifest.py --seed s`
  (20,000 pairs, 50/50). The seed-1 and seed-2 manifests are new; the
  seed-0 manifest is the existing `manifests/train_pairs_20000.jsonl`.
- Validation manifest: ALWAYS the seed-0 one
  (`manifests/val_pairs_2000.jsonl`) for every cell and every model —
  the plan requires the same validation set across the comparison;
  thresholds are tuned on it exactly as in M3–M5.
- Common head seed: s (`train_head.py --seed s`; already supported).
- EmbeddingGemma 2 partial-update training seed: s (new `--seed` arg,
  Part 1). All other M5 hyperparameters unchanged, including the scope
  from `configs/partial_embeddinggemma2.yaml`
  (`last_k:0.model.vision_tower.encoder.layers:2`), encoder lr 1e-5 /
  head lr 1e-3 parameter groups, FP32 master weights + BF16 autocast,
  micro-batch 8 pairs × grad accum 4.
- Qwen3-VL QLoRA training seed: 13 + s (new `--seed` arg, Part 1).
  Rationale: M4 Run B used training seed 13 with the seed-0 manifest,
  so 13 + s makes M4 Run B exactly the s = 0 cell and no seed-0 GPU
  work is repeated. All other M4 Run B hyperparameters unchanged.
- Qwen3-VL zero-shot is deterministic (greedy decoding): the M4 Run A
  numbers are a single reference row, NOT a seed cell, and are never
  averaged with anything.

Seed-0 cells are already measured and are reused, never re-run:
frozen SigLIP 2 / EmbeddingGemma 2 = M3 (`results/siglip2/`,
`results/embeddinggemma2/`), partial = M5
(`results/embeddinggemma2_partial/`), Qwen fine-tune = M4 Run B
(`results/qwen3vl8b_ft/`). Only seeds 1 and 2 need new GPU work.

Result layout for new runs: `results/m6/seed{s}/<condition>/` with
condition ∈ {siglip2, embeddinggemma2, embeddinggemma2_partial,
qwen3vl8b_ft}, mirroring the per-condition directory contents of
M3/M4/M5. Every run appends to `results/costs.jsonl` via record_run
with its seed recorded (existing behaviour — verify, do not change).

## Learning curves (scope fixed)

- Curves are produced for the two FROZEN encoders only (SigLIP 2,
  EmbeddingGemma 2 frozen): for each seed s ∈ {0, 1, 2} and each
  training size ∈ {2,000, 5,000, 10,000, 20,000} pairs, train the
  common head on that many pairs (subsample of the seed's manifest)
  and evaluate all three tasks with `eval_frozen.py` on cached
  features. The 20,000 point is the main cell itself — do not train it
  twice; the curve assembly (Part 4) reuses the main-cell metrics.
- The trained conditions (partial update, Qwen fine-tune) appear in
  the curves figure only as their 20,000-pair points. One curve point
  for them equals one full training run; multi-point curves for them
  are out of scope for M6 and are noted as a limitation in the runbook.

## Part 1 — seed plumbing (only these changes to existing scripts)

- `scripts/train_embeddinggemma_partial.py`: add `--seed` (int,
  default 0, preserving M5 behaviour). Replace the hardcoded seed 0
  (torch/random seeding, record_run, run_config) with the argument.
- `scripts/train_qwen_lora.py`: add `--seed` (int, default 13,
  preserving M4 behaviour). Replace the SEED constant's uses (torch
  seeding, TrainingArguments, record_run, run config) with the
  argument.
- Nothing else in existing scripts changes. `build_pair_manifest.py`,
  `train_head.py`, `extract_features.py`, `eval_frozen.py`,
  `build_qwen_train_jsonl.py`, `run_qwen_eval.py` are used as-is.

## Part 2 — new manifest helpers

- `scripts/build_seed_manifests.py`: for s ∈ {1, 2}, run the
  build_pair_manifest logic with seed s into
  `manifests/seeds/seed{s}/`, then DELETE the per-seed validation file
  it produced, so no consumer can accidentally use a non-seed-0
  validation set. Print pair counts and the train-manifest SHA-256.
  (Import from build_pair_manifest rather than duplicating its logic
  where practical; a thin subprocess call is also acceptable.)
- `scripts/build_union_images.py`: generalisation of
  `collect_images.py` — takes one or more `--pair-manifest` paths plus
  the val manifest and the official items dir, writes the union of all
  referenced image paths in the same JSONL format
  (`{"image_id": path, "path": path}`, sorted). Used to build:
  - `manifests/images_union_seeds012.jsonl`: union over the train
    manifests of seeds 0, 1, 2 + val + all task items (frozen-feature
    extraction is seed-independent, so frozen features are extracted
    ONCE over this union and shared by all three seed cells).
  - per-seed partial manifests `manifests/images_seed{s}.jsonl`:
    union over seed s's train manifest + val + all task items.
- `scripts/subsample_pairs.py`: nested training-size subsets of a
  20,000-pair manifest. The manifest's rows come in consecutive
  (label 1, label 0) pairs sharing an anchor model; subsample at that
  pair-group level: deterministically shuffle the 10,000 groups with
  the manifest's seed, take the first 1,000 / 2,500 / 5,000 groups for
  the 2,000 / 5,000 / 10,000-pair subsets. Output files
  `train_pairs_{size}_seed{s}.jsonl` next to the input. Subsets are
  nested by construction and exactly 50/50.

## Part 3 — pod scripts (trap + self-terminate pattern of M4/M5)

Each script: `pip install -e '.[gpu]'`, log under `results/`,
EXIT trap writes its DONE marker with success/failure + log tail and
self-terminates via the RunPod API unless `M6_NO_AUTOSTOP=1`.

- `scripts/pod_m6_frozen.sh` (marker `results/M6FROZEN_DONE`, log
  `results/m6_frozen.log`):
  1. Build seed manifests (Part 2) and the union images manifest.
  2. Extract frozen features over the union for both encoders →
     `results/m6/frozen_features/{siglip2,embeddinggemma2}.npz`.
  3. For s ∈ {0, 1, 2}, for each encoder: train the full-size head
     (`--seed s`, seed s's train manifest, seed-0 val) → eval →
     `results/m6/seed{s}/<encoder>/`.
  4. Learning-curve heads: for s ∈ {0, 1, 2}, each encoder, sizes
     {2000, 5000, 10000}: subsample, train head, eval →
     `results/m6/curves/seed{s}/<encoder>/size{size}/`.
- `scripts/pod_m6_partial.sh` (env `M6_SEED` ∈ {1, 2}, required — fail
  if unset; marker `results/M6PARTIAL_DONE`, log
  `results/m6_partial_seed{s}.log`): mirror `pod_m5_run.sh` with the
  seed's train manifest, `--seed s`, the scope from the config, the
  per-seed images manifest, outputs →
  `results/m6/seed{s}/embeddinggemma2_partial/`. Consistency check
  first, as in M5.
- `scripts/pod_m6_qwen.sh` (env `M6_SEED` ∈ {1, 2}, required; marker
  `results/M6QWEN_DONE`, log `results/m6_qwen_seed{s}.log`):
  build the Qwen train JSONL from the seed's manifest
  (`build_qwen_train_jsonl.py --train-manifest ... --out-dir
  results/m6/seed{s}/qwen3vl8b_ft/`), train with `--seed 13+s` and the
  M4 Run B recipe (`--load-4bit`), evaluate via `run_qwen_eval.py
  --adapter ...` into the same directory.

## Part 4 — summary, tables, figures

- `scripts/summarize_m6.py` (stdlib + numpy; NO torch): inputs are the
  metrics JSONs of every cell — seed-0 cells read from the M3/M4/M5
  result dirs, seed-1/2 cells from `results/m6/`, curve cells from
  `results/m6/curves/` — plus `results/costs.jsonl`. Outputs under
  `results/m6/summary/`:
  - `main_table.csv` + `main_table.md`: per condition × task —
    accuracy_all and macro-F1 per seed, mean ± std over the 3 seeds,
    answer rate, and mean measured cost per run (training + extraction
    + inference GPU-hours from the cost log, attributed per condition
    and seed as recorded). Literature / reproduced / measured values
    stay in separate columns/files per AGENTS.md (this table is
    measured-only; the Qwen zero-shot reference row is marked
    `reference, single deterministic run`).
  - `learning_curves.csv`: condition, seed, train size, per-task
    accuracy_all (frozen conditions; trained conditions contribute
    their 20,000-pair rows only).
  - `cumulative_cost.csv`: from the cost log, cumulative GPU-hours and
    cumulative evaluated items over time per condition — the data
    behind the cumulative-cost figure.
- `scripts/plot_m6.py` (matplotlib; runs on the VM from the summary
  CSVs — the VM has matplotlib, the repo core must not depend on it):
  writes PNGs to `results/m6/summary/`: (a) performance–cost scatter
  (Task 1 accuracy_all vs mean total GPU-hours, error bars = seed
  std), (b) learning curves (Task 1 accuracy_all vs training pairs,
  mean ± std band over seeds, frozen conditions; trained conditions
  as single 20,000-pair markers), (c) cumulative cost vs cumulative
  items. Axis labels and legends must state the protocol (all-questions
  denominator, common 20,000 pairs, val-tuned threshold).

## Part 5 — runbook

`docs/M6_RUNBOOK.md`: the seed semantics above and their rationale,
the learning-curve scope decision + limitation, the run list with
measured M3/M4/M5 unit costs as the estimate basis (frozen union
extraction ≈ 1.1 GPU-h; partial per seed ≈ 1.33 train + ≈ 0.95
extract GPU-h; Qwen per seed ≈ 3.22 GPU-h; heads/evals CPU-negligible),
GPU choice guidance (RTX PRO 4000 Blackwell Secure if available, else
RTX 4090 Secure; record actuals), the exact pod commands in order
(frozen → partial s=1 → partial s=2 → qwen s=1 → qwen s=2), and the
retrieval + summary/plot steps. Estimated total ≈ $8–10; a ≈ $10
top-up is required before the GPU phase (balance ≈ $2.5 at brief
time) — the GPU phase does not start until the owner confirms the
top-up.

## Unit tests (CPU, no torch model download)

- Seed plumbing: partial script's `--seed` default is 0 and the value
  reaches the run config; Qwen script's `--seed` default is 13.
  (Parse args / factor a small helper if needed — keep it testable
  without importing torch-heavy paths.)
- Subsample: subsets are nested (2,000 ⊂ 5,000 ⊂ 10,000 ⊂ 20,000),
  exactly 50/50 at every size, deterministic for a fixed seed,
  pair-groups never split.
- Union builder: deduplicates across manifests, includes val + item
  images, output sorted and in the established JSONL format.
- Seed-manifest builder: the per-seed val file is removed; the
  canonical seed-0 val manifest is untouched.
- Summarize: aggregation over synthetic metrics JSONs (mean/std over
  seeds, per-seed values preserved, zero-shot row flagged as
  reference), and cumulative-cost accumulation from synthetic cost
  records.
- Existing test suite must still pass.

## Out of scope

- No GPU runs, no pod creation, no re-running any seed-0 cell, no
  changes to M3/M4/M5 results or code paths beyond Part 1, no M7
  error analysis.
