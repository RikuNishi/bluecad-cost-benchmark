# M5 brief — EmbeddingGemma 2 partial update + common classifier

Read `AGENTS.md` first and follow it. CPU-only work: write code + unit
tests; GPU runs happen later on RunPod. Reuse the M2/M3 machinery
(`src/bluecad_cost/runwrap.py` record_run, `src/bluecad_cost/head.py`,
`scripts/extract_features.py`, `scripts/train_head.py`,
`scripts/eval_frozen.py`, `manifests/`).

M3 baseline this milestone is compared against (frozen encoder, common
head trained on the same 20,000 pairs, seed 0, val-tuned threshold):
Task 1 accuracy_all 0.740, Task 1-dim 0.664, Task 2 0.853. Features are
768-d per-image embeddings from `google/embeddinggemma-2` (rev
f9972ac2), loaded via SentenceTransformer in BF16 (FP16 produces NaNs —
BF16 only, everywhere in this milestone).

## Design (fixed — do not improvise alternatives)

The ONLY difference from M3 is the encoder weights. Protocol:

1. Partially fine-tune the encoder on the SAME 20,000 common train
   pairs (manifests/train_pairs_20000.jsonl), 1 epoch, seed 0. During
   encoder training, a common-head-architecture classifier
   (concat[d, r, |d−r|, d⊙r] → logistic, as in `src/bluecad_cost/head.py`)
   is attached and trained jointly with BCE loss. This joint head is a
   training scaffold ONLY.
2. Save the updated encoder in SentenceTransformer format.
3. Re-extract features for the SAME extraction union (36,366 images,
   the manifest the M3 extractor consumed) with the updated encoder.
4. Train a FRESH common head with the unmodified M3 protocol
   (`scripts/train_head.py`) on the new features. THE reported M5
   numbers come from this head, never from the joint scaffold head
   (record the scaffold head's val accuracy separately as a diagnostic).
5. Evaluate with `scripts/eval_frozen.py` (Task 1, Task 1-dim, Task 2)
   into `results/embeddinggemma2_partial/`.

There is no official recipe for this ablation (the paper used the
encoder frozen); our hyperparameters are our own choice and must be
documented as such in docs/M5_RUNBOOK.md: AdamW, lr 1e-5, weight
decay 0.0, cosine schedule, warmup_ratio 0.03, batch 32 pairs,
BF16, 1 epoch, seed 0.

## Part 1 — training script (`scripts/train_embeddinggemma_partial.py`)

- Load the model exactly as `scripts/extract_features.py` does
  (SentenceTransformer, BF16). The training forward MUST produce
  embeddings identical in form to the extractor's (same preprocessing,
  pooling and normalisation). Implement the forward through the loaded
  model's own modules so gradients flow; do not re-implement pooling
  from scratch unless the model's modules force it (document if so).
- `--check-consistency N` mode: encode N manifest images through both
  the extractor path and the training forward (no training) and assert
  max cosine deviation < 1e-3; print the result. This runs on the pod
  before any training.
- Pair dataset: from the train/val manifests, resolving image paths
  exactly like the M3 pipeline (reuse itemmap / manifest helpers).
  Each example = (drawing image, render image, label). Batch 32 pairs.
- Trainable scope, selected by `--scope`:
  - The script must DISCOVER the model's repeated-block lists on the
    image path (e.g. vision tower blocks, backbone blocks) and print a
    module summary (top-level children + block-list lengths + parameter
    counts) to stdout at startup.
  - Scope grammar: `last_k:<which>:<k>` where <which> names a
    discovered block list (your discovery code defines the names and
    documents them in the runbook), and `all`. Freeze everything, then
    unfreeze the last k blocks of the named list. Any final
    projection/pooling module on the image path that is NOT part of a
    block list is also trainable (document which). Everything else
    stays frozen; assert the trainable parameter count in the startup
    log and in the run config JSON.
  - Fail loudly if a named list cannot be found.
- Wrap training in record_run (stage `eg2_partial_train`), save the
  updated encoder (SentenceTransformer `save`) to
  `results/embeddinggemma2_partial/encoder/`, write the standard run
  config JSON (model id, resolved revision, scope spec, trainable
  parameter count, lr, batch, epochs, seed, precision bf16), and log
  loss every 20 steps plus final val-pair accuracy of the scaffold
  head (threshold 0.5 AND val-tuned, labelled as diagnostic only).
- `--probe` mode: for each candidate scope in
  [`last_k` k=1, k=2, k=4 for the principal image-path block list,
  plus `all`], run 3 training steps at batch 32 pairs, record peak
  VRAM (torch.cuda.max_memory_allocated) and seconds/step, and write
  `results/embeddinggemma2_partial/probe.json`. No saving, no full
  epoch. Wrap in record_run (stage `eg2_partial_probe`).

## Part 2 — wire the updated encoder into the M3 eval chain

- `scripts/extract_features.py`: add `--model-path` (local directory
  or HF id) overriding the MODELS map id for either encoder, keeping
  default behaviour unchanged. Re-extraction for M5 uses
  `--encoder embeddinggemma2 --model-path results/embeddinggemma2_partial/encoder`.
- `train_head.py` and `eval_frozen.py` are used unmodified.

## Part 3 — config + pod scripts + runbook

- `configs/partial_embeddinggemma2.yaml`: keep `trainable_scope: null`
  for now (it is filled after the probe), but add the fixed training
  hyperparameters from the Design section and a `probe` section listing
  the candidate scopes.
- `scripts/pod_m5_probe.sh`: mirror `scripts/pod_m4b_run.sh` (trap on
  EXIT writes `results/M5PROBE_DONE` with success/failure + log tail,
  then self-terminates the pod via the RunPod API unless
  M5_NO_AUTOSTOP=1). Steps: pip install -e '.[gpu]', run
  `--check-consistency 64`, then `--probe`. Log to
  `results/m5probe.log`.
- `scripts/pod_m5_run.sh`: same trap pattern, marker
  `results/M5_DONE`, log `results/m5.log`. Steps: pip install,
  `--check-consistency 64`, partial training with the scope taken from
  an env var `M5_SCOPE` (fail if unset — the scope is a human decision
  after the probe), re-extract features to
  `results/embeddinggemma2_partial/features/`, train the fresh head
  (`train_head.py --features ... --out results/embeddinggemma2_partial/head/`),
  evaluate (`eval_frozen.py --features ... --head ... --out-dir
  results/embeddinggemma2_partial/`), then cat the three metrics files.
- `docs/M5_RUNBOOK.md`: design, protocol, hyperparameters (marked as
  our choice, no official recipe), the scope decision rule (largest
  candidate k whose probe peak VRAM ≤ 20 GB at batch 32 pairs; `all`
  is measured for information only and is NOT eligible — M5 is a
  partial update by construction), and the exact pod commands.

## Unit tests (CPU, no torch model download)

- Scope-spec parsing: valid/invalid specs, k bounds.
- Scope resolution against a synthetic nested module tree (fake block
  lists): correct modules unfrozen, everything else frozen, trainable
  count matches hand-computed value.
- Pair batch builder: shapes/labels from a tiny synthetic manifest.
- Existing test suite must still pass; the extractor change must not
  alter default behaviour (test the argument default).

## Out of scope

- No GPU runs, no pod creation, no changes to M3/M4 code paths or
  results, no changes to `scripts/train_head.py` / `eval_frozen.py`.
