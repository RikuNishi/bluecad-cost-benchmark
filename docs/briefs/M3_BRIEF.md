# M3 implementation brief (for Codex)

Goal of M3: frozen-encoder reproduction — extract frozen image features with
SigLIP 2 and EmbeddingGemma 2, train ONE shared classifier head on the common
20,000-pair manifest, and evaluate on the official Task 1 (and Task 2 via
pair-score argmax). All runs must use src/bluecad_cost/runwrap.py record_run
for cost logging. Follow AGENTS.md rules exactly (all-questions denominator,
never call the trained head zero-shot, literature/reproduced/measured kept
separate).

## Fixed facts (verified, do not re-derive)

- View order for render files 000.png..005.png is defined by the official
  vfms code (retrieval_common.py): VIEW_NAMES = [front, back, left, right,
  top, bottom]. So 000=front, 001=back, 002=left, 003=right, 004=top,
  005=bottom. Cite this source in code comments; never guess from numbers.
- HF data layout under data/bluecad-30k/:
  - drawings/<model>/normal/<model>_<view>.png and drawings/<model>/dim/<model>_<view>_dim.png
  - renders/<model>/<NNN>.png (NNN per VIEW_NAMES order above)
  - splits/{train,val,test}.txt (model ids), shard_index.csv
- Official item files live in the sibling clone
  ~/workspace/research/blueprint2cad-official/vlms/src/benchmark/items/
  (task1_items.json, task1_dim_items.json, task2_items.json). Their JSON has
  {config, items}; each item has id, images (EC2 absolute paths), answer,
  group, meta. EC2 path mapping to HF layout:
  - .../renders/<group>/<model>/<NNN>.png -> renders/<model>/<NNN>.png
  - .../png_out/<group>/<model>/normal/<NNN>.png -> drawings/<model>/normal/<model>_<view>.png
    where <view> = VIEW_NAMES[int(NNN)]
  - Task 1 uses the front drawing; Task 2 uses the right drawing.
- Models (pin in configs, record revision at runtime from the loaded model):
  - SigLIP 2: google/siglip2-base-patch16-224 (the checkpoint named in the
    official extractor code). Use the vision tower image embedding
    (AutoModel / Siglip2Model get_image_features or vision_model pooled
    output per the official extractor), L2-normalized.
  - EmbeddingGemma 2: google/embeddinggemma-2 (multimodal, 768-dim).
    Use SentenceTransformer with the image input path if available, else
    the transformers EmbeddingGemma2Model image path. BF16 ONLY — FP16
    produces NaNs for this model (model card). Output = the 768-dim
    embedding, L2-normalized.
- Splits are disjoint; train.txt contains 4 phantom ids with no data
  (sample_888170/888176/888179/888180): ALWAYS intersect split ids with
  shard_index.csv before use (M1 finding).

## Deliverables

1. scripts/build_pair_manifest.py
   - Builds manifests/train_pairs_20000.jsonl and manifests/val_pairs_2000.jsonl.
   - Pair = (front drawing of model A, one render). Positive: render of A
     (view uniform over the 6 views). Negative: render of a different model
     B != A (view uniform). Balance exactly 50/50. Deterministic with
     --seed (default 0): positives cycle through distinct train models as
     far as possible; negatives sample B uniformly from train ids.
   - Each line: {pair_id, drawing_model, drawing_path, render_model,
     render_path, render_view, label (1 same / 0 different), split}.
   - Paths are repo-relative to data/bluecad-30k/. Print summary counts and
     verify every referenced file exists IF its shard has been extracted;
     report (do not fail on) files whose shard tar is present but not yet
     extracted — instead provide scripts/extract_shards.py (see 2).
2. scripts/extract_shards.py — extract selected shard tars under
   data/bluecad-30k/{drawings,renders}/ with tar --no-same-owner into the
   same directory (idempotent; skip models already extracted).
3. src/bluecad_cost/itemmap.py — parse official item JSONs, map image paths
   to HF-layout repo-relative paths, return per-item structured records for
   Task 1 (drawing path, render path, answer, group) and Task 2 (drawing
   path, 4 render paths in candidate order, answer letter). Unit-test the
   mapping against the real item files (paths must exist for test-split
   models present in shard-0000 extraction).
4. scripts/extract_features.py — CLI: --encoder siglip2|embeddinggemma2
   --images <jsonl of {image_id, path}> --out <npz>. Extracts frozen
   features on CUDA (BF16), batch size from config, saves npz with
   image_ids + float32 features matrix; wrapped in record_run with
   model/precision/batch metadata. Also scripts/collect_images.py that
   builds the deduplicated image list needed for: train pairs, val pairs,
   Task 1 items, Task 1-dim items, Task 2 items (union), writing
   manifests/images_all.jsonl.
5. src/bluecad_cost/head.py + scripts/train_head.py — the COMMON head:
   pair feature = concat[d, r, |d-r|, d*r] (d, r L2-normalized encoder
   outputs); standardize using train-pair statistics; logistic regression
   trained with torch (Adam, lr 1e-3, weight_decay 1e-4, 200 epochs full
   batch on GPU or CPU, seed from CLI). Save head weights + scaler +
   metadata (encoder id, train manifest hash, seed) to a .pt file.
   Threshold policy: primary threshold = value maximizing accuracy on the
   VAL pairs (documented as val-tuned, per plan); also record the 0.5
   threshold metrics. Never tune on test items.
6. scripts/eval_frozen.py — given encoder features npz + head .pt:
   - Task 1: score each item's (drawing, render) pair -> yes/no by
     threshold; write predictions JSONL in the M2 scorer's format; run the
     scorer (all-questions denominator); save metrics JSON.
   - Task 2: score the 4 candidates' pairs, predict argmax letter; same
     scoring path.
   - Wrap in record_run.
7. Pin model ids in configs/frozen_siglip2.yaml and
   configs/frozen_embeddinggemma2.yaml (replace the TODO), adding head
   hyperparameters and the threshold policy.
8. Unit tests (stdlib unittest, no GPU, no network, no model downloads):
   manifest builder on a tiny fake data tree; itemmap on the real item
   files; head training on synthetic separable features (accuracy > 0.9);
   eval argmax logic on synthetic scores. All existing tests must keep
   passing (python3 -m unittest discover -s tests).
9. Update docs/MILESTONES.md M3 line only when the code is complete and
   tests pass (leave the GPU run itself to the operator), and add
   docs/M3_RUNBOOK.md: exact commands to run on the pod in order
   (extract shards -> collect images -> extract features x2 -> train head
   x2 -> eval x2), with expected outputs and where results land.

Constraints: Python stdlib + torch/transformers/sentence-transformers only
on the GPU side; VM-side scripts (manifest, itemmap, tests) must run with
stdlib only (torch import guarded). Do not download model weights on the
VM. Keep diffs focused; one git commit at the end with a clear message.
