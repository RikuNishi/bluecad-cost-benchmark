#!/bin/bash
# Re-run only the M3 GPU steps (features -> heads -> eval) after a fix.
# Assumes data is downloaded+extracted and manifests are built (pod_m3_run.sh
# steps install/download/extract already done). Stops on the first error.
set -ex
cd /workspace/bluecad-cost-benchmark
export HF_HOME=/workspace/hf-cache
ITEMS="$PWD/data/official_items"
ls manifests/train_pairs_20000.jsonl manifests/val_pairs_2000.jsonl manifests/images_all.jsonl
echo "M3STEP features siglip2"
python3 scripts/extract_features.py --encoder siglip2 --images manifests/images_all.jsonl --out results/siglip2/features.npz
echo "M3STEP features embeddinggemma2"
python3 scripts/extract_features.py --encoder embeddinggemma2 --images manifests/images_all.jsonl --out results/embeddinggemma2/features.npz
echo "M3STEP heads"
python3 scripts/train_head.py --features results/siglip2/features.npz --out results/siglip2/head.pt --seed 0
python3 scripts/train_head.py --features results/embeddinggemma2/features.npz --out results/embeddinggemma2/head.pt --seed 0
echo "M3STEP eval"
python3 scripts/eval_frozen.py --features results/siglip2/features.npz --head results/siglip2/head.pt --items-dir "$ITEMS" --out-dir results/siglip2
python3 scripts/eval_frozen.py --features results/embeddinggemma2/features.npz --head results/embeddinggemma2/head.pt --items-dir "$ITEMS" --out-dir results/embeddinggemma2
echo "M3STEP summary"
for f in results/siglip2/task*_metrics.json results/embeddinggemma2/task*_metrics.json; do echo "== $f"; cat "$f"; done
echo "M3_DONE"
