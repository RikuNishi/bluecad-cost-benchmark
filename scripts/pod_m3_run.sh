#!/bin/bash
# Full M3 frozen-feature pipeline, run on the pod from the repo root.
# Progress markers (M3STEP ...) go to stdout; the caller redirects to a log.
set -ex
cd /workspace/bluecad-cost-benchmark
export HF_HOME=/workspace/hf-cache
ITEMS="$PWD/data/official_items"
echo "M3STEP install"
pip install -q -e '.[gpu]' || pip install -q sentence-transformers transformers accelerate pillow pyyaml
echo "M3STEP download"
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('alex-novineer/blueprint2cad', repo_type='dataset',
                  local_dir='data/bluecad-30k')
PY
echo "M3STEP extract"
python3 scripts/extract_shards.py all
python3 scripts/build_pair_manifest.py --seed 0
python3 scripts/collect_images.py --items-dir "$ITEMS"
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
