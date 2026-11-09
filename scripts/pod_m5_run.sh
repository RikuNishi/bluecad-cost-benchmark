#!/bin/bash
set -ex
ROOT=/workspace/bluecad-cost-benchmark
finish() {
    status=$?
    trap - EXIT
    set +ex
    mkdir -p "$ROOT/results"
    if [ "$status" -eq 0 ]; then outcome=success; else outcome=failure; fi
    { echo "$outcome (exit $status)"; tail -n 40 "$ROOT/results/m5.log" 2>/dev/null; } > "$ROOT/results/M5_DONE"
    sync
    if [ "${M5_NO_AUTOSTOP:-0}" != 1 ]; then
        curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' -d '{"action":"terminate"}' || true
    fi
    exit "$status"
}
trap finish EXIT
cd "$ROOT"
mkdir -p results
exec >> "results/m5.log" 2>&1
export HF_HOME=/workspace/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q -e '.[gpu]'
python3 scripts/train_embeddinggemma_partial.py --check-consistency 64
test -n "${M5_SCOPE:-}"
python3 scripts/train_embeddinggemma_partial.py --scope "$M5_SCOPE"
python3 scripts/extract_features.py --encoder embeddinggemma2 --model-path results/embeddinggemma2_partial/encoder --images manifests/images_all.jsonl --out results/embeddinggemma2_partial/features/features.npz
python3 scripts/train_head.py --features results/embeddinggemma2_partial/features/features.npz --out results/embeddinggemma2_partial/head/head.pt
python3 scripts/eval_frozen.py --features results/embeddinggemma2_partial/features/features.npz --head results/embeddinggemma2_partial/head/head.pt --items-dir data/official_items --out-dir results/embeddinggemma2_partial
for f in results/embeddinggemma2_partial/task*_metrics.json; do cat "$f"; done
