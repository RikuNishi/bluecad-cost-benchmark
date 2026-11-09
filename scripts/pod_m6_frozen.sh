#!/bin/bash
set -ex
ROOT=/workspace/bluecad-cost-benchmark
LOG="results/m6_frozen.log"
finish() {
    status=$?
    trap - EXIT
    set +ex
    mkdir -p "$ROOT/results"
    if [ "$status" -eq 0 ]; then outcome=success; else outcome=failure; fi
    { echo "$outcome (exit $status)"; tail -n 40 "$ROOT/$LOG" 2>/dev/null; } > "$ROOT/results/M6FROZEN_DONE"
    sync
    if [ "${M6_NO_AUTOSTOP:-0}" != 1 ]; then
        curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' -d '{"action":"terminate"}' || true
    fi
    exit "$status"
}
trap finish EXIT
cd "$ROOT"
mkdir -p results
exec >> "$LOG" 2>&1
export HF_HOME=/workspace/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q -e '.[gpu]'
# Each command writes a scoped log. Copy even failed runs to the global ledger;
# scope files preserve unambiguous attribution, including shared extraction.
run_logged() {
    local scope=$1; shift
    local log="results/m6/costs/${scope}.jsonl" tmp
    mkdir -p "$(dirname "$log")"
    tmp=$(mktemp)
    local status=0
    "$@" --cost-log "$tmp" || status=$?
    if [ "$status" -eq 0 ]; then cat "$tmp" >> "$log"; fi
    cat "$tmp" >> results/costs.jsonl
    rm -f "$tmp"
    return "$status"
}
python3 scripts/build_seed_manifests.py
python3 scripts/build_union_images.py --pair-manifest manifests/train_pairs_20000.jsonl --pair-manifest manifests/seeds/seed1/train_pairs_20000.jsonl --pair-manifest manifests/seeds/seed2/train_pairs_20000.jsonl --out manifests/images_union_seeds012.jsonl
for encoder in siglip2 embeddinggemma2; do
    features="results/m6/frozen_features/$encoder.npz"
    run_logged "shared/$encoder" python3 scripts/extract_features.py --encoder "$encoder" --images manifests/images_union_seeds012.jsonl --out "$features"
    for seed in 0 1 2; do
        if [ "$seed" = 0 ]; then train=manifests/train_pairs_20000.jsonl; else train="manifests/seeds/seed$seed/train_pairs_20000.jsonl"; fi
        if [ "$seed" != 0 ]; then
            out="results/m6/seed$seed/$encoder"
            run_logged "seed$seed/$encoder" python3 scripts/train_head.py --features "$features" --train "$train" --val manifests/val_pairs_2000.jsonl --out "$out/head.pt" --seed "$seed"
            run_logged "seed$seed/$encoder" python3 scripts/eval_frozen.py --features "$features" --head "$out/head.pt" --items-dir data/official_items --out-dir "$out"
        fi
        python3 scripts/subsample_pairs.py --train-manifest "$train" --seed "$seed"
        for size in 2000 5000 10000; do
            subset="$(dirname "$train")/train_pairs_${size}_seed${seed}.jsonl"
            out="results/m6/curves/seed$seed/$encoder/size$size"
            run_logged "curves/seed$seed/$encoder/size$size" python3 scripts/train_head.py --features "$features" --train "$subset" --val manifests/val_pairs_2000.jsonl --out "$out/head.pt" --seed "$seed"
            run_logged "curves/seed$seed/$encoder/size$size" python3 scripts/eval_frozen.py --features "$features" --head "$out/head.pt" --items-dir data/official_items --out-dir "$out"
        done
    done
done
