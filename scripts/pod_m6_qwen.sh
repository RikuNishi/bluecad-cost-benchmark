#!/bin/bash
set -ex
ROOT=/workspace/bluecad-cost-benchmark
LOG="results/m6_qwen_seed${M6_SEED:-unset}.log"
finish() {
    status=$?
    trap - EXIT
    set +ex
    mkdir -p "$ROOT/results"
    if [ "$status" -eq 0 ]; then outcome=success; else outcome=failure; fi
    { echo "$outcome (exit $status)"; tail -n 40 "$ROOT/$LOG" 2>/dev/null; } > "$ROOT/results/M6QWEN_DONE"
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
case "${M6_SEED:-}" in 1|2) ;; *) echo 'M6_SEED must be 1 or 2' >&2; exit 1 ;; esac
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
seed=$M6_SEED
train="manifests/seeds/seed$seed/train_pairs_20000.jsonl"
out="results/m6/seed$seed/qwen3vl8b_ft"
python3 scripts/build_qwen_train_jsonl.py --train-manifest "$train" --val-manifest manifests/val_pairs_2000.jsonl --out-dir "$out"
run_logged "seed$seed/qwen3vl8b_ft" python3 scripts/train_qwen_lora.py --train "$out/train.jsonl" --val "$out/val.jsonl" --train-manifest "$train" --val-manifest manifests/val_pairs_2000.jsonl --out "$out/adapter" --seed "$((13 + seed))" --load-4bit
run_logged "seed$seed/qwen3vl8b_ft" python3 scripts/run_qwen_eval.py --adapter "$out/adapter" --out-dir "$out" --tasks task1,task1_dim,task2
