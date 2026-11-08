#!/bin/bash
set -ex
ROOT=/workspace/bluecad-cost-benchmark
finish() {
    status=$?
    trap - EXIT
    set +ex
    mkdir -p "$ROOT/results"
    if [ "$status" -eq 0 ]; then outcome=success; else outcome=failure; fi
    { echo "$outcome (exit $status)"; tail -n 40 "$ROOT/results/m4b.log" 2>/dev/null; } > "$ROOT/results/M4B_DONE"
    sync
    if [ "${M4_NO_AUTOSTOP:-0}" != 1 ]; then
        curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' -d '{"action":"terminate"}' || true
    fi
    exit "$status"
}
trap finish EXIT
cd "$ROOT"
mkdir -p results
exec >> "results/m4b.log" 2>&1
export HF_HOME=/workspace/hf-cache
pip install -q -e '.[gpu]'
python3 scripts/build_qwen_train_jsonl.py
python3 scripts/train_qwen_lora.py --load-4bit
python3 scripts/run_qwen_eval.py --adapter results/qwen3vl8b_ft/adapter --tasks task1,task1_dim,task2
for f in results/qwen3vl8b_ft/task*_metrics.json; do cat "$f"; done
