#!/bin/bash
# M6 master runner: chains the five stage scripts on ONE pod.
# Stage scripts are invoked with M6_NO_AUTOSTOP=1 so they write their
# own DONE markers but never terminate the pod mid-chain. A failed
# stage does not stop the chain; every stage outcome is recorded in
# results/M6_ALL_DONE. This script self-terminates the pod at the end
# unless M6_ALL_NO_AUTOSTOP=1.
ROOT=/workspace/bluecad-cost-benchmark
LOG="results/m6_all.log"
declare -a OUTCOMES
finish() {
    status=$?
    trap - EXIT
    mkdir -p "$ROOT/results"
    { echo "master exit $status"; printf '%s\n' "${OUTCOMES[@]}"; } > "$ROOT/results/M6_ALL_DONE"
    sync
    if [ "${M6_ALL_NO_AUTOSTOP:-0}" != 1 ]; then
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
export M6_NO_AUTOSTOP=1

echo "=== stage frozen $(date -Is) ==="
bash scripts/pod_m6_frozen.sh
OUTCOMES+=("frozen:$?")

echo "=== stage partial seed1 $(date -Is) ==="
M6_SEED=1 bash scripts/pod_m6_partial.sh
OUTCOMES+=("partial_seed1:$?")

echo "=== stage partial seed2 $(date -Is) ==="
M6_SEED=2 bash scripts/pod_m6_partial.sh
OUTCOMES+=("partial_seed2:$?")

echo "=== stage qwen seed1 $(date -Is) ==="
M6_SEED=1 bash scripts/pod_m6_qwen.sh
OUTCOMES+=("qwen_seed1:$?")

echo "=== stage qwen seed2 $(date -Is) ==="
M6_SEED=2 bash scripts/pod_m6_qwen.sh
OUTCOMES+=("qwen_seed2:$?")

echo "=== all stages done $(date -Is) ==="
printf '%s\n' "${OUTCOMES[@]}"
