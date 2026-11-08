#!/usr/bin/env bash
# Run on Muse's VM. Syncs the repo (code + manifests + results) to a pod.
# Usage: scripts/sync_to_pod.sh root@<host> <port> [--with-data]
#   --with-data also syncs data/bluecad-30k (~24GB, HF-gated content that was
#   downloaded on the VM through the connector; ingress to RunPod is free).
set -euo pipefail

HOST="${1:?usage: sync_to_pod.sh root@<host> <port> [--with-data]}"
PORT="${2:?usage: sync_to_pod.sh root@<host> <port> [--with-data]}"
DEST="/workspace/bluecad-cost-benchmark"
cd "$(dirname "$0")/.."

rsync -az -e "ssh -p $PORT" \
  --exclude '.git' --exclude 'data' --exclude 'results' --exclude '__pycache__' \
  ./ "$HOST:$DEST/"

if [[ "${3:-}" == "--with-data" ]]; then
  rsync -az --progress -e "ssh -p $PORT" \
    data/bluecad-30k/ "$HOST:$DEST/data/bluecad-30k/"
fi
echo "Synced to $HOST:$DEST"
