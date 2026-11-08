#!/usr/bin/env bash
# Run ON the RunPod pod, once per pod (the network volume persists between pods).
# Idempotent. Verifies the GPU stack and installs the Python extras the
# official PyTorch template does not ship.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "$HF_HOME" /workspace/bluecad-cost-benchmark

echo "== GPU =="
nvidia-smi
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0),
      "| bf16", torch.cuda.is_bf16_supported())
PY

echo "== Python extras =="
pip install --quiet --upgrade \
  transformers accelerate peft safetensors sentencepiece pillow pyyaml
python3 -c "import transformers, peft, accelerate; print('extras ok')"

echo "Bootstrap done. Sync the repo and data from the VM with scripts/sync_to_pod.sh."
