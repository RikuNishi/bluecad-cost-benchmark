# scripts/ — entry points

`check_env.py`, `audit_data.py`, and `score.py` are implemented. The other
entry points below are specs to implement per milestone (docs/MILESTONES.md).
GPU experiment entry points will take a config from `configs/`, run unattended,
resume from checkpoints, and append a cost record (see
`src/bluecad_cost/costlog.py`) plus per-question predictions under `results/`.

| Script | Milestone | Runs on | Job |
|---|---|---|---|
| `audit_data.py` (implemented) | M1 | CPU | Verify official JSON ↔ image mapping, missing files, ID splits; audit 100–500 parts |
| `build_pair_manifest.py` | M4 | CPU | Regenerate the common 20,000-pair manifest with the official procedure, fixed seed; used by ALL compared models |
| `extract_features.py` | M3 | GPU | Frozen-encoder features for SigLIP 2 / EmbeddingGemma 2 (+ optional DINOv3), with cache + build-time logging |
| `train_head.py` | M3/M5 | CPU/GPU | Train the common classifier on cached features; partial fine-tuning variant for EmbeddingGemma 2 |
| `run_qwen_eval.py` | M4 | GPU 24GB | Qwen3-VL 8B zero-shot / fine-tuned inference on Task 1 & 2, batch=1, per-question predictions + timing |
| `train_qwen.py` | M4 | GPU 24GB | Official-style fine-tune on the common manifest; checkpointed, resumable |
| `score.py` (implemented) | M2 | CPU | Score predictions: accuracy over all questions, answered-only accuracy, answer rate, macro-F1; saves evaluated IDs |
| `summarize_costs.py` | M6 | CPU | Build the score/cost tables and performance–cost figures from `results/` |
