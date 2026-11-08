# M4 results — Qwen3-VL 8B zero-shot (Run A)

Run date: 2026-11-08. All artifacts on the RunPod network volume under
`results/qwen3vl8b_zs/` (predictions, metrics, run configs, cost records).

## Attempt 1 — failed (jinja2)

- Pod: `bluecad-bootstrap` (u7dj1bw0o54e5y), Secure RTX 4090.
- The eval script crashed on the first item: `ImportError:
  apply_chat_template requires jinja2>=3.1.6` — the RunPod base image
  ships jinja2 3.1.4 and the `[gpu]` extra did not pin it. The model had
  already downloaded and its weights loaded.
- Cost record: `qwen_zs_infer_task1`, 24.2s, 0.0067 GPU-h, peak VRAM
  17.5GB (partial run).
- The pod self-terminated on script exit via the `pod_m4a_run.sh` trap
  (RunPod injects `RUNPOD_API_KEY` + `RUNPOD_POD_ID` into pod env).
  This confirmed the auto-stop mechanism works on the failure path.
- Fix: `jinja2>=3.1.6` pinned in `pyproject.toml` `[gpu]` extra
  (local commit 7b4ca0a, pushed to GitHub e3fffbdf).

## Attempt 2 — success

- Pod: `bluecad-reader` (my692n22t2twre), Secure RTX 4090, code re-synced
  from GitHub, relaunched 2026-11-08 ~13:05 JST. `M4A_DONE`:
  `success (exit 0)`. The pod self-terminated when the script finished
  (~13:26 JST, ~21 min after launch).
- Model: `Qwen/Qwen3-VL-8B-Instruct`, resolved revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. BF16, batch 1, greedy
  (`do_sample=False`), `max_new_tokens=2000`, seed 13, no adapter.
  Official prompts from `PROMPTS` (task1_verification, retrieval
  prompt with labels A–D; item files carry `prompt_key`
  'task3_retrieval', mapped to the `task2_retrieval` text).
- Predictions: 1,000 per task (3,000 total), all saved under
  `results/qwen3vl8b_zs/`.

| Task | accuracy_all | accuracy_answered | macro-F1 (answered) | answer rate |
| ---- | -----------: | ----------------: | ------------------: | ----------: |
| Task 1 | 0.653 | 0.653 | 0.647 | 1.00 (1000/1000) |
| Task 1-dim | 0.632 | 0.632 | 0.630 | 1.00 (1000/1000) |
| Task 2 | 0.334 | 0.348 | 0.297 | 0.96 (960/1000; 40 unparseable) |

- Accuracy denominator is ALL questions (plan rule). Task 2's 40
  unparseable generations (no single-letter answer) count as wrong in
  `accuracy_all` and are excluded from `accuracy_answered`.

## Costs (measured, RTX 4090 @ $0.89/hr Secure)

| Stage | wall (s) | GPU-h | peak VRAM (GB) |
| ----- | -------: | ----: | -------------: |
| Task 1 inference | 175.1 | 0.0486 | 17.7 |
| Task 1-dim inference | 145.3 | 0.0404 | 17.7 |
| Task 2 inference | 313.2 | 0.0870 | 18.0 |
| **Total (v2)** | **633.6** | **0.176** | — |
| Failed v1 attempt | 24.2 | 0.0067 | 17.5 |

- Inference GPU cost v2 ≈ $0.16. Pod wall time (incl. pip install and
  model load) ≈ 21 min ≈ $0.31 at the Secure rate. Peak VRAM 17.5–
  18.0GB — fits a 24GB card with headroom.

## How to read these numbers

- These are ZERO-SHOT results (no trained head, no fine-tune). Do NOT
  compare them directly with the M3 frozen-encoder numbers — M3 used a
  trained logistic head on the same 20,000 pairs.
- The paper's VLM zero-shot Task 1 range is 0.52–0.67 across models;
  Qwen3-VL 8B at 0.653 sits at the high end of that range. (Literature
  value, different protocol — our numbers are measured on the official
  items with our scoring, all-questions denominator.)
- Task 2 (0.334 vs 0.25 chance) is the weak spot — retrieval with a
  single-letter answer from 4 renders, plus 40 generations that did not
  parse as a letter.
- Qwen zero-shot Task 1 (0.653) is BELOW both frozen encoders' trained-
  head Task 1 (SigLIP 2 0.725, EmbeddingGemma 2 0.740) — expected: the
  frozen models got 20,000 supervised pairs; the zero-shot model got
  none. The fair comparison is the fine-tuned Qwen (Run B).

## Remaining for M4

- Run B: QLoRA fine-tune on the common 20,000 pairs (rank 16, lr 1e-4,
  1 epoch, --load-4bit --grad-ckpt), then the same three-task eval
  through `run_qwen_eval.py --adapter`. Duration and cost unknown until
  the first training logs — estimate from steps/sec, check early.
