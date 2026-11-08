# M4 Qwen runs

Run on a BF16-capable 24 GB GPU pod with the repository at `/workspace/bluecad-cost-benchmark`, the extracted dataset on the volume at `data/bluecad-30k`, the official items at `data/official_items`, and both shared pair manifests in `manifests/`. All referenced images must be present. Set `HF_TOKEN` and `RUNPOD_API_KEY` as pod environment variables at creation; RunPod supplies `RUNPOD_POD_ID`. Do not put credentials into commands or files. `HF_HOME` is set to `/workspace/hf-cache` by the scripts.

Run A, zero-shot:

```bash
cd /workspace/bluecad-cost-benchmark
bash scripts/pod_m4a_run.sh
```

Run B, training followed by evaluation:

```bash
cd /workspace/bluecad-cost-benchmark
bash scripts/pod_m4b_run.sh
```

Each script logs to `results/m4a.log` or `results/m4b.log`, writes `results/M4A_DONE` or `results/M4B_DONE` with status and the log tail on exit, syncs the filesystem, then requests termination of its own pod via the RunPod API, including on failure. Check the marker and RunPod status; API or network failure can prevent termination. Set `M4_NO_AUTOSTOP=1` in the pod environment to disable termination during debugging. Results must be on persistent storage to survive termination.

Inference is expected to take roughly 1–1.5 hours; this is a planning estimate, not a measured result. Training duration is unknown: read the first training logging output for steps per second and estimate remaining time using the total steps. Training uses 4-bit NF4 with BF16 compute and gradient checkpointing. The common pair manifests are preserved exactly; the authors' separate recipe uses one negative per positive and 70% hard negatives. The official training system prompt mentions upper-limb prosthetic components verbatim. Training puts the drawing first, whereas the benchmark Task 1 items put the render first. Prompt tokens are masked for loss, as in the official collator. Unlike the official periodic checkpoints and validation, this recipe saves the adapter at the end.

For a smoke test, set `M4_NO_AUTOSTOP=1` and run directly after installing `pip install -q -e '.[gpu]'`:

```bash
python3 scripts/run_qwen_eval.py --tasks task1,task1_dim,task2 --limit 2 --out-dir results/qwen_smoke_zs
python3 scripts/build_qwen_train_jsonl.py
python3 scripts/train_qwen_lora.py --load-4bit --max-samples 32 --out results/qwen_smoke_ft/adapter
python3 scripts/run_qwen_eval.py --adapter results/qwen_smoke_ft/adapter --tasks task1,task1_dim,task2 --limit 2 --out-dir results/qwen_smoke_ft
```

Smoke results are partial and must not be reported as full benchmark results. Cost records are appended to `results/costs.jsonl`; task metrics include evaluated IDs, accuracy over all evaluated questions, answered-only accuracy and answer rate. Keep the run configs, predictions, adapter, training summary and costs together.
