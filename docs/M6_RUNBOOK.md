# M6 runbook

Do not start the GPU phase until the owner confirms the approximately $10
account top-up. The brief-time balance was approximately $2.50 and the
estimated total is $8–10, not a measured bill. Prefer RTX PRO 4000 Blackwell
Secure; otherwise use RTX 4090 Secure. Record the actual GPU and bill. The
estimate uses M3–M5 measurements: frozen union extraction about 1.1 GPU-h,
partial per seed about 1.33 GPU-h training plus 0.95 extraction, Qwen per
seed about 3.22 GPU-h; heads and evaluations are CPU-negligible. These are
planning estimates, not M6 measurements.

Each seed cell uses its own balanced 20,000 training pairs, the same seed-0
validation manifest, and head seed 0, 1, or 2. Partial training uses that
same seed and the M5 configured scope and hyperparameters. Qwen training uses
13 + seed: seed 0 is exactly the existing M4 Run B. Qwen zero-shot is one
deterministic reference, not a seed average. Reuse M3/M4/M5 seed-0 results;
never rerun those main cells. Frozen features for M6 are extracted once per
encoder on the union of all seeds, validation and official task images. The
full shared extraction cost is charged to each new frozen cell in the table;
it is counted only once in the cumulative ledger. The seed-0 table uses its
historical extraction. Validation labels tune classifier thresholds.

Frozen learning curves use nested, balanced 2,000, 5,000 and 10,000 subsets
and reuse each main 20,000 result. Partial and Qwen have only 20,000-point
markers: smaller trained-model curves are outside M6, a limitation.

Prepare the repository, the official items, extracted image files, canonical
seed-0 manifests and historical results/cost log on the pod. From
`/workspace/bluecad-cost-benchmark`, run the following in order on suitable
pods (each terminates itself on exit unless `M6_NO_AUTOSTOP=1`):

```sh
bash scripts/pod_m6_frozen.sh
M6_SEED=1 bash scripts/pod_m6_partial.sh
M6_SEED=2 bash scripts/pod_m6_partial.sh
M6_SEED=1 bash scripts/pod_m6_qwen.sh
M6_SEED=2 bash scripts/pod_m6_qwen.sh
```

Preserve and transfer the generated manifests, `results/m6/`,
`results/costs.jsonl`, logs and DONE markers after each pod, including the
scoped cost logs. Ensure later pods receive the generated seed manifests.
Each marker contains status and log tail; inspect it before proceeding. Retain
the M3/M4/M5 result directories and full original cost ledger on the VM.

Old records do not reliably distinguish seeds and conditions. Inspect the
historical logs and run metadata, including failed attempts, and create
`results/m6/legacy_cost_mapping.csv` with header `run_id,scope`, mapping only
verified successful records to `seed0/siglip2`, `seed0/embeddinggemma2`,
`seed0/embeddinggemma2_partial`, `seed0/qwen3vl8b_ft`, or
`seed0/qwen3vl8b_zs`. Include each applicable extraction, training and
inference record; do not guess. Missing attribution leaves costs blank. Scoped
M6 logs are joined by run ID to the global ledger; failed commands remain in
the global ledger but are not put into the successful scoped logs. Avoid
reusing a scope after a successful run, since duplicate successful records
would be included in its total.

```sh
python3 scripts/summarize_m6.py
python3 scripts/plot_m6.py
```

Outputs are under `results/m6/summary/`. Macro-F1 is over answered questions,
as recorded by the existing scorer; accuracy_all uses all questions. The
cumulative plot shows actual logged chronological costs and evaluated
questions, not a projected operating-cost curve. Check missing costs and
attribution before interpreting or publishing the plots.
