# M7 outputs

The analyses ran on the RunPod volume (reader pod, 2026-11-10).
Small CSVs were transcribed back verbatim and are kept here:
`qwen_task2_taxonomy.csv`, `error_overlap.csv`, `cost_accuracy.csv`.
`breakdowns.csv` (≈1.1 MB, includes per-item ID lists) and
`qwen_training_artifacts.csv` remain on the volume under
`results/m7/`; their aggregate values are reported in
`docs/M7_RESULTS.md`. Figures under `figures/` were rendered on the
VM from the transcribed M6 summary tables (see docs/M7_RESULTS.md,
Part 3 for why and for the seed-0 GPU-hour provenance); the PNG
files are kept in the local working tree only, as the snapshot push
used for this repo accepts text files only.
