# M7 reader-pod runbook

Use a CPU reader pod with the existing network volume mounted at `/workspace`.
The repository and volume paths below are `/workspace/bluecad-cost-benchmark`,
`data/official_items`, and `results`, relative to that repository. Transfer the
M7 code to that checkout before running it. Do not regenerate benchmark items,
run training or inference, or change M3–M6 artifacts. The estimated reader-pod
cost is approximately $0.5–1; actual charges depend on runtime and provider.

Run the following in order. Stop if verification fails; investigate missing
files, IDs, or mismatched metrics before analysis. Verification reads the saved
predictions and metrics for all three tasks and the 13 independent condition/seed
cells (including the one zero-shot reference), checks 1,000 unique official IDs
per task and prediction cell, recomputes saved headline metrics and evaluated
IDs, and checks each all-question accuracy against `docs/M6_RESULTS.md`.

```sh
cd /workspace/bluecad-cost-benchmark
python3 scripts/verify_m7.py --results-dir /workspace/bluecad-cost-benchmark/results --items-dir /workspace/bluecad-cost-benchmark/data/official_items
python3 scripts/analyze_m7.py --results-dir /workspace/bluecad-cost-benchmark/results --items-dir /workspace/bluecad-cost-benchmark/data/official_items --out-dir /workspace/bluecad-cost-benchmark/results/m7
python3 scripts/cost_reading_m7.py --summary-dir /workspace/bluecad-cost-benchmark/results/m6/summary --out /workspace/bluecad-cost-benchmark/results/m7/cost_accuracy.csv
python3 -m pip install matplotlib
python3 scripts/plot_m7.py --summary-dir /workspace/bluecad-cost-benchmark/results/m6/summary --out-dir /workspace/bluecad-cost-benchmark/results/m7/figures
```

The analysis repeats verification, so it cannot silently proceed after a failed
check. `qwen_task2_taxonomy.csv` contains counts and parsed letter frequencies;
`error_overlap.csv` counts per-seed all/one/none failures across five methods.
The single zero-shot run is reused as a reference for each seed, not represented
as three independent runs. `breakdowns.csv` includes overall and explicit
metadata group scores, item IDs for fix/break comparisons and the SigLIP outlier,
and 10k-to-20k changes where curve predictions exist. Task 2 curve changes are
also split at the median 10k top-two score margin, a descriptive uncertainty
proxy, not a calibrated probability. Missing curve predictions are recorded.
`qwen_training_artifacts.csv` records hashes of available training manifests and
adapter configurations and the evaluation configuration; missing artifacts are
explicit. Compare their recorded contents and hashes, and inspect training logs
if needed; equal configuration hashes do not establish equal adapter weights.

Taxonomy uses the existing Task 2 parser first. Parsed A–D choices are correct
or wrong against the official answer. For unparsed text, standalone letters
are counted case-insensitively: no standalone letter is `no_letter` (empty or
“Yes or No”); one outside A–D is `letter_out_of_range` (`E`); multiple are
`multiple_letters` (`A and E`). The parser accepts `ANSWER: B` and the last
standalone A–D letter, so `A and B` is parsed as B. This mechanical rule can
mistake an isolated article for a letter and does not infer intent.

`cost_accuracy.csv` identifies the cheapest known eligible condition for each
requested task and lists eligible conditions with missing mean GPU-hours. If
`status` is `incomplete_costs`, no overall cheapest conclusion is supported.
The zero-shot reference has a single measured cost, not a three-seed mean.
The figures omit points without measured mean costs, use measured values only,
and retain the reference as a single run. The cumulative plot represents the
recorded sequence in the summary table, including its cost-attribution gaps;
it is not a projection or a per-seed deployment cost curve. Outputs are under
`results/m7/figures/`; a missing or unsupported cumulative table is reported.

Transcribe the small CSV outputs back to the VM and interpret them before writing
`docs/M7_RESULTS.md` and `docs/M7_PAPER_OUTLINE.md`. Neither document has been
written in this implementation; both depend on the actual volume analysis.
