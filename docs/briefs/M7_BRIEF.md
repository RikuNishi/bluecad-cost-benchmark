# M7 brief — error analysis and writing (approved by owner 2026-11-10)

Read `AGENTS.md` first and follow it. CPU-only work: write code + unit
tests; no new GPU training in Parts 0–4. Any follow-up training the
analysis suggests is a separate decision with its own cost estimate —
it is NOT part of this brief.

Goal (research plan, weeks 13–20): explain *why* the M6 main comparison
came out the way it did, then write it up. M6 (2026-11-10, 3 seeds) is
complete: EmbeddingGemma 2 partial update leads on all three tasks and
is the most stable (Task 1 0.766 ± 0.003, Task 1-dim 0.759 ± 0.002,
Task 2 0.950 ± 0.004); Qwen3-VL 8B fine-tune collapses on Task 2 in
every seed (0.195 / 0.189 / 0.305, answer rate 0.806). The open
questions M6 left are the work list below. Final numbers live in
`docs/M6_RESULTS.md`; this brief does not re-measure them.

## Part 0 — retrieve the predictions (prerequisite, small cost)

The local repo has no `results/m6/` and no per-condition prediction
files — predictions live on the RunPod network volume only. Before any
analysis, verify them on the volume with a reader pod (the established
pattern; prior reader pods cost ~$0.25 each). There is no bulk file
channel from a pod back to the VM, so the Part 1–3 analyses run ON the
reader pod and only their small output CSVs are transcribed back;
Part 4 (writing) runs on the VM. Inventory verified 2026-11-10: every
cell below exists with `task*_predictions.jsonl` + `task*_metrics.json`,
Qwen rows are `{"id", "pred", "raw"}`, frozen/partial rows are
`{"id", "pred", "scores"}`; one spot check already shows fine-tuned
Task 2 rows with raw = "E" (Part 1, category 2):

- M6 cells: `results/m6/seed{s}/<condition>/` for s ∈ {1, 2},
  condition ∈ {siglip2, embeddinggemma2, embeddinggemma2_partial,
  qwen3vl8b_ft}, plus `results/m6/summary/` (main_table.csv,
  learning_curves.csv, cumulative_cost.csv).
- Seed-0 cells: `results/siglip2/`, `results/embeddinggemma2/`,
  `results/embeddinggemma2_partial/`, `results/qwen3vl8b_ft/`,
  `results/qwen3vl8b_zs/` (M3/M4/M5 dirs on the same volume).
- Verify after retrieval: every cell has 1,000 predictions per task
  (3,000 total), IDs match the official items, and the recomputed
  metrics match `docs/M6_RESULTS.md` exactly. If a file is missing or
  a metric mismatches, stop and report — do not analyse around it.
- Qwen predictions carry the `raw` generation text
  (`scripts/run_qwen_eval.py` saves `{'id', 'pred', 'raw'}`); frozen
  and partial predictions carry `{'id', 'pred', 'scores'}`. The
  analysis below must work from these saved fields only — no
  re-running inference.

## Part 1 — Qwen Task 2 failure taxonomy (the headline question)

Why does fine-tuned Qwen collapse on Task 2 (0.195/0.189/0.305) while
zero-shot scores 0.334? Classify every Task 2 generation (all 3 seeds
× fine-tuned + zero-shot) from the saved `raw` text, using the parser
in `src/bluecad_cost/scoring.py` (`parse_prediction`: needs an
`ANSWER: X` or a final single letter A–D) as the reference rule.
Categories (count each, report per seed):

1. `no_letter`: no A–D letter anywhere (e.g. empty, Yes/No answer,
   free-form description with no choice).
2. `letter_out_of_range`: a single letter is given but outside A–D.
3. `multiple_letters`: several candidate letters, no clear final one.
4. `parsed_but_wrong`: parses to A–D but is the wrong choice.
5. `parsed_correct`: parses and is right.

Then answer, with counts: (a) is the collapse mostly format failure
(categories 1–3) or wrong choices (category 4)? (b) among parsed
answers, is there a letter bias (e.g. over-predicting one option)?
(c) what is different about seed 2 (Task 2 0.305, answer rate 0.858
vs ~0.78 in seeds 0/1) — which category shrinks there, and does the
seed-2 training manifest or adapter differ in any recorded way?
(d) does zero-shot fail in the same categories, or differently?
Output: `results/m7/qwen_task2_taxonomy.csv` + a short section in
`docs/M7_RESULTS.md`. No new training.

## Part 2 — where do the methods fail? (error overlap and breakdowns)

Using the saved per-item predictions for all 5 conditions × 3 seeds:

- Overlap: for each task, how many items are failed by all methods,
  by only one method, and by none? Is there a shared hard core, or
  does each method fail different items? Report as a small table plus
  counts, not a figure alone.
- Task 1-dim breakdown: Task 1-dim items carry a `dim` variant
  (dimension-focused). Break Task 1-dim accuracy down by whatever
  grouping the official items carry (item metadata only — do not
  invent groups). If no grouping exists in the items, say so and
  report the overall rate only.
- Frozen vs partial: which items does the partial update fix relative
  to frozen EmbeddingGemma 2, and which (if any) does it break?
  Same question for Qwen fine-tune vs zero-shot on Task 1 / Task 1-dim.
- Learning-curve follow-up: Task 1 gains little from 10k→20k pairs
  while Task 2 gains a lot (M6 learning curves). Check whether the
  extra pairs fix a specific failure category in Part 1/2 terms
  (e.g. Task 2 errors concentrated where the head is uncertain) or
  whether the gain is spread evenly. Use the saved curve-cell
  predictions under `results/m6/curves/` if present; if they were not
  saved, note that and answer from the 20k cells only.
- Seed outliers: SigLIP 2 Task 1-dim seed 0 (0.670) vs seeds 1/2
  (0.707/0.701). Check whether the seed-0 gap is spread across items
  or driven by a small item subset, and whether the same items are
  weak in seeds 1/2.

Output: `results/m7/error_overlap.csv`, `results/m7/breakdowns.csv`,
plus the analysis section in `docs/M7_RESULTS.md`.

## Part 3 — cost–accuracy reading (no new runs)

From the M6 summary tables only: state the cheapest condition that
reaches within 0.01 of the best Task 1 accuracy, and the cheapest
condition that reaches within 0.01 of the best Task 2 accuracy, with
the measured GPU-hours per condition (mean over seeds). Do not
interpolate or extrapolate costs. If the answer differs by task, say
so plainly — that is a finding, not a problem to smooth over.

Figures (owner request, 2026-11-10): from the same M6 summary tables,
produce the cost–performance figures for the write-up and for the
owner to inspect — (1) accuracy vs mean GPU-hours scatter, one panel
per task, one point per condition (3-seed mean ± std where the table
carries it); (2) the frozen-encoder learning curves (accuracy vs
training pairs, per task); (3) cumulative GPU-hours vs evaluated
items per condition if `cumulative_cost.csv` supports it. Measured
values only, axes labelled with units, no smoothing or fitted lines.
`scripts/plot_m6.py` exists — extend or supersede it as needed;
matplotlib is available on the reader pod after `pip install
matplotlib` (it was missing during the M6 retrieval, which is why no
figures exist yet). Save PNGs under `results/m7/figures/`.

## Part 4 — writing outline

Draft `docs/M7_PAPER_OUTLINE.md`: a paper skeleton (title options,
abstract paragraph, methods paragraph using the common-protocol facts
— same 20,000 pairs, same validation set, all-questions denominator,
val-tuned threshold stated — results table with the M6 3-seed means,
the Part 1 taxonomy as the Task 2 explanation, limitations
including the M6 noted gaps such as the blank seed-0 cost cells in
`main_table.csv`). The outline quotes measured numbers only; anything
not yet measured is marked TODO, never filled with a guess.

## Part 5 — runbook

`docs/M7_RUNBOOK.md`: retrieval steps and verification, the analysis
scripts and their outputs, the taxonomy definitions with examples,
and the writing outline's status. Estimated cost: reader pod only
(~$0.5–1); Parts 1–3 run on the reader pod (CPU), Part 4 on the VM,
no GPU training or inference.

## Unit tests (CPU, no torch model download)

- Taxonomy: each category is triggered by a synthetic `raw` string
  (empty, Yes/No only, letter E, two letters, `ANSWER: B`, plain
  trailing letter); the parser-based rule and the taxonomy agree on
  what counts as parseable.
- Overlap/breakdown: aggregation over synthetic prediction files
  (shared-failure counts, per-group accuracy, fix/break counts).
- Retrieval check: a script that verifies prediction counts, ID
  coverage and metric reproduction against `docs/M6_RESULTS.md`
  values, failing loudly on mismatch.
- Existing test suite must still pass.

## Out of scope

- No new GPU training or inference runs, no pod creation beyond the
  Part 0 reader pod, no re-running any M6 cell, no changes to M3–M6
  results or code paths, no external evaluation or multi-view
  extension (those need a separate brief once this analysis lands).
