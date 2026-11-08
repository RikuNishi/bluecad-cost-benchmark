# AGENTS.md — instructions for Codex working in this repo

Mission: implement the study in `RESEARCH_PLAN.ja.md` — re-measure accuracy vs
compute cost on the official Blueprint2CAD / BlueCAD-30K tasks. Read the plan
before changing experiment code; it is the source of truth, this file is the
short contract.

## Non-negotiables (from the research plan)

- Use the official Task 1 / Task 2 JSON as distributed. Never regenerate the
  test combinations, candidate order, answers, or viewpoints. Only rewrite the
  absolute paths inside the JSON to the local image locations.
- Report accuracy over ALL questions as the denominator, plus accuracy over
  answered questions and the answer rate, separately. Save the evaluated IDs
  for every run.
- Never call a frozen encoder + trained classifier "zero-shot". If validation
  labels were used to pick a threshold, state that in the run metadata.
- Give every compared model the same classifier, the same 20,000 training
  pairs (same regenerated manifest if the original list is unavailable), and
  the same validation set.
- Literature values, reproduced values, and this study's measured values are
  three different things. Never merge them into one table column or figure.
- Every training / extraction / inference run must append a cost record
  (see `src/bluecad_cost/costlog.py`): GPU model, wall time, GPU hours,
  peak VRAM, batch size, precision, number of items. No cost record = run
  does not count.
- Do not plot or report a cost or improvement number that was not measured.

## Working style (the owner is on ChatGPT Plus — Codex usage is scarce)

- Work one milestone at a time (see `docs/MILESTONES.md`). Keep diffs small
  and complete the current milestone before starting the next.
- Run the tests after each change: `python -m unittest discover -s tests`.
  The base environment has no GPU and only the Python standard library, so
  core logic (scoring, manifests, cost logging) must import and test without
  torch. GPU-only imports go inside functions or GPU entry-point scripts.
- Do not read or summarise files under `data/` (large binaries) and do not
  add notebooks with embedded outputs.
- When a task is done, summarise: files changed, tests run and their result,
  and anything left unverified. Stop there; do not invent follow-up work.

## Layout

- `configs/` — one YAML per experiment; common settings in `common.yaml`.
- `scripts/` — runnable entry points (GPU ones are meant to run on the GPU
  machine, not here). Specs are in `scripts/README.md`.
- `src/bluecad_cost/` — shared library code (scoring, cost logging, manifests).
- `manifests/` — fixed split / pair manifests and their documentation.
- `results/` — per-run predictions, metrics and cost logs (JSONL/JSON/CSV).
- `data/` — dataset files, git-ignored, never commit.
