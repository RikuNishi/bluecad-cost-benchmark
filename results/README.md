# results/

Per-run outputs (git-ignored except this file):

- `cost_log.jsonl` — one cost record per run (see src/bluecad_cost/costlog.py)
- `<run_id>/predictions.jsonl` — per-question predictions, with evaluated IDs
- `<run_id>/metrics.json` — accuracy (all-questions denominator), answered-only
  accuracy, answer rate, macro-F1, plus full run metadata (model revision,
  preprocessing, image token settings, quantisation, library versions)

Literature values live in the paper tables, never in this directory.
