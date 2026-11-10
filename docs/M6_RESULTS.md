# M6 results — attempt 1 aborted (2026-11-09, no data)

## Attempt 1: chain terminated ~5 minutes after launch

- Pod: `9kz86266n3nglf` (bluecad-m6, Secure RTX PRO 4000 Blackwell
  24GB, $0.57/hr, EU-RO-1), launched 2026-11-09 ~19:06 JST running
  `scripts/pod_m6_all.sh` (chains frozen → partial seed 1 → partial
  seed 2 → qwen seed 1 → qwen seed 2, continuing past stage failures).
- The pod vanished ~19:11 JST. All four `_DONE` markers
  (`M6_ALL_DONE`, `M6FROZEN_DONE`, `M6PARTIAL_DONE`, `M6QWEN_DONE`)
  are 0-byte files; `m6_all.log` (200 B) ends at the frozen stage's
  pre-`exec` trace lines; `m6_frozen.log` (1,281 B) ends right after
  `build_seed_manifests.py` printed the seed-1 manifest lines
  (20,000 train pairs, 0 missing images). No extraction, training, or
  evaluation output exists; `results/m6/` contains only
  `legacy_cost_mapping.csv` (written later by the watchdog's reader
  pod); `results/costs.jsonl` has no M6 entries.
- Seed manifests survived: `manifests/seeds/seed1/` and
  `manifests/seeds/seed2/` each contain `train_pairs_20000.jsonl`
  (per-seed validation files were deleted by design — the canonical
  seed-0 validation manifest is used for all cells).
- Cost of the aborted attempt: pod alive ~5 min (~$0.05) plus two
  reader pods (~$0.5 total). Negligible against the ~$8 estimate.

## Cause: undetermined

No traceback, no error, no disk problem (`/workspace` 82% used),
no OOM evidence in the logs. The pod vanished abruptly. Two candidate
causes, neither proven:

1. Platform-side termination of the pod (host eviction / container
   kill). Secure Cloud pods are not normally preemptible, but the
   abrupt disappearance with no trap output is consistent with a
   SIGKILL-level event.
2. Web-terminal session teardown killing the background process
   group. The master script was started with `nohup ... &` from the
   launch terminal; the terminal session closed ~19:09 JST and the
   pod was gone ~19:11 JST. `nohup` protects against SIGHUP, not
   against a process-group SIGKILL on session teardown. (M4/M5
   launches kept the script in the foreground of an attached terminal
   and survived.)

## Seed-0 reference (from M3/M4/M5, via legacy_cost_mapping.csv)

Measured totals per condition (seed 0): SigLIP 2 frozen 0.165 GPU-h;
EmbeddingGemma 2 frozen 0.506 GPU-h; Qwen zero-shot 0.176 GPU-h;
Qwen fine-tune 3.222 GPU-h; EmbeddingGemma 2 partial 2.198 GPU-h.
17 cost records mapped, 11 probe/smoke records left unmapped
(deliberately — unmapped records leave costs blank rather than
guessed). The mapping file is
`results/m6/legacy_cost_mapping.csv` on the volume.

## Attempt 2: full chain completed (2026-11-09/10)

- Pod: `qx6jptxzkxid77` (bluecad-m6, Secure RTX PRO 4500 Blackwell
  32GB, $0.72/hr, EU-RO-1), launched 2026-11-09 ~22:49 JST via
  `setsid` (new session, no controlling terminal). Survival verified
  from a fresh terminal at 22:56 JST — the `setsid` fix worked; the
  attempt-1 failure mode (terminal teardown) did not recur. The
  volume had been expanded to 200 GB earlier that evening (it was
  full at 100 GB: ~182 GB used, EDQUOT on all writes; deleting the
  ~23 GB HF download-cache tars was not enough; expansion via console
  Storage → Edit, $7/mo → $14/mo).
- The chain ran to completion and the pod self-terminated.
  `results/M6_ALL_DONE`: master exit 0; frozen:0, partial_seed1:0,
  partial_seed2:0, qwen_seed1:0, qwen_seed2:0. Pod lifetime ≈15.7 h,
  ≈$11.3. Watchdog retrieval by reader pod `bluecad-reader8`
  (2026-11-10).
- Main table — `accuracy_all` (all-questions denominator, n=1000 per
  task), mean ± std over 3 seeds; per-seed values in parentheses
  (seed0 / seed1 / seed2). Answer rate 1.0 everywhere unless noted.
  GPU-hours per seed are measured totals for the condition's
  extract+train+infer in that seed; for the two trained conditions
  the seed-0 column in `main_table.csv` is blank (summarizer
  attribution gap) — the known seed-0 totals from M4/M5 are given
  here instead. The shared union extraction cost is charged in full
  to each new frozen seed cell; seed 0 uses its historical M3
  extraction (the documented convention).

  | condition | Task 1 | Task 1-dim | Task 2 | GPU-h (s0/s1/s2) |
  |---|---|---|---|---|
  | SigLIP 2 frozen | 0.727 ± 0.002 (0.725/0.728/0.728) | 0.693 ± 0.020 (0.670/0.707/0.701) | 0.880 ± 0.007 (0.872/0.885/0.883) | 0.165 / 0.449 / 0.449 |
  | EmbeddingGemma 2 frozen | 0.737 ± 0.005 (0.740/0.731/0.739) | 0.671 ± 0.007 (0.664/0.673/0.677) | 0.859 ± 0.011 (0.853/0.872/0.852) | 0.506 / 1.429 / 1.429 |
  | EmbeddingGemma 2 partial | 0.766 ± 0.003 (0.763/0.766/0.769) | 0.759 ± 0.002 (0.760/0.757/0.760) | 0.950 ± 0.004 (0.954/0.946/0.951) | 2.198* / 1.610 / 1.641 |
  | Qwen3-VL 8B FT | 0.750 ± 0.004 (0.748/0.748/0.755) | 0.706 ± 0.027 (0.675/0.719/0.725) | 0.230 ± 0.065 (0.195/0.189/0.305), rate 0.806 | 3.222* / 5.016 / 4.908 |
  | Qwen3-VL 8B zero-shot (reference, single deterministic run) | 0.653 | 0.632 | 0.334 (rate 0.96) | 0.176 |

  \* Seed-0 totals from M4/M5 docs (M5 partial 2.198 GPU-h, M4 Qwen
  FT 3.222 GPU-h); `main_table.csv` leaves those two cells blank.
- Learning curves (frozen encoders only, 3-seed means,
  `results/m6/summary/learning_curves.csv`): Task 1 accuracy at
  2k/5k/10k/20k pairs — SigLIP 2: 0.682 / 0.707 / 0.723 / 0.727;
  EmbeddingGemma 2: 0.693 / 0.720 / 0.730 / 0.737. Task 2 at
  2k→20k — SigLIP 2: 0.812 → 0.880; EmbeddingGemma 2: 0.794 →
  0.859. More pairs help, most clearly on Task 2; the 10k→20k gain
  on Task 1 is small for both encoders.
- New-run GPU-hours (seeds 1–2, measured): frozen 3.756, partial
  3.251, Qwen FT 9.924 — total ≈16.9 GPU-h. Including the reused
  seed-0 cells (6.267 GPU-h from M3/M4/M5), all M6 cells ≈23.2
  GPU-h.
- Findings: the partial update wins on all three tasks and is the
  most stable across seeds (std ≤ 0.004). The Qwen FT Task 2
  collapse reproduces in every seed (0.195/0.189/0.305), but seed 2
  is notably higher (0.305, answer rate 0.858 vs ~0.78) — the
  largest seed variance in the study (std 0.065); M7 should check
  what differs in seed 2. SigLIP 2 Task 1-dim seed 0 (0.670) is the
  low outlier against seeds 1/2 (0.707/0.701).

Source tables: `results/m6/summary/main_table.csv`,
`results/m6/summary/learning_curves.csv`,
`results/m6/summary/cumulative_cost.csv` (on the volume).
