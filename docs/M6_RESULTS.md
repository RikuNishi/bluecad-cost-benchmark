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

## Next step (not yet run)

Retry the same chain with a launch that does not depend on the
terminal session: start the master with `setsid` (new session, no
controlling terminal), close the terminal, then verify from a fresh
terminal a few minutes later that the process is alive and the logs
are advancing. If it dies again on terminal close, fall back to a
foreground launch with the terminal kept attached. The retry needs
the owner's go-ahead; no new GPU work is started automatically.
