# M5: EmbeddingGemma 2 partial update

M5 changes the encoder weights relative to M3. Train the encoder on the same
20,000 training pairs with a jointly trained logistic classifier over
`[d, r, |d-r|, d*r]`. That classifier is a scaffold: its validation accuracy
at 0.5 and at a threshold selected using validation labels is diagnostic
only. Save the encoder, extract the same `manifests/images_all.jsonl` union
(36,366 images), then train a **fresh** M3 common head on the same training
and validation manifests and evaluate all three official tasks. The fresh head
uses validation labels to tune its threshold. Report those evaluation results,
not the scaffold diagnostic. No official partial-update recipe exists; the
following hyperparameters are our choice: AdamW, learning rate 1e-5, weight
decay 0, cosine schedule, warmup ratio 0.03, effective batch 32 pairs
(micro-batch 8 pairs x gradient accumulation 4), BF16 autocast over FP32
master weights, one epoch, seed 0. The forward uses SentenceTransformer's own tokenization and modules,
then the same L2 normalization as extraction; no pooling is reimplemented.

At startup the script prints top-level module parameter counts and all
homogeneous ModuleList/Sequential lists whose named-module path contains
`vision`, `visual`, or `image`, with lengths and parameter counts. The list's
full named-module path is its scope name. The principal list is the one with
the most parameters (lexicographically largest path breaks ties). For
`last_k:<path>:<k>`, all encoder parameters are frozen first, then the last
k modules in that list and image-side projection/pooling modules outside any
discovered list are unfrozen. These extra modules have an image/vision/visual
or multi_modal path and a final path component containing `project` or `pool`;
the startup log and run config list them. `all` unfreezes the entire encoder.
The printed and saved trainable counts exclude the scaffold head.

Probe k=1, 2, and 4 on the principal list and also `all`, each for three
steps with a fresh encoder at batch 32. Select the largest k with peak VRAM
at most 20 GB; `all` is informational and is not eligible. If no candidate
fits, do not run M5 with these settings. Read `probe.json` and use the exact
scope string from the selected result. The consistency check compares the
training forward and `encode` for 64 manifest images before either operation.
Both checks and all runs append cost records; no GPU results are implied by
this runbook.

On the configured RunPod, with the M3 data, manifests, and official items
already present in the repository at `/workspace/bluecad-cost-benchmark`:

```bash
cd /workspace/bluecad-cost-benchmark
bash scripts/pod_m5_probe.sh
```

After inspecting `results/embeddinggemma2_partial/probe.json`, start a new pod
and set the selected scope explicitly (replace the example with the actual
printed path and selected k):

```bash
cd /workspace/bluecad-cost-benchmark
M5_SCOPE='last_k:<printed-principal-path>:<selected-k>' bash scripts/pod_m5_run.sh
```

The scripts write `results/M5PROBE_DONE` or `results/M5_DONE` with status and
log tail and terminate the pod on exit using `RUNPOD_POD_ID` and
`RUNPOD_API_KEY`. Set `M5_NO_AUTOSTOP=1` to disable termination. Outputs
include the encoder and source revision, run configuration and diagnostic,
features, fresh head, predictions, evaluated IDs and task metrics. A SHA-256
of local encoder files identifies extracted feature provenance.

## Deviations from the original brief (learned from probe attempts 1-2)

- Batch 32 pairs in a single forward/backward OOM'd on a 24 GB card
  (21.6 GiB allocated during the first training step of the k=1 scope;
  activations through the vision tower + per-layer-embedding backbone
  dominate). Training therefore uses micro-batch 8 pairs with gradient
  accumulation 4; the effective batch, schedule and step count (625)
  are unchanged.
- Pure-BF16 AdamW at lr 1e-5 would round most per-step updates away
  (the BF16 ulp at typical weight scales exceeds the update size), so
  the encoder is trained with FP32 master weights and BF16 autocast.
  Extraction afterwards casts the saved encoder to BF16, exactly as
  M3's BF16 extraction did. The consistency check compares the training
  forward against `encode` under the same autocast context.
- The probe runs each scope in a subprocess and appends to probe.json
  after every scope, so one scope's OOM cannot erase the others.
- Discovered structure (probe attempt 2): total 744,371,488 params;
  image block list `0.model.vision_tower.encoder.layers` (16 blocks,
  151,046,144 params); always-trainable extras are the vision pooler,
  `embed_vision.embedding_projection` (393,216) and
  `embed_vision.embedding_pre_projection_norm`. Consistency check
  passed with max cosine deviation 1.19e-07.
