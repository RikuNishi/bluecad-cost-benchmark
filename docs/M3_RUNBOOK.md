# M3 frozen-feature reproduction runbook

Run from the repository root on the GPU pod after installing `pip install -e '.[gpu]'`, obtaining the full BlueCAD-30K dataset and official item clone, and accepting model access terms. Set `ITEMS` if the official clone is elsewhere. A BF16-capable CUDA GPU is required. The scripts use repository-relative image paths, so run all commands from the repository root. Archives must already be downloaded under `data/bluecad-30k/{drawings,renders}/`.

```bash
ITEMS="$HOME/workspace/research/blueprint2cad-official/vlms/src/benchmark/items"
python3 scripts/extract_shards.py all
python3 scripts/build_pair_manifest.py --seed 0
python3 scripts/collect_images.py --items-dir "$ITEMS"
python3 scripts/extract_features.py --encoder siglip2 --images manifests/images_all.jsonl --out results/siglip2/features.npz
python3 scripts/extract_features.py --encoder embeddinggemma2 --images manifests/images_all.jsonl --out results/embeddinggemma2/features.npz
python3 scripts/train_head.py --features results/siglip2/features.npz --out results/siglip2/head.pt --seed 0
python3 scripts/train_head.py --features results/embeddinggemma2/features.npz --out results/embeddinggemma2/head.pt --seed 0
python3 scripts/eval_frozen.py --features results/siglip2/features.npz --head results/siglip2/head.pt --items-dir "$ITEMS" --out-dir results/siglip2
python3 scripts/eval_frozen.py --features results/embeddinggemma2/features.npz --head results/embeddinggemma2/head.pt --items-dir "$ITEMS" --out-dir results/embeddinggemma2
```

The pair manifests contain 20,000 train and 2,000 validation pairs with balanced labels. The image manifest is their union with the three official item sets. Each NPZ contains IDs, normalized float32 features, model ID and resolved revision; each head contains the train and validation manifest hashes, seed, scaler, weights, validation-tuned threshold and validation accuracy at both thresholds. Each result directory receives predictions and metrics for Task 1, Task 1-dim and Task 2. Metrics include all-question and answered-only accuracy, answer rate, and evaluated IDs. Task 1 also records the 0.5 threshold metrics. These are trained-head results, not zero-shot scores. Extraction, training and evaluation append measurements to `results/costs.jsonl`; evaluation uses cached features. GPU runs and their costs are not yet measured.

The official item paths determine the actual views. In the available Task 2 items the drawing reference is `000`, which maps to front under the official view mapping; do not replace it based on the brief's conflicting description of Task 2 as right-view.
