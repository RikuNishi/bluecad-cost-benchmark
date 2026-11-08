# manifests/

Fixed manifests that make runs comparable and reproducible:

- the official Task 1 / Task 2 question IDs as distributed (never regenerated);
- `train_pairs_20000.jsonl` — the common 20,000 training pairs. If the original
  pair list from the paper cannot be obtained, regenerate it once with the
  official procedure and a fixed seed, and use this same file for every
  compared model. Record the generation procedure and seed alongside the file.
