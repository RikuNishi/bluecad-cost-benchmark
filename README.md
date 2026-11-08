# BlueCAD Cost Benchmark

公式機械図面ベンチマーク **Blueprint2CAD / BlueCAD-30K** 上で、モデルの精度と計算コストを
同じ問題・同じ学習条件で再計測する研究のリポジトリ。

- 研究計画（正本）: [RESEARCH_PLAN.ja.md](RESEARCH_PLAN.ja.md)
- データ配布の確認記録: [DATA_ACCESS_AUDIT.md](DATA_ACCESS_AUDIT.md)
- GPU要件の見積もり: [docs/GPU_REQUIREMENTS.md](docs/GPU_REQUIREMENTS.md)
- Codex（Plusプラン）の節約運用: [docs/CODEX_USAGE.md](docs/CODEX_USAGE.md)
- 実施マイルストーン: [docs/MILESTONES.md](docs/MILESTONES.md)

## 比較対象（最低限、実測するもの）

1. Qwen3-VL 8B（事前学習済み）
2. Qwen3-VL 8B（公式方式で Task 1 追加学習、共通 20,000 ペア）
3. SigLIP 2 凍結特徴 + 共通分類器
4. EmbeddingGemma 2 凍結特徴 + 共通分類器
5. EmbeddingGemma 2 部分更新 + 共通分類器

## 実行環境の分担

| 場所 | 役割 |
|---|---|
| Muse の VM（GPUなし） | コード作成（Codex）、データ監査、採点、集計・作図 |
| GPU環境（24GB BF16 ×1、Colab L4 / レンタル 4090 等） | 特徴抽出、Qwen 推論・追加学習の実行専用 |

GPU側では `scripts/` のエントリポイントを config 指定で実行するだけの形にし、
全 run でコストログ（GPU時間・VRAM・料金換算の材料）を `results/` に残す。

## クイックスタート

```bash
python scripts/check_env.py          # CPU/GPU・VRAM・ディスクの確認（どこでも実行可）
python -m unittest discover -s tests # 標準ライブラリのみで動くスモークテスト
```

データ本体（約24GB、gated）は `data/` に置くが git 管理しない。
取得は Hugging Face 側の同意が前提（DATA_ACCESS_AUDIT.md 参照）。

## 状態

2026-11-07: 土台作成。データ取得・学習・評価は未実施。
