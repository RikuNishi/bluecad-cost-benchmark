# M5 結果：EmbeddingGemma 2 部分更新（2026-11-09 実行）

実行環境：RunPod Secure Cloud、NVIDIA RTX PRO 4000 Blackwell（24GB）、
EU-RO-1、$0.57/h。Attempt 2 の pod（zraema5s7nkr3a）は 2026-11-09
16:05 JST 頃に起動し、17:55:47 JST に M5_DONE（success, exit 0）を
書き込んで自己終了した。Pod 課金時間は約 1.85 時間で約 $1.05。
Attempt 1 の停止分（約 $0.15）は別途発生している（下記）。

## 設計（M3 との差分はエンコーダ重みのみ）

- M3 と同じ共通 20,000 ペア（正負 50/50、seed 0）でエンコーダを学習し、
  同じ 36,366 画像の特徴を再抽出してから、M3 と同じ共通 head
  （[d, r, |d-r|, d*r] 連結＋標準化＋ロジスティック回帰）を
  新規に学習して 3 タスクを評価した。head・manifest・評価手順は M3 と同一。
- モデル：google/embeddinggemma-2、revision f9972ac22e696a105c482202646979e33e4b5a1e。
- 更新範囲：`last_k:0.model.vision_tower.encoder.layers:2`
  （vision tower 16 ブロック中の最後の 2 ブロック＋画像側 projection/pooling
  モジュール：vision pooler、`embed_vision.embedding_projection`、
  `embed_vision.embedding_pre_projection_norm`）。
  学習可能パラメータ数は 19,273,984（scaffold head を除く）。
- ハイパーパラメータ：AdamW、encoder lr 1e-5、scaffold head lr 1e-3、
  weight decay 0、cosine schedule、warmup ratio 0.03、実効バッチ 32 ペア、
  1 epoch、seed 0。閾値は検証ペア 2,000 件で精度最大化した値を使用
  （検証ラベル使用を明記）。fresh head の閾値は 0.4325。
- 公式の部分更新レシピは存在しないため、上記は本研究の選択である。

## 当初計画からの逸脱（プローブで判明したもの）

- バッチ 32 ペアの単一 forward/backward は 24GB カードで OOM したため、
  micro-batch 8 ペア × gradient accumulation 4（実効バッチ 32、625 ステップは不変）。
- 純 BF16 の AdamW では lr 1e-5 の更新が丸めで消えるため、FP32 マスター重み＋
  BF16 autocast で学習した。抽出時は保存したエンコーダを BF16 にキャストし、
  M3 の BF16 抽出と同一条件にした。

## Attempt 1 の失敗と修正

- Attempt 1（pod s91foru8ujy73m）は step ~80/625 で手動停止した。
  scaffold loss が ln 2（0.692–0.694）に張り付いたままだった。
  原因：単一の AdamW グループ（lr 1e-5）が scaffold head にも適用され、
  M3 で lr 1e-3 で学習する head がほぼ初期値のままだったため、
  エンコーダにも有用な勾配が届かなかった。
- 修正：パラメータグループを分離（encoder 1e-5 / head 1e-3）して再実行
  （Attempt 2）。Attempt 1 の中間成果物は保存されておらず破棄した。
  Attempt 2 では loss は 0.695（step 20）→ 0.668（step 80）→
  終盤 0.56–0.60 まで低下した。

## 実測値（本研究の測定値。文献値とは別）

fresh head による評価結果。回答率はいずれも 1.0（全問に予測あり、
n=1000、n_error=0、n_unparseable=0）。Task 2 はペアスコアの argmax で予測。
いずれも学習済み head の結果であり zero-shot ではない。

| タスク | M5 accuracy_all | M5 macro-F1 | M3 凍結 EG2 accuracy_all | 差分 |
|---|---|---|---|---|
| Task 1 | 0.763 | 0.762 | 0.740 | +0.023 |
| Task 1-dim | 0.760 | 0.760 | 0.664 | +0.096 |
| Task 2 | 0.954 | 0.954 | 0.853 | +0.101 |

部分更新は 3 タスクすべてで凍結エンコーダを上回った。特に Task 1-dim
（+0.096）と Task 2（+0.101）の改善が大きい。

### Scaffold 診断（診断のみ。報告値ではない）

学習と同時に訓練した scaffold head の検証精度は、閾値 0.5 固定で 0.761、
検証ラベルで閾値調整後で 0.762（閾値 0.5005）。これは訓練過程の診断値で
あり、上記の評価結果は scaffold ではなく fresh head によるものである。

## コスト実測（RTX PRO 4000 Blackwell、results/costs.jsonl より）

| 工程 | 時間 | GPU-h | ピーク VRAM |
|---|---|---|---|
| 部分更新の学習（20,000 ペア、625 ステップ） | 4,779 秒 | 1.328 | 14.78 GB |
| 特徴再抽出（36,366 画像、BF16、batch 64） | 3,130 秒 | 0.870 | 6.42 GB |
| fresh head 学習（20,000 ペア） | 2.5 秒 | 0.001 | 0.97 GB |
| 評価（3,000 問、特徴キャッシュ） | 1.7 秒 | 0.0005 | 0.12 GB |

Attempt 2 のパイプライン合計は約 2.20 GPU-h。Pod 課金は約 1.85 時間 ×
$0.57/h で約 $1.05。比較のため、M3 の EmbeddingGemma 2 凍結抽出は
RTX 4090 で 1,820 秒（0.506 GPU-h）だった。

## 成果物の所在

予測・指標・特徴・head・エンコーダは Pod にマウントした Network Volume 上の
/workspace/bluecad-cost-benchmark/results/embeddinggemma2_partial/ に
保存されている。エンコーダの識別子は
local-sha256:090e9be6b89dc9cb4f24fcc65e65df5acdca46a98506d358764fabdb776a73bb。
実行設定は同ディレクトリの run_config.json、指標は task1_metrics.json /
task1_dim_metrics.json / task2_metrics.json、実行ログは results/m5.log。
