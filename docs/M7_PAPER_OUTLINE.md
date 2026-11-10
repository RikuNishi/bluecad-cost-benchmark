# 論文アウトライン（M7 Part 4、2026-11-10）

実測値のみを引用する。未測定の項目は TODO と明記する。

## タイトル案

1. Accuracy per GPU-Hour: A Controlled Cost Benchmark of Adaptation
   Strategies for CAD Drawing–Model Matching (Blueprint2CAD)
2. Frozen Encoders, Partial Updates, and VLM Fine-Tuning Compared
   under a Common Protocol and a Common Budget Ledger
3. When Fine-Tuning Breaks the Answer Format: Failure Taxonomy of a
   Vision-Language Model on a CAD Benchmark

## Abstract（草案）

We re-measure accuracy against compute cost on the official
Blueprint2CAD / BlueCAD-30K benchmark under one common protocol:
the same 20,000 training pairs, the same validation set, a common
classifier head for encoder-based methods, and accuracy computed
over all questions (unanswered questions count as wrong). Across
three seeds, partial updating of a frozen embedding encoder
(EmbeddingGemma 2, last-2 vision layers) achieves the best accuracy
on all three tasks (Task 1 0.766 ± 0.003, Task 1-dim 0.759 ± 0.002,
Task 2 0.950 ± 0.004) at a mean cost of 1.82 GPU-hours per run,
while QLoRA fine-tuning of an 8B vision-language model
(Qwen3-VL 8B) costs more (4.38 GPU-hours) and collapses on the
multiple-choice task (0.230 ± 0.065). A failure taxonomy of the
saved generations shows the collapse is twofold: 14–22% of answers
fall outside the A–D answer format, and the answers that do parse
are at chance level (0.242–0.250 in two of three seeds). Frozen
encoders with the common head are far cheaper (0.35–1.12 GPU-hours)
and remain competitive on Task 1, making the cheapest method within
0.01 of the best accuracy the partial update on both tasks we
checked. TODO: venue/length constraints decide how much of the
taxonomy and cost ledger fits the main text vs. appendix.

## Methods（共通プロトコルの要点）

- データ: 公式 Blueprint2CAD / BlueCAD-30K。公式 item JSON を
  配布どおり使用。train/val/test は互いに素（26,039 / 2,890 /
  テストは各タスク 1,000 問）。学習ペアは共通 manifest の
  20,000 ペア（seed ごとに再抽選、検証は常に seed 0 の
  2,000 ペア）。
- 採点: 分母は全問。回答率と回答分のみの精度を併記し、
  評価 ID を保存する。凍結系・部分更新のしきい値は検証集合で調整。
- 条件: (1) SigLIP 2 凍結特徴 + 共通 head、(2) EmbeddingGemma 2
  凍結特徴 + 共通 head、(3) EmbeddingGemma 2 部分更新
  （vision 最終 2 層 + 新規 head）、(4) Qwen3-VL 8B zero-shot
  （参照、単一ラン）、(5) Qwen3-VL 8B QLoRA 追加学習
  （r16、共通 20,000 ペア、1 epoch）。
- 共通 head: 特徴の concat [d, r, |d−r|, d⊙r] に対する
  ロジスティック回帰（Adam lr 1e-3、200 epoch）。
- コスト: 全 run で GPU 時間を実測記録。GPU は run ごとに異なる
  ため、費用換算ではなく GPU 時間を主指標とする。
  TODO: GPU 種別ごとの内訳表を付録に入れるか検討。
- 再現性: seed は 3 本（0/1/2）。Qwen の学習 seed は 13+seed、
  評価は決定的（greedy、評価 seed 13）。

## Results

### 主表（M6、3 seed 平均 ± 標準偏差、accuracy_all）

| 条件 | GPU-h 平均 | Task 1 | Task 1-dim | Task 2 |
|---|---|---|---|---|
| SigLIP 2 凍結 | 0.354 | 0.727 ± 0.002 | 0.693 ± 0.020 | 0.880 ± 0.007 |
| EG2 凍結 | 1.121 | 0.737 ± 0.005 | 0.671 ± 0.007 | 0.859 ± 0.011 |
| EG2 部分更新 | 1.816 | **0.766 ± 0.003** | **0.759 ± 0.002** | **0.950 ± 0.004** |
| Qwen FT | 4.382 | 0.750 ± 0.004 | 0.706 ± 0.027 | 0.230 ± 0.065 |
| Qwen ZS（単一） | 0.176 | 0.653 | 0.632 | 0.334 |

図: `results/m7/figures/cost_accuracy.png`（精度 vs GPU 時間）、
`results/m7/figures/learning_curves.png`（凍結系の学習曲線）。

### 費用対性能の読み

- 最高精度から 0.01 以内で最も安い条件は、Task 1・Task 2 とも
  部分更新（1.82 GPU-h）。圏内に入る他の条件はない。
- 予算優先なら SigLIP 2 凍結（0.354 GPU-h）が Task 1 −0.039 /
  Task 2 −0.070 の代償で約 1/5 のコスト。

### Task 2 崩壊の分類（Part 1）

- FT の形式失敗（A–D 範囲外の文字）は 219/220/142 件、
  解釈できたが不正解は 586/591/553 件、正解は 195/189/305 件。
- 解釈済み回答の正解率は seed 0/1 で 0.250/0.242（偶然水準）、
  seed 2 は 0.355。ZS は形式失敗 40 件のみで解釈済み正解率 0.348。
- 予測文字の偏り: FT は D 寄り、ZS は A・B 寄り。
- 設定記録に seed 間の異常なし（adapter 設定は並び順のみ相違）。

### 誤りの重なりと内訳（Part 2）

- 全条件が失敗する共通難問は Task 1 で 3–4%、Task 2 で約 1%。
- 部分更新の Task 2 改善はほぼ一方通行（直した 98〜121 /
  壊した 13〜24）。
- Task 1-dim: Qwen FT は否定グループで 0.85〜0.97 の一方、
  positive グループは 0.396〜0.558 と非対称に低下。
- 学習曲線: 10k→20k の Task 2 の伸びは低 margin の問題に集中。

## Limitations

- `main_table.csv` の seed 0 GPU 時間が部分更新・Qwen FT で空欄。
  本書の平均は M4/M5 の記録（2.198 / 3.222 GPU-h）で補完した。
- `cumulative_cost.csv` の列名と中身に不整合があり、累積費用の
  図は作成できなかった。
- zero-shot は単一ラン（決定的評価）のため seed 分散がない。
  重なり分析では同ランを 3 seed 分に再利用した。
- GPU 機種が run ごとに異なるため、GPU 時間は機種差を含む。
  TODO: 機種を揃えた再測が必要かは今後の判断。
- 単一ベンチマーク・単一データセットでの結果であり、外部評価は
  未実施（別ブリーフで扱う）。
- 凍結系の数値は学習済み head の精度であり、zero-shot の文献値や
  検索 Recall@1 とは直接比較できない。

## TODO（執筆前に要判断）

- 投稿先・形式（ワークショップ論文 / 技術ブログ / arXiv）の決定
- 累積費用図を作り直すか（元データの列不整合の修正が前提）
- 外部評価・多視点拡張を行うか（別ブリーフ）
