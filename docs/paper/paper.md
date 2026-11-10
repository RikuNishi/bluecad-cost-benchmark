# Accuracy per GPU-Hour: A Controlled Cost Benchmark of Adaptation Strategies for CAD Drawing–Render Matching

**Riku Imanishi**（所属: TODO）
草稿 — 2026-11-10。本稿の数値はすべて本研究の実測値であり、文献値は文献値として区別して表記する。
図 1–3 の PNG は docs/paper/figures/、図 4–5 は results/m7/figures/ にあり、いずれもローカル作業ツリーのみ（スナップショット push はテキストファイルのみ対象）。

## Abstract

We re-measure accuracy against compute cost on the official
Blueprint2CAD / BlueCAD-30K benchmark [1] under one common protocol:
the same 20,000 training pairs, the same validation set, a common
classifier head for encoder-based methods, and accuracy computed
over all questions (unanswered questions count as wrong). Across
three seeds, partial updating of a frozen embedding encoder
(EmbeddingGemma 2, last two vision layers) achieves the best
accuracy on all three tasks (Task 1 0.766 ± 0.003, Task 1-dim
0.759 ± 0.002, Task 2 0.950 ± 0.004) at a mean cost of 1.82
GPU-hours per run, while QLoRA fine-tuning of an 8B
vision-language model (Qwen3-VL 8B) costs more (4.38 GPU-hours)
and collapses on the multiple-choice task (0.230 ± 0.065). A
failure taxonomy of the saved generations shows the collapse is
twofold: 14–22% of generations fall outside the A–D answer format,
and the answers that do parse are at chance level in two of three
seeds (0.250 and 0.242, against a chance level of 0.25). On the
dimensioned-drawing task, the fine-tuned model is asymmetric: it
answers negative items almost perfectly (0.85–0.97) while positive
items drop to 0.40–0.56. Frozen encoders with the common head are
five to twenty times cheaper and remain competitive, but the
cheapest condition within 0.01 of the best accuracy is the partial
update on both tasks examined. All runs carry a measured cost
record; we report GPU-hours, not estimates.

**Keywords:** CAD, engineering drawings, benchmark, compute cost,
vision-language models, failure analysis

## 1. はじめに

工学図面と CAD レンダリングの対応付けは、部品検索・設計再利用・
図面のデジタル化の入口となるタスクである。Valverde ら [1] は
Blueprint2CAD ベンチマークとそのデータ層 BlueCAD-30K を公開し、
基盤モデルの現状を初めて体系的に測った。彼らの報告（文献値）では、
凍結エンコーダによる検索は視点が既知の場合で最大 65% の
Recall@1、未知の場合で 43% にとどまり、6 つの zero-shot
視覚言語モデル（VLM）の検証精度は 0.52–0.67 の範囲で、モデル固有の
Yes/No バイアスを伴う。さらに LoRA による短時間の追加学習は
検証精度を数ポイント改善する一方で、そのバイアスを強めることが
報告されている。

しかし実務家が次に知りたいのは「どの適応戦略が、どれだけの
計算コストで、どれだけ良くなるのか」であり、この問いは元の
ベンチマークでは主対象ではない。手法ごとの学習量・検証集合・
採点方法が揃っていなければ、精度差と費用差を同じ土俵で読むことは
できない。また、追加学習が「バイアスを強める」という所見も、
どのタスクで・どの形で失敗が増えるのかという item 単位の内訳は
明らかでない。

本稿はこの2点に答える再測定である。貢献は次の4点である。

1. **共通プロトコルによる費用対性能の主比較。** 全条件に同じ
   20,000 学習ペア、同じ検証集合、同じ採点（分母は全問）を適用し、
   全 run に GPU 時間の実測記録を付けた。3 seed の平均では、
   EmbeddingGemma 2 の部分更新が全タスクで最良かつ最も安定し、
   平均 1.82 GPU 時間であった。最も高コストな Qwen3-VL 8B の
   QLoRA 追加学習（4.38 GPU 時間）は Task 1 では2位につける一方、
   Task 2 では最下位に崩れた。
2. **Task 2 崩壊の失敗分類。** 保存済み生成文を共通パーサで
   分類し、崩壊が「形式の失敗」（14–22%）と「解釈できても偶然
   水準の選択」（seed 0/1）の二重構造であることを示した。
   zero-shot の失敗は形式失敗が 4% のみで質が異なる。
3. **非対称なバイアスの特定。** Task 1-dim のグループ別内訳で、
   追加学習モデルが否定項目では 0.85–0.97 に達する一方、
   肯定項目では 0.40–0.56 に落ちることを示した。[1] の
   「バイアスが強まる」という所見を、item グループ単位で
   具体化したものである。
4. **再現可能な記録一式。** 予測・メトリクス・コスト記録を
   run ごとに保存し、本稿の全数値を保存済み予測から再計算で
   検証した（39 セル全て一致）。

## 2. 関連研究

**Blueprint2CAD / BlueCAD-30K [1].** Zero-to-CAD の 29,925
モデルに描画パイプラインを適用し、6 視点のレンダリング、線画
図面（寸法なし／あり）、表面種別メタデータを付与したデータ層と、
図面–レンダ対応付けの検証・検索タスクからなるベンチマーク。
本稿はその公式 item を配布どおりに用い、Task 1（線画）、
Task 1-dim（寸法入り図面）、Task 2（4 択）を扱う。[1] の数値は
文献値であり、本稿の実測値とは表を分ける。特に、[1] の凍結
エンコーダの数値は検索 Recall@1 であって、本稿の学習済み head
による検証精度とは直接比較できない。同様に [1] の VLM 数値は
zero-shot 検証精度であり、本稿の zero-shot 条件（Qwen3-VL 8B
単独）とはモデル集合が異なる。

**適応戦略。** 凍結エンコーダへの軽量 head の学習、部分的な
パラメータ更新、LoRA [2] / QLoRA [3] による VLM の追加学習は、
いずれも現場で使われる代表的な適応手段である。本稿はこれらを
新規手法としてではなく、同一条件で測る対象として扱う。
エンコーダには SigLIP 2 [4] と EmbeddingGemma 2 [5]、VLM には
Qwen3-VL 8B [6] を用いた。

**コストを軸にした評価。** 精度だけでなく学習・推論コストを
併記する研究は増えているが、コストの定義（GPU 機種、実測か
見積りか、特徴抽出を含むか）が手法ごとに異なると比較は崩れる。
本稿では全 run に同一形式のコスト記録（GPU 機種、wall 時間、
GPU 時間、ピーク VRAM、バッチサイズ、精度、処理件数）を義務付け、
主指標を実測 GPU 時間に統一した。

## 3. 方法

### 3.1 データとタスク

公式 item JSON を配布どおりに使用し、画像パスだけを手元の配置に
書き換えた（AGENTS 契約どおり、問題の組み合わせ・候補順・正解・
視点は一切再生成していない）。データ監査（M1）の結果、split は
互いに素（train 26,039 / validation 2,890 / テストは各タスク
1,000 問）で、各タスクの正解分布はほぼ均等である。一方で
train リストには配布データの無い 4 件の ID が含まれ、shard-0000
の renders には transforms.json が欠けていることを確認した
（本稿の評価には影響しない）。

タスクは次の3つである。**Task 1**: 線画図面とレンダリングの
ペアが同一部品かどうかを Yes/No で答える。**Task 1-dim**: 寸法
入り図面を用いる同形式のタスクで、item には group
（simple / complex、各 500 件）と meta.tier（easy_neg 250 件 /
hard_neg 250 件 / positive 500 件）の公式グループ情報がある。
**Task 2**: 1 枚の図面に対し 4 枚の候補レンダリングから一致する
ものを選ぶ 4 択タスク。

### 3.2 共通プロトコル

![共通プロトコル](figures/fig_protocol.png)

*図 1: 共通プロトコル。全条件が同じ学習ペア・同じ検証集合・
同じ採点器で測られる。*

- 学習ペアは共通 manifest の 20,000 ペア（正例・負例 50/50）。
  seed s の run は seed s の manifest を使う。
- 検証は常に seed 0 の 2,000 ペア。しきい値調整はこの検証集合
  でのみ行い、その旨を run 記録に明記する。
- 採点の分母は全問。回答率と「回答分のみの精度」を併記し、
  評価した item ID を run ごとに保存する。
- seed は 3 本（0, 1, 2）。seed 0 のセルは M3–M5 の既存 run を
  再利用し、主比較で新規に GPU を使ったのは seed 1–2 のみである。

### 3.3 条件

![エンコーダ条件のアーキテクチャ](figures/fig_arch_encoder.png)

*図 2: エンコーダ条件。共有エンコーダと共通ペア head。*

**エンコーダ条件（図 2）。** 図面とレンダリングを同じ視覚
エンコーダ E で特徴量 d, r に変換し、concat [d, r, |d−r|, d⊙r]
を共通のロジスティック head（Adam lr 1e-3、weight decay 1e-4、
200 epoch）に入力してペアスコアを得る。Task 1 / 1-dim は検証
集合で調整したしきい値で二値化し、Task 2 は 4 候補のスコアの
argmax で選ぶ。この head は学習されるため、凍結条件を
zero-shot とは呼ばない。

- **SigLIP 2 凍結**（google/siglip2-base-patch16-224）:
  E を固定し head のみ学習。
- **EmbeddingGemma 2 凍結**（google/embeddinggemma-2）: 同上。
  特徴抽出は BF16 のみで実施。
- **EmbeddingGemma 2 部分更新**: E の vision エンコーダ最終
  2 層（学習可能 19,273,984 パラメータ）を新規 head と同時に
  学習する。AdamW、encoder lr 1e-5 / head lr 1e-3、cosine
  スケジュール、warmup 3%、実効バッチ 32 ペア
  （micro-batch 8 × 累積 4）、1 epoch、FP32 マスタ重み +
  BF16 autocast。更新範囲は小規模プローブで VRAM 上限
  （≤20 GB）から決めた。学習後は特徴を再抽出して評価する。

![VLM 条件と失敗分類](figures/fig_arch_qwen.png)

*図 3: VLM 条件。生成・パース・失敗分類の流れ。*

**VLM 条件（図 3）。** Qwen3-VL 8B Instruct を BF16・greedy・
batch 1 で使う。生成文は item ごとに原文のまま保存する。

- **Qwen zero-shot**（参照条件、単一ラン）: 追加学習なし。
- **Qwen QLoRA 追加学習**: 共通 20,000 ペア（Task 1 形式の
  Yes/No ペア）で 1 epoch。LoRA rank 16、α 32、dropout 0.05、
  対象 369 モジュール（vision 117 / language 252）、学習可能
  約 52.5M パラメータ（約 0.6%）、lr 1e-4。学習 seed は
  13 + s。公式の学習重み・生ログは非公開のため、本条件は
  「公式方式に倣った追加学習」であり公式再現とは呼ばない。

### 3.4 パースと失敗分類

VLM の生成文は共通パーサで解釈する。パーサは「ANSWER: X」の
形式、または末尾の単独文字を受け付け、Task 2 では A–D のみを
有効とする。解釈できない生成は不正解として数える（分母は
全問のまま）。M7 では保存済み生成文を次の 5 分類に分けた:
no_letter（文字なし）、letter_out_of_range（A–D 外の単独文字）、
multiple_letters（複数の候補文字）、parsed_but_wrong
（解釈できたが不正解）、parsed_correct（解釈できて正解）。

### 3.5 コスト計測

全 run（特徴抽出・学習・推論）にコスト記録を付けた: GPU 機種、
wall 時間、GPU 時間、ピーク VRAM、バッチサイズ、精度区分、
処理件数。主指標は実測 GPU 時間（GPU-h）で、金額換算は
レンタル単価に依存するため本文では扱わない。条件の GPU 時間は
「その条件の評価一式（学習 + 特徴抽出 + 全タスク評価）」の
合計であり、学習曲線セルでは union 特徴抽出の全量を各セルに
計上する規約とした（M6 記録に明記）。

### 3.6 実装と検証

実験コード・設定・manifest は公開予定のリポジトリ
（bluecad-cost-benchmark）にあり、各マイルストーンの結果文書
（M1–M7）が run の一次記録である。M7 では保存済み予測
39 セル（5 条件 × 3 seed、zero-shot は seed 0）について、
件数・ID 集合・保存メトリクスの再計算・M6 報告値との一致を
検証し、全て一致した。本稿の分析は再推論ではなく、この
保存済み予測のみから行った。

## 4. 結果

### 4.1 主比較

表 1 に 3 seed の主比較を示す（seed 別の値は付録 A）。

**表 1: 主比較（3 seed 平均 ± 標準偏差、accuracy_all、回答率は
Task 2 の Qwen FT を除き全て 1.0）**

| 条件 | GPU-h 平均 | Task 1 | Task 1-dim | Task 2 |
|---|---|---|---|---|
| SigLIP 2 凍結 | 0.354 | 0.727 ± 0.002 | 0.693 ± 0.020 | 0.880 ± 0.007 |
| EG2 凍結 | 1.121 | 0.737 ± 0.005 | 0.671 ± 0.007 | 0.859 ± 0.011 |
| EG2 部分更新 | 1.816 | **0.766 ± 0.003** | **0.759 ± 0.002** | **0.950 ± 0.004** |
| Qwen FT | 4.382 | 0.750 ± 0.004 | 0.706 ± 0.027 | 0.230 ± 0.065 † |
| Qwen ZS（単一ラン） | 0.176 | 0.653 | 0.632 | 0.334 |

† Qwen FT の Task 2 回答率は 0.806（seed 別 0.781 / 0.780 / 0.858）。
GPU-h の seed 0 値について、部分更新と Qwen FT は集計表
（main_table.csv）で空欄のため、M5 / M4 の一次記録の合計
（2.198 / 3.222 GPU-h）を用いた。この補完は制限事項にも記す。

部分更新は全タスクで最良かつ標準偏差が最小である。Qwen FT は
Task 1 では部分更新に次ぐ 0.750 を記録するが、Task 2 では
zero-shot（0.334）を下回る 0.230 まで崩れた。凍結条件は
Task 2 でも 0.86–0.88 に達し、エンコーダ間の差は Task 1 では
EG2、Task 2 では SigLIP 2 が上回るなどタスク依存である。

![精度 vs GPU 時間](../../results/m7/figures/cost_accuracy.png)

*図 4: 精度 vs 平均 GPU 時間（タスク別、3 seed 平均 ± 標準偏差）。
部分更新が左上の効率フロンティアに位置し、Qwen FT は Task 2 で
右下に沈む。*

### 4.2 費用対性能の読み

「最高精度から 0.01 以内に入る最も安い条件」は、Task 1
（最高 0.766）でも Task 2（最高 0.950）でも部分更新だけであり、
他条件は圏内に入らない。次点との差は Task 1 で Qwen FT が
0.016、Task 2 で SigLIP 2 凍結が 0.070 である。一方、絶対的な
安さでは SigLIP 2 凍結（0.354 GPU-h、部分更新の約 1/5）が
別格で、精度の代償は Task 1 で −0.039、Task 2 で −0.070 に
とどまる。Qwen FT は最高コストで Task 2 最下位であり、
費用対性能の観点では全タスクで割に合わない。

### 4.3 学習曲線（凍結条件）

![凍結系の学習曲線](../../results/m7/figures/learning_curves.png)

*図 5: 凍結条件の学習曲線（学習ペア数 vs 精度、3 seed 平均）。*

学習ペアを 2,000 から 20,000 に増やすと、Task 1 は SigLIP 2 が
0.682 → 0.727、EG2 が 0.693 → 0.737 と伸びが鈍化するのに対し、
Task 2 は SigLIP 2 が 0.812 → 0.880、EG2 が 0.794 → 0.859 と
10,000 以降も伸び続ける。4.7 節で見るように、Task 2 の後半の
伸びは head が迷っている item に集中している。

### 4.4 Task 2 崩壊の失敗分類

**表 2: Qwen Task 2 の失敗分類（1,000 件中、保存済み生成文から）**

| 条件 | seed | 範囲外の文字 | 解釈できたが不正解 | 解釈できて正解 | 解釈済みの正解率 |
|---|---|---|---|---|---|
| FT | 0 | 219 | 586 | 195 | 0.250 |
| FT | 1 | 220 | 591 | 189 | 0.242 |
| FT | 2 | 142 | 553 | 305 | 0.355 |
| ZS | 0 | 40 | 626 | 334 | 0.348 |

no_letter と multiple_letters は全条件・全 seed で 0 件だった。
範囲外の文字は "E" のような A–D 外の単独文字である。

読み取りは3点ある。第一に、崩壊は形式だけの問題ではない。
seed 0 の誤り 805 件のうち形式失敗は 219 件（27%）で、残りは
解釈できたのに選択が誤っている。しかも解釈済みに限った正解率が
0.250 と 4 択の偶然水準そのものであり、seed 1 も 0.242 と
同様である。第二に、予測文字には偏りがある。FT は D に偏り
（解釈済みの約 32%）、ZS は A・B に強く偏る（B が 44%）。
第三に、seed 2 は形式失敗が 142 件に減り解釈済み正解率も
0.355 と偶然を上回るなど質的に異なるが、学習 manifest・
adapter 設定・評価設定の記録に seed 間の異常は無く（adapter
設定の差は target_modules の並び順のみ）、設定ミスではなく
学習のばらつきと判断する。ZS の失敗は形式失敗 4% で大半が
選択ミスであり、FT の崩壊とは別種である。

### 4.5 誤りの重なりとグループ別内訳

5 条件の予測の重なり（表は付録 C）では、全条件が失敗する
共通の難問は Task 1 で 30–39 件（3–4%）、Task 2 で 6–10 件
（約 1%）しかなく、失敗は条件ごとに異なる item で起きている。
条件間の item 単位の変化を見ると、部分更新の Task 2 改善は
ほぼ一方通行である（凍結に対して直した 98–121 件 / 壊した
13–24 件）。対照的に Qwen FT は Task 2 で ZS に対して
直した 85–157 件 / 壊した 186–233 件と純減が大きい。

Task 1-dim の公式グループ別（seed 0、表 3）では、凍結・部分
更新は hard_neg が最も難しいという共通の傾向を示し、部分更新は
全グループを底上げする。一方 Qwen FT は easy_neg 0.972 /
hard_neg 0.936 と否定側で ZS（0.708 / 0.684）を大きく上回る
反面、positive は 0.396 と ZS（0.568）を大きく下回る。
seed 1/2 でも同じ形で、positive は 0.558 / 0.546、否定側は
0.85–0.94 の範囲にある。Yes/No 形式のペアだけでの追加学習が
「No と答える」方向にモデルを寄せた結果と解釈できる。
Task 1-dim の seed 間ばらつき（0.675 / 0.719 / 0.725）の
主因はこの positive 側のばらつきである。

**表 3: Task 1-dim グループ別精度（seed 0）**

| 条件 | simple | complex | easy_neg | hard_neg | positive |
|---|---|---|---|---|---|
| SigLIP 2 凍結 | 0.662 | 0.678 | 0.772 | 0.580 | 0.664 |
| EG2 凍結 | 0.652 | 0.676 | 0.700 | 0.624 | 0.666 |
| EG2 部分更新 | 0.748 | 0.772 | 0.768 | 0.684 | 0.794 |
| Qwen FT | 0.636 | 0.714 | 0.972 | 0.936 | 0.396 |
| Qwen ZS | 0.590 | 0.674 | 0.708 | 0.684 | 0.568 |

### 4.6 seed 間のばらつきと外れ値

SigLIP 2 の Task 1-dim は seed 0（0.670）が seed 1/2
（0.707 / 0.701）より低い外れ値である。seed 0 が誤った
330 件の内訳は、seed 1・2 も誤る共通難問が 120 件、
seed 0 だけの誤りが 138 件、片方だけが誤るものが 72 件で、
少数の壊れた item ではなく広く散った誤りであり、head 全体が
seed 0 で少し弱い形の外れ方だった。

### 4.7 学習曲線のフォローアップ

curve セルの保存済み予測を使い、10,000 ペア時点と 20,000
ペア時点を item 単位で比べた。Task 2 の伸びは、10,000 時点で
head の上位 2 候補のスコア差（margin）が小さい半分の item に
ほぼ集中する: 低 margin 側では直した 38–50 件・壊れた
16–35 件が動いたのに対し、高 margin 側では直した 0–5 件・
壊れた 0–4 件しか動いていない。確信のある item は 10,000
ペアで既に安定しており、追加データは境界付近を押し上げている。
Task 1 は入れ替わりが激しい割に純増が小さく、SigLIP 2 の
seed 0 では純減（直した 69 / 壊した 84、曲線の 0.740 → 0.725
に対応）も見られた。

## 5. 考察

**なぜ部分更新が勝つのか。** 部分更新は、凍結特徴の上に
タスク固有の方向へエンコーダ末端だけを動かすため、汎用の
視覚表現を壊さずに図面ドメインへの適応が効いたと考えられる。
item 単位でも改善がほぼ一方通行（4.5 節）で、標準偏差が
最小であることと整合する。コストも学習 1.33 GPU 時間 +
再抽出 0.87 GPU 時間（seed 0 実測）と、VLM の追加学習より
大幅に小さい。

**なぜ VLM の追加学習が Task 2 で崩れるのか。** 学習データが
Task 1 形式の Yes/No ペアのみであったため、(i) A–D で答える
形式そのものが弱まり（範囲外の文字 "E" など）、(ii) 形式が
保たれた回答でも選択の信号が偶然水準まで落ちた。さらに
Task 1-dim では「No 寄り」の非対称性が現れた。[1] は LoRA
追加学習について「検証精度は数ポイント改善するがバイアスが
強まる」と報告しており、本稿の Task 1 の改善（0.653 →
0.750）とバイアスの具体化（肯定項目の崩壊）はその所見と
整合する。ただし本稿の条件は学習ペア・採点を統制した再測定
であり、[1] の「quick LoRA」と同一条件ではない点に注意する。
実務的な含意として、複数タスク形式で使う VLM を単一形式の
ペアで追加学習する場合は、形式の混在（A–D 形式の例を含める）
と、形式失敗率の監視が最低限必要である。

**実務家への指針。** 予算が極端に小さい場合は SigLIP 2 凍結 +
共通 head（0.35 GPU 時間）が十分に実用的であり、精度を
最大化したい場合は部分更新（1.82 GPU 時間）が費用対性能の
最適点である。8B 級 VLM の追加学習は、少なくとも本条件では
どちらの目的にも合わない。

## 6. 制限事項

- 単一ベンチマーク・単一データセットでの結果であり、外部
  評価は未実施である。
- GPU 機種は run ごとに異なる（RTX 4090、RTX PRO
  4000 / 4500、L4 級）。GPU 時間は機種差を含むため、機種を
  揃えた再測は今後の課題である。
- 集計表 main_table.csv の seed 0 GPU 時間（部分更新・
  Qwen FT）が空欄であり、4.1 節の平均は M4 / M5 の一次記録で
  補完した。自動集計スクリプトはこの欠落を検出して
  incomplete_costs を返す。
- cumulative_cost.csv の列名と中身に不整合があり（実行 ID が
  GPU 時間の列名の下に入っている）、累積費用の図は作成
  できなかった。
- zero-shot は決定的評価の単一ランであり seed 分散を持たない。
  重なり分析では同一ランを 3 seed 分に再利用した。
- Qwen の追加学習は公式の重み・ログが非公開のため「公式方式に
  倣った」条件であり、公式結果の再現ではない。
- 凍結系の数値は学習済み head の精度であり、[1] の検索
  Recall@1 や zero-shot 文献値とは直接比較できない。

## 7. 結論

共通プロトコルと実測コスト記録のもとで適応戦略を比べると、
勝者は最大のモデルでも最も高価な手法でもなく、凍結
エンコーダの部分更新であった。VLM の追加学習は Yes/No
タスクでは有効でも、学習に使わなかった回答形式のタスクでは
形式と中身の両面で崩壊しうる。精度の数字だけでなく、
コスト記録と失敗の内訳まで含めて初めて、適応戦略の選択は
根拠を持てる。本稿の記録一式（予測・コスト・分析コード）は
リポジトリで公開する予定である。

## データと帰属

本稿は Blueprint2CAD / BlueCAD-30K（Valverde ら [1]、
CC BY 4.0）の公式 item と画像を用いた再測定であり、
データセット自体の権利は原著者らに帰属する。モデルは
各提供元のライセンスに従って使用した。

## 参考文献

[1] A. Valverde, D. Xie-Li, Z. Wang, M. V. Conde.
"Blueprint2CAD: Benchmarking Foundation Models on Engineering
Drawing-to-CAD Render Correspondence." arXiv:2611.02433, 2026.
（WACV 2027 採択、著者らによる）

[2] E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large
Language Models." ICLR 2022. arXiv:2106.09685.

[3] T. Dettmers et al. "QLoRA: Efficient Finetuning of
Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.

[4] M. Tschannen et al. "SigLIP 2: Multilingual
Vision-Language Encoders with Improved Semantic Understanding,
Localization, and Dense Features." 2025. arXiv:2502.14786.

[5] EmbeddingGemma 2 model card, Google.（モデルカードの
正式な書誌情報は TODO: 投稿前に確認）

[6] Qwen3-VL model card / technical report, Qwen Team.
（正式な書誌情報は TODO: 投稿前に確認）

## 付録 A: seed 別精度（accuracy_all）

| 条件 | タスク | seed 0 | seed 1 | seed 2 | GPU-h (s0/s1/s2) |
|---|---|---|---|---|---|
| SigLIP 2 凍結 | Task 1 | 0.725 | 0.728 | 0.728 | 0.165 / 0.449 / 0.449 |
| | Task 1-dim | 0.670 | 0.707 | 0.701 | （同左） |
| | Task 2 | 0.872 | 0.885 | 0.883 | （同左） |
| EG2 凍結 | Task 1 | 0.740 | 0.731 | 0.739 | 0.506 / 1.429 / 1.429 |
| | Task 1-dim | 0.664 | 0.673 | 0.677 | （同左） |
| | Task 2 | 0.853 | 0.872 | 0.852 | （同左） |
| EG2 部分更新 | Task 1 | 0.763 | 0.766 | 0.769 | 2.198* / 1.610 / 1.641 |
| | Task 1-dim | 0.760 | 0.757 | 0.760 | （同左） |
| | Task 2 | 0.954 | 0.946 | 0.951 | （同左） |
| Qwen FT | Task 1 | 0.748 | 0.748 | 0.755 | 3.222* / 5.016 / 4.908 |
| | Task 1-dim | 0.675 | 0.719 | 0.725 | （同左） |
| | Task 2 | 0.195 | 0.189 | 0.305 | （同左） |
| Qwen ZS | Task 1 | 0.653 | — | — | 合計 0.176 |
| | Task 1-dim | 0.632 | — | — | （同左） |
| | Task 2 | 0.334 | — | — | （同左） |

\* seed 0 の GPU-h は M5 / M4 一次記録の合計（集計表では空欄）。
Qwen FT の Task 2 回答率は 0.781 / 0.780 / 0.858、ZS は 0.96、
他は全て 1.0。回答分のみの macro-F1 平均は、SigLIP 2 凍結
0.726 / 0.684 / 0.880、EG2 凍結 0.734 / 0.671 / 0.859、
部分更新 0.765 / 0.759 / 0.950、Qwen FT 0.743 / 0.692 / 0.276、
Qwen ZS 0.647 / 0.630 / 0.297（Task 1 / 1-dim / 2 の順）。

## 付録 B: 失敗分類の全数値

表 2 の分類に、予測文字分布を加えた全数値は
`results/m7/qwen_task2_taxonomy.csv` にある。要点: FT の
解釈済み予測分布は seed 0 で A 169 / B 157 / C 203 / D 252、
seed 1 で A 139 / B 156 / C 224 / D 261、seed 2 で
A 243 / B 140 / C 204 / D 271。ZS は A 315 / B 427 /
C 97 / D 121。正解分布はほぼ均等である。

## 付録 C: 誤りの重なり（1,000 件中）

| タスク | seed | 全条件が失敗 | ちょうど1条件だけ失敗 | 全条件が正解 |
|---|---|---|---|---|
| Task 1 | 0 | 30 | 215 | 381 |
| Task 1 | 1 | 39 | 222 | 395 |
| Task 1 | 2 | 39 | 206 | 404 |
| Task 1-dim | 0 | 16 | 234 | 266 |
| Task 1-dim | 1 | 36 | 238 | 320 |
| Task 1-dim | 2 | 28 | 249 | 314 |
| Task 2 | 0 | 9 | 249 | 86 |
| Task 2 | 1 | 10 | 259 | 82 |
| Task 2 | 2 | 6 | 287 | 115 |

条件は 5 つ（ZS は seed 0 の参照ランを再利用）。Task 2 で
「全条件が正解」が少ないのは Qwen FT の崩壊による。

## 付録 D: 再現性チェックリスト

- モデル revision: SigLIP 2 = 75de2d55（TODO: 完全な
  ハッシュを一次記録から転記）、EmbeddingGemma 2 =
  f9972ac2（同）、Qwen3-VL 8B Instruct =
  0c351dd01ed87e9c1b53cbc748cba10e6187ff3b。
- 学習ペア manifest は seed ごとに固定され、ハッシュは
  リポジトリの manifests/ に記録。
- 全 run のコスト記録は results/cost_log.jsonl（pod 側
  ボリュームに完全な記録、要約は各 M 文書）。
- 分析コード: scripts/verify_m7.py、scripts/analyze_m7.py、
  scripts/cost_reading_m7.py（単体テスト 38 件、1 件 skip）。
- 本稿の検証: 保存済み予測 39 セルの再計算が M6 報告値と
  完全一致（docs/M7_RESULTS.md）。
