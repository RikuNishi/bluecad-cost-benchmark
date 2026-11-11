#set page(
  paper: "us-letter",
  margin: (x: 1.7cm, top: 1.9cm, bottom: 2.1cm),
  columns: 2,
  numbering: "1",
  number-align: center,
)
#set text(font: "Noto Serif CJK JP", size: 9.5pt, lang: "ja")
#set par(justify: true, leading: 0.85em, first-line-indent: 1em, spacing: 0.55em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 1.15em, below: 0.5em,
  text(size: 11.5pt, weight: "bold")[#it.body],
)
#show heading.where(level: 2): it => block(
  above: 0.95em, below: 0.4em,
  text(size: 10pt, weight: "bold")[#it.body],
)
#set list(indent: 0.4em, body-indent: 0.7em, spacing: 0.45em)
#set enum(indent: 0.4em, body-indent: 0.7em, spacing: 0.45em)
#show figure.where(kind: table): set text(size: 8pt)
#show figure.caption: set text(size: 8.5pt)


#place(top, scope: "parent", float: true)[
  #set par(first-line-indent: 0em)
  #align(center)[#text(size: 15.5pt, weight: "bold")[Accuracy per GPU-Hour: A Controlled Cost Benchmark of Adaptation Strategies for CAD Drawing–Render Matching]]
  #v(0.5em)
  #align(center)[#text(size: 11pt)[Riku Imanishi（所属: TODO）]]
  #align(center)[#text(size: 8.5pt, fill: gray)[草稿 — 2026-11-10、2026-11-11 改訂（外部レビューと追加診断を反映）。本稿の数値はすべて本研究の実測値であり、文献値は文献値として区別して表記する。]]
  #v(0.7em)
  #block(inset: (x: 1.1cm))[
    #text(size: 9.5pt, weight: "bold")[Abstract]


    #par(first-line-indent: 0em)[#text(size: 9pt)[We re-measure accuracy against compute cost on the official Blueprint2CAD / BlueCAD-30K benchmark [1] under one common protocol: the same 20,000 training pairs, the same validation set, a common classifier head for encoder-based methods, and accuracy computed over all questions (unanswered questions count as wrong). Across three seeds, partial updating of a frozen embedding encoder (EmbeddingGemma 2; last two vision blocks plus image-side projection and pooling) achieves the best accuracy on all three tasks (Task 1 0.766 ± 0.003, Task 1-dim 0.759 ± 0.002, Task 2 0.950 ± 0.004) at a mean cost of 1.82 GPU-hours per run, while QLoRA fine-tuning of an 8B vision-language model (Qwen3-VL 8B) costs more (4.38 GPU-hours) and collapses on the multiple-choice task (0.230 ± 0.065). A failure taxonomy of the saved generations, extended with a letter-shift diagnostic, shows the collapse is primarily a label-mapping failure: 60–78% of the fine-tuned model's answers are exactly one letter past the ground truth — the same failure mode the benchmark paper reports for its own fine-tune — and shifting the answers back by one letter (a diagnostic only, not a score) recovers 0.60–0.78 accuracy. In addition, 14–22% of generations fall outside the A–D answer format. On the dimensioned-drawing task, the fine-tuned model is asymmetric: it answers negative items almost perfectly (0.85–0.97) while positive items drop to 0.40–0.56. Frozen encoders with the common head remain competitive at lower recorded cost (0.35–1.12 GPU-hours per run under our recording conventions, against 1.82 for the partial update and 4.38 for the QLoRA fine-tune). We also disclose and quantify an overlap between our training pairs and the official evaluation items (Section 4.9): on items that reference no training or validation part, the partial update remains best on every task, while the fine-tuned model's Task 1 accuracy drops from 0.750 to 0.718. All runs carry a measured cost record; we report GPU-hours, not estimates, and every cost claim is scoped to the hardware and accounting conventions actually used.]]


    #par(first-line-indent: 0em)[#text(size: 9pt)[*Keywords:* CAD, engineering drawings, benchmark, compute cost, vision-language models, failure analysis]]


  ]
]


= 1. はじめに

工学図面と CAD レンダリングの対応付けは、部品検索・設計再利用・図面のデジタル化の入口となるタスクである。Valverde ら [1] はBlueprint2CAD ベンチマークとそのデータ層 BlueCAD-30K を公開し、基盤モデルの現状を初めて体系的に測った。彼らの報告（文献値）では、凍結エンコーダによる検索は視点が既知の場合で最大 65% のRecall\@1、未知の場合で 43% にとどまり、6 つの zero-shot視覚言語モデル（VLM）の検証精度は 0.52–0.67 の範囲で（寸法条件を含む著者らの要約値）、モデル固有の Yes/No バイアスを伴う。さらに LoRA による短時間の追加学習は検証精度を数ポイント改善する一方で、そのバイアスを強めることが報告されている。

しかし実務家が次に知りたいのは「どの適応戦略が、どれだけの計算コストで、どれだけ良くなるのか」であり、この問いは元のベンチマークでは主対象ではない。手法ごとの学習量・検証集合・採点方法が揃っていなければ、精度差と費用差を同じ土俵で読むことはできない。なお [1] の v2 は、追加学習後の Task 2 精度が偶然水準まで低下すること、その失敗の形（範囲外の文字 "E" の回答、正解より一文字後ろの文字を出す傾向）、Task 1 の正例・負例別の内訳まで報告している。本稿の失敗分析はこの既報を踏まえ、共通プロトコルでの再測定として位置づける。

本稿はこの2点に答える再測定である。貢献は次の4点である。

+ *共通プロトコルによる費用対性能の主比較。* 全条件に同じ20,000 学習ペア、同じ検証集合、同じ採点（分母は全問）を適用し、全 run に GPU 時間の実測記録を付けた。3 seed の平均では、EmbeddingGemma 2 の部分更新が全タスクで最良の平均精度を示し、平均 1.82 GPU 時間であった。最も高コストな Qwen3-VL 8B のQLoRA 追加学習（4.38 GPU 時間）は Task 1 では2位につける一方、Task 2 では最下位に崩れた。

+ *Task 2 崩壊の失敗分類と文字ずれ診断。* 保存済み生成文を共通パーサで分類し、さらに予測文字と正解文字のずれを数えた。崩壊の主因は選択能力の喪失ではなく、[1] も報告する「正解より一文字後ろを答える」ラベル対応の失敗であった（追加学習条件の60–78%）。文字を 1 つ戻す診断では精度が 0.60–0.78 まで回復する。zero-shot の失敗は形式失敗が 4% のみで質が異なる。

+ *非対称なバイアスの特定。* Task 1-dim のグループ別内訳で、追加学習モデルが否定項目では 0.85–0.97 に達する一方、肯定項目では 0.40–0.56 に落ちることを示した。[1] の「バイアスが強まる」という所見を、本稿の統制条件でも追認し、公式 tier 単位で定量化したものである。

+ *再現可能な記録一式。* 予測・メトリクス・コスト記録をrun ごとに保存し、主比較 39 セルの予測由来指標を保存済み予測から再計算で検証した（全て一致）。あわせて、本研究の学習ペアと公式評価 item の部品重複を全量照合し、層別精度まで開示する。

= 2. 関連研究

*Blueprint2CAD / BlueCAD-30K [1].* Zero-to-CAD の 29,925モデルに描画パイプラインを適用し、6 視点のレンダリング、線画図面（寸法なし／あり）、表面種別メタデータを付与したデータ層と、図面–レンダ対応付けの検証・検索タスクからなるベンチマーク。本稿はその公式 item を配布どおりに用い、Task 1（線画）、Task 1-dim（寸法入り図面）、Task 2（4 択）を扱う（Task 1-dimは [1] の Task 1 + DIM 条件に当たる）。[1] の数値は文献値であり、本稿の実測値とは表を分ける。特に、[1] の凍結エンコーダの数値は検索 Recall\@1 であって、本稿の学習済み headによる検証精度とは直接比較できない。同様に [1] の VLM 数値はzero-shot 検証精度であり、本稿の zero-shot 条件（Qwen3-VL 8B単独）とはモデル集合が異なる。また [1] の凍結エンコーダ検索は10,000 部品のギャラリー（各 6 視点）に対するもので、用いるSigLIP 2 も So400m/14・384px と、本稿の base-patch16-224 より大きい。さらに [1] には Task 3（complexity probe）があるが、本稿では扱わない。

*適応戦略。* 凍結エンコーダへの軽量 head の学習、部分的なパラメータ更新、LoRA [2] / QLoRA [3] による VLM の追加学習は、いずれも現場で使われる代表的な適応手段である。本稿はこれらを新規手法としてではなく、同一条件で測る対象として扱う。エンコーダには SigLIP 2 [4] と EmbeddingGemma 2 [5]、VLM にはQwen3-VL 8B [6] を用いた。

*コストを軸にした評価。* 精度だけでなく学習・推論コストを併記する研究は増えているが、コストの定義（GPU 機種、実測か見積りか、特徴抽出を含むか）が手法ごとに異なると比較は崩れる。本稿では全 run に同一形式のコスト記録（GPU 機種、wall 時間、GPU 時間、ピーク VRAM、バッチサイズ、精度、処理件数）を義務付け、主指標を実測 GPU 時間に統一した。

= 3. 方法

== 3.1 データとタスク

公式 item JSON を配布どおりに使用し、画像パスだけを手元の配置に書き換えた（AGENTS 契約どおり、問題の組み合わせ・候補順・正解・視点は一切再生成していない）。データ監査（M1）の結果、配布split の ID 数は train 26,039 / validation 2,890 / test 1,000で、[1] v2 が記す 27,479 / 1,446 / 1,000 とは異なる（配布物と論文記載の差として記録する）。各タスクは 1,000 問で、正解分布はほぼ均等である。一方で train リストには配布データの無い4 件の ID が含まれ、shard-0000 の標本 500 部品では renders のtransforms.json が欠けていた（本稿の評価には影響しない）。

もう 1 点、学習データと評価問題の関係を開示する。公式評価item は主対象の 1,000 部品以外も負例・候補として参照しており（Task 1 系は 1,216 部品、Task 2 は 1,714 部品）、本研究の学習 manifest に現れる部品と重なる。seed 0 では Task 1 の121 問・Task 2 の 549 問が学習に使った部品を参照し、検証manifest との重なりも Task 1 で 16 問ある。したがって「評価入力は全て未見の部品」とは言えない。この重なりの影響は4.9 節で層別に示す。なお本件は、データセットが train / val に分かれているかどうかとは別の話で、こちらの学習ペアと公式の評価問題が、参照する部品のレベルで重なっているという関係である。

タスクは次の3つである。*Task 1*: 線画図面とレンダリングのペアが同一部品かどうかを Yes/No で答える。*Task 1-dim*: 寸法入り図面を用いる同形式のタスクで、item には group（simple / complex、各 500 件）と meta.tier（easy\_neg 250 件 /hard\_neg 250 件 / positive 500 件）の公式グループ情報がある。*Task 2*: 1 枚の図面に対し 4 枚の候補レンダリングから一致するものを選ぶ 4 択タスク（[1] 本文は right-view と記すが、配布JSON の画像は図面・候補とも 000.png であり、公式コードの候補順も front から始まる。本稿は配布 JSON をそのまま用い、視点名の物理的な保証はしない）。

== 3.2 共通プロトコル

#place(top, scope: "parent", float: true)[
  #align(center)[#image("figures/fig_protocol.png", width: 82%)]
  #text(size: 8.5pt)[図 1: 共通プロトコル。全条件が同じ学習ペア・同じ検証集合・同じ採点器で測られる。]
]

- 学習ペアは共通 manifest の 20,000 ペア（正例・負例 50/50）。seed s の run は seed s の manifest を使う。

- 検証は常に seed 0 の 2,000 ペア。しきい値調整にこの検証集合を使うのはエンコーダ条件の head のみで、その旨を run 記録に明記する。Qwen 条件は検証によるモデル選択を行わず、最終epoch の adapter をそのまま評価する。

- 採点の分母は全問。回答率と「回答分のみの精度」を併記し、評価した item ID を run ごとに保存する。

- seed は 3 本（0, 1, 2）。seed 0 のセルは M3–M5 の既存 run を再利用し、主比較で新規に GPU を使ったのは seed 1–2 のみである。

== 3.3 条件

#place(top, scope: "parent", float: true)[
  #align(center)[#image("figures/fig_head_detail.png", width: 82%)]
  #text(size: 8.5pt)[図 2: エンコーダ条件。エンコーダは既製の事前学習モデルとして1 つの箱に省略し、本研究が設計・学習する共通ペア head を詳しく描く。部分更新では vision エンコーダ最終 2 ブロックと画像側projection / pooling が学習される。]
]

*エンコーダ条件（図 2）。* 図面とレンダリングを同じ視覚エンコーダ E で特徴量 d, r に変換し、concat [d, r, |d−r|, d⊙r]を共通のロジスティック head（Adam lr 1e-3、weight decay 1e-4、200 epoch）に入力してペアスコアを得る。Task 1 / 1-dim は検証集合で調整したしきい値で二値化し、Task 2 は 4 候補のスコアのargmax で選ぶ。この head は学習されるため、凍結条件をzero-shot とは呼ばない。

- *SigLIP 2 凍結*（google/siglip2-base-patch16-224）:E を固定し head のみ学習。

- *EmbeddingGemma 2 凍結*（google/embeddinggemma-2）: 同上。特徴抽出は BF16 のみで実施。

- *EmbeddingGemma 2 部分更新*: E の vision エンコーダ最終2 ブロックに加え、画像側の pooler・embedding projection・事前 norm を更新する（エンコーダ側の学習可能パラメータ19,273,984）。学習時は仮の head（scaffold）と共同学習し、学習後にその head は捨て、特徴を再抽出してから新規の共通head を学習する二段階である。AdamW、encoder lr 1e-5 /head lr 1e-3、cosine スケジュール、warmup 3%、実効バッチ32 ペア（micro-batch 8 × 累積 4）、1 epoch、FP32 マスタ重み + BF16 autocast。更新範囲は小規模プローブで VRAM上限（≤20 GB）から決めた。

#place(top, scope: "parent", float: true)[
  #align(center)[#image("figures/fig_lora_detail.png", width: 82%)]
  #text(size: 8.5pt)[図 3: VLM 条件。ベースモデル（Qwen3-VL の視覚エンコーダと言語モデル）は塔として省略し、追加学習条件でのみ入る LoRAアダプタ（凍結した射影 W に対する低ランクの並列経路）を拡大して描く。生成文は共通パーサで解釈され、解釈できない生成は不正解として数える。失敗分類の内訳は表 2 に示す。]
]

*VLM 条件（図 3）。* Qwen3-VL 8B Instruct を BF16・greedy・batch 1 で使う。生成文は item ごとに原文のまま保存する。画像処理については、学習時は公式 fine-tune 設定の processor（min / max pixels = 200,704 / 802,816）を用い、評価時はzero-shot が base の processor、追加学習が学習時に保存したprocessor を使う。512 × 512 への明示的なリサイズは行っておらず、実行時の実画像 grid / token 数は記録していない（[1] は入力 512 × 512 と記す。この差は制限事項に挙げる）。

- *Qwen zero-shot*（参照条件、単一ラン）: 追加学習なし。本研究では greedy で評価したが、[1] の zero-shot は vLLM の既定 sampling であり、生成設定は同一ではない。

- *Qwen QLoRA 追加学習*: 共通 20,000 ペア（Task 1 形式のYes/No ペア）で 1 epoch。LoRA rank 16、α 32、dropout 0.05、対象 369 モジュール（vision 117 / language 252）、学習可能約 52.5M パラメータ（約 0.6%）、lr 1e-4。学習 seed は13 + s。LoRA は BF16 の base に merge してから評価する。学習時の system / user prompt と画像順（図面→レンダリング）は [1] の Table 21 と公式コードに合わせた。一方、学習ペアは[1] の生成方式（各部品 2 視点の図面、silhouette 近傍からのhard negative 70%、500 ペア検証での best-loss checkpoint選択）ではなく本研究の共通 manifest（front 図面のみ、負例はランダム）であり、検証による checkpoint 選択も行わない。したがって本条件は「LoRA / QLoRA の主要設定を [1] に倣った追加学習」であって、公式再現とは呼ばない。

== 3.4 パースと失敗分類

VLM の生成文は本研究の共通パーサ（公式実装を参考にした独自scorer）で解釈する。Task 1 では最初の Yes / No を取り、Task 2 では「ANSWER: X」の形式を優先し、なければ末尾の単独文字が A–D の場合に受理する。公式実装の normalize とは細部が異なる（公式は A–Y を受理する等）ため、同一の生成文でも正誤が変わりうる。解釈できない生成は不正解として数える（分母は全問のまま）。M7 では保存済み生成文を次の 5 分類に分けた: no\_letter（文字なし）、letter\_out\_of\_range（A–D 外の単独文字）、multiple\_letters（複数の候補文字）、parsed\_but\_wrong（解釈できたが不正解）、parsed\_correct（解釈できて正解）。

== 3.5 コスト計測

全 run（特徴抽出・学習・推論）にコスト記録を付けた: GPU 機種、wall 時間、GPU 時間、ピーク VRAM、バッチサイズ、精度区分、処理件数。主指標は実測 GPU 時間（GPU-h）で、金額換算はレンタル単価に依存するため本文では扱わない。ただし GPU 機種はrun ごとに異なり、工程の計測境界（モデルロードの扱い等）も完全には揃っていないため、GPU 時間は「本研究の記録規約の下での wall 時間の記録」として読み、機種を揃えた厳密な効率比較とは位置づけない。条件の GPU 時間は「その条件の評価一式（学習 + 特徴抽出 + 全タスク評価）」の合計であり、凍結条件のseed 1–2 では 3 seed 共通の union 特徴抽出の全量を各セルに計上する規約とした（M6 記録に明記）。これは実際に消費したGPU 時間の配賦であり、条件間で共有された抽出を seed ごとに重ねて数えている点に注意する。

== 3.6 実装と検証

実験コード・設定・manifest は公開予定のリポジトリ（bluecad-cost-benchmark）にあり、各マイルストーンの結果文書（M1–M7）が run の一次記録である。M7 では保存済み予測39 セル（5 条件 × 3 seed、zero-shot は seed 0）について、件数・ID 集合・保存メトリクスの再計算・M6 報告値との一致を検証し、全て一致した。本稿の分析は再推論ではなく、この保存済み予測のみから行った。なお、この検証の対象は予測由来の指標であり、コスト記録や画像内容そのものは含まない。

= 4. 結果

== 4.1 主比較

表 1 に 3 seed の主比較を示す（seed 別の値は付録 A）。

#figure(
  table(columns: 5, stroke: none,
    table.hline(stroke: 0.8pt),
    [*条件*], align(center)[*GPU-h 平均*], align(center)[*Task 1*], align(center)[*Task 1-dim*], align(center)[*Task 2*],
    table.hline(stroke: 0.4pt),
    [SigLIP 2 凍結], align(center)[0.354], align(center)[0.727 ± 0.002], align(center)[0.693 ± 0.020], align(center)[0.880 ± 0.007],
    [EG2 凍結], align(center)[1.121], align(center)[0.737 ± 0.005], align(center)[0.671 ± 0.007], align(center)[0.859 ± 0.011],
    [EG2 部分更新], align(center)[1.816], align(center)[*0.766 ± 0.003*], align(center)[*0.759 ± 0.002*], align(center)[*0.950 ± 0.004*],
    [Qwen FT], align(center)[4.382], align(center)[0.750 ± 0.004], align(center)[0.706 ± 0.027], align(center)[0.230 ± 0.065 †],
    [Qwen ZS（単一ラン）], align(center)[0.176], align(center)[0.653], align(center)[0.632], align(center)[0.334],
    table.hline(stroke: 0.8pt),
  ),
  caption: [表 1: 主比較（3 seed 平均 ± 標準偏差、accuracy\_all。回答率はTask 2 で Qwen FT 平均 0.806 / Qwen ZS 0.960、他は全て 1.0）],
)

† Qwen FT の Task 2 回答率は 0.806（seed 別 0.781 / 0.780 / 0.858）。GPU-h の seed 0 値について、部分更新と Qwen FT は集計表（main\_table.csv）で空欄のため、M5 / M4 の一次記録の合計（2.198 / 3.222 GPU-h）を用いた。この補完は制限事項にも記す。

部分更新は全タスクで最高の平均精度である。seed 間の標準偏差も小さいが、Task 1 では SigLIP 2 凍結（±0.002）が最小である。Qwen FT はTask 1 では部分更新に次ぐ 0.750 を記録するが、Task 2 ではzero-shot（0.334）を下回る 0.230 まで崩れた。凍結条件はTask 2 でも 0.86–0.88 に達し、エンコーダ間の差は Task 1 ではEG2、Task 2 では SigLIP 2 が上回るなどタスク依存である。

#figure[
  #align(center)[#image("../../results/m7/figures/cost_accuracy.png", width: 100%)]
  #text(size: 8.5pt)[図 4: 精度 vs 平均 GPU 時間（タスク別、3 seed 平均 ± 標準偏差）。部分更新が左上の効率フロンティアに位置し、Qwen FT は Task 2 で右下に沈む。]
]

== 4.2 費用対性能の読み

本節の読みは、本研究の記録規約（機種混在・配賦規約を含む）の下での記述的な観察であり、運用上の最適性を一般化するものではない。「最高精度から 0.01 以内」という物差しも、実務上の許容差を実証した基準ではなく、本稿での条件選定のための記述的な規則である。その規則では、Task 1（最高 0.766）でも Task 2（最高 0.950）でも部分更新だけが圏内に入る。次点との差はTask 1 で Qwen FT が 0.016、Task 2 で SigLIP 2 凍結が 0.070である。記録上の絶対的な安さでは SigLIP 2 凍結（0.354 GPU-h）が最も小さく、精度の代償は Task 1 で −0.039、Task 2 で−0.070 にとどまる。Qwen FT は記録上の GPU 時間が最大で、Task 2 精度は最低であった。

== 4.3 学習曲線（凍結条件）

#figure[
  #align(center)[#image("../../results/m7/figures/learning_curves.png", width: 100%)]
  #text(size: 8.5pt)[図 5: 凍結条件の学習曲線（学習ペア数 vs 精度、3 seed 平均）。]
]

学習ペアを 2,000 から 20,000 に増やすと、Task 1 は SigLIP 2 が0.682 → 0.727、EG2 が 0.693 → 0.737 と伸びが鈍化するのに対し、Task 2 は SigLIP 2 が 0.812 → 0.880、EG2 が 0.794 → 0.859 と10,000 以降も伸び続ける。4.8 節で見るように、Task 2 の後半の伸びは head が迷っている item に集中している。

== 4.4 Task 2 崩壊の失敗分類

#figure(
  table(columns: 6, stroke: none,
    table.hline(stroke: 0.8pt),
    [*条件*], align(center)[*seed*], align(center)[*範囲外の文字*], align(center)[*解釈できたが不正解*], align(center)[*解釈できて正解*], align(center)[*解釈済みの正解率*],
    table.hline(stroke: 0.4pt),
    [FT], align(center)[0], align(center)[219], align(center)[586], align(center)[195], align(center)[0.250],
    [FT], align(center)[1], align(center)[220], align(center)[591], align(center)[189], align(center)[0.242],
    [FT], align(center)[2], align(center)[142], align(center)[553], align(center)[305], align(center)[0.355],
    [ZS], align(center)[0], align(center)[40], align(center)[626], align(center)[334], align(center)[0.348],
    table.hline(stroke: 0.8pt),
  ),
  caption: [表 2: Qwen Task 2 の失敗分類（1,000 件中、保存済み生成文から）],
)

no\_letter と multiple\_letters は全条件・全 seed で 0 件だった。範囲外の文字は "E" のような A–D 外の単独文字である。

読み取りは3点ある。第一に、崩壊は形式だけの問題ではない。seed 0 の誤り 805 件のうち形式失敗は 219 件（27%）で、残りは解釈できたのに選択が誤っている。解釈済みに限った正解率が0.250 と 4 択の偶然水準そのものであり、seed 1 も 0.242 と同様である。ただしこれは「選択の信号が消えた」ことを意味しない。次節の文字ずれ診断が示すように、誤りの大半は正解より一文字後ろへの系統的なずれである。第二に、予測文字には偏りがある。FT は D に偏り（解釈済みの約 32%）、ZS は A・B に強く偏る（B が 44%）。第三に、seed 2 は形式失敗が 142 件に減り解釈済み正解率も 0.355 と偶然を上回るなど質的に異なるが、学習 manifest・adapter 設定・評価設定の記録に seed 間の異常は見当たらなかった（adapter 設定の差は target\_modules の並び順のみ）。seed 2 が他と異なる理由は未特定である。ZS の失敗は形式失敗 4% で大半が選択ミスであり、FT の崩壊とは別種である。

== 4.5 文字ずれ診断（Task 2）

[1] は自らの追加学習モデルについて、範囲外の "E" 回答が12.9%、正解より一文字後ろの文字を出す回答が 54.1% あったと報告し、図面自体を最初のラベル付き候補として数える挙動を示唆した（文献値）。本稿の保存済み生成文でも同じ集計を行った（表 3、全数値は results/review\_followup/）。

#figure(
  table(columns: 6, stroke: none,
    table.hline(stroke: 0.8pt),
    [*条件*], align(center)[*seed*], align(center)[*公式スコア*], align(center)[*一文字後ろ*], align(center)[*一文字前*], align(center)[*戻し採点（診断値）*],
    table.hline(stroke: 0.4pt),
    [FT], align(center)[0], align(center)[0.195], align(center)[733], align(center)[24], align(center)[0.733],
    [FT], align(center)[1], align(center)[0.189], align(center)[777], align(center)[12], align(center)[0.777],
    [FT], align(center)[2], align(center)[0.305], align(center)[599], align(center)[20], align(center)[0.599],
    [ZS], align(center)[0], align(center)[0.334], align(center)[332], align(center)[99], align(center)[0.332],
    table.hline(stroke: 0.8pt),
  ),
  caption: [表 3: Qwen Task 2 の文字ずれ（1,000 件中）],
)

本研究の追加学習条件では、全回答に占める「一文字後ろ」の割合は 73.3% / 77.7% / 59.9% と、[1] の報告よりさらに強く現れた。"E" 回答（219 / 220 / 142 件）は、正解が D のときに一文字後ろへずれた結果と一致する（混同行列はresults/review\_followup/shift\_confusion.csv）。予測文字を一律に 1 つ戻す診断採点（公式スコアではなく診断値）では、精度は 0.733 / 0.777 / 0.599 まで回復する。zero-shot では一文字後ろが 33.2% あるものの、戻し採点は 0.332 と元の0.334 と変わらず、系統的なずれとは言えない。以上から、Task 2 崩壊の主因は幾何的な対応能力の喪失ではなく、ラベル対応の失敗であると結論する。追加学習モデルは候補の選択自体は概ねできているが、文字との対応付けが 1 つずれている。

== 4.6 誤りの重なりとグループ別内訳

5 条件の予測の重なり（表は付録 C）では、全条件が失敗する共通の難問は Task 1 で 30–39 件（3–4%）、Task 2 で 6–10 件（約 1%）しかなく、失敗は条件ごとに異なる item で起きている。条件間の item 単位の変化を見ると、部分更新の Task 2 改善はほぼ一方通行である（凍結に対して直した 98–121 件 / 壊した13–24 件）。対照的に Qwen FT は Task 2 で ZS に対して直した 85–157 件 / 壊した 186–233 件と純減が大きい。

Task 1-dim の公式グループ別（seed 0、表 4）では、凍結・部分更新は hard\_neg が最も難しいという共通の傾向を示し、部分更新は全グループを底上げする。一方 Qwen FT は easy\_neg 0.972 /hard\_neg 0.936 と否定側で ZS（0.708 / 0.684）を大きく上回る反面、positive は 0.396 と ZS（0.568）を大きく下回る。seed 1/2 でも同じ形で、positive は 0.558 / 0.546、否定側は0.85–0.94 の範囲にある。Yes/No 形式のペアだけでの追加学習が「No と答える」方向にモデルを寄せた結果と解釈できる。Task 1-dim の seed 間ばらつき（0.675 / 0.719 / 0.725）の主因はこの positive 側のばらつきである。

#figure(
  table(columns: 6, stroke: none,
    table.hline(stroke: 0.8pt),
    [*条件*], align(center)[*simple*], align(center)[*complex*], align(center)[*easy\_neg*], align(center)[*hard\_neg*], align(center)[*positive*],
    table.hline(stroke: 0.4pt),
    [SigLIP 2 凍結], align(center)[0.662], align(center)[0.678], align(center)[0.772], align(center)[0.580], align(center)[0.664],
    [EG2 凍結], align(center)[0.652], align(center)[0.676], align(center)[0.700], align(center)[0.624], align(center)[0.666],
    [EG2 部分更新], align(center)[0.748], align(center)[0.772], align(center)[0.768], align(center)[0.684], align(center)[0.794],
    [Qwen FT], align(center)[0.636], align(center)[0.714], align(center)[0.972], align(center)[0.936], align(center)[0.396],
    [Qwen ZS], align(center)[0.590], align(center)[0.674], align(center)[0.708], align(center)[0.684], align(center)[0.568],
    table.hline(stroke: 0.8pt),
  ),
  caption: [表 4: Task 1-dim グループ別精度（seed 0）],
)

== 4.7 seed 間のばらつきと外れ値

SigLIP 2 の Task 1-dim は seed 0（0.670）が seed 1/2（0.707 / 0.701）より低い外れ値である。seed 0 が誤った330 件の内訳は、seed 1・2 も誤る共通難問が 120 件、seed 0 だけの誤りが 138 件、片方だけが誤るものが 72 件で、少数の壊れた item ではなく広く散った誤りであり、head 全体がseed 0 で少し弱い形の外れ方だった。

== 4.8 学習曲線のフォローアップ

curve セルの保存済み予測を使い、10,000 ペア時点と 20,000ペア時点を item 単位で比べた。Task 2 の伸びは、10,000 時点でhead の上位 2 候補のスコア差（margin）が小さい半分の item にほぼ集中する: 低 margin 側では直した 38–50 件・壊れた16–35 件が動いたのに対し、高 margin 側では直した 0–5 件・壊れた 0–4 件しか動いていない。確信のある item は 10,000ペアで既に安定しており、追加データは境界付近を押し上げている。Task 1 は入れ替わりが激しい割に純増が小さく、SigLIP 2 のseed 0 では純減（直した 69 / 壊した 84、曲線の 0.740 → 0.725に対応）も見られた。

== 4.9 学習データと評価問題の重なり（層別精度）

3.1 節で開示した部品の重なりについて、各 item を「学習manifest の部品を参照する（train 層）」「検証 manifest のみ（val 層）」「どちらも参照しない（clean 層）」に分け、全条件の精度を層別に再集計した。全 39 セルの全体精度は M6 報告値と完全一致した（詳細は results/review\_followup/overlap\_strata.csv）。

要点は 3 つある。第一に、エンコーダ条件では train 層の精度がclean 層を上回ることはなく（Task 1 の 3 seed 平均で SigLIP 2 0.680 vs 0.740、EG2 0.663 vs 0.756、部分更新 0.735 vs 0.777）、重なりが凍結・部分更新の数字を持ち上げた形跡はない。clean 層だけでも部分更新が全タスクで最良である。第二に、Qwen 追加学習は対照的で、Task 1 の train 層では 0.93–0.98 に達する一方、clean 層では 0.718（全体 0.750）まで下がる。学習したペアそのもの、あるいは同じ部品のペアへの接触がTask 1 の見かけを押し上げており、未見部品だけでの比較ではQwen FT は EG2 凍結（clean 0.756）も下回る。第三に、Task 2ではどの条件でも層間の差は小さく（部分更新 train 0.952 /clean 0.949）、4.5 節の文字ずれという主因の説明と整合する。val 層は件数が少なく（Task 1 系 16 件、Task 2 34–44 件）、解釈は留保する。

= 5. 考察

*なぜ部分更新が勝つのか。* 部分更新は、凍結特徴の上にタスク固有の方向へエンコーダ末端だけを動かすため、汎用の視覚表現を壊さずに図面ドメインへの適応が効いたと考えられる。item 単位でも改善がほぼ一方通行（4.6 節）で、seed 変動の小ささと整合する。コストも学習 1.33 GPU 時間 +再抽出 0.87 GPU 時間（seed 0 実測）と、VLM の追加学習より大幅に小さい。

*なぜ VLM の追加学習が Task 2 で崩れるのか。* 学習データがTask 1 形式の Yes/No ペアのみであったため、4 択の回答形式との対応付けが壊れた。4.5 節の診断では、失敗の主因は選択能力の喪失ではなく「正解より一文字後ろ」を出すラベル対応のずれであり、これは [1] が自らの追加学習モデルで報告した失敗の形と一致する。[1] は LoRA 追加学習について「検証精度は数ポイント改善するがバイアスが強まる」と報告しており、本稿のTask 1 の改善（0.653 → 0.750、ただし clean 層では 0.718）とTask 1-dim の非対称性（肯定項目の落ち込み）はその所見と整合する。ただし本稿の条件は学習ペア・採点を統制した再測定であり、[1] の追加学習と同一条件ではない点に注意する。実務的な含意として、複数タスク形式で使う VLM を単一形式のペアで追加学習する場合は、形式の混在（A–D 形式の例を含める）と、形式失敗率に加えて文字ずれの監視が必要である。

*実務家への指針。* 本研究の記録条件の範囲では、記録コストの小さい SigLIP 2 凍結 + 共通 head（0.35 GPU 時間）と、精度最大の部分更新（1.82 GPU 時間）が両端の候補になる。8B 級VLM の追加学習は、本条件ではコスト・精度のどちらでも優位が無かった。ただし外部評価や実運用の遅延・累積費用は測っていないため、この指針は本ベンチマーク内の観察に限る。

= 6. 制限事項

- 単一ベンチマーク・単一データセットでの結果であり、外部評価は未実施である。

- 本研究の学習ペアと公式評価 item の間には参照部品の重なりがある（3.1 節、4.9 節）。主結論は clean 層でも変わらないが、Qwen 追加学習の Task 1 は全体値が重なりに押し上げられている。

- 配布 split の ID 数は [1] v2 の記載と異なる。どちらが正かは配布元の今後の更新を待つ。

- GPU 機種は run ごとに異なる（RTX 4090、RTX PRO4000 / 4500）。GPU 時間は機種差と計測境界の差を含むため、効率の厳密比較には機種を揃えた再測が必要である。

- M5（部分更新 seed 0）の工程別 GPU 時間の合計（約 2.20GPU-h）と、当時記録した pod 実働（約 1.85 時間）が整合しない。コスト台帳（ボリューム側 results/costs.jsonl）には時刻・pod の識別情報が無く、原因は未解消である。部分更新の平均 GPU 時間はこの不確実さを含む。

- Qwen の評価入力の実サイズ（grid / token 数）は未記録で、[1] の記す 512 × 512 と一致していたかは確認できない。

- 本研究の scorer は公式実装と細部が異なり、同じ生成文でも正誤が変わりうる（3.4 節）。

- 集計表 main\_table.csv の seed 0 GPU 時間（部分更新・Qwen FT）が空欄であり、4.1 節の平均は M4 / M5 の一次記録で補完した。自動集計スクリプトはこの欠落を検出してincomplete\_costs を返す。

- cumulative\_cost.csv の列名と中身に不整合があり（実行 ID がGPU 時間の列名の下に入っている）、累積費用の図は作成できなかった。

- zero-shot は決定的評価の単一ランであり seed 分散を持たない。重なり分析では同一ランを 3 seed 分に再利用した。

- Qwen の追加学習は公式の重み・ログが非公開のため「LoRA の主要設定を [1] に倣った」条件であり、公式結果の再現ではない。学習ペアの生成方式と checkpoint 選択も [1] と異なる（3.3 節）。

- 凍結系の数値は学習済み head の精度であり、[1] の検索Recall\@1 や zero-shot 文献値とは直接比較できない。

= 7. 結論

共通プロトコルと実測コスト記録のもとで適応戦略を比べると、勝者は最大のモデルでも最も高価な手法でもなく、凍結エンコーダの部分更新であった。VLM の追加学習は Yes/Noタスクでは有効でも、学習に使わなかった回答形式のタスクでは、選択能力ではなくラベル対応の形で崩壊しうる。その崩壊は、失敗分類と文字ずれ診断まで行って初めて正しく理解できる。精度の数字だけでなく、コスト記録と失敗の内訳まで含めて初めて、適応戦略の選択は根拠を持てる。本稿の記録一式（予測・コスト・分析コード）はリポジトリで公開する予定である。

= データと帰属

本稿は Blueprint2CAD / BlueCAD-30K（Valverde ら [1]、CC BY 4.0）の公式 item と画像を用いた再測定であり、データセット自体の権利は原著者らに帰属する。モデルは各提供元のライセンスに従って使用した。

= 参考文献

[1] A. Valverde, D. Xie-Li, Z. Wang, M. V. Conde. "Blueprint2CAD: Benchmarking Foundation Models on Engineering Drawing-to-CAD Render Correspondence." arXiv:2611.02433v2, 5 November 2026.（venue は著者らの README によれば WACV 2027 LVR workshop）

[2] E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.

[3] T. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.

[4] M. Tschannen et al. "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features." 2025. arXiv:2502.14786.

[5] EmbeddingGemma 2 model card, Google.（モデルカードの正式な書誌情報は TODO: 投稿前に確認）

[6] Qwen3-VL model card / technical report, Qwen Team.（正式な書誌情報は TODO: 投稿前に確認）

= 付録 A: seed 別精度（accuracy\_all）

#figure(
  table(columns: 6, stroke: none,
    table.hline(stroke: 0.8pt),
    [*条件*], align(center)[*タスク*], align(center)[*seed 0*], align(center)[*seed 1*], align(center)[*seed 2*], align(center)[*GPU-h (s0/s1/s2)*],
    table.hline(stroke: 0.4pt),
    [SigLIP 2 凍結], align(center)[Task 1], align(center)[0.725], align(center)[0.728], align(center)[0.728], align(center)[0.165 / 0.449 / 0.449],
    [], align(center)[Task 1-dim], align(center)[0.670], align(center)[0.707], align(center)[0.701], align(center)[（同左）],
    [], align(center)[Task 2], align(center)[0.872], align(center)[0.885], align(center)[0.883], align(center)[（同左）],
    [EG2 凍結], align(center)[Task 1], align(center)[0.740], align(center)[0.731], align(center)[0.739], align(center)[0.506 / 1.429 / 1.429],
    [], align(center)[Task 1-dim], align(center)[0.664], align(center)[0.673], align(center)[0.677], align(center)[（同左）],
    [], align(center)[Task 2], align(center)[0.853], align(center)[0.872], align(center)[0.852], align(center)[（同左）],
    [EG2 部分更新], align(center)[Task 1], align(center)[0.763], align(center)[0.766], align(center)[0.769], align(center)[2.198\* / 1.610 / 1.641],
    [], align(center)[Task 1-dim], align(center)[0.760], align(center)[0.757], align(center)[0.760], align(center)[（同左）],
    [], align(center)[Task 2], align(center)[0.954], align(center)[0.946], align(center)[0.951], align(center)[（同左）],
    [Qwen FT], align(center)[Task 1], align(center)[0.748], align(center)[0.748], align(center)[0.755], align(center)[3.222\* / 5.016 / 4.908],
    [], align(center)[Task 1-dim], align(center)[0.675], align(center)[0.719], align(center)[0.725], align(center)[（同左）],
    [], align(center)[Task 2], align(center)[0.195], align(center)[0.189], align(center)[0.305], align(center)[（同左）],
    [Qwen ZS], align(center)[Task 1], align(center)[0.653], align(center)[—], align(center)[—], align(center)[合計 0.176],
    [], align(center)[Task 1-dim], align(center)[0.632], align(center)[—], align(center)[—], align(center)[（同左）],
    [], align(center)[Task 2], align(center)[0.334], align(center)[—], align(center)[—], align(center)[（同左）],
    table.hline(stroke: 0.8pt),
  ),
)

\* seed 0 の GPU-h は M5 / M4 一次記録の合計（集計表では空欄）。Qwen FT の Task 2 回答率は 0.781 / 0.780 / 0.858、ZS は 0.96、他は全て 1.0。回答分のみの精度（accuracy\_answered）はTask 2 の Qwen FT で seed 別 0.250 / 0.242 / 0.355、ZS で0.348 である。回答分のみの macro-F1 平均は、SigLIP 2 凍結0.726 / 0.684 / 0.880、EG2 凍結 0.734 / 0.671 / 0.859、部分更新 0.765 / 0.759 / 0.950、Qwen FT 0.743 / 0.692 / 0.276、Qwen ZS 0.647 / 0.630 / 0.297（Task 1 / 1-dim / 2 の順）。

= 付録 B: 失敗分類の全数値

表 2 の分類に、予測文字分布を加えた全数値は`results/m7/qwen_task2_taxonomy.csv` にある。要点: FT の解釈済み予測分布は seed 0 で A 169 / B 157 / C 203 / D 252、seed 1 で A 139 / B 156 / C 224 / D 261、seed 2 でA 243 / B 140 / C 204 / D 271。ZS は A 315 / B 427 /C 97 / D 121。正解分布はほぼ均等である。

= 付録 C: 誤りの重なり（1,000 件中）

#figure(
  table(columns: 5, stroke: none,
    table.hline(stroke: 0.8pt),
    [*タスク*], align(center)[*seed*], align(center)[*全条件が失敗*], align(center)[*ちょうど1条件だけ失敗*], align(center)[*全条件が正解*],
    table.hline(stroke: 0.4pt),
    [Task 1], align(center)[0], align(center)[30], align(center)[215], align(center)[381],
    [Task 1], align(center)[1], align(center)[39], align(center)[222], align(center)[395],
    [Task 1], align(center)[2], align(center)[39], align(center)[206], align(center)[404],
    [Task 1-dim], align(center)[0], align(center)[16], align(center)[234], align(center)[266],
    [Task 1-dim], align(center)[1], align(center)[36], align(center)[238], align(center)[320],
    [Task 1-dim], align(center)[2], align(center)[28], align(center)[249], align(center)[314],
    [Task 2], align(center)[0], align(center)[9], align(center)[249], align(center)[86],
    [Task 2], align(center)[1], align(center)[10], align(center)[259], align(center)[82],
    [Task 2], align(center)[2], align(center)[6], align(center)[287], align(center)[115],
    table.hline(stroke: 0.8pt),
  ),
)

条件は 5 つ（ZS は seed 0 の参照ランを再利用）。Task 2 で「全条件が正解」が少ないのは Qwen FT の崩壊による。

= 付録 D: 再現性チェックリスト

- モデル revision: SigLIP 2 = 75de2d5…（一次記録 M3 は7 桁まで。完全なハッシュは未確認）、EmbeddingGemma 2 =f9972ac22e696a105c482202646979e33e4b5a1e、Qwen3-VL 8BInstruct = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b。

- 学習ペア manifest は seed ごとに固定され、ハッシュはリポジトリの manifests/ に記録。

- 全 run のコスト記録はボリューム側 results/costs.jsonl が完全な台帳で、要約は各 M 文書にある（ローカル repo のresults/cost\_log.jsonl は検証用ダミー 1 件のみ）。

- 分析コード: scripts/verify\_m7.py、scripts/analyze\_m7.py、scripts/cost\_reading\_m7.py、scripts/analyze\_review\_followup.py（単体テスト 38 件、1 件 skip）。

- 本稿の検証: 保存済み予測 39 セルの再計算が M6 報告値と完全一致（docs/M7\_RESULTS.md）。
