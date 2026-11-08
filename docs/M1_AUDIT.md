# M1 監査結果（2026-11-07）

対象：公式 item ファイル全件（GitHub `alexvalgui/blueprint2cad`、commit `3061b37`）と、
HF データセット `alex-novineer/blueprint2cad` の splits 全件＋shard-0000
（drawings / renders 各 1,000 部品）。監査スクリプトは `scripts/audit_data.py`、
生レポートは `results/m1_audit_no_data.json` と `results/m1_audit_shard0.json`。

## item ファイル（本体データなしで確定したこと）

- Task 1 / Task 1（寸法付き）/ Task 2 は各 1,000 問。回答分布は Task 1 が yes 500 / no 500、
  Task 2 が A 258 / B 241 / C 262 / D 239。group は simple / complex 各 500。
- 重複 item ID なし。group ラベルの不整合なし。
- 参照モデル ID：Task 1 系は 1,216 件（通常・寸法付きで完全一致）、Task 2 は 1,714 件、
  Task 1 との重複は 1,169 件。参照 ID は全 30 シャードに分散している。

## splits の監査

- train 26,039 / val 2,890 / test 1,000。split 間の重複は 0、split 内の重複も 0。
  ID 分割は正しく行われている。
- ids_10k（検索ギャラリー）は 10,000 件、重複なし、全件が split の和集合に含まれる。
- **注意点**：split の和集合は 29,929 件で、実データ（shard_index.csv）の 29,925 件より 4 件多い。
  差分は train.txt のみの `sample_888170`、`sample_888176`、`sample_888179`、`sample_888180` で、
  データが配布されていない。Task の item からは参照されていない。
  → 20,000 ペアの manifest を train から作る際は、shard_index に存在する ID のみに絞ること。

## shard-0000 の実データ監査（500 部品をサンプル）

- パス対応付けは実データで動作確認済み：item 参照 9,000 件中、shard-0000 に含まれる
  286 件（Task 1：66、Task 1 寸法付き：66、Task 2：154）が found、残りは未取得シャード分。
- サンプル 500 部品は、renders 6 枚＋drawings normal 6 枚＋dim 6 枚が全て揃っていた。
- **注意点**：HF の README にある `renders/<id>/transforms.json` が shard-0000 の
  tar 内に存在しない（全 1,000 部品で欠落）。監査上の「不完全」はこの 1 件のみが理由。
  → レンダリング視点の対応は transforms.json に依存できない。視点が必要な実験では、
  公式コード側の視点順序の定義を別途確認してから使うこと（番号からの推測はしない）。
- 完全一致（sha256）の重複は 85 グループ、全て同一モデル内の対向視点ペア
  （front/back、left/right など）。異なるモデル間の重複は 0 件。
  対称形状では妥当な結果で、汚染ではない。

## M1 の完了条件に対する判定

「公式問題を正しく読み、欠損と ID 分割を監査できる」は達成。
残り 29 シャードのダウンロードは、特徴抽出（M3）や Qwen 実験（M4）で
必要になるシャードから順に行う。
