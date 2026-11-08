# 公開データ配布状況の確認記録

確認日：2026年11月6日。確認したものは資料・APIメタデータ・一部の公開データ行であり、学習可能性の最終検証やモデル学習は実施していない。

| 対象 | 実際に確認したこと | 未確認のこと |
|---|---|---|
| BlueCAD-30K | HF API上で公開リポジトリ、gated=auto、CC BY 4.0。drawings 30 shard、renders 30 shard、splitファイルの一覧。未認証でsplit取得はHTTP 401 | 同意後の本体取得、split実件数、各部品の画像数、画像とIDの対応、近重複 |
| BlueCAD容量 | drawings 4,551,168,000 bytes、renders 19,417,507,840 bytes。合計23,968,675,840 bytes | 展開後容量、派生特徴・学習チェックポイントの容量 |
| Blueprint2CADコード | 著者GitHubのREADMEを取得し、データ先・評価条件・コードの案内を確認 | スクリプトの実行、依存ライブラリの互換性、論文表の再現 |
| TriView-CAD | 公開APIにPNG 1,347件、合計25,876,913 bytes。独立したラベル／split／READMEファイルはこの一覧になし。CC BY 4.0 | フォルダや画像命名からラベル・part ID・splitが復元できるか。配布説明との整合 |
| TechMB | データカード、TSVのファイル一覧、Dataset Viewerの行を確認。947 QA、180図面というカード記載 | 質問別件数、加工法ラベル頻度、図面ID重複をまとめた全体統計、元モデル単位の分割 |
| OrthoCAD-322K | 著者GitHubのtree APIではREADME.mdのみ | データ・学習コードの別配布先、取得可能性、利用条件 |
| SPARE3D・DeepPatent | 著者公式READMEのデータ配布リンクと課題・分割の説明を確認 | 配布アーカイブの取得・展開、利用条件の詳細、ID監査 |

APIスナップショットには検査時点のメタデータを保存した。ダウンロードURLなどは将来変わる可能性があるため、取得時には公式配布先から再確認する。

主な証跡：

- `evidence/bluecad_metadata.json`
- `evidence/bluecad_drawings_tree.json`
- `evidence/bluecad_renders_tree.json`
- `evidence/blueprint_readme.md`
- `evidence/triview_metadata.json`
- `evidence/techmb_card.md`
- `evidence/techmb_firstrows.json`（画像base64を除いたメタデータ）
- `evidence/orthocad_repo_tree.json`
- `evidence/access_audit_summary.json`
- `evidence/additional_access_audit.json`
