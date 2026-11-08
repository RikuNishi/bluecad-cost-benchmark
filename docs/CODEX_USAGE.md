# Codex の節約運用（ChatGPT Plus）

Plus の Codex は 5時間枠＋週次制限の可能性あり。公表目安（ローカルメッセージ/5時間）：
Sol 15〜160、Terra 35〜285、Luna 350〜3,000。重いモデルほど早く溶ける前提で運用する。

## 実行バイナリの注意（2026-11-07 判明）

プリインストールの Codex 0.164.0-alpha.9 は `codex-code-mode-host` が欠けており、
ツール実行（ファイル編集・コマンド実行）を伴う作業が失敗する。そのため実作業には
npm 版 0.167.0 を使うこと：

```bash
~/workspace/tools/bin/codex exec ...   # symlink to workspace/tools/codex-npm/bin/codex
```

ログイン・設定（~/.codex/）は共通で、そのまま引き継がれる。

## 設定済みのプロファイル（2026-11-07）

`~/.codex/config.toml` のデフォルトを一番安い組み合わせにしてある：

| 使い方 | コマンド | モデル | 用途 |
|---|---|---|---|
| デフォルト | `codex` / `codex exec "..."` | GPT-6 Luna / low | 通常のコーディング、定型作業 |
| 中プロファイル | `codex -p mid` / `codex exec -p mid "..."` | GPT-6.1 Terra / medium | しっかりした実装作業 |
| 深プロファイル | `codex -p deep` / `codex exec -p deep "..."` | GPT-6.1 Sol / high | 設計レビュー、難所のデバッグだけ（短く） |

プロファイル実体：`~/.codex/mid.config.toml`、`~/.codex/deep.config.toml`。
残量確認は Codex 内で `/status`。

## この研究での回し方

1. マイルストーン単位（docs/MILESTONES.md）で `codex exec` を一発実行し、
   仕様は RESEARCH_PLAN.ja.md と AGENTS.md に固定して往復を減らす。
2. 終わるごとに git commit する。制限で切れても成果が残るようにする。
3. Codex Cloud タスクと Fast モードは消費が大きいので使わない。
4. GPU が必要な実行は Codex にやらせない（この VM に GPU はない）。
   Codex が書いた `scripts/` を GPU 環境で実行し、結果（予測・コストログ）だけ持ち帰る。
5. 5時間枠を使い切ったら、枠のリセットを待つか、その間は Muse 側で
   採点・集計など Codex を使わない作業を進める。

## マイルストーン実行の例

```bash
cd ~/workspace/research/bluecad-cost-benchmark
codex exec -p mid "MILESTONES.md の M1（データ監査スクリプト）を実装して。AGENTS.md の規約に従い、テストも実行すること。"
git add -A && git commit -m "M1: data audit script"
```
