# RunPod 運用手順

2026-11-07 に実行環境として RunPod を採用。設定は `configs/runpod_4090.yaml`。

## ユーザー側の初期設定（一度だけ）

1. https://www.runpod.io でアカウント作成
2. Billing でクレジットをチャージ（初回は $10〜20 で十分。秒単位課金で、
   Pod を terminate すれば計算課金は止まる）
3. Settings → SSH Public Keys に Muse VM の公開鍵を登録する
   （`ssh-ed25519 AAAAC3...zodj hatch`、全文は VM の `~/.ssh/id_ed25519.pub`）

## Pod 起動（Muse 側の手順）

- GPU: RTX 4090 24GB / Community Cloud（在庫がなければ Secure、$0.89/時）
- テンプレート: 公式 PyTorch
- Network Volume 100GB を `/workspace` にアタッチ（月 $7、拡張は後から可能）
  （ボリュームはデータセンター固定。Pod は同じ DC で起動すること）
- 起動後、接続情報（`ssh root@<host> -p <port>`）を Muse に渡す

## 制約（2026-11-07 実測）

- Muse の VM から外部への SSH / 直接 TCP 接続はプラットフォーム側で禁止されており、
  Pod への SSH は使えない。Pod の操作は API（起動・終了・ボリューム管理）と、
  ブラウザ経由の Web ターミナル / Jupyter、または起動時コマンドで代替する。
- Web ターミナル方式は 2026-11-07 に検証済み：メール+パスワードでコンソールに
  サインイン（Google ログインはパスキー確認でブラウザから進めない）、
  Web ターミナルで GPU / ボリューム / torch CUDA / BF16 を確認し、
  ボリュームへの書き込みも成功。長いコマンドは一括貼り付けが反映されない
  ことがあり、1文字ずつ入力すると確実。
- Community Cloud の RTX 4090 は 2026-11-07 時点で全 DC 在庫なし（400 capacity）。
  Secure Cloud（$0.89/時）は EU-RO-1 で即時起動できた。長時間 run の前に
  カタログ API で在庫を再確認すること。
- 作成済みリソース：Network Volume `bluecad-data`（id `2hjzeri7z0`、100GB、EU-RO-1）。
  Pod は検証後に terminate 済み。ボリュームは月 $7 で保持される。

## セッションの流れ

1. `scripts/bootstrap_pod.sh` を Pod 上で実行（環境検証＋追加ライブラリ）
2. VM から `scripts/sync_to_pod.sh root@<host> <port> --with-data` で
   コードとデータを同期（データは HF ゲート付きのため VM 側で取得済みのものを送る）
3. 実験スクリプトを実行（`record_run` でコスト記録、単価は実請求ベースで記入）
4. 結果（`results/`）を VM に rsync で回収
5. **Pod を terminate**（stop のまま放置するとボリュームディスクが
   $0.20/GB/月で課金され続ける。Network Volume 側は $0.07/GB/月で残る）

## 注意

- Community Cloud は在庫変動が大きい。長時間 run の前日に在庫を確認する
- HF トークンは Pod に置かない。モデル取得が必要な場合は VM で取得して同期する
- 課金の実績（請求画面の金額）をコストログの備考に残す
