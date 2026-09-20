# experiments — 実験記録

`experiments/` は、各モデルについて「事前の狙い」「再現可能な数値」「実行履歴」
「結果の解釈」を分けて保存する場所です。実装は `src/<ID>_run.py` に置き、
`experiments/<ID>/` にはコード以外の記録と成果物を置きます。

## 1実験の構成

```text
experiments/EXPxx/
├── strategy.md       # 実行前: 仮説、比較条件、採否基準
├── metrics.json      # 実行後: 比較に使う数値の正本
├── .log              # 実行日時、工程別時間、主要結果、生成物
├── report.md         # 実行後: 結果の表・図、解釈、採否
├── outputs/          # 提出TSVと、レポートで使う集計CSV
└── figures/          # 分布、誤差、候補比較の図
```

役割を重複させないことが基本です。設定値とスコアは `metrics.json`、人が読む考察は
`report.md`、実際に何を実行したかは `.log` を参照します。`report.md` の値は
`metrics.json` を読みやすい桁に丸めたものです。

各実験はリポジトリルートから次の1コマンドで実行します。

```bash
python3 src/EXP00_run.py
```

ローカル検証、集計表・図・`metrics.json` の更新、提出物の生成と形式検査を順番に行います。

## report.md の構成

原則として、次の見出しと順序で記録します。Stageや候補は各章の小見出しにします。

1. コード、事前設計、metrics、主要成果物、再現コマンド
2. `仮説`
3. `手法`
4. `検証設計`
5. `結果`（Stage別・候補別・期間別・成分別スコア）
6. `誤差分析`（市町別・セル別など）
7. `図と考察`（`figures/`内の相対パスを見出しに書き、図から読める事実を説明）
8. `解釈`
9. `失敗した試み`
10. `採否判定`
11. `制約・未解決`

`figures/`に残す図はすべてreportへ埋め込み、各図の相対パス、図から直接読める傾向、採用判断
との関係を記載します。考察に使わない重複図は生成・保存しません。

実験ごとのレポートの焦点は次のとおりです。

| 実験 | 手法 | レポートで重点的に確認する内容 |
|---|---|---|
| [EXP00](EXP00/report.md) | 学習期間の両端2日による線形補間 | ベースラインの総合・期間別・市町別誤差 |
| [EXP01](EXP01/report.md) | 同じ曜日の前後2点による線形補間 | 曜日対応による改善、曜日別・市町別誤差 |
| [EXP02](EXP02/report.md) | 出現確率と出現時代表値を分けた非対角予測 | 確率校正、A窓と代表値の4候補比較、総量保存の影響 |
| [EXP03](EXP03/report.md) | 対角の状態空間モデル＋非対角の総量×配分 | 成分別アブレーション、外部特徴、収束診断、CatBoost残差補正 |
| [EXP04](EXP04/report.md) | 非対角の出発側・到着側モデルと1:1結合 | N-F / N-R / N-B比較、日別・誤差種別・市町別の差 |

## metrics.json

機械的な比較に使う数値の正本です。上位キーは読みやすい順に揃えています。

| キー | 内容 |
|---|---|
| `experiment` / `date` / `status` | 実験ID、実験日、採否 |
| `compared_to` | 差分を取る比較元の実験ID |
| `results` | 総合、jan/apr、成分別、候補別などの評価値 |
| `config` | 手法とハイパーパラメータなど、再現に必要な設定 |

浮動小数点数は原則小数6桁に丸め、`1e-8` などの微小な設定値は保持します。
評価境界、正規化定数、指標の定義は重複記載せず、
`src/common/competition_metric.py` を正本とします。

入力・実装ファイルのhashは保存しません。コメント整理でも値が変わる一方で、結果の意味を
説明しないためです。再現条件は固定入力パス、`config`、依存バージョン、実行コマンドで管理します。

## .log

1回の実行を1ブロックで記録します。ブロックには次だけを残します。

- ヘッダー: 日時、実験ID、実行種別、成功・失敗
- `command`: 実行コマンド
- `stages`: 工程名、所要時間、開始時刻
- `results`: Combined NRMSEなど、その実行の主要値
- `artifacts`: 生成した主要ファイル
- `duration`: 全体の所要時間

過去の実験で使っていた `--validate` / `--submit` がログに残る場合がありますが、現在の
実行方法は引数なしの1コマンドです。ログの書式は `src/common/run_log.py` が正本です。

## outputs/ のファイル

再生成できる検証予測TSVや、`metrics.json` と同じ内容を並べ直した候補別・日別CSVは
保存しません。提出物と、レポートで直接比較・集計する表だけを残します。

| ファイル | 実験 | 内容 |
|---|---|---|
| `submission.tsv` | 全実験 | 提出対象58日の日付と予測OD辞書。ローカル生成物でGit管理外 |
| `municipality_errors.csv` | 全実験 | 採用モデルの検証14日を、市町・対角・非対角の起終点別に集計 |
| `fit_diagnostics.csv` | EXP03 | 状態空間モデルの候補ごとの規模、反復、収束、分散、所要時間 |
| `candidate_municipality_errors.csv` | EXP04 | N-F / N-R / N-Bそれぞれの市町別非対角誤差 |

`EXP02/outputs/shelter_reweight/` は、避難所の有無による到着地重み付けを試した追加診断の
記録です。現在の `EXP02_run.py` が生成する成果物ではなく、独立した `strategy.md`、
`metrics.json`、`report.md` を持つ不採用試行として保存しています。

### submission.tsv

ヘッダーなしのTSVで、1列目は `YYYYMMDD`、2列目は
`{origin: {destination: value}}` 形式の疎なOD辞書です。出力後に日付、セルID、値の有限性・
非負性、日次総量を検査します。

### municipality_errors.csv

`area` ごとに以下の3成分を持ちます。

- `diag_*`: セル内移動。セルの所属市町で集計
- `off_origin_*`: セル間移動。出発地の所属市町で集計
- `off_destination_*`: セル間移動。到着地の所属市町で集計

共通列と、各成分の接尾辞は次の意味です。

| 列 | 内容 |
|---|---|
| `experiment`, `area` | 実験ID、市町名 |
| `n_cells`, `cell_share_pct` | 対象セル数と全評価セルに占める割合 |
| `n_days` | 集計した検証日数 |
| `*_sse` | 二乗誤差和 |
| `*_sse_share_pct` | 全地域の同成分SSEに占める割合 |
| `*_zero_sse` | 全ゼロ予測のSSE（観測値の二乗和） |
| `*_sse_vs_zero` | `SSE / zero_sse`。1未満なら全ゼロ予測より良い |
| `*_rmse_daily_mean` | 固定母数で計算した日別RMSEの平均 |
| `*_nrmse` | 公式の成分別正規化定数で割った値 |

`off_origin_*` と `off_destination_*` は同じ非対角誤差を別の地域軸で見たものなので、
両者を足しません。地域の難しさを比較するときは、流動規模の影響を除いた
`*_sse_vs_zero` を主に使います。

### fit_diagnostics.csv

| 列 | 内容 |
|---|---|
| `candidate`, `kind` | 候補名と対象成分（対角・非対角など） |
| `fixed`, `groups`, `active`, `observations` | 固定効果、グループ、使用系列、観測数 |
| `q`, `r` | 状態・観測ノイズ分散 |
| `iterations`, `converged`, `objective` | 最適化の反復数、収束有無、目的関数値 |
| `negative_fraction` | 補正前予測に占める負値の割合 |
| `max_state_variance` | 最大状態分散 |
| `seconds` | 候補の計算時間 |

### candidate_municipality_errors.csv

先頭の `candidate` が `N-F`（出発側）、`N-R`（到着側）、`N-B`（1:1結合）を示します。
残りの列は `municipality_errors.csv` と同じです。

## figures/ の主要図

| 実験 | 図 |
|---|---|
| EXP00 | `EXP00_pred_dist.png`, `local_error_map.png` |
| EXP01 | `EXP01_pred_dist.png`, `local_error_map.png` |
| EXP02 | `EXP02_pred_dist.png`, `probability_calibration.png`, `local_error_map.png` |
| EXP03 | `diag_ablation_scores.png`, `off_ablation_scores.png`, `EXP03_pred_dist.png`, `local_error_map.png` |
| EXP04 | `direction_comparison.png`, `EXP04_pred_dist.png`, `local_error_map.png` |

`*_pred_dist.png` は対角・非対角の日次推移、`local_error_map.png` はセル単位の誤差分布、
その他は各実験の候補比較を示します。図の解釈と採用判断は各 `report.md` に記載します。
