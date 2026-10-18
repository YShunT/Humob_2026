# src — 最終モデル（EXP04 N-B）の実装

このディレクトリには、HuMob Challenge 2026 で最終的に採用したモデル（EXP04 の N-B）を
学習し、ローカル検証と提出ファイル生成を行うコードだけを置いている。
前処理済みCSVの作り方、図の描画、ほかの実験のコードは含まない。

## モデルの概要

予測するのは、日ごとの OD 行列（出発セル→到着セルの人流量）である。評価境界内の
1,476 セルについて、セル内滞留（対角）とセル間移動（非対角）を別々にモデル化する。

| 成分 | 内容 |
|---|---|
| 対角 | セル別のローカルレベル＋曜日・年末年始＋外部特徴（避難者率・断水率・降水量）。log1p 尺度で推定 |
| 非対角・出発側（F） | 出発セルの総流出量 × 到着先への配分比 |
| 非対角・到着側（R） | 到着セルの総流入量 × 出発元からの配分比。同じ有向ODを列方向に分解したもの（対称化はしない） |
| 非対角の最終値 | `0.5 × F + 0.5 × R` |
| 評価境界外のOD | 震災後の観測日の曜日平均（採点には影響しない） |

総量と配分の外部特徴は、避難者率・断水率・降水量・積雪・避難所の5群。
対角・総量・配分の各成分は、状態空間モデル（カルマンフィルター＋RTSスムーザー）とリッジ回帰を
EM で交互に推定する（最大200回）。状態は地震の発生日（2024-01-01）でリセットする。

## 実行方法

リポジトリのルートから実行する。

```bash
python3 src/main.py --validate        # CV14日の検証。metrics.json と CV予測TSV を保存（約50秒）
python3 src/main.py --submit          # テスト58日の提出TSVを生成し、公式validatorで検査（約55秒）
python3 src/main.py --inspect         # 設定と学習データの概要を表示するだけ（ファイルを書かない）
python3 src/main.py --check [PATH]    # 既存の提出TSVを公式validatorで検査
```

- モードは必ず1つ指定する（引数なしや複数指定はエラーになる）。
- 出力はすべて `experiments_final/` に書く。
- `.log` には1実行1ブロックで追記する（失敗時は FAIL を残す）。`.log` と `outputs/` のTSVは git 管理外。

### 必要なもの

| 種類 | 内容 |
|---|---|
| Python | 3.9 以上。ライブラリは [`requirements.txt`](../requirements.txt) |
| 配布データ | `data/raw/humob2026-dataset.tsv` |
| 前処理済みCSV | 下の5本。作成コードはこのリポジトリでは公開しない |
| 公式validator | `src/evaluation/humob2026_validator.py`。主催者配布のため git 管理外。無い場合、`--submit` は検査を skip と記録する |

| 前処理済みCSV | 使い道 |
|---|---|
| `data/processed/events/population_cell_map.csv` | セルの担当市町と緯度経度 |
| `data/processed/避難者率/市町別避難者率.csv` | 市町別の避難者率（公表日ごと） |
| `data/processed/断水率/市町別断水率.csv` | 市町別の断水世帯率（公表日ごと） |
| `data/processed/避難所/避難所_セル別集計.csv` | セルごとの避難所施設数 |
| `data/processed/weather/weather.csv` | 観測所ごとの日降水量・最深積雪 |

## ディレクトリ構成

```text
src/
├── README.md
├── main.py                       エントリポイント（--validate / --submit / --inspect / --check）
├── config.py                     入出力パスとハイパーパラメータ
├── dataset/                      モデルに渡すデータを作る
│   ├── grid.py                   評価境界・セルID・評価セル一覧
│   ├── periods.py                学習カレンダー・CV14日・テスト58日・曜日・年末年始・地震の発生日
│   ├── od_matrix.py              配布OD → 対角・非対角の日次配列、出発側/到着側の分解、評価外の曜日平均
│   └── external_features.py      前処理済みCSV → セル別の日次外部特徴 → 成分別の説明変数・標準化
├── models/                       モデル本体
│   ├── kalman_smoother.py        ローカルレベル模型のフィルター＋RTSスムーザー
│   ├── ridge_regression.py       系列別の暦効果＋全系列共通の外部係数のリッジ回帰
│   ├── state_space_em.py         1成分の EM 推定（平滑化 ⇄ 回帰）
│   └── od_forecaster.py          5成分を学習 → 出発側/到着側を1:1結合 → OD辞書に戻す
├── evaluation/
│   ├── metric.py                 公式 Combined NRMSE と期間別スコア
│   └── submission_check.py       公式validatorの呼び出し
└── utils/
    ├── od_io.py                  OD TSV の読み書き、metrics.json の保存
    └── run_log.py                experiments_final/.log への追記
```

## 処理の流れ

```text
data/raw/humob2026-dataset.tsv ──► dataset/od_matrix.load_od_matrix ──► ODMatrix
data/processed/*.csv ─────────────► dataset/external_features.load_external_features ──► ExternalFeatures
                                                     │
                     models/od_forecaster.predict ◄──┘
                       ├─ 対角 D5            : state_space_em.fit_component（kalman_smoother ⇄ ridge_regression）
                       ├─ 出発側 F：総量・配分 : fit_component ×2 → 配分を正規化 → 総量×配分
                       └─ 到着側 R：総量・配分 : fit_component ×2 → 配分を正規化 → 総量×配分
                                     → 非対角 = 0.5F + 0.5R
                                                     ▼
                     models/od_forecaster.to_od_dicts（日付→{origin: {destination: value}}）
                       ├─ --validate : evaluation/metric.score_periods → metrics.json、CV予測TSV
                       └─ --submit   : submission.tsv → evaluation/submission_check
```

`--validate` では CV14日を学習データから除き、正解として別に持つ。
`--submit` では CV14日を含む全観測日で学習し直す。

## import の関係

左のファイルが右のモジュールを import する。依存は
`config` / `dataset` → `models` → `main` の一方向で、逆向きの import はない。

| ファイル | import する src 内モジュール |
|---|---|
| `main.py` | config, dataset.{external_features,od_matrix,periods}, evaluation.{metric,submission_check}, models.od_forecaster, utils.{od_io,run_log} |
| `config.py` | なし |
| `dataset/grid.py`・`periods.py` | なし |
| `dataset/od_matrix.py` | config, dataset.{grid,periods}, utils.od_io |
| `dataset/external_features.py` | config, dataset.{od_matrix,periods} |
| `models/kalman_smoother.py` | config, dataset.periods |
| `models/ridge_regression.py` | config, dataset.periods |
| `models/state_space_em.py` | config, dataset.periods, models.{kalman_smoother,ridge_regression} |
| `models/od_forecaster.py` | config, dataset.{external_features,od_matrix,periods}, models.state_space_em, utils.{od_io,run_log} |
| `evaluation/metric.py` | dataset.{grid,periods}, utils.od_io |
| `evaluation/submission_check.py` | config |
| `utils/od_io.py` | なし |
| `utils/run_log.py` | config |

## 配列の形状の記号

各ファイル冒頭の docstring の「形状」と、`return x  # (T, C)` の注記では次の記号を使う。

| 記号 | 意味 | 値 |
|---|---|---|
| `T` | 学習カレンダーの日数（2023-11-01〜2024-10-31） | 366 |
| `C` | 評価セル数（x 30〜70 × y 35〜70） | 1,476 |
| `P` | 候補ODペア数（学習データで一度でも正の値をとった、向きつきの非対角OD） | CV時 525 |
| `S` | 1成分の系列数（対角・総量なら `C`、配分なら `P`） | — |
| `K` | 外部特徴の列数（標準化後。欠損フラグ列を含み、定数列は除く） | 成分ごとに異なる |
| `N` | 予測対象日数 | CV 14 / テスト 58 |
| `M` | 気象観測所の数 | 6 |

## 検査はそれぞれ1か所で1回だけ行う

同じ検査を複数の場所に書かない。後段の処理は、前段の検査結果を前提にする。

| 検査 | 場所 |
|---|---|
| 予測対象日が学習データに入っていない（リーク防止） | `models/od_forecaster.predict` |
| log1p 尺度の平均が発散していない | `models/state_space_em.fit_component` |
| 回帰の正規行列が不定値でない | `models/ridge_regression.symmetric_inverse` |
| 予測値がすべて有限 | `models/od_forecaster.predict` |
| 評価外ODの曜日平均がある | `models/od_forecaster.to_od_dicts` |
| 提出TSVの形式・日付・値の範囲 | 公式validator（`evaluation/submission_check`） |

前処理済みCSVの中身（重複・負値・市町のそろい）は、CSVを作る側で検査済みという前提で読む。
そのため、CSVを手で編集しないこと。

## 注意点

- **定数を書く場所はそれぞれ1か所だけ**
  - 評価境界：`dataset/grid.py`
  - 正規化定数：`evaluation/metric.py`
  - 日付：`dataset/periods.py`
  - パスとハイパーパラメータ：`config.py`
  - JSON や md に写さない。
- **入出力パスは引数で変えない。** 変えるときは `config.py` を直す。
- **サブディレクトリに `__init__.py` は置かない。** 名前空間パッケージとして import するため、`utils`・`models`・`dataset` などと同じ名前のパッケージを環境にインストールすると、そちらが優先されて import に失敗する。
- **`data/raw/` は読み取り専用。**
- **CV14日は独立したテストではない。** EXP00〜EXP04 でモデルの選択に繰り返し使った期間である。
- **対角の予測値は、実行するプロセスによって最下位桁が変わることがある。** 計算内容は EXP04 と同一だが（同じプロセス内ではビット単位で一致）、別プロセスで実行すると相対で約 1e-11 の差が出る。スコアは小数6桁まで一致し、非対角はビット単位で一致する。検証の詳細は [experiments_final/report.md](../experiments_final/report.md)。
- **コードの書き方**
  - 各ファイルの1行目：ファイル全体の説明1行
  - 各関数・クラスの直下：1行の docstring
  - 引数と戻り値：型注釈を付ける
  - `#`：配列の形状注記（`# (T, C)`）にだけ使う
