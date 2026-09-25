# src/common — 実験共通モジュール

どの実験（`src/<ID>_run.py`）からも同じ結果になってほしいものだけを置く。
**予測ロジックはここに置かない**。スコアの定義・検証データの切り出し・図の描き方を
共通化し、実験間で数値が比較可能であることを保証するのがこのディレクトリの役割。

| モジュール | 役割 | 依存 |
|---|---|---|
| [`competition_metric.py`](competition_metric.py) | 公式 Combined NRMSE の計算。評価境界・正規化定数の正本 | 標準ライブラリのみ |
| [`grid_geo_map.py`](grid_geo_map.py) | グリッド座標と緯度経度の変換、陸海判定、地理対応図 | numpy, matplotlib |
| [`local_error_map.py`](local_error_map.py) | ローカル検証14日の抽出と、空間誤差マップ3枚の描画 | numpy, matplotlib |
| [`municipality_error.py`](municipality_error.py) | セル―市町対応に基づく地域別SSE寄与率・NRMSEのCSV出力 | 標準ライブラリのみ |
| [`pred_dist.py`](pred_dist.py) | 提出物の予測を日次推移の折れ線1本に要約した図 | matplotlib |
| [`run_log.py`](run_log.py) | 実行ログ `experiments/<ID>/.log` を読みやすいブロックで追記 | 標準ライブラリのみ |
| [`experiment_io.py`](experiment_io.py) | OD TSV入出力、提出対象日、形式検査、metrics保存 | 標準ライブラリのみ |

---

## 呼び出し方法

実行は**必ずリポジトリルートから**。`python3 src/<ID>_run.py` とすると `sys.path[0]` が
`src/` になるので、`common.<module>` で解決できる。

```python
from common.competition_metric import day_errors, score
from common.experiment_io import read_od_days, submission_days, write_metrics, write_od_tsv
from common.local_error_map import save_local_error_map
from common.municipality_error import write_municipality_errors
from common.pred_dist import save_pred_dist
from common.run_log import run_log
```

`src/preprocessing/` のように `src/` 直下でないスクリプトから呼ぶ場合は、`src/` を明示的に載せる。

```python
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # src/ を載せる
```

numpy / matplotlib を使うモジュール（`local_error_map` / `pred_dist`）は、検証時だけ
必要になることが多い。関数の中で import すれば、提出物生成だけのときに重い依存を
読み込まずに済む。

---

## competition_metric.py — 公式スコア

一般的な `RMSE / mean(y)` ではなく、コンペ公式の手順（日ごとに diagonal /
off-diagonal を別々にRMSE → 日次平均 → 主催者提供の固定RMSで正規化して単純平均）を
実装する。**評価境界も正規化定数もこのファイルが正本**で、他所（metrics.json など）に
写しを作らない。

**定数**

| 名前 | 値 | 意味 |
|---|---|---|
| `EVAL_X` / `EVAL_Y` | `(30, 70)` / `(35, 70)` | 採点対象の評価境界（両端含む） |
| `N_CELL` = `N_DIAG` | 1476 | 境界内セル数＝対角ペア数 |
| `N_OFF` | 2177100 | 非対角ペア数（母数は常に固定） |
| `NORM_DIAG` / `NORM_OFF` | 26.57 / 0.0176 | 主催者提供の正規化RMS |
| `NX`, `NY` | 100, 70 | グリッド全体のサイズ |
| `OOB` | `"-1_-1"` | 域外を表す特殊ID。評価対象外 |

**関数**

| 関数 | 返り値 |
|---|---|
| `day_errors(pred, truth)` | 1日分の誤差。`rmse_diag` / `rmse_off` と、内訳 `hit_*` / `miss_*` / `extra_*`、観測・予測ペア数 |
| `score(day_results)` | 日次の平均。`nrmse_diag` / `nrmse_off` / `combined` |
| `calculate_competition_nrmse(predictions, truths, days=None)` | 上2つをまとめて実行。`days` 省略時は両方に存在する日の共通部分 |
| `observed_rms(rows)` | 観測ODから正規化定数と同じ定義のRMSを再計算 |
| `parse_gid` / `valid_gid` / `in_eval_box` | セルIDの解釈・提出形式の検査・評価境界内判定 |
| `pct(part, whole)` | 内訳表示用の百分率（分母0なら0） |

```python
# 日ごとに誤差を溜めて最後にまとめる（窓別の集計もできる）
per_day = [day_errors(pred[d], truth[d]) for d in days]
sc = score(per_day)
print(sc["combined"], sc["nrmse_diag"], sc["nrmse_off"])

# 1発でよければ
sc = calculate_competition_nrmse(pred, truth)
```

**注意**

- `pred` / `truth` は `{origin: {destination: value}}` の疎な辞書。辞書に無いペアは0
  として扱うが、**母数は常に `N_DIAG` / `N_OFF` で固定**する。予測を絞っても母数は減らない。
- 全ゼロ予測の Combined NRMSE がおよそ 1.0。**小さいほど良い**。
- 誤差内訳の 3分類: `hit`（両方にある）/ `miss`（正解にあり予測に無い）/
  `extra`（予測にあり正解に無い）。どこで損しているかの診断に使う。

---

## experiment_io.py — 実験共通の入出力

予測手法とは独立な、配布TSVの読み込み、提出TSVの保存・形式検査、提出対象日の定義、
JSON・集計CSVの保存をまとめる。実験間で別実装を持たない。

| 関数 | 役割 |
|---|---|
| `read_od_days(path, keep)` | 必要な日だけを読み、観測ODとNA日を返す |
| `write_od_tsv(predictions, path)` | 日付→OD辞書を提出形式で保存 |
| `check_od_tsv(path, expected_days, expected_daily_total=None)` | 日付、構造、ID、有限非負値、任意の日次総量を検査 |
| `submission_days()` | 除外日を抜いた提出対象58日を返す |
| `write_metrics(path, data)` | 上位キーを統一し、floatを小数6桁（微小値は保持）で保存 |
| `write_json` / `write_records_csv` | 小さなJSON・CSV成果物の共通処理 |

検証予測は実験内で評価・描画へ直接渡す。再生成できる`validation_pred.tsv`は保存しない。

---

## local_error_map.py — 検証期間と空間誤差マップ

テスト期間 2024-02〜03 を前後から挟む2窓14日（`jan` = 1/25〜1/31, `apr` = 4/1〜4/7）が
ローカル検証の既定。この窓の定義もここが正本。

**関数**

| 関数 | 返り値 |
|---|---|
| `validation_days()` | 検証14日を `YYYYMMDD` 整数のリストで |
| `error_surfaces(predictions, truths, days)` | 誤差を diagonal / off-origin / off-destination の3つの2次元面へ集約 |
| `save_local_error_map(predictions, truths, out, experiment="model", language="ja")` | 上の3面を1枚のPNGにして保存し、保存先 Path を返す。`language="en"`で英語表記 |

```python
from common.experiment_io import RAW_TSV, read_od_days
from common.local_error_map import save_local_error_map, validation_days

wanted = set(validation_days())
rows, _ = read_od_days(RAW_TSV, wanted.__contains__)
save_local_error_map(predictions, rows,
                     ROOT / "experiments" / "EXP00" / "figures" / "local_error_map.png",
                     experiment="EXP00")
```

---

## municipality_error.py — 市町別の検証誤差

`data/processed/events/population_cell_map.csv`をセル―地域定義の正本として、ローカル検証14日の
誤差を8市町と「対象8市町外・富山県」へ分解する。市町外セルも残すため、SSE寄与率は各列で
必ず100%になる。

- `diagonal`: セル自身の担当市町
- `off_origin`: 非対角ODを起点セルの担当市町で集計
- `off_destination`: 同じ非対角ODを終点セルの担当市町で集計

非対角の起点列と終点列は別々の診断軸であり、列をまたいで足さない。公式Combined NRMSE自体は
地域へ加算分解できないため、地域別Combinedは作らない。

**3つの指標は用途が違う。**

| 列 | 何を見るか | 地域間で比較できるか |
|---|---|---|
| `*_sse_share_pct` | 誤差がどの地域で発生しているか | ✗ 流動規模に比例する |
| `*_nrmse` | 地域のセル数を母数にした誤差の大きさ | ✗ セルあたりの流動密度に比例する |
| `*_sse_vs_zero` | **全ゼロ予測を1.0とした相対誤差（= SSE / Σ観測値²）** | **✓ 規模で正規化済み** |

`*_sse_share_pct`と`*_nrmse`は、流動の多い地域ほど大きく出る。**この2つだけを見て「誤差の
大きい地域＝苦手な地域」と判断してはいけない。** 難易度の比較には`*_sse_vs_zero`を使う。
1.0を超える地域は、その地域について何も予測しない方が誤差が小さいことを意味する。
`*_zero_sse`はその分母（観測ペアの二乗和）で、検算用に残してある。

`metrics.json`の`results.*.vs_zero`とは別物なので混同しないこと。あちらは
`Combined NRMSE / 全ゼロ予測のCombined NRMSE`で、平方根と日次平均を通した後の比である。
本CSVの`*_sse_vs_zero`は平方根を取る前のSSE比なので、おおむね二乗の関係になる
（EXP02: 対角0.0134・非対角0.1158 → 各平方根 0.116・0.340 → 平均 0.228 ≒ `vs_zero` 0.2299）。

```python
from common.municipality_error import write_municipality_errors

write_municipality_errors(predictions, truths, "EXP00")
# → experiments/EXP00/outputs/municipality_errors.csv
```

---

## pred_dist.py — 評価グリッド内の予測日次推移

`experiments/<ID>/outputs/submission.tsv`を読み、origin・destinationがともに評価グリッド内の
人流だけを集計する。対角と非対角を上下2段に分け、予測を橙、配布データの正解を黒で描く。
背景はCV期間とテスト期間を示し、予測不要日は線を切る。

提出TSV全体の日次総量は約189,190.25で一定だが、この図は評価グリッド内だけを集計するため、
その総量とは一致しない。公式指標と同じ空間範囲で、対角・非対角の配分と曜日周期を比較するための図である。

| 関数 | 返り値 |
|---|---|
| `save_pred_dist(exp_id, tsv=None, out=None)` | 図を保存して Path を返す。submission.tsv が無ければメッセージを出して `None` |
| `load_daily_stats(tsv, keep=None, evaluation_only=True)` | `{date: {total, n_pairs, diag}}`。既定は評価グリッド内だけ |
| `submission_path(exp_id)` / `figure_path(exp_id)` | 既定の入出力パス |

```python
from common.pred_dist import save_pred_dist
save_pred_dist("EXP00")
# → experiments/EXP00/figures/EXP00_pred_dist.png
```

```bash
python3 src/common/pred_dist.py EXP00
```

`tsv` / `out` は既定パスを上書きしたいとき（テストなど）だけ渡す。通常は実験IDだけでよい。

`language="en"`を渡すとタイトル・軸・凡例を英語にする（既定は`"ja"`）。
EXP04は英語表記を使用する。CVの予測線も含める場合は、`validation_predictions`に検証予測辞書を渡す。

---

## run_log.py — 実行ログ

`experiments/<ID>/.log` に1実行＝1ブロックを追記する。書式は `run_log.py` の
`stage` / `result` / `note` / `error` を正本とする。

| 関数 / メソッド | 役割 |
|---|---|
| `run_log(exp_id, mode, command)` | `with` で使うコンテキストマネージャ。ブロックの開始と計時 |
| `log.stage(name, label)` | 直前の stage を閉じて所要時間を確定し、次の stage を開始 |
| `log.result(**kv)` | `results` に主要値を1項目ずつ追加。floatは小数4桁で出る |
| `log.note(text)` | `artifacts` に生成物のパスなどを追加 |

```python
from common.run_log import run_log

with run_log("EXP00", "run", "python3 src/EXP00_run.py") as log:
    log.stage("stage0", "アンカー2日の読み込み")
    ...
    log.stage("stage1", "検証14日の抽出")
    ...
    log.result(combined=0.2737, jan=0.2748, apr=0.2726)
    log.note("→ experiments/EXP00/figures/local_error_map.png")
```

ブロックは `with` を抜けるときに1回で追記する。例外が出た場合はその時点の stage に
`FAIL` を付け、例外の型とメッセージを1行だけ残して再送出するので、**落ちても記録が残る**。

stage 名は実験をまたいで固定する（`stage0`=読み込み, `stage1`=前処理, ...）。揃えておくと
実験間で「どの工程が重くなったか」を比較できる。

---

## モジュールを足すときのルール

1. **予測ロジックを持たせない。** 共通モジュールが受け取るのは、日付をキーにした
   予測ODと正解ODだけ。どう予測したかは実験側の責任。
2. **実験IDを引数で受け、出力先は `experiments/<ID>/` 以下に固定する。** 呼び出し側で
   パスを組み立てさせない。
3. **定数の正本をここに置き、他所に写さない。** 評価境界や正規化定数を JSON や
   report.md にコピーすると、片方だけ直したときに検出できなくなる。
4. **重い依存（numpy / matplotlib）は関数の中で import する。** 提出物生成だけのときに
   読み込まれないようにする。
