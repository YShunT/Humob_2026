# src/preprocessing — 外部データの前処理

公開資料、地理データ、気象データを、実験コードから読み込めるCSVへ変換する。
このディレクトリは**入力データを作る処理**だけを持ち、予測モデルやスコア計算は置かない。
実験共通の評価処理は [`src/common/`](../common/README.md)、予測処理は
`src/<ID>_run.py` が担当する。

`data/` と `events/` は公開リポジトリに含まれない。各スクリプトは必要な原本がローカルの
所定位置にあることを前提とし、データのダウンロードは行わない。`data/raw/` 以下は
**参照専用**であり、変換結果は必ず `data/processed/` または `figures/` に保存する。

| モジュール | 役割 | 主な出力 |
|---|---|---|
| [`population_cell_mapping.py`](population_cell_mapping.py) | 評価セルを能登8市町へ割り当てる | `population_cell_map.csv`, `cell_area_map.csv` |
| [`shelter_extract.py`](shelter_extract.py) | 被害情報PDFから市町別の1次避難所推移を抽出 | `events/避難所.csv` |
| [`water_extract.py`](water_extract.py) | 被害情報PDFから市町別の断水推移を抽出 | `events/断水.csv` |
| [`evacuation_rate.py`](evacuation_rate.py) | 避難者数を人口で割って市町別避難者率を作る | `避難者率/市町別避難者率.csv` |
| [`water_outage_rate.py`](water_outage_rate.py) | 断水戸数を世帯数で割って市町別断水率を作る | `断水率/市町別断水率.csv` |
| [`shelter_features.py`](shelter_features.py) | 施設単位の避難所情報を評価セル単位に集計する | `避難所/避難所_セル別集計.csv` |
| [`weather_csv.py`](weather_csv.py) | 気象庁CSVを1日×1観測所の縦長形式へ変換する | `weather/weather.csv` |

---

## 実行方法と処理順序

コマンドは**必ずリポジトリルートから**実行する。

```bash
# 1. 評価セルと市町の対応
python3 src/preprocessing/population_cell_mapping.py

# 2. 被害情報PDFから市町別時系列を抽出・検算
python3 src/preprocessing/shelter_extract.py
python3 src/preprocessing/water_extract.py

# 3. 人口・世帯数で正規化
python3 src/preprocessing/evacuation_rate.py
python3 src/preprocessing/water_outage_rate.py

# 4. 施設単位の避難所データをセル集計
python3 src/preprocessing/shelter_features.py

# 5. 気象庁CSVを変換（上記処理とは独立）
python3 src/preprocessing/weather_csv.py
```

依存関係は次のとおり。

```text
municipality boundaries + coastline + population
  └─ population_cell_mapping.py
       ├─ population_cell_map.csv
       └─ cell_area_map.csv ── shelter_features.py

damage-report PDFs
  ├─ shelter_extract.py ── events/避難所.csv ── evacuation_rate.py
  └─ water_extract.py   ── events/断水.csv   ── water_outage_rate.py

JMA raw CSV ── weather_csv.py
```

PDF抽出にはPopplerの`pdftotext`が必要。それ以外のPython依存はルートの
[`requirements.txt`](../../requirements.txt)で管理する。

すべて固定パスで動かし、CLI引数は持たない。入力や出力を変える場合は、各ファイル冒頭の
パス定数を明示的に変更する。抽出・変換と検算は同じ実行内で行う。

---

## population_cell_mapping.py — 評価セルと市町の対応

公式評価範囲の1,476セルを、七尾市、珠洲市、穴水町、能登町、輪島市、志賀町、
中能登町、羽咋市へ割り当てる。市町境界、海岸線、評価グリッドの位置関係から判定し、
人口値そのものをセルへ按分する処理ではない。

**入力**

- `data/processed/events/population.csv`
- `data/processed/GeoJson/municipality_boundaries_2024.geojson`
- `data/processed/GeoJson/japan.geojson`

**出力**

- `data/processed/events/population_cell_map.csv`: 座標、担当市町、割当方法などを含む詳細表
- `data/processed/events/cell_area_map.csv`: `grid_id, area`だけのモデル結合用表
- `figures/population_cell_map.png`: セル割当の確認図

割当規則は次の順で適用する。

1. 陸上の境界セルは、セル矩形との重なり面積が最大の市町へ割り当てる。
2. 石川県側の陸セルは、8市町のいずれかへ必ず割り当てる。
3. 海上の沿岸セルは、最も近い本土が石川県で、行政区域セルから2マス以内の場合だけ
   最短の市町へ割り当てる。
4. 最も近い本土が富山県、沖合の小島、沿岸から2マス以上離れたセルは対象外にする。

`population_cell_map.csv`の主要列:

| 列 | 意味 |
|---|---|
| `grid_id` | `y_x`形式のセルID |
| `area` | 割り当てた市町。空欄は対象外 |
| `assignment_method` | 重なり面積、陸上最近傍、沿岸最近傍、対象外の別 |
| `municipality_overlap_ratio` | セル面積に占める採用市町境界の重なり比率 |
| `is_land` | セル中心が陸か |
| `within_coastal_range` | 沿岸セルとして許容距離内か |

---

## shelter_extract.py — 市町別1次避難所の時系列

`events/被害情報_危機管理監室/*_被害情報.pdf`の「避難所の開設状況」から、
「市町1次避難所」の表を抽出する。1行は**1公表時点×1市町**であり、個別施設の一覧ではない。

**出力**: `data/processed/events/避難所.csv`

| 列 | 意味 |
|---|---|
| `日付` | PDF本文の「現在」日時。読めない場合はファイル名の日付 |
| `市町名` | 原表に掲載された市町 |
| `開設数(箇所)` / `避難者数(人)` | 市町1次避難所の集計値 |
| `広域避難所(箇所)` / `広域避難者数(人)` | 備考から分離した広域避難分 |
| `閉鎖` | 表中の閉鎖記載 |

市町行の合計を表末尾の「計」と照合し、PDFの読取件数、不一致、日付差を表示してから
CSVを書き出す。

```bash
python3 src/preprocessing/shelter_extract.py
```

---

## water_extract.py — 市町別断水の時系列

同じ被害情報PDFの生活環境部ページから、市町別の断水戸数を抽出する。1行は
**1公表時点×1市町**。PDFの表には断水中の市町だけが残るため、表から消えたことを
自動的に0とは解釈しない。

**出力**: `data/processed/events/断水.csv`

| `状態` | `断水戸数` | 意味 |
|---|---:|---|
| `断水中` | 数値 | 当該PDFの表に掲載 |
| `解消済み` | 0 | 脚注から解消日を確認でき、その日以降 |
| `未報告` | 空欄 | 表にも解消脚注にもなく状態を確定できない |

脚注の解消日は後日のPDFに初めて現れる場合がある。そのため全PDFを先に読み、解消日を
集約してから各日の行を作る。`未報告`を0へ置換しないことが重要。

```bash
python3 src/preprocessing/water_extract.py
```

---

## evacuation_rate.py — 市町別避難者率

`events/避難所.csv`の避難者数を、`events/population.csv`の2023年12月人口で割る。

```text
避難者率 = 避難者数 ÷ 2023年12月人口
```

対象は能登8市町で、1行は**1公表日×1市町**。報告日の間を補間・前方補完せず、
公表値がある日だけを保存する。

**出力**

- `data/processed/避難者率/市町別避難者率.csv`
- `data/processed/避難者率/figures/避難者率_<市町名>.png`（8枚）
- `data/processed/避難者率/figures/避難者率_8市町全体.png`

8市町全体の率は市町率の単純平均ではなく、`8市町の避難者数合計 ÷ 8市町の人口合計`。
CSVには元の人数、人口、人口1万人当たり人数、開設数、広域避難者数も残す。

---

## water_outage_rate.py — 市町別断水世帯率

`events/断水.csv`の断水戸数を、`events/世帯数.csv`の2023年12月世帯数で割る。

```text
断水世帯率 = 断水戸数 ÷ 2023年12月世帯数
```

**出力**

- `data/processed/断水率/市町別断水率.csv`
- `data/processed/断水率/figures/断水率_<市町名>.png`（8枚）
- `data/processed/断水率/figures/断水率_8市町全体.png`

`未報告`の断水戸数と率は空欄のまま保持する。8市町全体の図は、8市町すべての値が
揃う公表日だけを使い、`断水戸数合計 ÷ 世帯数合計`で計算する。市町率の単純平均ではない。

---

## shelter_features.py — 施設単位避難所のセル集計

`data/processed/避難所/避難所.csv`にある**個別施設**を評価セル単位に集計する。
`shelter_extract.py`が作る市町別時系列の`data/processed/events/避難所.csv`とは、粒度も
用途も異なる。

**入力**

- `data/processed/避難所/避難所.csv`: 施設名、状態、収容人数、座標、対応セルなど
- `data/processed/events/cell_area_map.csv`: 評価セルと担当市町
- 市町境界・海岸線GeoJSON

**出力**

- `data/processed/避難所/避難所_セル別集計.csv`
- `figures/避難所_EDA.png`
- `figures/避難所_セル別施設数.png`

セル集計CSVは施設が0件の評価セルも含む。主要列は`facility_count`、`open_count`、
`closed_count`、`unavailable_count`、収容可能人数と避難者数の既知件数・合計。
`facility_count`は特定日の開設数ではなく、閉鎖済みを含む**登録施設の所在数**である。

入力について、必須列、セルID、緯度経度、元地域とセル担当市町、状態値を検査してから
集計する。

---

## weather_csv.py — 気象庁CSVの縦長化

気象庁「過去の気象データ・ダウンロード」のCP932・複数行ヘッダーCSVを読み、
1行を**1日×1観測所**とするUTF-8（BOM付き）のCSVへ変換する。

**既定入力**: `data/raw/data.csv`（読み取りのみ）  
**既定出力**: `data/processed/weather/weather.csv`

変換対象:

- 平均・最高・最低気温
- 日降水量
- 日照時間
- 最深積雪
- 平均風速

各数値列に対応する`<feature>_quality`列も保存する。全期間を通して値がない項目や、
対象項目がすべて空の観測所は除外する。出力前に`date, station`の重複、日付の連続性、
期待行数を検査する。空欄を0へ変換しない。

```bash
python3 src/preprocessing/weather_csv.py
```

---

## 前処理を変更するときのルール

1. `data/raw/`は読み取り専用とし、作成・編集・削除・移動・名前変更をしない。
2. 原本の「0」と「未報告・欠損」を区別する。根拠なく0埋め、補間、前方補完をしない。
3. 人口・世帯数で作る率は、分子・分母の元列も出力に残して再計算可能にする。
4. 市町集計と施設集計、時系列特徴と静的特徴を同じ列として混同しない。
5. 出力前に主キーの重複、行数、範囲、合計値を検査する。
6. 図は確認用であり、モデル入力の正本はCSVとする。
7. 新しい前処理を追加したら、このREADMEの一覧、入出力、欠損規則、再現コマンドを更新する。
