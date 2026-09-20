"""気象庁CSVを1日×1観測所の縦長CSVへ変換する。

実行: python3 src/preprocessing/weather_csv.py
入力: data/raw/data.csv
出力: data/processed/weather/weather.csv
依存: 標準ライブラリのみ
"""

import csv
import datetime as dt
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # src/preprocessing/ から見たリポジトリルート
INPUT_CSV = ROOT / "data" / "raw" / "data.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "weather" / "weather.csv"

DATE_LABEL = "年月日"
KNOWN_SUBHEADERS = {"", "現象なし情報", "品質情報", "均質番号"}

# 気象庁の項目名 -> 可読性のある英語列名。
# 現在の実験で直接使いやすい日別の数値項目に限定する。
FEATURES = [
    ("平均気温(℃)", "temperature_mean_c"),
    ("最高気温(℃)", "temperature_max_c"),
    ("最低気温(℃)", "temperature_min_c"),
    ("降水量の合計(mm)", "precipitation_total_mm"),
    ("日照時間(時間)", "sunshine_hours"),
    ("最深積雪(cm)", "snow_depth_max_cm"),
    ("平均風速(m/s)", "wind_speed_mean_m_s"),
]


# 入力とヘッダー解析
def read_source(path):
    """CP932の気象庁CSVを行列として読み込む。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"入力CSVがありません: {path}")

    with path.open("r", encoding="cp932", newline="") as source:
        rows = list(csv.reader(source))
    if not rows:
        raise ValueError(f"入力CSVが空です: {path}")
    return rows


def locate_headers(rows):
    """地点・項目・補助情報ヘッダーとデータ開始位置を返す。"""
    metric_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row and row[0].strip() == DATE_LABEL
        ),
        None,
    )
    if metric_index is None or metric_index == 0:
        raise ValueError("'年月日'を含む項目ヘッダーが見つかりません")

    station_index = metric_index - 1
    subheader_index = next(
        (
            index
            for index in range(metric_index + 1, len(rows))
            if any(cell.strip() in KNOWN_SUBHEADERS - {""} for cell in rows[index])
        ),
        None,
    )
    if subheader_index is None:
        raise ValueError("品質情報を含む補助ヘッダーが見つかりません")

    width = len(rows[metric_index])
    for label, index in (
        ("観測所", station_index),
        ("補助情報", subheader_index),
    ):
        if len(rows[index]) != width:
            raise ValueError(
                f"{label}ヘッダーの列数が一致しません: "
                f"expected={width}, actual={len(rows[index])}"
            )
    return station_index, metric_index, subheader_index, subheader_index + 1


def parse_dates(rows, data_start, width):
    """データ行を検証し、日付と元行の組を返す。"""
    dated_rows = []
    seen_dates = set()
    for line_number, row in enumerate(rows[data_start:], data_start + 1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != width:
            raise ValueError(
                f"{line_number}行目の列数が一致しません: "
                f"expected={width}, actual={len(row)}"
            )
        try:
            day = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d").date()
        except ValueError as error:
            raise ValueError(
                f"{line_number}行目の日付を解釈できません: {row[0]!r}"
            ) from error
        if day in seen_dates:
            raise ValueError(f"日付が重複しています: {day.isoformat()}")
        seen_dates.add(day)
        dated_rows.append((day, row))

    if not dated_rows:
        raise ValueError("有効な日付データがありません")
    dated_rows.sort(key=lambda item: item[0])
    return dated_rows


def build_column_map(rows, station_index, metric_index, subheader_index):
    """(観測所, 気象項目)ごとに値・品質列の位置を整理する。"""
    stations = rows[station_index]
    metrics = rows[metric_index]
    subheaders = rows[subheader_index]
    wanted_metrics = {japanese for japanese, _ in FEATURES}

    column_map = {}
    station_order = []
    current_station = ""
    for column in range(1, len(metrics)):
        if stations[column].strip():
            current_station = stations[column].strip()
        if not current_station:
            raise ValueError(f"{column + 1}列目の観測所名がありません")
        if current_station not in station_order:
            station_order.append(current_station)

        metric = metrics[column].strip()
        if metric not in wanted_metrics:
            continue
        subheader = subheaders[column].strip()
        if subheader not in KNOWN_SUBHEADERS:
            raise ValueError(
                f"未対応の補助ヘッダーです: station={current_station}, "
                f"metric={metric}, subheader={subheader}"
            )

        component = {
            "": "value",
            "現象なし情報": "no_phenomenon",
            "品質情報": "quality",
            "均質番号": "homogeneity",
        }[subheader]
        key = (current_station, metric)
        if component in column_map.setdefault(key, {}):
            raise ValueError(
                f"列が重複しています: station={current_station}, "
                f"metric={metric}, component={component}"
            )
        column_map[key][component] = column
    return column_map, station_order


# 値の変換
def parse_number(text, context):
    """空欄をNone、数値文字列をintまたはfloatに変換する。"""
    text = text.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"数値を解釈できません: {context}, value={text!r}") from error
    return int(number) if number.is_integer() else number


def active_features(dated_rows, column_map):
    """少なくとも1件の実測値がある気象項目だけを返す。"""
    active = []
    for japanese, output_name in FEATURES:
        value_columns = [
            parts["value"]
            for (station, metric), parts in column_map.items()
            if metric == japanese and "value" in parts
        ]
        if any(row[column].strip() for _, row in dated_rows for column in value_columns):
            active.append((japanese, output_name))
    if not active:
        raise ValueError("選択した気象項目に実測値がありません")
    return active


def usable_stations(dated_rows, column_map, station_order, features):
    """選択項目のどれかに実測値がある観測所を返す。"""
    feature_names = {japanese for japanese, _ in features}
    usable = []
    dropped = []
    for station in station_order:
        value_columns = [
            parts["value"]
            for (candidate, metric), parts in column_map.items()
            if candidate == station
            and metric in feature_names
            and "value" in parts
        ]
        has_value = any(
            row[column].strip()
            for _, row in dated_rows
            for column in value_columns
        )
        (usable if has_value else dropped).append(station)

    if not usable:
        raise ValueError("利用可能な観測所がありません")
    return usable, dropped


def make_output_rows(dated_rows, column_map, stations, features):
    """1日×1観測所の出力行を生成する。"""
    output_rows = []
    for day, source_row in dated_rows:
        for station in stations:
            output = {"date": day.isoformat(), "station": station}
            for japanese, output_name in features:
                parts = column_map.get((station, japanese), {})
                value_column = parts.get("value")
                quality_column = parts.get("quality")
                context = (
                    f"date={day.isoformat()}, station={station}, metric={japanese}"
                )
                output[output_name] = (
                    parse_number(source_row[value_column], context)
                    if value_column is not None
                    else None
                )
                output[f"{output_name}_quality"] = (
                    parse_number(source_row[quality_column], context + ", quality")
                    if quality_column is not None
                    else None
                )
            output_rows.append(output)
    return output_rows


# 検査と出力
def validate_output(output_rows, dates, stations):
    """行数、主キー、日付範囲を検証する。"""
    expected_rows = len(dates) * len(stations)
    if len(output_rows) != expected_rows:
        raise ValueError(
            f"出力行数が不正です: expected={expected_rows}, "
            f"actual={len(output_rows)}"
        )

    keys = {(row["date"], row["station"]) for row in output_rows}
    if len(keys) != expected_rows:
        raise ValueError("出力の(date, station)に重複があります")

    expected_dates = {
        dates[0] + dt.timedelta(days=offset)
        for offset in range((dates[-1] - dates[0]).days + 1)
    }
    missing_dates = expected_dates - set(dates)
    if missing_dates:
        first_missing = min(missing_dates).isoformat()
        raise ValueError(
            f"入力期間内に{len(missing_dates)}日の欠落があります: "
            f"first={first_missing}"
        )


def write_output(path, output_rows, features):
    """UTF-8 BOM付き・LF改行のCSVとして保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "station"]
    for _, output_name in features:
        fieldnames.extend([output_name, f"{output_name}_quality"])

    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)


# 実行
def convert(input_path, output_path):
    """入力CSVを検証・変換し、処理結果の要約を返す。"""
    rows = read_source(input_path)
    station_index, metric_index, subheader_index, data_start = locate_headers(rows)
    width = len(rows[metric_index])
    dated_rows = parse_dates(rows, data_start, width)
    column_map, station_order = build_column_map(
        rows,
        station_index,
        metric_index,
        subheader_index,
    )
    features = active_features(dated_rows, column_map)
    stations, dropped_stations = usable_stations(
        dated_rows,
        column_map,
        station_order,
        features,
    )
    output_rows = make_output_rows(dated_rows, column_map, stations, features)
    dates = [day for day, _ in dated_rows]
    validate_output(output_rows, dates, stations)
    write_output(output_path, output_rows, features)

    return {
        "start": dates[0],
        "end": dates[-1],
        "days": len(dates),
        "stations": stations,
        "dropped_stations": dropped_stations,
        "features": [output_name for _, output_name in features],
        "rows": len(output_rows),
    }


def run():
    summary = convert(INPUT_CSV, OUTPUT_CSV)

    print("== 気象庁CSV変換完了 ==")
    print(f"入力      : {INPUT_CSV}")
    print(f"出力      : {OUTPUT_CSV}")
    print(f"期間      : {summary['start']}〜{summary['end']}")
    print(f"日数      : {summary['days']}")
    print(f"観測所    : {', '.join(summary['stations'])}")
    print(f"気象項目  : {', '.join(summary['features'])}")
    print(f"出力行数  : {summary['rows']:,}")
    if summary["dropped_stations"]:
        print(
            "全項目空で除外: "
            + ", ".join(summary["dropped_stations"])
        )


if __name__ == "__main__":
    run()
