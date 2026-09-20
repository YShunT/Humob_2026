"""HuMob実験で共通に使う日付・OD TSV入出力・提出検査。

使い方:
  from common.experiment_io import read_od_days, submission_days, write_od_tsv

入力 : data/raw/humob2026-dataset.tsv または同形式のTSV
出力 : 呼び出し側が指定する提出TSV
依存 : common/competition_metric.py、標準ライブラリ

予測手法は置かない。実験コードから重複しやすい、データ形式に依存した処理だけを扱う。
"""

import ast
import csv
import datetime as dt
import json
import math
from pathlib import Path

from common.competition_metric import valid_gid


ROOT = Path(__file__).resolve().parents[2]
RAW_TSV = ROOT / "data" / "raw" / "humob2026-dataset.tsv"
TEST_SPAN = (dt.date(2024, 2, 1), dt.date(2024, 3, 31))
TEST_EXCLUDE = {20240202, 20240305}
DAILY_TOTAL = 189190.2509


def to_int(day):
    return int(day.strftime("%Y%m%d"))


def to_date(day):
    return dt.date(day // 10000, day // 100 % 100, day % 100)


def date_range(start, end):
    """start〜end（両端含む）をYYYYMMDD整数で返す。"""
    return [to_int(start + dt.timedelta(days=offset))
            for offset in range((end - start).days + 1)]


def submission_days():
    """提出対象58日を返す。"""
    return [day for day in date_range(*TEST_SPAN) if day not in TEST_EXCLUDE]


def read_od_days(path, keep=lambda _: True):
    """同形式TSVから必要な観測日だけ読み、(rows, na_days)を返す。"""
    rows, na_days = {}, []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip() or "\t" not in line:
                continue
            date_text, payload = line.rstrip("\n").split("\t", 1)
            try:
                day = int(date_text)
            except ValueError:
                continue
            if not keep(day):
                continue
            if payload.strip() == "NA":
                na_days.append(day)
            else:
                if day in rows:
                    raise ValueError(f"日付が重複しています: {day}")
                rows[day] = ast.literal_eval(payload)
    return rows, sorted(na_days)


def write_od_tsv(predictions, path):
    """日付→OD辞書を配布データと同じTSV形式で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for day in sorted(predictions):
            destination.write(f"{day}\t{predictions[day]!r}\n")
    print(f"[write] {path} ({len(predictions)}日, {path.stat().st_size / 1e6:.1f}MB)")
    return path


def write_json(path, value):
    """JSONをUTF-8・NaN禁止で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _round_floats(value, decimals):
    if isinstance(value, float):
        if value and abs(value) < 10 ** -decimals:
            return float(f"{value:.{decimals}g}")
        return round(value, decimals)
    if isinstance(value, dict):
        return {key: _round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, decimals) for item in value]
    return value


def write_metrics(path, metrics):
    """metrics.jsonを共通順序・小数6桁（微小値は保持）で保存する。"""
    order = (
        "experiment", "date", "status", "compared_to",
        "results", "config",
    )
    readable = {key: metrics[key] for key in order if key in metrics}
    readable.update({key: value for key, value in metrics.items() if key not in readable})
    return write_json(path, _round_floats(readable, 6))


def write_records_csv(path, rows):
    """同じキーを持つ辞書列をCSVへ保存する。"""
    rows = list(rows)
    if not rows:
        raise ValueError("CSVへ保存する行がありません")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def check_od_tsv(path, expected_days, expected_daily_total=None):
    """日付・辞書構造・ID・有限非負値と、必要なら日次総量を検査する。"""
    path = Path(path)
    rows, _ = read_od_days(path)
    if set(rows) != set(expected_days):
        missing = sorted(set(expected_days) - set(rows))
        extra = sorted(set(rows) - set(expected_days))
        raise ValueError(f"提出日が不正です: missing={missing}, extra={extra}")

    totals = []
    for day, od in rows.items():
        if not isinstance(od, dict):
            raise ValueError(f"{day}: ODが辞書ではありません")
        total = 0.0
        for origin, destinations in od.items():
            if not valid_gid(origin) or not isinstance(destinations, dict):
                raise ValueError(f"{day}: 起点または行が不正です: {origin}")
            for destination, value in destinations.items():
                if not valid_gid(destination):
                    raise ValueError(f"{day}: 終点IDが不正です: {destination}")
                if (not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < 0):
                    raise ValueError(f"{day}: 値が有限非負ではありません")
                total += value
        totals.append(total)

    if expected_daily_total is not None:
        worst = max(abs(total / expected_daily_total - 1.0) for total in totals)
        if worst > 1e-9:
            raise ValueError(f"日次総量が不正です: 最大相対誤差={worst:.2e}")
    print(f"[check] PASS {path}: {len(rows)}日")
    return rows
