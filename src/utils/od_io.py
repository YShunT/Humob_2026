"""配布データと同じ形式のOD TSV（日付\t{origin: {destination: value}}）の読み書きと、metrics.json の保存を行う。

使い方 : from utils.od_io import OD, iter_od_days, write_od_tsv, write_metrics
入力   : OD形式のTSV
出力   : 呼び出し側が指定するTSV・JSON
依存   : 標準ライブラリのみ
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterator

OD = dict[str, dict[str, float]]
METRICS_KEY_ORDER = ("experiment", "date", "status", "compared_to", "results", "config")


def iter_od_days(path: Path) -> Iterator[tuple[int, OD]]:
    """OD形式TSVを1行ずつ読み、NA以外の日について (YYYYMMDD, OD辞書) を返す。"""
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            date_text, payload = line.rstrip("\n").split("\t", 1)
            if payload.strip() != "NA":
                yield int(date_text), ast.literal_eval(payload)


def write_od_tsv(predictions: dict[int, OD], path: Path) -> Path:
    """日付→OD辞書を配布データと同じ形式のTSVへ日付順に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for day in sorted(predictions):
            destination.write(f"{day}\t{predictions[day]!r}\n")
    print(f"[write] {path} ({len(predictions)}日, {path.stat().st_size / 1e6:.1f}MB)")
    return path


def round_floats(value: Any, decimals: int = 6) -> Any:
    """入れ子の辞書・リスト内の float を小数 decimals 桁へ丸め、微小値は有効桁で残す。"""
    if isinstance(value, float):
        if value and abs(value) < 10 ** -decimals:
            return float(f"{value:.{decimals}g}")
        return round(value, decimals)
    if isinstance(value, dict):
        return {key: round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, decimals) for item in value]
    return value


def write_metrics(path: Path, metrics: dict[str, Any]) -> Path:
    """metrics.json を共通のキー順・小数6桁・NaN禁止で保存する。"""
    ordered = {key: metrics[key] for key in METRICS_KEY_ORDER if key in metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(ordered), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(f"[write] {path}")
    return path
