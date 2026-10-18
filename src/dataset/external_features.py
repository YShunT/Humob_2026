"""前処理済みCSVを読み、評価セルごとの日次外部特徴（避難者率・断水率・降水・積雪）と避難所施設数を、成分別の説明変数に組み立てる。
形状   : T = 暦日数、C = 評価セル数、P = 候補OD対数、S = 系列数（C か P）、K = 特徴列数、M = 観測所数（6）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CELL_MAP_CSV, CONSTANT_SD, EVACUATION_RATE_CSV, SHELTER_CELL_CSV, STATIONS,
                    WATER_OUTAGE_RATE_CSV, WEATHER_CSV)
from dataset.od_matrix import FlowView
from dataset.periods import CALENDAR, QUAKE_INDEX

CALENDAR_NS = np.array(CALENDAR, dtype="datetime64[ns]")
RATE_SOURCES = {"evac": (EVACUATION_RATE_CSV, "避難者率"), "water": (WATER_OUTAGE_RATE_CSV, "断水世帯率")}
WEATHER_SOURCES = {"rain": "precipitation_total_mm", "snow": "snow_depth_max_cm"}


@dataclass(frozen=True)
class ExternalFeatures:
    """評価セルごとの日次外部特徴（evac・water・rain・snow）と静的な避難所施設数。"""

    daily: dict[str, np.ndarray]
    facility: np.ndarray  # (C,)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """2点間の大円距離をkmで返す。"""
    a, b = np.radians(lat1), np.radians(lat2)
    dlat, dlon = b - a, np.radians(lon2) - np.radians(lon1)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(
        np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2, 0, 1)))  # (C,)


def area_rate_matrix(path: Path, column: str, area: np.ndarray) -> np.ndarray:
    """市町別の公表値を各日の直近公表値で前方補完してセルへ配り、震災前は0にする。"""
    table = pd.read_csv(path)
    table["date"] = pd.to_datetime(table["日付"])
    values = np.full((len(CALENDAR), len(area)), np.nan)
    for name in np.unique(area):
        report = table.loc[table["市町名"] == name].sort_values("date")
        if report.empty:
            continue
        position = np.searchsorted(report.date.to_numpy("datetime64[ns]"), CALENDAR_NS, side="right") - 1
        reported = pd.to_numeric(report[column], errors="coerce").to_numpy()
        value = np.where(position >= 0, reported[np.maximum(position, 0)], np.nan)
        value[:QUAKE_INDEX] = 0
        values[:, area == name] = value[:, None]
    return values  # (T, C)


def nearest_station_weather(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """各日・各セルに、その日に有効値のある最寄り観測所の log1p(降水量・積雪) を割り当てる。"""
    weather = pd.read_csv(WEATHER_CSV)
    weather["date"] = pd.to_datetime(weather.date)
    names = list(STATIONS)
    distance = np.stack([haversine_km(lat, lon, *STATIONS[name]) for name in names], axis=1)  # (C, M)
    result = {}
    for group, column in WEATHER_SOURCES.items():
        table = weather.pivot(index="date", columns="station", values=column)
        station_values = table.reindex(index=pd.DatetimeIndex(CALENDAR_NS), columns=names).to_numpy(float)
        values = np.full((len(CALENDAR), len(lat)), np.nan)
        for t, row in enumerate(station_values):
            usable = np.isfinite(row) & (row >= 0)
            if usable.any():
                nearest = np.argmin(np.where(usable[None, :], distance, np.inf), axis=1)
                values[t] = np.log1p(row[nearest])
        result[group] = values
    return result  # (T, C)


def load_external_features(cells: np.ndarray) -> ExternalFeatures:
    """評価セルの並びに合わせて全外部特徴を読み込む。"""
    cell_table = pd.read_csv(CELL_MAP_CSV).set_index("grid_id").loc[cells]
    area = cell_table.area.fillna("対象外").to_numpy(str)
    daily = {group: area_rate_matrix(path, column, area) for group, (path, column) in RATE_SOURCES.items()}
    daily.update(nearest_station_weather(cell_table.latitude.to_numpy(float), cell_table.longitude.to_numpy(float)))
    facility = pd.read_csv(SHELTER_CELL_CSV).set_index("grid_id").loc[cells].facility_count.to_numpy(float)
    return ExternalFeatures(daily, facility)


def cell_features(external: ExternalFeatures, groups: tuple[str, ...]) -> np.ndarray:
    """セル単位の系列（対角・総量）用に、各群の特徴を並べる（shelter は避難者率×log1p施設数）。"""
    columns = [external.daily["evac"] * np.log1p(external.facility) if group == "shelter" else external.daily[group]
               for group in groups]
    return np.stack(columns, axis=-1)  # (T, C, K)


def pair_features(external: ExternalFeatures, groups: tuple[str, ...], view: FlowView) -> np.ndarray:
    """OD対単位の配分系列用に、anchor 側・partner 側の特徴と partner 側の避難所特徴を並べる。"""
    shape = view.ratio.shape
    columns = []
    for group in groups:
        if group == "shelter":
            log_facility = np.log1p(external.facility[view.partner])
            columns.extend([np.broadcast_to(log_facility, shape),
                            np.broadcast_to((external.facility[view.partner] > 0).astype(float), shape),
                            external.daily["evac"][:, view.partner] * log_facility])
        else:
            columns.extend([external.daily[group][:, view.anchor], external.daily[group][:, view.partner]])
    return np.stack(columns, axis=-1)  # (T, P, K)


def standardize(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """学習マスク上の統計だけで欠損を平均補完し、欠損フラグを足して標準化する（定数列は落とす）。"""
    columns = []
    for j in range(raw.shape[-1]):
        x = raw[:, :, j]
        finite = np.isfinite(x)
        training = x[mask & finite]
        if training.size == 0:
            continue
        for column in (np.where(finite, x, float(training.mean())), (~finite).astype(float)):
            mu = float(column[mask].mean())
            sd = float(column[mask].std())
            if sd >= CONSTANT_SD:
                columns.append((column - mu) / sd)
    return np.stack(columns, axis=-1)  # (T, S, K)
