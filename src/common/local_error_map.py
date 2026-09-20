"""ローカル検証の期間定義と、モデル共通の空間誤差マップ生成。

各実験の検証コードから ``save_local_error_map`` を呼ぶ。予測ロジックはこの
ファイルに持たせず、日付ごとの予測ODと正解ODだけを受け取る。

入力 : 呼び出し側が渡す予測OD・正解OD
出力 : experiments/<ID>/figures/local_error_map.png（呼び出し側が指定）
依存 : numpy, matplotlib
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP",
    "IPAexGothic", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from common.competition_metric import (
    EVAL_X, EVAL_Y, N_CELL, NORM_DIAG, NORM_OFF, NX, NY,
    in_eval_box, parse_gid,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent   # src/common/ から見たリポジトリルート
COAST_GEOJSON = ROOT / "data" / "processed" / "GeoJson" / "japan.geojson"

LON0, LON1 = 136.029, 138.042
LAT0, LAT1 = 36.203, 37.646
DLON = (LON1 - LON0) / (NX - 1)
DLAT = (LAT1 - LAT0) / (NY - 1)

VAL_WINDOWS = [
    ("jan", dt.date(2024, 1, 25), dt.date(2024, 1, 31)),
    ("apr", dt.date(2024, 4, 1), dt.date(2024, 4, 7)),
]


def to_int(day):
    return int(day.strftime("%Y%m%d"))


def validation_days():
    """既定の2窓14日を YYYYMMDD 整数で返す。"""
    return [
        to_int(start + dt.timedelta(offset))
        for _, start, end in VAL_WINDOWS
        for offset in range((end - start).days + 1)
    ]


def _add_error(diag_sse, out_sse, in_sse, origin, dest, error2):
    yo, xo = parse_gid(origin)
    yd, xd = parse_gid(dest)
    if origin == dest:
        diag_sse[yo - 1, xo - 1] += error2
    else:
        out_sse[yo - 1, xo - 1] += error2
        in_sse[yd - 1, xd - 1] += error2


def error_surfaces(predictions, truths, days):
    """指定日の誤差を3種類の2次元NRMSE面へ集約する。

    diagonal は各セルの時系列RMSE、off-origin / off-destination は各セルを
    起点・終点とする全評価内ペアのRMSE。欠損ペアは0として固定母数で割る。
    """
    days = [day for day in days if day in truths and day in predictions]
    if not days:
        raise ValueError("予測と正解がそろう検証日がありません")

    diag_sse = np.zeros((NY, NX), dtype=np.float64)
    out_sse = np.zeros((NY, NX), dtype=np.float64)
    in_sse = np.zeros((NY, NX), dtype=np.float64)

    for day in days:
        pred, truth = predictions[day], truths[day]

        for origin, truth_dests in truth.items():
            if not in_eval_box(origin):
                continue
            pred_dests = pred.get(origin, {})
            for dest, actual in truth_dests.items():
                if not in_eval_box(dest):
                    continue
                error2 = (pred_dests.get(dest, 0.0) - actual) ** 2
                _add_error(diag_sse, out_sse, in_sse, origin, dest, error2)

        for origin, pred_dests in pred.items():
            if not in_eval_box(origin):
                continue
            truth_dests = truth.get(origin, {})
            for dest, value in pred_dests.items():
                if not in_eval_box(dest) or dest in truth_dests:
                    continue
                _add_error(diag_sse, out_sse, in_sse, origin, dest, value ** 2)

    n_days = len(days)
    off_per_cell = N_CELL - 1
    return {
        "diagonal": np.sqrt(diag_sse / n_days) / NORM_DIAG,
        "off_origin": np.sqrt(out_sse / (n_days * off_per_cell)) / NORM_OFF,
        "off_destination": np.sqrt(in_sse / (n_days * off_per_cell)) / NORM_OFF,
    }


def _load_coast_rings(path=COAST_GEOJSON):
    path = Path(path)
    if not path.exists():
        return []
    gj = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feature in gj.get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates", [])
        polygons = coords if geometry.get("type") == "MultiPolygon" else [coords]
        for polygon in polygons:
            if polygon:
                rings.extend(np.asarray(ring, dtype=float) for ring in polygon if ring)
    return rings


def _grid_to_lon(x):
    return LON0 + (np.asarray(x, dtype=float) - 1) * DLON


def _grid_to_lat(y):
    return LAT0 + (np.asarray(y, dtype=float) - 1) * DLAT


def save_local_error_map(predictions, truths, out, experiment="model"):
    """14日統合の空間誤差マップ3枚を保存する。

    ``predictions`` と ``truths`` はどちらも ``{YYYYMMDD: nested_od}``。
    戻り値は保存先Path。
    """
    surfaces = error_surfaces(predictions, truths, validation_days())

    panels = [
        ("diagonal", "同一セル内誤差", "同一セル内における人流のNRMSE"),
        ("off_origin", "出発点における誤差", "各セルを出発する人流のNRMSE"),
        ("off_destination", "到着点における誤差", "各セルへ到着する人流のNRMSE"),
    ]
    lon_edges = _grid_to_lon(np.arange(EVAL_X[0] - 0.5, EVAL_X[1] + 1.0))
    lat_edges = _grid_to_lat(np.arange(EVAL_Y[0] - 0.5, EVAL_Y[1] + 1.0))
    coast = _load_coast_rings()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    for ax, (metric, title, subtitle) in zip(axes, panels):
        vmax = float(np.nanmax(surfaces[metric]))
        norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=max(vmax, 1e-12))
        grid = surfaces[metric][
            EVAL_Y[0] - 1:EVAL_Y[1], EVAL_X[0] - 1:EVAL_X[1]
        ]
        mesh = ax.pcolormesh(
            lon_edges, lat_edges, grid, cmap="Reds", norm=norm,
            shading="flat", rasterized=True, zorder=1,
        )
        for ring in coast:
            ax.plot(ring[:, 0], ring[:, 1], color="#374151", lw=0.55, zorder=2)
        ax.set_xlim(lon_edges[0], lon_edges[-1])
        ax.set_ylim(lat_edges[0], lat_edges[-1])
        ax.set_aspect(1.0 / np.cos(np.radians((LAT0 + LAT1) / 2)))
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("経度")
        ax.set_ylabel("緯度")
        ax.grid(color="#9ca3af", alpha=0.18, linewidth=0.35)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("NRMSE（濃い赤ほど誤差が大きい）")

    fig.suptitle(
        f"{experiment}_CV期間における空間誤差分布",
        fontsize=15,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out
