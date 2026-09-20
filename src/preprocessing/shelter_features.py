"""登録避難所を評価セル単位に集計し、確認図を作る。

実行: python3 src/preprocessing/shelter_features.py
入力: data/processed/{避難所/避難所.csv,events/cell_area_map.csv,GeoJson/}
出力: data/processed/避難所/避難所_セル別集計.csv、figures/避難所_*.png
依存: preprocessing/population_cell_mapping.py、matplotlib、numpy
"""

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

# common.* / preprocessing.* を解決するため src/ を sys.path に載せる。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.competition_metric import EVAL_X, EVAL_Y
from preprocessing import population_cell_mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SHELTER_DIR = ROOT / "data" / "processed" / "避難所"
INPUT_CSV = SHELTER_DIR / "避難所.csv"
CELL_MAP_CSV = ROOT / "data" / "processed" / "events" / "cell_area_map.csv"
BOUNDARIES_GEOJSON = (
    ROOT / "data" / "processed" / "GeoJson" / "municipality_boundaries_2024.geojson"
)
COASTLINE_GEOJSON = ROOT / "data" / "processed" / "GeoJson" / "japan.geojson"
OUTPUT_CSV = SHELTER_DIR / "避難所_セル別集計.csv"
EDA_FIGURE = ROOT / "figures" / "避難所_EDA.png"
MAP_FIGURE = ROOT / "figures" / "避難所_セル別施設数.png"

REQUIRED_COLUMNS = {
    "避難所ID",
    "避難所名称",
    "地域",
    "利用可否",
    "開設日時",
    "閉鎖日時",
    "収容可能人員",
    "避難世帯数",
    "避難者数",
    "緯度（WGS84）",
    "経度（WGS84）",
    "対応セル",
    "セル定義上の地区",
}

STATUS_ORDER = ["開設済", "閉鎖", "使用不可", "欠損"]
STATUS_COLORS = {
    "開設済": "#2F80ED",
    "閉鎖": "#9AA5B1",
    "使用不可": "#D64545",
    "欠損": "#D9B44A",
}


# 入力と検査
def read_csv(path, required_columns=None):
    """UTF-8 CSVを読み、必須列と空データを検査する。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSVがありません: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing = set(required_columns or []) - set(fieldnames)
        if missing:
            raise ValueError(f"{path.name}に必要な列がありません: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSVにデータ行がありません: {path}")
    return fieldnames, rows


def parse_number(value):
    """空欄をNone、それ以外をfloatへ変換する。"""
    text = str(value).strip().replace(",", "")
    return None if not text else float(text)


def parse_datetime(value):
    """避難所CSVの日付時刻をdatetimeへ変換する。"""
    text = str(value).strip()
    if not text:
        return None
    for date_format in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    raise ValueError(f"解釈できない日時です: {text}")


def parse_grid_id(grid_id):
    """y_x形式のグリッドIDを(y, x)へ変換する。"""
    parts = str(grid_id).strip().split("_")
    if len(parts) != 2:
        raise ValueError(f"不正なグリッドIDです: {grid_id}")
    try:
        y, x = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"不正なグリッドIDです: {grid_id}") from error
    return y, x


def load_cell_areas(path):
    """評価範囲1,476セルのIDと担当市町を読み込む。"""
    _, rows = read_csv(path, {"grid_id", "area"})
    areas = {}
    for row in rows:
        grid_id = row["grid_id"].strip()
        y, x = parse_grid_id(grid_id)
        if not (EVAL_X[0] <= x <= EVAL_X[1] and EVAL_Y[0] <= y <= EVAL_Y[1]):
            raise ValueError(f"評価範囲外のセルがcell_area_mapにあります: {grid_id}")
        if grid_id in areas:
            raise ValueError(f"cell_area_mapに重複セルがあります: {grid_id}")
        areas[grid_id] = row["area"].strip()

    expected_count = (
        (EVAL_X[1] - EVAL_X[0] + 1) * (EVAL_Y[1] - EVAL_Y[0] + 1)
    )
    if len(areas) != expected_count:
        raise ValueError(f"評価セル数が不正です: {len(areas)} != {expected_count}")
    return areas


def validate_shelters(rows, cell_areas):
    """ID、座標、セル、市町対応の整合性を検査する。"""
    shelter_ids = [row["避難所ID"].strip() for row in rows]
    if any(not shelter_id for shelter_id in shelter_ids):
        raise ValueError("避難所IDに空欄があります。")
    duplicate_ids = [
        shelter_id for shelter_id, count in Counter(shelter_ids).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"避難所IDが重複しています: {duplicate_ids[:5]}")

    mismatches = []
    for row in rows:
        parse_number(row["緯度（WGS84）"])
        parse_number(row["経度（WGS84）"])
        grid_id = row["対応セル"].strip()
        if not grid_id:
            raise ValueError(f"対応セルが空欄です: {row['避難所ID']}")
        y, x = parse_grid_id(grid_id)
        in_eval = EVAL_X[0] <= x <= EVAL_X[1] and EVAL_Y[0] <= y <= EVAL_Y[1]
        recorded_area = row["セル定義上の地区"].strip()
        if in_eval:
            expected_area = cell_areas.get(grid_id)
            if expected_area is None:
                raise ValueError(f"対応セルがcell_area_mapにありません: {grid_id}")
            if recorded_area != expected_area:
                mismatches.append((row["避難所ID"], grid_id, recorded_area, expected_area))
        elif recorded_area != "評価範囲外":
            mismatches.append((row["避難所ID"], grid_id, recorded_area, "評価範囲外"))

    if mismatches:
        example = "; ".join("/".join(item) for item in mismatches[:3])
        raise ValueError(f"避難所CSVとセル定義が一致しません: {example}")


# セル集計とCSV出力
def empty_cell_summary(grid_id, area):
    return {
        "grid_id": grid_id,
        "area": area,
        "facility_count": 0,
        "open_count": 0,
        "closed_count": 0,
        "unavailable_count": 0,
        "status_missing_count": 0,
        "capacity_known_count": 0,
        "capacity_sum": 0.0,
        "evacuee_known_count": 0,
        "evacuee_sum": 0.0,
    }


def aggregate_by_cell(rows, cell_areas):
    """評価セル別の施設数、状態、既知の収容・避難者数を集計する。"""
    summaries = {
        grid_id: empty_cell_summary(grid_id, area)
        for grid_id, area in cell_areas.items()
    }
    outside_count = 0
    for row in rows:
        grid_id = row["対応セル"].strip()
        if grid_id not in summaries:
            outside_count += 1
            continue

        summary = summaries[grid_id]
        summary["facility_count"] += 1
        status = row["利用可否"].strip()
        if status == "開設済":
            summary["open_count"] += 1
        elif status == "閉鎖":
            summary["closed_count"] += 1
        elif status == "使用不可":
            summary["unavailable_count"] += 1
        else:
            summary["status_missing_count"] += 1

        capacity = parse_number(row["収容可能人員"])
        if capacity is not None:
            summary["capacity_known_count"] += 1
            summary["capacity_sum"] += capacity

        evacuees = parse_number(row["避難者数"])
        if evacuees is not None:
            summary["evacuee_known_count"] += 1
            summary["evacuee_sum"] += evacuees

    return summaries, outside_count


def write_cell_summary(summaries, output_path):
    """全評価セルを含むモデル結合用の集計CSVを保存する。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "grid_id",
        "area",
        "facility_count",
        "open_count",
        "closed_count",
        "unavailable_count",
        "status_missing_count",
        "capacity_known_count",
        "capacity_sum",
        "evacuee_known_count",
        "evacuee_sum",
    ]
    ordered = sorted(
        summaries.values(),
        key=lambda row: parse_grid_id(row["grid_id"]),
    )
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            output = dict(row)
            output["capacity_sum"] = f"{output['capacity_sum']:.0f}"
            output["evacuee_sum"] = f"{output['evacuee_sum']:.0f}"
            writer.writerow(output)
    print(f"[write] {output_path}")


def records_in_evaluation_grid(rows, cell_areas):
    return [row for row in rows if row["対応セル"].strip() in cell_areas]


# 可視化
def add_bar_labels(ax, bars, horizontal=False, integer=True):
    """棒グラフへ値ラベルを付ける。"""
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        label = f"{int(round(value))}" if integer else f"{value:.1f}%"
        if horizontal:
            ax.text(
                bar.get_width() + max(ax.get_xlim()[1] * 0.01, 0.2),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )


def save_eda_figure(rows, evaluation_rows, output_path):
    """地域分布、状態、欠損、開閉月の4面EDA図を保存する。"""
    population_cell_mapping.configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle("避難所データの基礎集計", fontsize=18, y=0.98)

    area_counts = Counter(row["セル定義上の地区"] for row in evaluation_rows)
    areas = population_cell_mapping.AREA_ORDER
    values = [area_counts[area] for area in areas]
    bars = axes[0, 0].barh(
        areas[::-1],
        values[::-1],
        color=[population_cell_mapping.AREA_COLORS[area] for area in areas[::-1]],
    )
    axes[0, 0].set_title("評価範囲内の担当市町別施設数")
    axes[0, 0].set_xlabel("登録施設数")
    axes[0, 0].set_xlim(0, max(values) * 1.16)
    add_bar_labels(axes[0, 0], bars, horizontal=True)

    status_counts = Counter(row["利用可否"].strip() or "欠損" for row in rows)
    status_values = [status_counts[status] for status in STATUS_ORDER]
    bars = axes[0, 1].bar(
        STATUS_ORDER,
        status_values,
        color=[STATUS_COLORS[status] for status in STATUS_ORDER],
    )
    axes[0, 1].set_title("全データの利用可否")
    axes[0, 1].set_ylabel("登録施設数")
    axes[0, 1].set_ylim(0, max(status_values) * 1.14)
    add_bar_labels(axes[0, 1], bars)

    missing_columns = [
        "利用可否",
        "開設日時",
        "閉鎖日時",
        "収容可能人員",
        "避難世帯数",
        "避難者数",
    ]
    missing_rates = [
        100 * sum(not row[column].strip() for row in rows) / len(rows)
        for column in missing_columns
    ]
    bars = axes[1, 0].barh(
        missing_columns[::-1],
        missing_rates[::-1],
        color="#4E79A7",
    )
    axes[1, 0].set_title("主要列の欠損率")
    axes[1, 0].set_xlabel("欠損率（%）")
    axes[1, 0].set_xlim(0, max(missing_rates) * 1.25)
    add_bar_labels(axes[1, 0], bars, horizontal=True, integer=False)

    opened = Counter()
    closed = Counter()
    for row in rows:
        opened_at = parse_datetime(row["開設日時"])
        closed_at = parse_datetime(row["閉鎖日時"])
        if opened_at:
            opened[opened_at.strftime("%Y-%m")] += 1
        if closed_at:
            closed[closed_at.strftime("%Y-%m")] += 1
    months = sorted(set(opened) | set(closed))
    positions = np.arange(len(months))
    width = 0.38
    axes[1, 1].bar(
        positions - width / 2,
        [opened[month] for month in months],
        width,
        label="開設日時あり",
        color="#4E79A7",
    )
    axes[1, 1].bar(
        positions + width / 2,
        [closed[month] for month in months],
        width,
        label="閉鎖日時あり",
        color="#A0AEC0",
    )
    axes[1, 1].set_title("月別の開設・閉鎖記録数")
    axes[1, 1].set_ylabel("記録数")
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels([month.replace("2024-", "") + "月" for month in months])
    axes[1, 1].legend(frameon=False)

    for index, ax in enumerate(axes.ravel()):
        grid_axis = "x" if index in (0, 2) else "y"
        ax.grid(axis=grid_axis, color="#D8DEE6", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelrotation=0)

    fig.text(
        0.5,
        0.015,
        f"全{len(rows)}施設／評価グリッド内{len(evaluation_rows)}施設。施設数は閉鎖済みを含む登録件数。",
        ha="center",
        fontsize=10,
        color="#4A5568",
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.95), h_pad=2.2, w_pad=2.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


def summaries_to_grids(summaries):
    """セル集計を地図描画用の市町配列と施設数配列へ変換する。"""
    height = EVAL_Y[1] - EVAL_Y[0] + 1
    width = EVAL_X[1] - EVAL_X[0] + 1
    area_grid = np.full((height, width), "", dtype=object)
    facility_grid = np.full((height, width), np.nan, dtype=float)
    for summary in summaries.values():
        y, x = parse_grid_id(summary["grid_id"])
        row = y - EVAL_Y[0]
        col = x - EVAL_X[0]
        area = summary["area"]
        area_grid[row, col] = area
        if area in population_cell_mapping.AREA_ORDER:
            facility_grid[row, col] = summary["facility_count"]
    return area_grid, facility_grid


def save_facility_map(summaries, area_polygons, coastline, output_path):
    """評価セルごとの登録避難所数を行政境界・海岸線と重ねて保存する。"""
    population_cell_mapping.configure_plot_style()
    area_grid, facility_grid = summaries_to_grids(summaries)
    occupied = np.isfinite(facility_grid) & (facility_grid > 0)
    maximum = int(np.nanmax(facility_grid))
    total_facilities = int(np.nansum(facility_grid))

    colors = [
        "#F3F4F6",
        "#FFF3BF",
        "#FFD166",
        "#F4A261",
        "#E76F51",
        "#D1495B",
        "#8F1D2C",
    ]
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 9.5, max(10.5, maximum + 0.5)]
    cmap = ListedColormap(colors)
    cmap.set_bad(population_cell_mapping.OUTSIDE_COLOR)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(13.5, 10.8))
    population_cell_mapping.draw_basemap(ax, coastline)
    x_edges = np.arange(EVAL_X[0] - 0.5, EVAL_X[1] + 1.0)
    y_edges = np.arange(EVAL_Y[0] - 0.5, EVAL_Y[1] + 1.0)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        facility_grid,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=population_cell_mapping.GRID_COLOR,
        linewidth=0.18,
        antialiased=True,
        rasterized=True,
        zorder=2,
    )
    population_cell_mapping.draw_coastline(ax, coastline)
    population_cell_mapping.draw_area_boundaries(ax, area_polygons)

    for area in population_cell_mapping.AREA_ORDER:
        mask = area_grid == area
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        ax.text(
            float(np.mean(xs + EVAL_X[0])),
            float(np.mean(ys + EVAL_Y[0])),
            area,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1F2933",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=7,
        )

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        fraction=0.034,
        pad=0.025,
        boundaries=boundaries,
        ticks=[0, 1, 2, 3, 4.5, 7.5, (10 + maximum) / 2],
    )
    colorbar.ax.set_yticklabels(["0", "1", "2", "3", "4–5", "6–9", "10以上"])
    colorbar.set_label("登録避難所数（施設）")

    ax.set_xlim(EVAL_X[0] - 0.5, EVAL_X[1] + 0.5)
    ax.set_ylim(EVAL_Y[0] - 0.5, EVAL_Y[1] + 0.5)
    ax.set_xticks(np.arange(30, 71, 5))
    ax.set_yticks(np.arange(35, 71, 5))
    ax.set_aspect(population_cell_mapping.DLAT / (population_cell_mapping.DLON * np.cos(np.radians(37.25))))
    ax.set_xlabel("グリッド x（セル番号）")
    ax.set_ylabel("グリッド y（セル番号）")
    ax.set_title("評価グリッド内の避難所施設数（セル別）", fontsize=17, pad=14)
    ax.tick_params(labelsize=9)
    ax.text(
        0.01,
        0.01,
        f"登録{total_facilities}施設／施設あり{int(np.sum(occupied))}セル／最大{maximum}施設",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#1F2933",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#A0AEC0",
            "alpha": 0.88,
        },
        zorder=8,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.93)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")


# 実行と要約
def print_summary(rows, evaluation_rows, summaries, outside_count):
    """主要なEDA結果を端末へ表示する。"""
    occupied = [row for row in summaries.values() if row["facility_count"] > 0]
    status_counts = Counter(row["利用可否"].strip() or "欠損" for row in rows)
    area_counts = Counter(row["セル定義上の地区"] for row in evaluation_rows)
    source_area_mismatches = sum(
        row["地域"].strip() != row["セル定義上の地区"].strip()
        for row in evaluation_rows
    )

    print("\n== 避難所データ ==")
    print(f"  登録施設              : {len(rows):>4}件")
    print(f"  評価グリッド内        : {len(evaluation_rows):>4}件")
    print(f"  評価グリッド外        : {outside_count:>4}件")
    print(f"  施設がある評価セル    : {len(occupied):>4}セル")
    print(f"  元地域とセル地区の差  : {source_area_mismatches:>4}件")

    print("\n== 利用可否 ==")
    for status in STATUS_ORDER:
        print(f"  {status:<8}: {status_counts[status]:>4}件")

    print("\n== 評価範囲内の担当市町別施設数 ==")
    for area in population_cell_mapping.AREA_ORDER:
        print(f"  {area:<5}: {area_counts[area]:>3}件")

    print("\n== 施設数が多いセル ==")
    for summary in sorted(
        occupied,
        key=lambda row: (-row["facility_count"], row["grid_id"]),
    )[:10]:
        print(
            f"  {summary['grid_id']:<5} {summary['area']:<5} "
            f"{summary['facility_count']:>2}施設"
        )


def run():
    _, rows = read_csv(INPUT_CSV, REQUIRED_COLUMNS)
    cell_areas = load_cell_areas(CELL_MAP_CSV)
    validate_shelters(rows, cell_areas)
    summaries, outside_count = aggregate_by_cell(rows, cell_areas)
    evaluation_rows = records_in_evaluation_grid(rows, cell_areas)

    write_cell_summary(summaries, OUTPUT_CSV)
    save_eda_figure(rows, evaluation_rows, EDA_FIGURE)

    area_polygons = population_cell_mapping.load_area_polygons(BOUNDARIES_GEOJSON)
    coastline = population_cell_mapping.load_coastline_polygons(COASTLINE_GEOJSON)
    save_facility_map(
        summaries,
        area_polygons,
        coastline,
        MAP_FIGURE,
    )
    print_summary(rows, evaluation_rows, summaries, outside_count)


if __name__ == "__main__":
    run()
