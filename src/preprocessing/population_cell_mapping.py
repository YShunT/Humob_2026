"""評価セルを地理条件で能登8市町へ割り当て、CSVと確認図を作る。

実行: python3 src/preprocessing/population_cell_mapping.py
入力: data/processed/{events/population.csv,GeoJson/}
出力: data/processed/events/{population_cell_map.csv,cell_area_map.csv}、figures/population_cell_map.png
依存: common/{competition_metric,grid_geo_map}.py、matplotlib、numpy
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

# common.* を解決するため src/ を sys.path に載せる。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.competition_metric import EVAL_X, EVAL_Y, NX, NY
from common import grid_geo_map as e1


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVENTS_DIR = ROOT / "data" / "processed" / "events"

POPULATION_CSV = EVENTS_DIR / "population.csv"
GEOJSON_DIR = ROOT / "data" / "processed" / "GeoJson"
BOUNDARIES_GEOJSON = GEOJSON_DIR / "municipality_boundaries_2024.geojson"
COASTLINE_GEOJSON = GEOJSON_DIR / "japan.geojson"
OUTPUT_CSV = EVENTS_DIR / "population_cell_map.csv"
OUTPUT_AREA_CSV = EVENTS_DIR / "cell_area_map.csv"
OUTPUT_FIGURE = ROOT / "figures" / "population_cell_map.png"
OUTSIDE_AREA = "対象8市町外・富山県"

# HuMob配布データのグリッド定義。
LON0, LON1 = 136.029, 138.042
LAT0, LAT1 = 36.203, 37.646
DLON = (LON1 - LON0) / (NX - 1)
DLAT = (LAT1 - LAT0) / (NY - 1)

MAX_AREA_CELL_DISTANCE = 2

AREA_ORDER = [
    "七尾市",
    "珠洲市",
    "穴水町",
    "能登町",
    "輪島市",
    "志賀町",
    "中能登町",
    "羽咋市",
]
AREA_COLORS = {
    "七尾市": "#76B7B2",
    "珠洲市": "#4E79A7",
    "穴水町": "#F28E2B",
    "能登町": "#59A14F",
    "輪島市": "#E15759",
    "志賀町": "#B07AA1",
    "中能登町": "#EDC948",
    "羽咋市": "#9C755F",
}

SEA_COLOR = "#DCECF6"
LAND_COLOR = "#E4EFDF"
COAST_COLOR = "#365A73"
BOUNDARY_COLOR = "#273B47"
GRID_COLOR = "#455A64"
OUTSIDE_COLOR = "#D4D7DA"


# 座標と入力
def configure_plot_style():
    """日本語表示用のフォント候補と基本設定を適用する。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def grid_to_lon(x):
    return LON0 + (np.asarray(x, dtype=float) - 1) * DLON


def grid_to_lat(y):
    return LAT0 + (np.asarray(y, dtype=float) - 1) * DLAT


def lon_to_grid(lon):
    return 1 + (np.asarray(lon, dtype=float) - LON0) / DLON


def lat_to_grid(lat):
    return 1 + (np.asarray(lat, dtype=float) - LAT0) / DLAT


def load_population(path):
    """対象8市町の2023年12月末人口を読み込む。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"人口CSVがありません: {path}")

    populations = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"area", "population(2023/12)"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"人口CSVに必要な列がありません: {sorted(missing)}")
        for row in reader:
            area = row["area"].strip()
            if area in AREA_ORDER:
                populations[area] = int(row["population(2023/12)"])

    missing_areas = set(AREA_ORDER) - set(populations)
    if missing_areas:
        raise ValueError(f"人口CSVに対象市町がありません: {sorted(missing_areas)}")
    return populations


def polygon_rings(geometry):
    """Polygon / MultiPolygonを外周・穴の組へ分解する。"""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        raw_polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        raw_polygons = coordinates
    else:
        return

    for polygon in raw_polygons:
        if not polygon:
            continue
        exterior = np.asarray(polygon[0], dtype=float)
        holes = [
            np.asarray(ring, dtype=float)
            for ring in polygon[1:]
            if len(ring) >= 3
        ]
        yield exterior, holes


def load_area_polygons(path):
    """8市町の行政区域GeoJSONを市町別に読み込む。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"行政区域GeoJSONがありません: {path}")

    geojson = json.loads(path.read_text(encoding="utf-8"))
    by_area = {area: [] for area in AREA_ORDER}
    for feature in geojson.get("features", []):
        properties = feature.get("properties") or {}
        area = properties.get("area") or properties.get("N03_004")
        if area not in by_area:
            continue
        by_area[area].extend(
            polygon_rings(feature.get("geometry") or {})
        )

    missing_areas = [area for area, polygons in by_area.items() if not polygons]
    if missing_areas:
        raise ValueError(
            f"行政区域GeoJSONに対象市町がありません: {missing_areas}"
        )
    return by_area


def load_coastline_polygons(path):
    """背景表示用GeoJSONを外周・穴の組として読み込む。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"海岸線GeoJSONがありません: {path}")
    geojson = json.loads(path.read_text(encoding="utf-8"))
    polygons = []
    for feature in geojson.get("features", []):
        polygons.extend(polygon_rings(feature.get("geometry") or {}))
    return polygons


def load_prefecture_polygons(path):
    """背景GeoJSONを都道府県別のポリゴンとして読み込む。"""
    path = Path(path)
    geojson = json.loads(path.read_text(encoding="utf-8"))
    by_prefecture = {}
    for feature in geojson.get("features", []):
        properties = feature.get("properties") or {}
        prefecture = properties.get("nam_ja", "")
        if not prefecture:
            continue
        by_prefecture.setdefault(prefecture, []).extend(
            polygon_rings(feature.get("geometry") or {})
        )
    return by_prefecture


# セルと行政区域の幾何計算
def clip_ring_to_rectangle(ring, bounds):
    """ポリゴンリングを軸平行なセル矩形でクリップする。"""
    xmin, xmax, ymin, ymax = bounds
    vertices = [tuple(point[:2]) for point in np.asarray(ring, dtype=float)]
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]

    def clip(vertices, inside, intersect):
        if not vertices:
            return []
        output = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def vertical_intersection(start, end, x_value):
        dx = end[0] - start[0]
        ratio = 0.0 if dx == 0 else (x_value - start[0]) / dx
        return (x_value, start[1] + ratio * (end[1] - start[1]))

    def horizontal_intersection(start, end, y_value):
        dy = end[1] - start[1]
        ratio = 0.0 if dy == 0 else (y_value - start[1]) / dy
        return (start[0] + ratio * (end[0] - start[0]), y_value)

    vertices = clip(
        vertices,
        lambda point: point[0] >= xmin,
        lambda start, end: vertical_intersection(start, end, xmin),
    )
    vertices = clip(
        vertices,
        lambda point: point[0] <= xmax,
        lambda start, end: vertical_intersection(start, end, xmax),
    )
    vertices = clip(
        vertices,
        lambda point: point[1] >= ymin,
        lambda start, end: horizontal_intersection(start, end, ymin),
    )
    return clip(
        vertices,
        lambda point: point[1] <= ymax,
        lambda start, end: horizontal_intersection(start, end, ymax),
    )


def ring_area(vertices):
    """クリップ後リングの符号なし面積を返す。"""
    if len(vertices) < 3:
        return 0.0
    points = np.asarray(vertices, dtype=float)
    return 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )


def polygons_overlap_area(polygons, bounds):
    """行政区域ポリゴンとセル矩形の交差面積を経緯度平面上で求める。"""
    xmin, xmax, ymin, ymax = bounds
    total = 0.0
    for exterior, holes in polygons:
        if (
            exterior[:, 0].max() < xmin
            or exterior[:, 0].min() > xmax
            or exterior[:, 1].max() < ymin
            or exterior[:, 1].min() > ymax
        ):
            continue
        total += ring_area(clip_ring_to_rectangle(exterior, bounds))
        for hole in holes:
            total -= ring_area(clip_ring_to_rectangle(hole, bounds))
    return max(total, 0.0)


def cell_bounds(x, y):
    """グリッドセルの経緯度境界を返す。"""
    return (
        float(grid_to_lon(x - 0.5)),
        float(grid_to_lon(x + 0.5)),
        float(grid_to_lat(y - 0.5)),
        float(grid_to_lat(y + 0.5)),
    )


def build_geometry_masks(coastline):
    """E1地図と同じ幾何定義で陸・沿岸範囲のマスクを作る。"""
    land = e1.build_land_mask(coastline)
    coast = e1.build_coast_cells(coastline)
    near_coast = e1.dilate8(coast)

    y_slice = slice(EVAL_Y[0] - 1, EVAL_Y[1])
    x_slice = slice(EVAL_X[0] - 1, EVAL_X[1])
    return {
        "land": land[y_slice, x_slice],
        "coast": coast[y_slice, x_slice],
        "near_coast": near_coast[y_slice, x_slice],
    }


def nearest_area_for_cell(x, y, administrative_area, max_distance=None):
    """行政区域セルとの距離が最小の市町を返す。"""
    best = None
    for area_index, area in enumerate(AREA_ORDER):
        area_ys, area_xs = np.where(administrative_area == area)
        if len(area_xs) == 0:
            continue
        cell_xs = area_xs + EVAL_X[0]
        cell_ys = area_ys + EVAL_Y[0]
        chebyshev = np.maximum(np.abs(cell_xs - x), np.abs(cell_ys - y))
        allowed = (
            np.ones_like(chebyshev, dtype=bool)
            if max_distance is None
            else chebyshev <= max_distance
        )
        if not np.any(allowed):
            continue
        euclidean2 = (cell_xs - x) ** 2 + (cell_ys - y) ** 2
        distance2 = int(np.min(euclidean2[allowed]))
        candidate = (distance2, area_index, area)
        if best is None or candidate < best:
            best = candidate
    return "" if best is None else best[2]


def nearest_prefecture_for_cell(x, y, mainland_prefecture):
    """海上セルに最も近い本土セルの県を返す。同距離では富山県を優先する。"""
    best = None
    for priority, prefecture_name in enumerate(["富山県", "石川県"]):
        prefecture_ys, prefecture_xs = np.where(
            mainland_prefecture == prefecture_name
        )
        if len(prefecture_xs) == 0:
            continue
        cell_xs = prefecture_xs + EVAL_X[0]
        cell_ys = prefecture_ys + EVAL_Y[0]
        distance2 = int(np.min((cell_xs - x) ** 2 + (cell_ys - y) ** 2))
        candidate = (distance2, priority, prefecture_name)
        if best is None or candidate < best:
            best = candidate
    return "" if best is None else best[2]


# セル割り当て
def assign_cells(area_polygons, prefecture_polygons, geometry_masks):
    """最大重なり面積と沿岸距離により評価セルを8市町へ割り当てる。"""
    xs = np.arange(EVAL_X[0], EVAL_X[1] + 1)
    ys = np.arange(EVAL_Y[0], EVAL_Y[1] + 1)
    x_grid, y_grid = np.meshgrid(xs, ys)
    lon_grid, lat_grid = np.meshgrid(grid_to_lon(xs), grid_to_lat(ys))
    administrative = np.full(x_grid.shape, "", dtype=object)
    prefecture = np.full(x_grid.shape, "", dtype=object)
    overlap_ratio = np.zeros(x_grid.shape, dtype=float)
    for row, col in np.ndindex(x_grid.shape):
        x = int(x_grid[row, col])
        y = int(y_grid[row, col])
        bounds = cell_bounds(x, y)
        cell_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])

        area_overlaps = [
            polygons_overlap_area(area_polygons[area], bounds)
            for area in AREA_ORDER
        ]
        largest_area_index = int(np.argmax(area_overlaps))
        largest_area_overlap = area_overlaps[largest_area_index]
        if largest_area_overlap > 0:
            administrative[row, col] = AREA_ORDER[largest_area_index]
            overlap_ratio[row, col] = largest_area_overlap / cell_area

        prefecture_names = list(prefecture_polygons)
        prefecture_overlaps = [
            polygons_overlap_area(prefecture_polygons[name], bounds)
            for name in prefecture_names
        ]
        largest_prefecture_index = int(np.argmax(prefecture_overlaps))
        if prefecture_overlaps[largest_prefecture_index] > 0:
            prefecture[row, col] = prefecture_names[largest_prefecture_index]

    assigned = np.full(x_grid.shape, "", dtype=object)
    method = np.full(x_grid.shape, "outside", dtype=object)
    nearest_coastal_prefecture = np.full(x_grid.shape, "", dtype=object)

    mainland_administrative = administrative.copy()
    mainland_administrative[~geometry_masks["land"]] = ""
    mainland_prefecture = prefecture.copy()
    mainland_prefecture[~geometry_masks["land"]] = ""

    land_assignment = (
        geometry_masks["land"]
        & (mainland_administrative != "")
        & (prefecture != "富山県")
    )
    assigned[land_assignment] = mainland_administrative[land_assignment]
    method[land_assignment] = "largest_overlap_area"

    unassigned_land = (
        geometry_masks["land"]
        & (prefecture != "")
        & (prefecture != "富山県")
        & (assigned == "")
    )
    for row, col in zip(*np.where(unassigned_land)):
        area = nearest_area_for_cell(
            int(x_grid[row, col]),
            int(y_grid[row, col]),
            mainland_administrative,
        )
        if area:
            assigned[row, col] = area
            method[row, col] = "nearest_land_area"

    coastal_candidates = (
        (~geometry_masks["land"])
        & geometry_masks["near_coast"]
        & (prefecture != "富山県")
        & (assigned == "")
    )
    for row, col in zip(*np.where(coastal_candidates)):
        coastal_prefecture = nearest_prefecture_for_cell(
            int(x_grid[row, col]),
            int(y_grid[row, col]),
            mainland_prefecture,
        )
        nearest_coastal_prefecture[row, col] = coastal_prefecture
        if coastal_prefecture != "石川県":
            continue
        area = nearest_area_for_cell(
            int(x_grid[row, col]),
            int(y_grid[row, col]),
            mainland_administrative,
            max_distance=MAX_AREA_CELL_DISTANCE,
        )
        if area:
            assigned[row, col] = area
            method[row, col] = "nearest_coastal_area"

    return {
        "x": x_grid,
        "y": y_grid,
        "lon": lon_grid,
        "lat": lat_grid,
        "area": assigned,
        "administrative_area": administrative,
        "mainland_administrative_area": mainland_administrative,
        "municipality_overlap_ratio": overlap_ratio,
        "prefecture": prefecture,
        "nearest_coastal_prefecture": nearest_coastal_prefecture,
        "assignment_method": method,
        **geometry_masks,
    }


# CSV出力
def write_cell_map(cell_map, populations, output_path):
    """全評価セルと市町割り当てをCSVへ保存する。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "grid_id",
            "grid_x",
            "grid_y",
            "longitude",
            "latitude",
            "area",
            "population_2023_12",
            "assignment_method",
            "prefecture",
            "nearest_coastal_prefecture",
            "municipality_overlap_ratio",
            "is_land",
            "within_coastal_range",
        ])
        for x, y, lon, lat, area, method, prefecture, coastal_prefecture, overlap, land, near_coast in zip(
            cell_map["x"].ravel(),
            cell_map["y"].ravel(),
            cell_map["lon"].ravel(),
            cell_map["lat"].ravel(),
            cell_map["area"].ravel(),
            cell_map["assignment_method"].ravel(),
            cell_map["prefecture"].ravel(),
            cell_map["nearest_coastal_prefecture"].ravel(),
            cell_map["municipality_overlap_ratio"].ravel(),
            cell_map["land"].ravel(),
            cell_map["near_coast"].ravel(),
        ):
            writer.writerow([
                f"{int(y)}_{int(x)}",
                int(x),
                int(y),
                f"{lon:.7f}",
                f"{lat:.7f}",
                area,
                populations.get(area, ""),
                method,
                prefecture,
                coastal_prefecture,
                f"{overlap:.6f}",
                int(land),
                int(near_coast),
            ])
    print(f"[write] {output_path}")
    return output_path


def write_cell_area_map(cell_map, output_path):
    """全評価セルについて、グリッドIDと担当地域名だけをCSVへ保存する。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["grid_id", "area"])
        for x, y, area in zip(
            cell_map["x"].ravel(),
            cell_map["y"].ravel(),
            cell_map["area"].ravel(),
        ):
            writer.writerow([
                f"{int(y)}_{int(x)}",
                area or OUTSIDE_AREA,
            ])
    print(f"[write] {output_path}")
    return output_path


# 可視化
def draw_basemap(ax, polygons):
    ax.set_facecolor(SEA_COLOR)
    for exterior, holes in polygons:
        ax.fill(
            lon_to_grid(exterior[:, 0]),
            lat_to_grid(exterior[:, 1]),
            facecolor=LAND_COLOR,
            edgecolor="none",
            zorder=0,
        )
        for hole in holes:
            ax.fill(
                lon_to_grid(hole[:, 0]),
                lat_to_grid(hole[:, 1]),
                facecolor=SEA_COLOR,
                edgecolor="none",
                zorder=0,
            )


def draw_coastline(ax, polygons):
    for exterior, holes in polygons:
        ax.plot(
            lon_to_grid(exterior[:, 0]),
            lat_to_grid(exterior[:, 1]),
            color=COAST_COLOR,
            linewidth=0.8,
            zorder=4,
        )
        for hole in holes:
            ax.plot(
                lon_to_grid(hole[:, 0]),
                lat_to_grid(hole[:, 1]),
                color=COAST_COLOR,
                linewidth=0.55,
                zorder=4,
            )


def draw_area_boundaries(ax, area_polygons):
    for polygons in area_polygons.values():
        for exterior, holes in polygons:
            ax.plot(
                lon_to_grid(exterior[:, 0]),
                lat_to_grid(exterior[:, 1]),
                color=BOUNDARY_COLOR,
                linewidth=1.0,
                zorder=5,
            )
            for hole in holes:
                ax.plot(
                    lon_to_grid(hole[:, 0]),
                    lat_to_grid(hole[:, 1]),
                    color=BOUNDARY_COLOR,
                    linewidth=0.6,
                    zorder=5,
                )


def save_figure(cell_map, area_polygons, coastline, output_path):
    configure_plot_style()
    counts = Counter(cell_map["area"].ravel())
    values = np.zeros(cell_map["area"].shape, dtype=int)
    for index, area in enumerate(AREA_ORDER, start=1):
        values[cell_map["area"] == area] = index

    colors = [OUTSIDE_COLOR] + [AREA_COLORS[area] for area in AREA_ORDER]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(13.5, 10.8))
    draw_basemap(ax, coastline)
    x_edges = np.arange(EVAL_X[0] - 0.5, EVAL_X[1] + 1.0)
    y_edges = np.arange(EVAL_Y[0] - 0.5, EVAL_Y[1] + 1.0)
    ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        alpha=0.58,
        edgecolors=GRID_COLOR,
        linewidth=0.18,
        antialiased=True,
        rasterized=True,
        zorder=2,
    )
    draw_coastline(ax, coastline)
    draw_area_boundaries(ax, area_polygons)

    for area in AREA_ORDER:
        mask = cell_map["area"] == area
        if not np.any(mask):
            continue
        label_x = float(np.mean(cell_map["x"][mask]))
        label_y = float(np.mean(cell_map["y"][mask]))
        ax.text(
            label_x,
            label_y,
            area,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1F2933",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            zorder=7,
        )

    ax.set_xlim(EVAL_X[0] - 0.5, EVAL_X[1] + 0.5)
    ax.set_ylim(EVAL_Y[0] - 0.5, EVAL_Y[1] + 0.5)
    ax.set_xticks(np.arange(30, 71, 5))
    ax.set_yticks(np.arange(35, 71, 5))
    ax.set_aspect(DLAT / (DLON * np.cos(np.radians(37.25))))
    ax.set_xlabel("グリッド x（セル番号）")
    ax.set_ylabel("グリッド y（セル番号）")
    ax.set_title("評価グリッドと能登8市町の地理的セル定義", fontsize=17, pad=14)
    ax.tick_params(labelsize=9)

    handles = [
        Patch(
            facecolor=AREA_COLORS[area],
            edgecolor=BOUNDARY_COLOR,
            label=f"{area}（{counts[area]}セル）",
        )
        for area in AREA_ORDER
    ]
    handles.append(
        Patch(
            facecolor=OUTSIDE_COLOR,
            edgecolor=GRID_COLOR,
            label=f"対象8市町外・富山県（{counts['']}セル）",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.93)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {output_path}")
    return output_path


# 実行と要約
def print_summary(cell_map, populations):
    counts = Counter(cell_map["area"].ravel())
    methods = Counter(cell_map["assignment_method"].ravel())
    print("\n== 能登8市町の評価セル割り当て ==")
    for area in AREA_ORDER:
        print(
            f"  {area:<4}: {counts[area]:>3}セル / "
            f"人口 {populations[area]:>6,}人"
        )
    print(f"  対象外: {counts['']:>3}セル")
    print(f"  合計  : {cell_map['area'].size:>3}セル")
    print("\n== 割り当て方法 ==")
    print(f"  最大重なり面積   : {methods['largest_overlap_area']:>3}セル")
    print(f"  陸上の最近傍市町 : {methods['nearest_land_area']:>3}セル")
    print(f"  沿岸の最近傍市町 : {methods['nearest_coastal_area']:>3}セル")
    print(f"  対象外           : {methods['outside']:>3}セル")


def run():
    populations = load_population(POPULATION_CSV)
    area_polygons = load_area_polygons(BOUNDARIES_GEOJSON)
    coastline = load_coastline_polygons(COASTLINE_GEOJSON)
    prefecture_polygons = load_prefecture_polygons(COASTLINE_GEOJSON)
    geometry_masks = build_geometry_masks(coastline)
    cell_map = assign_cells(area_polygons, prefecture_polygons, geometry_masks)
    print_summary(cell_map, populations)
    write_cell_map(cell_map, populations, OUTPUT_CSV)
    write_cell_area_map(cell_map, OUTPUT_AREA_CSV)
    save_figure(cell_map, area_polygons, coastline, OUTPUT_FIGURE)


if __name__ == "__main__":
    run()
