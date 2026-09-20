"""
グリッド区画 (y_x) が実地図（緯度経度）上のどこに当たるかを可視化する。

コマンド:
  python3 src/common/grid_geo_map.py --month 2023-12   # 区画の地理的対応
  python3 src/common/grid_geo_map.py --month 2024-01
  python3 src/common/grid_geo_map.py --month 2024-04

入力 : data/raw/humob2026-dataset.tsv
出力 : figures/E1_<YYYY-MM>_geomap.png
依存 : matplotlib のみ（海岸線 GeoJSON は data/processed/GeoJson/japan.geojson。
       無ければ "geo.jsonがありません" と表示し、海判定なしで続行）
"""

import argparse
import ast
import json
import urllib.request
from pathlib import Path as FsPath

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

# 日本語フォント。rcParams への代入はフォント名を検証しない
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro",        # macOS
    "Yu Gothic", "Meiryo",                              # Windows
    "Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic",  # Linux
    "DejaVu Sans",                                      # 最終手段（日本語は出ない）
]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = FsPath(__file__).resolve().parent
ROOT = HERE.parent.parent   # src/common/ から見たリポジトリルート
DATA = ROOT / "data" / "raw" / "humob2026-dataset.tsv"
GEOJSON_CACHE = ROOT / "data" / "processed" / "GeoJson" / "japan.geojson"
GEOJSON_URL = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
FIGDIR = ROOT / "figures"

# --- グリッド定義 -------------------------------------------------------
NX, NY = 100, 70
LON0, LON1 = 136.029, 138.042
LAT0, LAT1 = 36.203, 37.646
DLON = (LON1 - LON0) / (NX - 1)
DLAT = (LAT1 - LAT0) / (NY - 1)

BX = (30, 70)
BY = (35, 70)

# 海岸線 GeoJSON は全国分（47都道府県）だが、グリッドは石川だけでなく富山・新潟・
# 長野・岐阜・福井にまたがる。陸セルを生まないポリゴンはキャッシュ書き出し時点で
# 捨てる（13MB/1158ポリゴン → 数百KB/7ポリゴン）。GEO_MARGIN は bbox 粗フィルタの余白。
GEO_MARGIN = 0.3

# 5色ヒートマップ（画像参考: blue→cyan→green→yellow→red）
HEAT = LinearSegmentedColormap.from_list(
    "bcgyr", ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"])
SEA_COLOR = "#c2ddf1"    # 薄い水色（海・0区画）
LAND_COLOR = "#cfe8cf"   # 薄い緑（陸・0区画）
NOISE_COLOR = "#b3b3b3"  # 薄いグレー（GPSノイズ＝沿岸2マス外の海上活動）

CITIES = [
    (136.656, 36.561, "金沢"),
    (136.967, 37.043, "七尾"),
    (136.906, 37.229, "穴水"),
    (136.900, 37.391, "輪島"),
    (137.259, 37.436, "珠洲"),
    (137.150, 37.300, "能登町"),
]


def x_to_lon(x): return LON0 + (np.asarray(x, float) - 1) * DLON
def y_to_lat(y): return LAT0 + (np.asarray(y, float) - 1) * DLAT
def lon_to_x(lon): return 1 + (np.asarray(lon, float) - LON0) / DLON
def lat_to_y(lat): return 1 + (np.asarray(lat, float) - LAT0) / DLAT


def load_cell_activity(target_ym="2024-04"):
    grid = np.zeros((NY, NX), float)
    ndays = 0
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            date_str, payload = line.split("\t", 1)
            if payload.strip() == "NA":
                continue
            if f"{date_str[:4]}-{date_str[4:6]}" != target_ym:
                continue
            ndays += 1
            for o_gid, dest_map in ast.literal_eval(payload).items():
                p = o_gid.split("_")
                if len(p) != 2:
                    continue
                try:
                    y, x = int(p[0]), int(p[1])
                except ValueError:
                    continue
                if 1 <= x <= NX and 1 <= y <= NY:
                    grid[y - 1, x - 1] += sum(dest_map.values())
    if ndays:
        grid /= ndays
    return grid, ndays


def in_map_range(ring):
    """外環リングの bbox が地図範囲（＋余白）と交わるか。"""
    ext = np.asarray(ring, float)
    if ext.ndim != 2 or ext.shape[0] < 3:
        return False
    return not (ext[:, 0].max() < LON0 - GEO_MARGIN or ext[:, 0].min() > LON1 + GEO_MARGIN or
                ext[:, 1].max() < LAT0 - GEO_MARGIN or ext[:, 1].min() > LAT1 + GEO_MARGIN)


def affects_grid(poly):
    """このポリゴンがグリッド内に陸セルを1つでも生むか。

    1セル≈2km に対し 1km 未満の微小ポリゴン（七ツ島などの沖合小島・岩礁）は
    セル中心を1つも含まないため陸として塗られず、海上に輪郭線だけが残って
    可視化のノイズになる。bbox が重なるだけの県（山梨県）も同様に落ちる。
    """
    ext = np.asarray(poly[0], float)
    holes = [np.asarray(h, float) for h in poly[1:]
             if np.asarray(h, float).ndim == 2 and len(h) >= 3]
    return bool(build_land_mask([(ext, holes)]).any())


def clip_geojson(gj):
    """図に実際に効くポリゴンだけ残した FeatureCollection を返す。"""
    feats = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        raw = coords if gtype == "MultiPolygon" else [coords] if gtype == "Polygon" else []
        # bbox で粗く絞ってから（重いので）実効性判定
        kept = [p for p in raw if p and in_map_range(p[0]) and affects_grid(p)]
        if not kept:
            continue
        geom = ({"type": "Polygon", "coordinates": kept[0]} if gtype == "Polygon"
                else {"type": "MultiPolygon", "coordinates": kept})
        feats.append({"type": "Feature", "properties": feat.get("properties", {}),
                      "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}


def load_land_polygons():
    """海岸線ポリゴンを読む。ファイルが無ければ何もせず空リストを返す。"""
    if not GEOJSON_CACHE.exists():
        print("geo.jsonがありません")
        return []
    try:
        gj = json.loads(GEOJSON_CACHE.read_text())
    except Exception as e:
        print(f"[warn] 海岸線 GeoJSON 読み込み失敗（海判定なしで続行）: {e}")
        return []
    polys = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        raw = coords if gtype == "MultiPolygon" else [coords] if gtype == "Polygon" else []
        for poly in raw:
            if not poly or not in_map_range(poly[0]):
                continue
            ext = np.asarray(poly[0], float)
            holes = [np.asarray(h, float) for h in poly[1:]
                     if np.asarray(h, float).ndim == 2 and len(h) >= 3]
            polys.append((ext, holes))
    return polys


def build_land_mask(polys):
    lon_c = x_to_lon(np.arange(1, NX + 1))
    lat_c = y_to_lat(np.arange(1, NY + 1))
    LonC, LatC = np.meshgrid(lon_c, lat_c)
    pts = np.column_stack([LonC.ravel(), LatC.ravel()])
    inside = np.zeros(len(pts), bool)
    for ext, holes in polys:
        pe_ = MplPath(ext).contains_points(pts)
        for h in holes:
            pe_ &= ~MplPath(h).contains_points(pts)
        inside |= pe_
    return inside.reshape(NY, NX)


def build_coast_cells(polys, step_deg=0.005):
    """沿岸線（＝陸ポリゴン境界）が通過するセルを True にした (NY,NX)。"""
    coast = np.zeros((NY, NX), bool)
    for ext, holes in polys:
        for ring in [ext] + list(holes):
            for i in range(len(ring) - 1):
                lon0, lat0 = ring[i]
                lon1, lat1 = ring[i + 1]
                n = max(1, int(max(abs(lon1 - lon0), abs(lat1 - lat0)) / step_deg))
                lons = np.linspace(lon0, lon1, n + 1)
                lats = np.linspace(lat0, lat1, n + 1)
                xs = np.rint(lon_to_x(lons)).astype(int)
                ys = np.rint(lat_to_y(lats)).astype(int)
                ok = (xs >= 1) & (xs <= NX) & (ys >= 1) & (ys <= NY)
                coast[ys[ok] - 1, xs[ok] - 1] = True
    return coast


def dilate8(mask):
    """8近傍で1回膨張（沿岸セル＋その隣接セル）。"""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def report_sea_activity(grid, land, topk=15):
    """海判定なのに活動があるセルを大きい順に列挙（診断）。"""
    sea_act = (~land) & (grid > 0)
    ys, xs = np.where(sea_act)
    vals = grid[ys, xs]
    order = np.argsort(-vals)
    print(f"\n[diag] 海判定セルの活動: {sea_act.sum()}セル / 活動総量シェア "
          f"{100*vals.sum()/grid[grid>0].sum():.1f}%")
    print("   rank  cell(y_x)   経度      緯度     日平均フロー")
    for i in order[:topk]:
        y, x = ys[i] + 1, xs[i] + 1
        print(f"   {list(order).index(i)+1:>4}  {y:>2}_{x:<3}  "
              f"{float(x_to_lon(x)):.3f}  {float(y_to_lat(y)):.3f}  {vals[i]:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description="区画の地理的対応マップ（月指定）")
    ap.add_argument("--month", default="2024-04", help="対象月 YYYY-MM（例 2023-12, 2024-01）")
    args = ap.parse_args()
    month = args.month
    mm = int(month.split("-")[1])
    out = FIGDIR / f"E1_{month}_geomap.png"

    grid, ndays = load_cell_activity(month)
    polys = load_land_polygons()
    land = build_land_mask(polys) if polys else np.ones((NY, NX), bool)
    print(f"[load] 活動量 {month}: {ndays}日平均, 非ゼロセル {int((grid>0).sum())}")
    print(f"[load] 陸ポリゴン {len(polys)}個, 陸セル {int(land.sum())} / {NX*NY}")
    report_sea_activity(grid, land)

    # GPSノイズ判定：海判定 & 活動あり & 沿岸線から2マス以上（沿岸セル＋隣接を除外）
    coast = build_coast_cells(polys) if polys else np.zeros((NY, NX), bool)
    near_coast = dilate8(coast)                 # 沿岸セル＋その隣接（=距離1以内）
    noise = (~land) & (grid > 0) & (~near_coast)  # 距離2マス以上の海上活動
    print(f"[noise] 沿岸セル {int(coast.sum())} / 沿岸2マス外の海上活動(グレー) "
          f"{int(noise.sum())}セル（海上活動 {int(((~land)&(grid>0)).sum())}中）")

    lon_edges = x_to_lon(np.arange(0.5, NX + 1.0))
    lat_edges = y_to_lat(np.arange(0.5, NY + 1.0))

    fig, ax = plt.subplots(figsize=(13, 9))

    # 下地：海(0)/陸(1)を薄色で
    ax.pcolormesh(lon_edges, lat_edges, land.astype(float),
                  cmap=ListedColormap([SEA_COLOR, LAND_COLOR]),
                  vmin=0, vmax=1, shading="flat", zorder=0)

    # GPSノイズセルをグレーで（活動ヒートマップより下、下地より上）
    Ng = np.ma.masked_where(~noise, np.ones((NY, NX)))
    ax.pcolormesh(lon_edges, lat_edges, Ng,
                  cmap=ListedColormap([NOISE_COLOR]), vmin=0, vmax=1,
                  shading="flat", zorder=1)

    # 活動量（0とノイズは透過→下地/グレーが残る）
    Z = np.ma.masked_where((grid == 0) | noise, np.log1p(grid))
    mesh = ax.pcolormesh(lon_edges, lat_edges, Z, cmap=HEAT, shading="flat", zorder=2)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.09)
    cb.set_label("log(1 + セル総フロー日平均)")

    # 海岸線
    for ext, holes in polys:
        ax.plot(ext[:, 0], ext[:, 1], color="#4a6f90", lw=0.7, zorder=3)
        for h in holes:
            ax.plot(h[:, 0], h[:, 1], color="#4a6f90", lw=0.5, zorder=3)

    # 評価ボックス（赤枠のみ。説明は凡例＝地図外へ）
    # セル外縁(±0.5)に合わせ、境界セル(x=30,70 / y=35,70)ごと内側に囲む
    bl, br = x_to_lon(BX[0] - 0.5), x_to_lon(BX[1] + 0.5)
    bb, bt = y_to_lat(BY[0] - 0.5), y_to_lat(BY[1] + 0.5)
    ax.add_patch(plt.Rectangle((bl, bb), br - bl, bt - bb,
                               fill=False, edgecolor="red", lw=2.2, zorder=5))

    # 都市（白文字・サイズ据え置き。薄地でも読めるよう細い縁取り）
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, "o", color="orange", ms=6, mec="black", mew=0.6, zorder=7)
        ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(5, 3),
                    fontsize=10, color="white", zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#333333")])

    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    ax.set_xlabel("経度 (°E)")
    ax.set_ylabel("緯度 (°N)")
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (LAT0 + LAT1))))

    secx = ax.secondary_xaxis("top", functions=(lon_to_x, x_to_lon))
    secx.set_xlabel("グリッド x（セル番号）")
    secx.set_xticks(np.arange(0, NX + 1, 10))
    secy = ax.secondary_yaxis("right", functions=(lat_to_y, y_to_lat))
    secy.set_ylabel("グリッド y（セル番号）")
    secy.set_yticks(np.arange(0, NY + 1, 10))

    # 凡例（地図外・下）に 海/陸/評価ボックス説明をまとめる
    ax.legend(
        handles=[
            Patch(facecolor=SEA_COLOR, edgecolor="#4a6f90", label="海"),
            Patch(facecolor=LAND_COLOR, edgecolor="gray", label="陸・データ0（疎/伏字）"),
            Patch(facecolor=NOISE_COLOR, edgecolor="#555555",
                  label="GPSノイズ（沿岸2マス外の海上活動）"),
            Line2D([0], [0], color="red", lw=2.2,
                   label="評価ボックス x∈[30,70], y∈[35,70]"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
        fontsize=10, frameon=True,
    )

    ax.set_title(f"{mm}月における区画と地理的対応", fontsize=14)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
