"""対象8市町の断水戸数を世帯数で正規化し、CSVと推移図を作る。

実行: python3 src/preprocessing/water_outage_rate.py
入力: data/processed/events/{断水.csv,世帯数.csv}
出力: data/processed/断水率/{市町別断水率.csv,figures/}
依存: matplotlib
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVENTS_DIR = ROOT / "data" / "processed" / "events"
WATER_CSV = EVENTS_DIR / "断水.csv"
HOUSEHOLDS_CSV = EVENTS_DIR / "世帯数.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "断水率" / "市町別断水率.csv"
FIGURE_DIR = OUTPUT_CSV.parent / "figures"

TARGET_AREAS = (
    "七尾市",
    "珠洲市",
    "穴水町",
    "能登町",
    "輪島市",
    "志賀町",
    "中能登町",
    "羽咋市",
)
AREA_ORDER = {area: index for index, area in enumerate(TARGET_AREAS)}
FIELDNAMES = (
    "日付",
    "市町名",
    "世帯数(2023/12)",
    "断水戸数",
    "断水世帯率",
    "1万世帯当たり断水戸数",
    "概数",
    "状態",
    "備考",
)


# 入力と集計
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_households() -> dict[str, int]:
    households = {}
    for row in read_csv(HOUSEHOLDS_CSV):
        area = row["area"]
        if area in households:
            raise ValueError(f"世帯数の市町名が重複しています: {area}")
        households[area] = int(row["households(2023/12)"])

    missing = set(TARGET_AREAS) - set(households)
    if missing:
        raise ValueError(f"世帯数がない対象市町があります: {sorted(missing)}")
    if any(households[area] <= 0 for area in TARGET_AREAS):
        raise ValueError("対象市町の世帯数は正である必要があります")
    return households


def build_rows() -> list[dict[str, object]]:
    households = load_households()
    output = []
    seen = set()

    for source_row in read_csv(WATER_CSV):
        area = source_row["市町名"]
        if area not in AREA_ORDER:
            continue

        day = date.fromisoformat(source_row["日付"])
        key = (day, area)
        if key in seen:
            raise ValueError(f"日付×市町名が重複しています: {day}/{area}")
        seen.add(key)

        value_text = source_row["断水戸数"].strip()
        water_outages = int(value_text) if value_text else None
        if water_outages is not None and water_outages < 0:
            raise ValueError(f"断水戸数が負です: {day}/{area}")
        if source_row["状態"] == "未報告" and water_outages is not None:
            raise ValueError(f"未報告なのに断水戸数があります: {day}/{area}")
        if source_row["状態"] == "解消済み" and water_outages != 0:
            raise ValueError(f"解消済みの断水戸数が0ではありません: {day}/{area}")

        n_households = households[area]
        rate = None if water_outages is None else water_outages / n_households
        output.append(
            {
                "日付": day.isoformat(),
                "市町名": area,
                "世帯数(2023/12)": n_households,
                "断水戸数": "" if water_outages is None else water_outages,
                "断水世帯率": "" if rate is None else f"{rate:.10f}",
                "1万世帯当たり断水戸数": "" if rate is None else f"{rate * 10_000:.6f}",
                "概数": source_row["概数"],
                "状態": source_row["状態"],
                "備考": source_row["備考"],
            }
        )

    output.sort(key=lambda row: (row["日付"], AREA_ORDER[row["市町名"]]))
    dates = {row["日付"] for row in output}
    expected = {(day, area) for day in dates for area in TARGET_AREAS}
    actual = {(row["日付"], row["市町名"]) for row in output}
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"報告日×対象8市町が揃っていません: {missing[:5]}")
    if len(output) != len(dates) * len(TARGET_AREAS):
        raise ValueError("出力行数が報告日数×8市町と一致しません")

    for row in output:
        if row["断水戸数"] == "":
            if row["断水世帯率"] != "":
                raise ValueError(f"断水戸数欠損なのに率があります: {row}")
            continue
        restored = float(row["断水世帯率"]) * int(row["世帯数(2023/12)"])
        if abs(restored - int(row["断水戸数"])) > 1e-4:
            raise ValueError(f"断水率から戸数を再現できません: {row}")
    return output


# 可視化
def save_rate_figure(
    dates: list[date],
    rates: list[float],
    title: str,
    output: Path,
) -> None:
    """公表日の断水率を、説明注記なし・水平の日付ラベルで保存する。"""
    rate_percent = [rate * 100 if math.isfinite(rate) else math.nan for rate in rates]
    fig, axis = plt.subplots(figsize=(10, 5.4))
    axis.plot(
        dates,
        rate_percent,
        color="#2563A6",
        marker="o",
        markersize=3.0,
        linewidth=1.7,
    )
    axis.fill_between(dates, rate_percent, color="#2563A6", alpha=0.10)
    axis.set_title(title, fontsize=15, pad=13)
    axis.set_xlabel("公表日")
    axis.set_ylabel("断水世帯率（%）")
    axis.set_ylim(bottom=0)
    axis.grid(axis="both", alpha=0.22)
    # 年月を最大7目盛り程度に抑え、水平表示でも重ならない間隔にする。
    first, last = min(dates), max(dates)
    month_count = (last.year - first.year) * 12 + last.month - first.month + 1
    month_interval = max(1, math.ceil(month_count / 7))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", labelrotation=0)
    plt.setp(axis.get_xticklabels(), ha="center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_figures(rows: list[dict[str, object]]) -> list[Path]:
    """8市町別と、全市町の値が揃う日の8市町合計断水率図を保存する。"""
    outputs = []
    for area in TARGET_AREAS:
        selected = [row for row in rows if row["市町名"] == area]
        dates = [date.fromisoformat(str(row["日付"])) for row in selected]
        rates = [
            math.nan if row["断水世帯率"] == "" else float(row["断水世帯率"])
            for row in selected
        ]
        output = FIGURE_DIR / f"断水率_{area}.png"
        save_rate_figure(dates, rates, f"{area}の断水世帯率", output)
        outputs.append(output)

    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["日付"]), []).append(row)
    total_households = sum(
        int(next(row["世帯数(2023/12)"] for row in rows if row["市町名"] == area))
        for area in TARGET_AREAS
    )
    complete_dates = sorted(
        day
        for day, day_rows in by_date.items()
        if all(row["断水戸数"] != "" for row in day_rows)
    )
    overall_outages = [
        sum(int(row["断水戸数"]) for row in by_date[day])
        for day in complete_dates
    ]
    overall_rates = [value / total_households for value in overall_outages]
    overall_output = FIGURE_DIR / "断水率_8市町全体.png"
    save_rate_figure(
        [date.fromisoformat(day) for day in complete_dates],
        overall_rates,
        "対象8市町全体の断水世帯率",
        overall_output,
    )
    outputs.append(overall_output)
    return outputs


# 実行
def run() -> None:
    rows = build_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    figures = save_figures(rows)
    dates = sorted({row["日付"] for row in rows})
    missing_values = sum(row["断水戸数"] == "" for row in rows)
    print(
        f"[write] {OUTPUT_CSV.relative_to(ROOT)}: "
        f"{len(rows):,}行 / {len(dates)}報告日 / {len(TARGET_AREAS)}市町 / "
        f"{dates[0]}〜{dates[-1]} / 断水戸数欠損{missing_values}件"
    )
    print(f"[figures] {FIGURE_DIR.relative_to(ROOT)}: {len(figures)}枚")


if __name__ == "__main__":
    run()
