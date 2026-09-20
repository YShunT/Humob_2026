"""対象8市町の避難者数を人口で正規化し、CSVと推移図を作る。

実行: python3 src/preprocessing/evacuation_rate.py
入力: data/processed/events/{避難所.csv,population.csv}
出力: data/processed/避難者率/{市町別避難者率.csv,figures/}
依存: matplotlib
"""

from __future__ import annotations

import csv
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
SHELTER_CSV = EVENTS_DIR / "避難所.csv"
POPULATION_CSV = EVENTS_DIR / "population.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "避難者率" / "市町別避難者率.csv"
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
    "人口(2023/12)",
    "避難者数(人)",
    "避難者率",
    "人口1万人当たり避難者数(人)",
    "開設数(箇所)",
    "広域避難者数(人)",
    "広域避難所数(箇所)",
    "閉鎖",
)


# 入力と集計
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_populations() -> dict[str, int]:
    populations = {}
    for row in read_csv(POPULATION_CSV):
        area = row["area"]
        if area in populations:
            raise ValueError(f"人口の市町名が重複しています: {area}")
        populations[area] = int(row["population(2023/12)"])

    missing = set(TARGET_AREAS) - set(populations)
    if missing:
        raise ValueError(f"人口がない対象市町があります: {sorted(missing)}")
    if any(populations[area] <= 0 for area in TARGET_AREAS):
        raise ValueError("対象市町の人口は正である必要があります")
    return populations


def build_rows() -> list[dict[str, object]]:
    populations = load_populations()
    output = []
    seen = set()

    for source_row in read_csv(SHELTER_CSV):
        area = source_row["市町名"]
        if area not in AREA_ORDER:
            continue

        day = date.fromisoformat(source_row["日付"])
        key = (day, area)
        if key in seen:
            raise ValueError(f"日付×市町名が重複しています: {day}/{area}")
        seen.add(key)

        population = populations[area]
        evacuees = int(source_row["避難者数(人)"])
        if evacuees < 0:
            raise ValueError(f"避難者数が負です: {day}/{area}")
        rate = evacuees / population

        output.append(
            {
                "日付": day.isoformat(),
                "市町名": area,
                "人口(2023/12)": population,
                "避難者数(人)": evacuees,
                "避難者率": f"{rate:.10f}",
                "人口1万人当たり避難者数(人)": f"{rate * 10_000:.6f}",
                "開設数(箇所)": int(source_row["開設数(箇所)"]),
                "広域避難者数(人)": int(source_row["広域避難者数(人)"]),
                "広域避難所数(箇所)": int(source_row["広域避難所(箇所)"]),
                "閉鎖": source_row["閉鎖"],
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
        restored = float(row["避難者率"]) * int(row["人口(2023/12)"])
        if abs(restored - int(row["避難者数(人)"])) > 1e-4:
            raise ValueError(f"避難者率から人数を再現できません: {row}")
    return output


# 可視化
def save_rate_figure(
    dates: list[date],
    rates: list[float],
    title: str,
    output: Path,
) -> None:
    """公表日の避難者率を、説明注記なし・水平の日付ラベルで保存する。"""
    fig, axis = plt.subplots(figsize=(10, 5.4))
    axis.plot(
        dates,
        [rate * 100 for rate in rates],
        color="#C74343",
        marker="o",
        markersize=2.8,
        linewidth=1.7,
    )
    axis.fill_between(
        dates,
        [rate * 100 for rate in rates],
        color="#C74343",
        alpha=0.10,
    )
    axis.set_title(title, fontsize=15, pad=13)
    axis.set_xlabel("公表日")
    axis.set_ylabel("避難者率（%）")
    axis.set_ylim(bottom=0)
    axis.grid(axis="both", alpha=0.22)
    # 年月を最大7目盛り程度に抑え、水平表示でも重ならない間隔にする。
    first, last = min(dates), max(dates)
    month_count = (last.year - first.year) * 12 + last.month - first.month + 1
    month_interval = max(1, (month_count + 6) // 7)
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", labelrotation=0)
    plt.setp(axis.get_xticklabels(), ha="center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_figures(rows: list[dict[str, object]]) -> list[Path]:
    """8市町別と8市町合計の避難者率図を保存する。"""
    outputs = []
    for area in TARGET_AREAS:
        selected = [row for row in rows if row["市町名"] == area]
        dates = [date.fromisoformat(str(row["日付"])) for row in selected]
        rates = [float(row["避難者率"]) for row in selected]
        output = FIGURE_DIR / f"避難者率_{area}.png"
        save_rate_figure(dates, rates, f"{area}の避難者率", output)
        outputs.append(output)

    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["日付"]), []).append(row)
    overall_dates = sorted(by_date)
    total_population = sum(
        int(next(row["人口(2023/12)"] for row in rows if row["市町名"] == area))
        for area in TARGET_AREAS
    )
    overall_evacuees = [
        sum(int(row["避難者数(人)"]) for row in by_date[day])
        for day in overall_dates
    ]
    overall_rates = [value / total_population for value in overall_evacuees]
    overall_output = FIGURE_DIR / "避難者率_8市町全体.png"
    save_rate_figure(
        [date.fromisoformat(day) for day in overall_dates],
        overall_rates,
        "対象8市町全体の避難者率",
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
    print(
        f"[write] {OUTPUT_CSV.relative_to(ROOT)}: "
        f"{len(rows):,}行 / {len(dates)}報告日 / {len(TARGET_AREAS)}市町 / "
        f"{dates[0]}〜{dates[-1]}"
    )
    print(f"[figures] {FIGURE_DIR.relative_to(ROOT)}: {len(figures)}枚")


if __name__ == "__main__":
    run()
