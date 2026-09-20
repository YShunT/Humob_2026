"""ローカル検証誤差をセルの担当市町別に集計し、CSVへ保存する。

入力 : data/processed/events/population_cell_map.csv
       呼び出し側が渡す予測OD・正解OD
出力 : experiments/<ID>/outputs/municipality_errors.csv
依存 : common/competition_metric.py, common/local_error_map.py

対角誤差はセルの担当市町へ、非対角誤差は起点市町・終点市町の2通りで集計する。
各集計内では全評価セルを必ずどこか1地域へ割り当てるため、地域別SSE比率の合計は100%に
なる。8市町に割り当てられていないセルは「対象8市町外・富山県」として残す。

``*_sse_share_pct`` と ``*_nrmse`` はどちらも地域の流動規模に比例して大きくなるため、
地域間の予測難易度の比較には使えない。難易度は全ゼロ予測を基準にした相対誤差
``*_sse_vs_zero``（= SSE / Σ観測値²、1.0で全ゼロ予測と同等、小さいほど良い）で比べる。
"""

import csv
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from common.competition_metric import (
    EVAL_X,
    EVAL_Y,
    N_CELL,
    NORM_DIAG,
    NORM_OFF,
    day_errors,
    in_eval_box,
    score,
)


CELL_AREA_CSV = ROOT / "data" / "processed" / "events" / "population_cell_map.csv"

AREA_ORDER = (
    "七尾市",
    "珠洲市",
    "穴水町",
    "能登町",
    "輪島市",
    "志賀町",
    "中能登町",
    "羽咋市",
)
OUTSIDE_AREA = "対象8市町外・富山県"
ALL_AREA = "全体"

FIELDNAMES = (
    "experiment",
    "area",
    "n_cells",
    "cell_share_pct",
    "n_days",
    "diag_sse",
    "diag_sse_share_pct",
    "diag_zero_sse",
    "diag_sse_vs_zero",
    "diag_rmse_daily_mean",
    "diag_nrmse",
    "off_origin_sse",
    "off_origin_sse_share_pct",
    "off_origin_zero_sse",
    "off_origin_sse_vs_zero",
    "off_origin_rmse_daily_mean",
    "off_origin_nrmse",
    "off_destination_sse",
    "off_destination_sse_share_pct",
    "off_destination_zero_sse",
    "off_destination_sse_vs_zero",
    "off_destination_rmse_daily_mean",
    "off_destination_nrmse",
)


def output_path(experiment):
    """実験IDに対応する既定の地域別誤差CSVパスを返す。"""
    return ROOT / "experiments" / experiment / "outputs" / "municipality_errors.csv"


def _expected_grid_ids():
    return {
        f"{y}_{x}"
        for y in range(EVAL_Y[0], EVAL_Y[1] + 1)
        for x in range(EVAL_X[0], EVAL_X[1] + 1)
    }


def load_cell_areas(path=CELL_AREA_CSV):
    """評価セルID→担当市町を読み、1,476セルが過不足なくあることを検査する。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"セル―市町対応CSVがありません: {path}")

    cell_areas = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"grid_id", "area"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"セル―市町対応CSVに必要な列がありません: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            grid_id = row["grid_id"].strip()
            if grid_id in cell_areas:
                raise ValueError(f"セル―市町対応CSVに重複IDがあります: {grid_id}（{row_number}行目）")
            area = row["area"].strip() or OUTSIDE_AREA
            if area not in AREA_ORDER and area != OUTSIDE_AREA:
                raise ValueError(f"未定義の市町名です: {area}（{row_number}行目）")
            cell_areas[grid_id] = area

    expected = _expected_grid_ids()
    missing_ids = sorted(expected - set(cell_areas))
    extra_ids = sorted(set(cell_areas) - expected)
    if missing_ids or extra_ids:
        raise ValueError(
            "セル―市町対応CSVが公式評価範囲と一致しません: "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )
    if len(cell_areas) != N_CELL:
        raise ValueError(f"セル数が不正です: {len(cell_areas)} != {N_CELL}")
    return cell_areas


def _empty_area_values():
    return {area: 0.0 for area in (*AREA_ORDER, OUTSIDE_AREA)}


def _add_error(day_sse, cell_areas, origin, destination, error2):
    """1 ODペアの二乗誤差を対角・起点・終点の担当市町へ加算する。"""
    if origin == destination:
        day_sse["diagonal"][cell_areas[origin]] += error2
    else:
        day_sse["off_origin"][cell_areas[origin]] += error2
        day_sse["off_destination"][cell_areas[destination]] += error2


def _day_area_sse(prediction, truth, cell_areas):
    """1日分のSSEと、全ゼロ予測時のSSEを市町別に返す。

    公式指標と同じく未記録ペアは0とする。全ゼロ予測のSSEは観測ペアの二乗和 Σ観測値²
    そのもので、予測しか無いペア（extra）は0を予測しても誤差を生まないため寄与しない。
    """
    result = {
        "diagonal": _empty_area_values(),
        "off_origin": _empty_area_values(),
        "off_destination": _empty_area_values(),
    }
    zero_result = {
        "diagonal": _empty_area_values(),
        "off_origin": _empty_area_values(),
        "off_destination": _empty_area_values(),
    }

    for origin, truth_destinations in truth.items():
        if not in_eval_box(origin):
            continue
        prediction_destinations = prediction.get(origin, {})
        for destination, actual in truth_destinations.items():
            if not in_eval_box(destination):
                continue
            predicted = prediction_destinations.get(destination, 0.0)
            _add_error(
                result,
                cell_areas,
                origin,
                destination,
                (predicted - actual) ** 2,
            )
            _add_error(zero_result, cell_areas, origin, destination, actual ** 2)

    for origin, prediction_destinations in prediction.items():
        if not in_eval_box(origin):
            continue
        truth_destinations = truth.get(origin, {})
        for destination, predicted in prediction_destinations.items():
            if not in_eval_box(destination) or destination in truth_destinations:
                continue
            _add_error(
                result,
                cell_areas,
                origin,
                destination,
                predicted ** 2,
            )
    return result, zero_result


def _percent(part, whole):
    return 100.0 * part / whole if whole else 0.0


def _rounded(value):
    return round(float(value), 6)


def _ratio(sse, zero_sse):
    """全ゼロ予測を1.0としたときの相対誤差。観測が無い地域は ``None``。

    地域の流動規模で正規化されるので、``*_nrmse`` と違って地域間で難易度を比べられる。
    1.0を超える地域は、その地域については何も予測しない方が誤差が小さい。
    """
    if not zero_sse:
        return None
    return round(float(sse) / float(zero_sse), 6)


def municipality_error_rows(predictions, truths, experiment, cell_areas=None):
    """市町別SSE・誤差比率・公式定義に沿う日次平均NRMSEを返す。

    非対角は起点担当と終点担当を別列にする。同一の非対角SSEを2回足すのではなく、
    起点別の列内、終点別の列内でそれぞれ全体を100%に分解している。
    """
    prediction_days = set(predictions)
    truth_days = set(truths)
    if prediction_days != truth_days:
        raise ValueError(
            "地域別評価の日付集合が一致しません: "
            f"prediction_only={sorted(prediction_days-truth_days)}, "
            f"truth_only={sorted(truth_days-prediction_days)}"
        )
    if not truth_days:
        raise ValueError("地域別評価の対象日がありません")

    cell_areas = cell_areas or load_cell_areas()
    areas = (*AREA_ORDER, OUTSIDE_AREA)
    cell_counts = {
        area: sum(assigned == area for assigned in cell_areas.values())
        for area in areas
    }
    totals = {
        component: _empty_area_values()
        for component in ("diagonal", "off_origin", "off_destination")
    }
    zero_totals = {
        component: _empty_area_values()
        for component in ("diagonal", "off_origin", "off_destination")
    }
    rmse_sums = {
        component: _empty_area_values()
        for component in ("diagonal", "off_origin", "off_destination")
    }
    official_errors = []

    for day in sorted(truth_days):
        day_sse, day_zero_sse = _day_area_sse(
            predictions[day], truths[day], cell_areas
        )
        official_errors.append(day_errors(predictions[day], truths[day]))
        for area in areas:
            diagonal_pairs = cell_counts[area]
            off_pairs = cell_counts[area] * (N_CELL - 1)
            for component in ("diagonal", "off_origin", "off_destination"):
                zero_totals[component][area] += day_zero_sse[component][area]
            totals["diagonal"][area] += day_sse["diagonal"][area]
            totals["off_origin"][area] += day_sse["off_origin"][area]
            totals["off_destination"][area] += day_sse["off_destination"][area]
            rmse_sums["diagonal"][area] += math.sqrt(
                day_sse["diagonal"][area] / diagonal_pairs
            )
            rmse_sums["off_origin"][area] += math.sqrt(
                day_sse["off_origin"][area] / off_pairs
            )
            rmse_sums["off_destination"][area] += math.sqrt(
                day_sse["off_destination"][area] / off_pairs
            )

    n_days = len(truth_days)
    component_totals = {
        component: sum(totals[component].values())
        for component in totals
    }
    zero_component_totals = {
        component: sum(zero_totals[component].values())
        for component in zero_totals
    }
    official_sse_diag = sum(item["se_diag"] for item in official_errors)
    official_sse_off = sum(item["se_off"] for item in official_errors)
    if not math.isclose(
        component_totals["diagonal"], official_sse_diag, rel_tol=1e-12, abs_tol=1e-8
    ):
        raise RuntimeError("地域別diagonal SSEが公式集計と一致しません")
    for component in ("off_origin", "off_destination"):
        if not math.isclose(
            component_totals[component], official_sse_off, rel_tol=1e-12, abs_tol=1e-8
        ):
            raise RuntimeError(f"地域別{component} SSEが公式集計と一致しません")
    if not math.isclose(
        zero_component_totals["off_origin"],
        zero_component_totals["off_destination"],
        rel_tol=1e-12,
        abs_tol=1e-8,
    ):
        raise RuntimeError("全ゼロ予測SSEの起点集計と終点集計が一致しません")

    rows = []
    for area in areas:
        diagonal_rmse = rmse_sums["diagonal"][area] / n_days
        origin_rmse = rmse_sums["off_origin"][area] / n_days
        destination_rmse = rmse_sums["off_destination"][area] / n_days
        rows.append({
            "experiment": experiment,
            "area": area,
            "n_cells": cell_counts[area],
            "cell_share_pct": round(_percent(cell_counts[area], N_CELL), 3),
            "n_days": n_days,
            "diag_sse": _rounded(totals["diagonal"][area]),
            "diag_sse_share_pct": round(
                _percent(totals["diagonal"][area], component_totals["diagonal"]), 3
            ),
            "diag_zero_sse": _rounded(zero_totals["diagonal"][area]),
            "diag_sse_vs_zero": _ratio(
                totals["diagonal"][area], zero_totals["diagonal"][area]
            ),
            "diag_rmse_daily_mean": _rounded(diagonal_rmse),
            "diag_nrmse": _rounded(diagonal_rmse / NORM_DIAG),
            "off_origin_sse": _rounded(totals["off_origin"][area]),
            "off_origin_sse_share_pct": round(
                _percent(totals["off_origin"][area], component_totals["off_origin"]), 3
            ),
            "off_origin_zero_sse": _rounded(zero_totals["off_origin"][area]),
            "off_origin_sse_vs_zero": _ratio(
                totals["off_origin"][area], zero_totals["off_origin"][area]
            ),
            "off_origin_rmse_daily_mean": _rounded(origin_rmse),
            "off_origin_nrmse": _rounded(origin_rmse / NORM_OFF),
            "off_destination_sse": _rounded(totals["off_destination"][area]),
            "off_destination_sse_share_pct": round(
                _percent(
                    totals["off_destination"][area],
                    component_totals["off_destination"],
                ),
                3,
            ),
            "off_destination_zero_sse": _rounded(
                zero_totals["off_destination"][area]
            ),
            "off_destination_sse_vs_zero": _ratio(
                totals["off_destination"][area], zero_totals["off_destination"][area]
            ),
            "off_destination_rmse_daily_mean": _rounded(destination_rmse),
            "off_destination_nrmse": _rounded(destination_rmse / NORM_OFF),
        })

    overall_score = score(official_errors)
    rows.append({
        "experiment": experiment,
        "area": ALL_AREA,
        "n_cells": N_CELL,
        "cell_share_pct": 100.0,
        "n_days": n_days,
        "diag_sse": _rounded(component_totals["diagonal"]),
        "diag_sse_share_pct": 100.0,
        "diag_zero_sse": _rounded(zero_component_totals["diagonal"]),
        "diag_sse_vs_zero": _ratio(
            component_totals["diagonal"], zero_component_totals["diagonal"]
        ),
        "diag_rmse_daily_mean": _rounded(overall_score["rmse_diag"]),
        "diag_nrmse": _rounded(overall_score["nrmse_diag"]),
        "off_origin_sse": _rounded(component_totals["off_origin"]),
        "off_origin_sse_share_pct": 100.0,
        "off_origin_zero_sse": _rounded(zero_component_totals["off_origin"]),
        "off_origin_sse_vs_zero": _ratio(
            component_totals["off_origin"], zero_component_totals["off_origin"]
        ),
        "off_origin_rmse_daily_mean": _rounded(overall_score["rmse_off"]),
        "off_origin_nrmse": _rounded(overall_score["nrmse_off"]),
        "off_destination_sse": _rounded(component_totals["off_destination"]),
        "off_destination_sse_share_pct": 100.0,
        "off_destination_zero_sse": _rounded(
            zero_component_totals["off_destination"]
        ),
        "off_destination_sse_vs_zero": _ratio(
            component_totals["off_destination"],
            zero_component_totals["off_destination"],
        ),
        "off_destination_rmse_daily_mean": _rounded(overall_score["rmse_off"]),
        "off_destination_nrmse": _rounded(overall_score["nrmse_off"]),
    })
    return rows


def write_municipality_errors(
    predictions,
    truths,
    experiment,
    out=None,
    cell_area_csv=CELL_AREA_CSV,
):
    """地域別評価をCSVへ保存し、保存先Pathを返す。"""
    rows = municipality_error_rows(
        predictions,
        truths,
        experiment,
        cell_areas=load_cell_areas(cell_area_csv),
    )
    out = Path(out) if out else output_path(experiment)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write] {out}")
    print("  地域別SSE比率（対角 / 非対角・起点 / 非対角・終点）と、非対角の全ゼロ比")
    for row in rows[:-1]:
        vs_zero = row["off_origin_sse_vs_zero"]
        vs_zero_text = "     -" if vs_zero is None else f"{vs_zero:>6.3f}"
        print(
            f"    {row['area']:<12} "
            f"{row['diag_sse_share_pct']:>7.3f}% / "
            f"{row['off_origin_sse_share_pct']:>7.3f}% / "
            f"{row['off_destination_sse_share_pct']:>7.3f}%"
            f"   非対角SSE/Σy² {vs_zero_text}"
        )
    return out
