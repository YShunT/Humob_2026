"""EXP02: 非対角ODの出現確率×代表値候補を比較し、提出物を生成する。

実行: python3 src/EXP02_run.py
入力: data/raw/humob2026-dataset.tsv
出力: experiments/EXP02/{metrics.json,.log,outputs/,figures/}
依存: EXP01_run.py、common/{competition_metric,experiment_io,local_error_map,municipality_error,pred_dist,run_log}.py
"""

import datetime as dt
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean, median

import EXP01_run as exp01
from common.competition_metric import day_errors, in_eval_box, pct, score
from common.experiment_io import (
    DAILY_TOTAL, check_od_tsv, read_od_days, submission_days,
    write_metrics, write_od_tsv,
)
from common.run_log import run_log


EXP_ID = "EXP02"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TSV = ROOT / "data" / "raw" / "humob2026-dataset.tsv"
EXP = ROOT / "experiments" / EXP_ID
OUT = EXP / "outputs" / "submission.tsv"
METRICS = EXP / "metrics.json"
ERROR_MAP = EXP / "figures" / "local_error_map.png"
CALIBRATION_FIGURE = EXP / "figures" / "probability_calibration.png"
BASE_METRICS = ROOT / "experiments" / "EXP01" / "metrics.json"

WD = "月火水木金土日"

VAL_WINDOWS = [
    ("jan", dt.date(2024, 1, 25), dt.date(2024, 1, 31)),
    ("apr", dt.date(2024, 4, 1), dt.date(2024, 4, 7)),
]

A_WINDOWS = ("full", "post_quake")
VALUE_STATISTICS = ("median", "mean")

VAL_SETUPS = {
    "full": {
        "A": (dt.date(2023, 11, 1), dt.date(2024, 1, 24)),
        "B": (dt.date(2024, 4, 8), dt.date(2024, 6, 30)),
        "bound": (dt.date(2024, 1, 24), dt.date(2024, 4, 8)),
    },
    "post_quake": {
        "A": (dt.date(2024, 1, 1), dt.date(2024, 1, 24)),
        "B": (dt.date(2024, 4, 8), dt.date(2024, 6, 30)),
        "bound": (dt.date(2024, 1, 24), dt.date(2024, 4, 8)),
    },
}

SUB_SETUPS = {
    "full": {
        "A": (dt.date(2023, 11, 1), dt.date(2024, 1, 31)),
        "B": (dt.date(2024, 4, 1), dt.date(2024, 6, 30)),
        "bound": (dt.date(2024, 1, 31), dt.date(2024, 4, 1)),
    },
    "post_quake": {
        "A": (dt.date(2024, 1, 1), dt.date(2024, 1, 31)),
        "B": (dt.date(2024, 4, 1), dt.date(2024, 6, 30)),
        "bound": (dt.date(2024, 1, 31), dt.date(2024, 4, 1)),
    },
}

CALIBRATION_BINS = [
    (0.00, 0.10, "0.00-0.10"),
    (0.10, 0.25, "0.10-0.25"),
    (0.25, 0.50, "0.25-0.50"),
    (0.50, 1.000001, "0.50-1.00"),
]


# ---------------------------------------------------------------- 共通ユーティリティ

def to_int(day):
    return int(day.strftime("%Y%m%d"))


def to_date(day):
    return dt.date(day // 10000, day // 100 % 100, day % 100)


def actual_command():
    return "python3 src/EXP02_run.py"


def candidate_id(a_window, value_statistic):
    return f"{a_window}_{value_statistic}"


def candidate_specs():
    return [
        (candidate_id(a_window, statistic), a_window, statistic)
        for a_window in A_WINDOWS
        for statistic in VALUE_STATISTICS
    ]


def in_period(day, period):
    date = to_date(day)
    return period[0] <= date <= period[1]


def interpolation_weight(day, bounds):
    date = to_date(day)
    return (date - bounds[0]).days / (bounds[1] - bounds[0]).days


def validation_days():
    return [
        day
        for _, start, end in VAL_WINDOWS
        for day in exp01.daterange(start, end)
    ]


def validation_window(day):
    date = to_date(day)
    return next(
        name for name, start, end in VAL_WINDOWS if start <= date <= end
    )


def off_support(od, evaluation_only=True):
    """正の非対角ペア集合を返す。校正時だけ評価グリッド内に限定する。"""
    return {
        (origin, destination)
        for origin, destinations in od.items()
        if not evaluation_only or in_eval_box(origin)
        for destination, value in destinations.items()
        if origin != destination
        and value > 0
        and (not evaluation_only or in_eval_box(destination))
    }


# ---------------------------------------------------------------- Stage 1: 二値出現確率

def build_templates(rows, setup, a_window, value_statistic):
    """側・曜日・非対角ペアごとの確率、代表値、期待人流を作る。"""
    if a_window not in A_WINDOWS:
        raise ValueError(f"unknown A window: {a_window}")
    if value_statistic not in {"median", "mean"}:
        raise ValueError("value_statistic must be 'median' or 'mean'")
    values = {
        side: [defaultdict(list) for _ in range(7)] for side in ("A", "B")
    }
    observed_days = {side: [[] for _ in range(7)] for side in ("A", "B")}

    for day, od in rows.items():
        side = next(
            (name for name in ("A", "B") if in_period(day, setup[name])),
            None,
        )
        if side is None:
            continue
        weekday = to_date(day).weekday()
        observed_days[side][weekday].append(day)
        for origin, destinations in od.items():
            for destination, value in destinations.items():
                if origin != destination and value > 0:
                    values[side][weekday][(origin, destination)].append(value)

    probability = {side: [dict() for _ in range(7)] for side in ("A", "B")}
    expected = {side: [dict() for _ in range(7)] for side in ("A", "B")}
    positive_value = {side: [dict() for _ in range(7)] for side in ("A", "B")}

    for side in ("A", "B"):
        for weekday in range(7):
            n_days = len(observed_days[side][weekday])
            if n_days == 0:
                raise RuntimeError(f"{side}側の{WD[weekday]}曜に有効観測日がありません")
            for pair, samples in values[side][weekday].items():
                occurrence_probability = len(samples) / n_days
                representative_value = (
                    median(samples) if value_statistic == "median" else fmean(samples)
                )
                probability[side][weekday][pair] = occurrence_probability
                positive_value[side][weekday][pair] = representative_value
                expected[side][weekday][pair] = (
                    occurrence_probability * representative_value
                )

    return {
        "probability": probability,
        "positive_value": positive_value,
        "expected": expected,
        "observed_days": observed_days,
        "setup": setup,
        "a_window": a_window,
        "value_statistic": value_statistic,
    }


def probability_at(templates, weekday, pair, alpha):
    probability = templates["probability"]
    return (
        (1.0 - alpha) * probability["A"][weekday].get(pair, 0.0)
        + alpha * probability["B"][weekday].get(pair, 0.0)
    )


def template_summary(templates):
    return {
        "valid_days": {
            side: {
                f"{weekday}_{WD[weekday]}": len(
                    templates["observed_days"][side][weekday]
                )
                for weekday in range(7)
            }
            for side in ("A", "B")
        },
        "candidate_pairs": {
            f"{weekday}_{WD[weekday]}": len(
                set(templates["expected"]["A"][weekday])
                | set(templates["expected"]["B"][weekday])
            )
            for weekday in range(7)
        },
    }


def show_template_summary(templates, label):
    print(
        f"=== {label}: 学習テンプレート "
        f"(A={templates['a_window']}, 値={templates['value_statistic']}) ==="
    )
    setup = templates["setup"]
    print(
        f"  A側 {setup['A'][0]}〜{setup['A'][1]} / "
        f"B側 {setup['B'][0]}〜{setup['B'][1]}"
    )
    print("  曜日     " + "  ".join(f"{WD[w]:>4}" for w in range(7)))
    for side in ("A", "B"):
        counts = [len(templates["observed_days"][side][w]) for w in range(7)]
        print(f"  {side}側日数 " + "  ".join(f"{value:>4}" for value in counts))
    candidates = [
        len(
            set(templates["expected"]["A"][w])
            | set(templates["expected"]["B"][w])
        )
        for w in range(7)
    ]
    print("  候補ペア " + "  ".join(f"{value:>4}" for value in candidates))


def bin_name(probability):
    return next(
        label
        for low, high, label in CALIBRATION_BINS
        if low <= probability < high
    )


def evaluate_calibration(templates, truths, base_predictions):
    """候補ペア上で確率校正、coverage、EXP01とのsupport比較を集計する。"""
    bins = {
        label: {"n_pairs": 0, "sum_probability": 0.0, "n_present": 0}
        for _, _, label in CALIBRATION_BINS
    }
    present_probabilities = []
    absent_probabilities = []
    covered = total_truth = 0
    selected = selected_hit = 0
    base_selected = base_hit = 0

    for day in sorted(truths):
        weekday = to_date(day).weekday()
        alpha = interpolation_weight(day, templates["setup"]["bound"])
        candidates = {
            pair
            for pair in (
                set(templates["probability"]["A"][weekday])
                | set(templates["probability"]["B"][weekday])
            )
            if in_eval_box(pair[0]) and in_eval_box(pair[1])
        }
        truth_pairs = off_support(truths[day])
        base_pairs = off_support(base_predictions[day])
        covered += len(truth_pairs & candidates)
        total_truth += len(truth_pairs)
        base_selected += len(base_pairs)
        base_hit += len(base_pairs & truth_pairs)

        for pair in candidates:
            probability = probability_at(templates, weekday, pair, alpha)
            present = pair in truth_pairs
            bucket = bins[bin_name(probability)]
            bucket["n_pairs"] += 1
            bucket["sum_probability"] += probability
            bucket["n_present"] += int(present)
            (present_probabilities if present else absent_probabilities).append(
                probability
            )
            if probability >= 0.20:
                selected += 1
                selected_hit += int(present)

    calibration_bins = {}
    for _, _, label in CALIBRATION_BINS:
        bucket = bins[label]
        n_pairs = bucket["n_pairs"]
        calibration_bins[label] = {
            "n_pairs": n_pairs,
            "mean_probability": round(bucket["sum_probability"] / n_pairs, 4),
            "actual_rate": round(bucket["n_present"] / n_pairs, 4),
        }

    return {
        "valid_days_by_side_weekday": template_summary(templates)["valid_days"],
        "candidate_coverage": round(covered / total_truth, 4),
        "covered_truth_pairs": covered,
        "total_truth_pairs": total_truth,
        "mean_probability": {
            "present_candidates": round(
                sum(present_probabilities) / len(present_probabilities), 4
            ),
            "absent_candidates": round(
                sum(absent_probabilities) / len(absent_probabilities), 4
            ),
        },
        "bins": calibration_bins,
        "support_comparison": {
            "exp01": {
                "predicted_pairs_per_day": round(base_selected / len(truths), 3),
                "precision": round(base_hit / base_selected, 4),
                "recall": round(base_hit / total_truth, 4),
            },
            "probability_at_0.20": {
                "predicted_pairs_per_day": round(selected / len(truths), 3),
                "precision": round(selected_hit / selected, 4),
                "recall": round(selected_hit / total_truth, 4),
            },
        },
    }


def show_calibration(calibration):
    print("\n=== Stage 1: 出現確率の校正 ===")
    means = calibration["mean_probability"]
    print(
        f"  正解日に出現した候補ペアの平均確率 {means['present_candidates']:.4f} / "
        f"非出現候補 {means['absent_candidates']:.4f}"
    )
    print(
        f"  正解ペアcoverage {calibration['candidate_coverage']:.4f} "
        f"({calibration['covered_truth_pairs']:,}/{calibration['total_truth_pairs']:,})"
    )
    print("  確率帯       ペア数  平均推定確率  実出現率")
    for _, _, label in CALIBRATION_BINS:
        item = calibration["bins"][label]
        print(
            f"  {label:>10} {item['n_pairs']:>7,} "
            f"{item['mean_probability']:>12.4f} {item['actual_rate']:>9.4f}"
        )
    print("\n  同程度の予測ペア数でのsupport比較")
    for label, key in (("EXP01", "exp01"), ("確率>=0.20", "probability_at_0.20")):
        item = calibration["support_comparison"][key]
        print(
            f"    {label:<12} {item['predicted_pairs_per_day']:>6.1f}ペア/日  "
            f"precision {item['precision']:.4f}  recall {item['recall']:.4f}"
        )


def save_calibration_figure(calibration, out=CALIBRATION_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP",
        "IPAexGothic", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    labels = [label for _, _, label in CALIBRATION_BINS]
    expected = [calibration["bins"][label]["mean_probability"] for label in labels]
    actual = [calibration["bins"][label]["actual_rate"] for label in labels]
    counts = [calibration["bins"][label]["n_pairs"] for label in labels]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.2, label="完全校正")
    ax.plot(
        expected, actual, "o-", color="#2563eb", lw=2.0, ms=7,
        label="EXP02出現確率",
    )
    for x, y, count in zip(expected, actual, counts):
        ax.annotate(f"n={count:,}", (x, y), xytext=(5, 7),
                    textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("平均推定確率")
    ax.set_ylabel("検証日の実出現率")
    ax.set_title("EXP02 非対角ODペアの確率校正")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[figure] {out}")
    return out


# ---------------------------------------------------------------- Stage 2: ソフト予測

def predict_day(base_prediction, templates, day):
    """非対角期待人流を作り、EXP01のorigin別行合計を対角へ戻す。"""
    weekday = to_date(day).weekday()
    alpha = interpolation_weight(day, templates["setup"]["bound"])
    expected_a = templates["expected"]["A"][weekday]
    expected_b = templates["expected"]["B"][weekday]
    new_off = defaultdict(dict)

    for origin, destination in set(expected_a) | set(expected_b):
        value = (
            (1.0 - alpha) * expected_a.get((origin, destination), 0.0)
            + alpha * expected_b.get((origin, destination), 0.0)
        )
        if value > 0:
            new_off[origin][destination] = value

    prediction = {}
    scaled_origins = 0
    for origin in set(base_prediction) | set(new_off):
        row_total = sum(base_prediction.get(origin, {}).values())
        if row_total <= 0:
            continue
        destinations = dict(new_off.get(origin, {}))
        off_total = sum(destinations.values())
        if off_total > row_total:
            factor = row_total / off_total
            destinations = {
                destination: value * factor
                for destination, value in destinations.items()
                if value * factor > 0
            }
            off_total = row_total
            scaled_origins += 1
        diagonal = max(0.0, row_total - off_total)
        if diagonal > 0:
            destinations[origin] = diagonal
        if destinations:
            prediction[origin] = destinations

    return prediction, alpha, scaled_origins


# ---------------------------------------------------------------- metrics・表示

def period_config(setup):
    return {
        side: [to_int(setup[side][0]), to_int(setup[side][1])]
        for side in ("A", "B")
    } | {
        "interpolation_bounds": [
            to_int(setup["bound"][0]), to_int(setup["bound"][1])
        ]
    }


def select_records(records, predicate):
    selected = [record for record in records if predicate(record)]
    return (
        [record["error"] for record in selected],
        [record["zero"] for record in selected],
    )


def candidate_metrics(records):
    """4候補の比較に必要な数値を保存用にまとめる。"""
    errors, zeros = select_records(records, lambda _: True)
    return {
        "overall": exp01.score_entry(errors, zeros),
        "validation": {
            name: exp01.score_entry(
                *select_records(
                    records,
                    lambda record, name=name: record["window"] == name,
                )
            )
            for name, _, _ in VAL_WINDOWS
        },
        "error_breakdown": exp01.error_breakdown(errors),
    }


def update_metrics(records, calibration, templates, candidate_records, selected):
    all_errors, all_zero = select_records(records, lambda _: True)
    windows = {
        name: exp01.score_entry(
            *select_records(records, lambda record, name=name: record["window"] == name)
        )
        for name, _, _ in VAL_WINDOWS
    }
    groups = {
        "weekday": exp01.score_entry(
            *select_records(records, lambda record: record["weekday"] < 5)
        ),
        "weekend": exp01.score_entry(
            *select_records(records, lambda record: record["weekday"] >= 5)
        ),
    }
    weekdays = {
        f"{weekday}_{WD[weekday]}": exp01.score_entry(
            *select_records(
                records,
                lambda record, weekday=weekday: record["weekday"] == weekday,
            )
        )
        for weekday in range(7)
    }
    selected_a_window = templates["a_window"]
    selected_statistic = templates["value_statistic"]
    comparisons = {}
    for name, a_window, statistic in candidate_specs():
        candidate = candidate_metrics(candidate_records[name])
        comparisons[name] = {
            "a_window": a_window,
            "value_statistic": statistic,
            "combined_nrmse": candidate["overall"]["combined_nrmse"],
            "nrmse_diag": candidate["overall"]["nrmse_diag"],
            "nrmse_offdiag": candidate["overall"]["nrmse_offdiag"],
            "jan_combined_nrmse": candidate["validation"]["jan"]["combined_nrmse"],
            "apr_combined_nrmse": candidate["validation"]["apr"]["combined_nrmse"],
        }
    payload = {
        "experiment": EXP_ID,
        "date": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "compared_to": "EXP01",
        "status": "評価済み",
        "config": {
            "method": "weekday binary occurrence probability x selected positive statistic; A-window selected by validation",
            "validation_periods": period_config(VAL_SETUPS[selected_a_window]),
            "submission_periods": period_config(SUB_SETUPS[selected_a_window]),
            "a_window_candidates": {
                name: {
                    "validation": [
                        to_int(VAL_SETUPS[name]["A"][0]),
                        to_int(VAL_SETUPS[name]["A"][1]),
                    ],
                    "submission": [
                        to_int(SUB_SETUPS[name]["A"][0]),
                        to_int(SUB_SETUPS[name]["A"][1]),
                    ],
                }
                for name in A_WINDOWS
            },
            "probability": "presence count / valid observed days; NA excluded",
            "value_statistic_candidates": list(VALUE_STATISTICS),
            "selected_candidate": {
                "id": selected,
                "a_window": selected_a_window,
                "value_statistic": selected_statistic,
            },
            "selection_metric": "14-day overall combined_nrmse",
            "row_mass_conservation": "EXP01 origin row total; residual assigned to diagonal",
            "hyperparameters": None,
        },
        "results": {
            "overall": exp01.score_entry(all_errors, all_zero),
            "validation": windows,
            "groups": groups,
            "weekdays": weekdays,
            "error_breakdown": exp01.error_breakdown(all_errors),
            "stage1_calibration": calibration,
            "template": template_summary(templates),
            "candidate_comparison": {
                "selected": selected,
                "candidates": comparisons,
            },
        },
    }
    write_metrics(METRICS, payload)
    print(f"[write] {METRICS}")
    return payload


def print_score_results(records):
    print("\n=== Stage 2: 公式スコア ===")
    print(
        f"  {'区分':>8} {'日数':>4} {'NRMSE_diag':>11} "
        f"{'NRMSE_off':>10} {'Combined':>10} {'全ゼロ比':>10}"
    )

    def one(label, selected):
        errors = [record["error"] for record in selected]
        zeros = [record["zero"] for record in selected]
        result = score(errors)
        zero = score(zeros)
        print(
            f"  {label:>8} {len(selected):>4} {result['nrmse_diag']:>11.4f} "
            f"{result['nrmse_off']:>10.4f} {result['combined']:>10.4f} "
            f"{result['combined']/zero['combined']:>10.4f}"
        )
        return result

    window_results = {}
    for name, _, _ in VAL_WINDOWS:
        window_results[name] = one(
            name, [record for record in records if record["window"] == name]
        )
    overall = one("統合", records)
    weekday = one("平日", [record for record in records if record["weekday"] < 5])
    weekend = one("週末", [record for record in records if record["weekday"] >= 5])

    by_weekday = {
        weekday_index: score(
            [
                record["error"]
                for record in records
                if record["weekday"] == weekday_index
            ]
        )
        for weekday_index in range(7)
    }
    print("\n  曜日別Combined（各2日）")
    print(
        "  " + "  ".join(
            f"{WD[index]}={by_weekday[index]['combined']:.4f}"
            for index in range(7)
        )
    )

    totals = {
        key: sum(record["error"][key] for record in records)
        for key in exp01.ERROR_KEYS
    }
    print("\n  統合誤差の内訳")
    for label, suffix, se_key, pair_key in (
        ("diagonal", "d", "se_diag", "diag"),
        ("offdiag ", "o", "se_off", "off"),
    ):
        print(
            f"    {label}  hit {pct(totals[f'hit_{suffix}'], totals[se_key]):5.1f}%  "
            f"miss {pct(totals[f'miss_{suffix}'], totals[se_key]):5.1f}%  "
            f"extra {pct(totals[f'extra_{suffix}'], totals[se_key]):5.1f}%  "
            f"observed {totals[f'n_{pair_key}_obs']/len(records):.1f}/日  "
            f"predicted {totals[f'n_{pair_key}_pred']/len(records):.1f}/日"
        )

    if BASE_METRICS.exists():
        baseline = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
        baseline_score = baseline["results"]["overall"]["combined_nrmse"]
        print(
            f"\n  EXP01比: {overall['combined'] - baseline_score:+.4f} "
            f"({baseline_score:.4f} → {overall['combined']:.4f})"
        )

    return {
        "overall": overall,
        "windows": window_results,
        "weekday": weekday,
        "weekend": weekend,
    }


# ---------------------------------------------------------------- 実行モード

def validate(tsv, log=None):
    from common.local_error_map import save_local_error_map
    from common.municipality_error import write_municipality_errors

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:validate", actual_command()
    )
    with context as log:
        log.stage("validate-load", "観測データとEXP01基準予測の準備")
        rows, _ = read_od_days(tsv)
        templates_by_candidate = {
            name: build_templates(rows, VAL_SETUPS[a_window], a_window, statistic)
            for name, a_window, statistic in candidate_specs()
        }
        base_anchors = exp01.build_anchors(rows, exp01.VAL_BOUND)
        for name, _, _ in candidate_specs():
            show_template_summary(templates_by_candidate[name], f"候補 {name}")

        expected_days = set(validation_days())
        truths, _ = read_od_days(tsv, expected_days.__contains__)
        if set(truths) != expected_days:
            raise RuntimeError(
                f"検証日不一致: missing={sorted(expected_days-set(truths))}, "
                f"extra={sorted(set(truths)-expected_days)}"
            )
        for name, templates in templates_by_candidate.items():
            training_days = {
                day
                for side in ("A", "B")
                for weekday in range(7)
                for day in templates["observed_days"][side][weekday]
            }
            leaked = sorted(training_days & expected_days)
            if leaked:
                raise RuntimeError(
                    f"{name}の学習期間が検証日に触れています: {leaked}"
                )
            print(
                f"  リーク検査 [OK] {name}: "
                f"学習{len(training_days)}日は検証14日の外"
            )

        base_predictions = {
            day: exp01.predict(base_anchors, day)[0] for day in sorted(truths)
        }

        log.stage("validate-calibration", "A窓2種の二値出現確率を校正")
        calibrations_by_window = {}
        for a_window in A_WINDOWS:
            templates = templates_by_candidate[candidate_id(a_window, "median")]
            print(f"\n######## A窓: {a_window} ########")
            calibration = evaluate_calibration(
                templates, truths, base_predictions
            )
            calibrations_by_window[a_window] = calibration
            show_calibration(calibration)

        log.stage("validate-candidates", "A窓2種×代表値2種の4候補評価と採用判定")
        predictions_by_candidate = {}
        records_by_candidate = {}
        summaries_by_candidate = {}
        for name, a_window, statistic in candidate_specs():
            predictions = {}
            records = []
            total_scaled_origins = 0
            for day in sorted(truths):
                prediction, alpha, scaled_origins = predict_day(
                    base_predictions[day], templates_by_candidate[name], day
                )
                error = day_errors(prediction, truths[day])
                zero = day_errors({}, truths[day])
                predictions[day] = prediction
                records.append({
                    "day": day,
                    "window": validation_window(day),
                    "weekday": to_date(day).weekday(),
                    "alpha": alpha,
                    "error": error,
                    "zero": zero,
                })
                total_scaled_origins += scaled_origins
            print(
                f"\n######## 候補: {name} "
                f"(A={a_window}, 値={statistic}) ########"
            )
            summaries_by_candidate[name] = print_score_results(records)
            print(
                f"  非対角を比例縮小したorigin: "
                f"{total_scaled_origins}件（14日合計）"
            )
            predictions_by_candidate[name] = predictions
            records_by_candidate[name] = records

        selected = min(
            (name for name, _, _ in candidate_specs()),
            key=lambda name: summaries_by_candidate[name]["overall"]["combined"],
        )
        predictions = predictions_by_candidate[selected]
        records = records_by_candidate[selected]
        summaries = summaries_by_candidate[selected]
        templates = templates_by_candidate[selected]
        calibration = calibrations_by_window[templates["a_window"]]
        print("\n=== 4候補の採用判定 ===")
        for name, _, _ in candidate_specs():
            print(
                f"  {name:<20}: Combined "
                f"{summaries_by_candidate[name]['overall']['combined']:.6f}"
            )
        print(f"  採用: {selected}（14日統合Combined NRMSE最小）")

        log.stage("validate-artifacts", "採用版の誤差図と市町別誤差CSV")
        save_local_error_map(predictions, truths, ERROR_MAP, experiment=EXP_ID)
        save_calibration_figure(calibration)
        municipality_error_csv = write_municipality_errors(
            predictions, truths, EXP_ID
        )

        log.stage("validate-metrics", "metrics.jsonを更新")
        update_metrics(
            records, calibration, templates, records_by_candidate, selected
        )

        result_values = {
            name: summaries_by_candidate[name]["overall"]["combined"]
            for name, _, _ in candidate_specs()
        }
        log.result(
            selected=selected,
            combined=summaries["overall"]["combined"],
            **result_values,
        )
        log.note(f"→ {METRICS.relative_to(ROOT)}")
        log.note(f"→ {ERROR_MAP.relative_to(ROOT)}")
        log.note(f"→ {municipality_error_csv.relative_to(ROOT)}")
    return predictions


def load_selected_candidate():
    """直近の検証で採用されたA窓と代表値をmetrics.jsonから読む。"""
    if not METRICS.exists():
        raise RuntimeError("先に `python3 src/EXP02_run.py` を実行してください")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    selected = metrics.get("config", {}).get("selected_candidate", {})
    name = selected.get("id")
    a_window = selected.get("a_window")
    statistic = selected.get("value_statistic")
    valid_names = {item[0] for item in candidate_specs()}
    if (
        name not in valid_names
        or a_window not in A_WINDOWS
        or statistic not in VALUE_STATISTICS
        or name != candidate_id(a_window, statistic)
    ):
        raise RuntimeError(
            "metrics.jsonに採用候補がありません。"
            "`python3 src/EXP02_run.py` を再実行してください"
        )
    return name, a_window, statistic


def build(tsv, out, validation_predictions, log=None):
    from common.pred_dist import save_pred_dist

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:submit", actual_command()
    )
    with context as log:
        log.stage("submit-load", "観測データと提出用テンプレートの準備")
        rows, _ = read_od_days(tsv)
        selected, a_window, statistic = load_selected_candidate()
        templates = build_templates(
            rows, SUB_SETUPS[a_window], a_window, statistic
        )
        base_anchors = exp01.build_anchors(rows, exp01.SUB_BOUND)
        show_template_summary(templates, "提出・採用候補")
        print(f"  採用候補: {selected}（metrics.jsonの検証結果）")

        log.stage("submit-write", "採用候補だけで提出予測を書き出し")
        predictions = {}
        total_scaled_origins = 0
        for day in submission_days():
            base_prediction = exp01.predict(base_anchors, day)[0]
            prediction, _, scaled = predict_day(
                base_prediction, templates, day
            )
            predictions[day] = prediction
            total_scaled_origins += scaled
        print(
            f"  {selected}: 非対角を比例縮小したorigin "
            f"{total_scaled_origins}件（58日合計）"
        )
        output = write_od_tsv(predictions, out)

        log.stage("submit-check", "形式検査")
        print("\n=== 形式検査 ===")
        check_od_tsv(output, submission_days(), DAILY_TOTAL)

        log.stage("submit-figure", "日次推移図")
        save_pred_dist(EXP_ID, tsv=output,
                       validation_predictions=validation_predictions)

        log.result(
            selected=selected,
            days=len(predictions),
            size=f"{output.stat().st_size/1e6:.1f}MB",
            check="PASS",
        )
        try:
            note = output.relative_to(ROOT)
        except ValueError:
            note = output
        log.note(f"→ {note}")
    return output


def run():
    """EXP02を比較・採用し、提出物まで一度に再生成する。"""
    with run_log(EXP_ID, "run", actual_command()) as log:
        validation_predictions = validate(TSV, log)
        build(TSV, OUT, validation_predictions, log)


if __name__ == "__main__":
    run()
