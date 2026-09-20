"""EXP01: 同曜日アンカーの線形補間を検証し、提出物を生成する。

実行: python3 src/EXP01_run.py
入力: data/raw/humob2026-dataset.tsv
出力: experiments/EXP01/{metrics.json,.log,outputs/,figures/}
依存: common/{competition_metric,experiment_io,local_error_map,municipality_error,pred_dist,run_log}.py
"""

import datetime as dt
import json
import unicodedata
from contextlib import nullcontext
from pathlib import Path

from common.competition_metric import (
    EVAL_X, EVAL_Y, N_CELL, N_DIAG, N_OFF,
    day_errors, in_eval_box, pct, score,
)
from common.experiment_io import (
    DAILY_TOTAL, check_od_tsv, date_range as daterange,
    read_od_days as read_days, submission_days, to_date, to_int,
    write_metrics, write_od_tsv,
)
from common.run_log import run_log

EXP_ID = "EXP01"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TSV = ROOT / "data" / "raw" / "humob2026-dataset.tsv"
EXP = ROOT / "experiments" / EXP_ID
OUT = EXP / "outputs" / "submission.tsv"
METRICS = EXP / "metrics.json"
ERROR_MAP = EXP / "figures" / "local_error_map.png"
BASE_METRICS = ROOT / "experiments" / "EXP00" / "metrics.json"

WD = "月火水木金土日"

# 境界。アンカーはこの外側からしか取らない。
#   検証時は検証窓(1/25-31, 4/1-7)を hold out するので内側に寄る
#   提出時は学習データを全部使えるので窓ぎりぎりまで寄せられる
VAL_BOUND = (dt.date(2024, 1, 24), dt.date(2024, 4, 8))
SUB_BOUND = (dt.date(2024, 1, 31), dt.date(2024, 4, 1))

VAL_WINDOWS = [
    ("jan", dt.date(2024, 1, 25), dt.date(2024, 1, 31)),
    ("apr", dt.date(2024, 4, 1), dt.date(2024, 4, 7)),
]


# ---------------------------------------------------------------- 共通ユーティリティ

def rjust(s, n):
    """全角文字を2桁として数えて右詰めする（表の桁ずれ防止）。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return " " * max(0, n - w) + s


def npairs(od):
    return sum(len(dm) for dm in od.values())


def actual_command():
    return "python3 src/EXP01_run.py"


# ---------------------------------------------------------------- 曜日アンカー

def nominal_day(boundary, step, wd):
    """boundaryから外側へ進んだ最初の指定曜日を返す。"""
    day = boundary
    while day.weekday() != wd:
        day += dt.timedelta(step)
    return day


def pick_day(rows, boundary, step, wd):
    """曜日 wd のアンカーに使う観測日を1日返す。

    step=-1 なら boundary から過去方向（A側）、+1 なら未来方向（B側）。
    その日が NA なら **7日刻みで**さらに外へずらす（曜日を保つため）。
    """
    d = nominal_day(boundary, step, wd)
    for _ in range(60):
        if to_int(d) in rows:
            return to_int(d)
        d += dt.timedelta(step * 7)
    raise RuntimeError(
        f"{WD[wd]}曜のアンカーが取れません "
        f"(boundary={boundary}, step={step})"
    )


def build_union(od_a, od_b, box_only=False):
    """2アンカーの OD を {origin: {dest: [a, b]}} に畳む。片方に無い側は 0。"""
    u = {}
    for src, idx in ((od_a, 0), (od_b, 1)):
        for o, dm in src.items():
            if box_only and not in_eval_box(o):
                continue
            t = u.setdefault(o, {})
            for d, v in dm.items():
                if box_only and not in_eval_box(d):
                    continue
                if d not in t:
                    t[d] = [0.0, 0.0]
                t[d][idx] = v
    return {o: dm for o, dm in u.items() if dm}


def interp(union, w):
    """重み w（0でアンカーA、1でアンカーB）の OD 行列を作る。"""
    return {o: {d: a + (b - a) * w for d, (a, b) in dm.items()}
            for o, dm in union.items()}


def build_anchors(rows, bound, box_only=False):
    """曜日ごとのアンカーを組む。{weekday: (da, db, union)} を返す。"""
    out = {}
    for wd in range(7):
        da = pick_day(rows, bound[0], -1, wd)
        db = pick_day(rows, bound[1], +1, wd)
        out[wd] = (da, db, build_union(rows[da], rows[db], box_only))
    return out


def predict(anchors, day):
    """指定日の OD 行列を、その曜日のアンカーから補間して作る。"""
    da, db, union = anchors[to_date(day).weekday()]
    w = (to_date(day) - to_date(da)).days / (to_date(db) - to_date(da)).days
    return interp(union, w), w


def show_anchors(anchors, bound, label):
    print(f"[anchor] {label}  境界 {to_int(bound[0])} / {to_int(bound[1])}")
    print(f"  {rjust('曜日', 4)} {rjust('アンカーA', 10)} {rjust('アンカーB', 10) } "
          f"{rjust('区間', 6)} {rjust('A→境界', 8)} {rjust('ペア', 8)}")
    for wd in range(7):
        da, db, union = anchors[wd]
        lag = (bound[0] - to_date(da)).days
        nominal_a = nominal_day(bound[0], -1, wd)
        nominal_b = nominal_day(bound[1], +1, wd)
        shift_a = (nominal_a - to_date(da)).days
        shift_b = (to_date(db) - nominal_b).days
        notes = []
        if shift_a:
            notes.append(f"AをNAから{shift_a}日外へ")
        if shift_b:
            notes.append(f"BをNAから{shift_b}日外へ")
        print(f"  {rjust(WD[wd], 4)} {da:>10} {db:>10} "
              f"{(to_date(db)-to_date(da)).days:>5}日 {lag:>6}日 {npairs(union):>8,}"
              + (f"  ※{' / '.join(notes)}" if notes else ""))


# ---------------------------------------------------------------- metrics.json

ERROR_KEYS = (
    "se_diag", "se_off", "hit_d", "miss_d", "extra_d",
    "hit_o", "miss_o", "extra_o",
    "n_diag_obs", "n_off_obs", "n_diag_pred", "n_off_pred",
)


def aggregate_errors(day_results):
    """日別誤差から公式スコアと誤差内訳合計を返す。"""
    if not day_results:
        raise ValueError("集計対象の日別誤差がありません")
    return score(day_results), {
        key: sum(item[key] for item in day_results) for key in ERROR_KEYS
    }


def score_entry(day_results, zero_results):
    """metrics.jsonへ保存するスコア行を作る。"""
    sc = score(day_results)
    sz = score(zero_results)
    return {
        "n_days": len(day_results),
        "combined_nrmse": round(sc["combined"], 4),
        "nrmse_diag": round(sc["nrmse_diag"], 4),
        "nrmse_offdiag": round(sc["nrmse_off"], 4),
        "vs_zero": round(sc["combined"] / sz["combined"], 4),
    }


def error_breakdown(day_results):
    """hit/miss/extraと観測・予測ペア数の日平均を保存用にまとめる。"""
    _, totals = aggregate_errors(day_results)
    n_days = len(day_results)

    def component(suffix, se_key, pair_key):
        denominator = totals[se_key]
        return {
            "hit_pct": round(pct(totals[f"hit_{suffix}"], denominator), 3),
            "miss_pct": round(pct(totals[f"miss_{suffix}"], denominator), 3),
            "extra_pct": round(pct(totals[f"extra_{suffix}"], denominator), 3),
            "observed_pairs_per_day": round(
                totals[f"n_{pair_key}_obs"] / n_days, 3
            ),
            "predicted_pairs_per_day": round(
                totals[f"n_{pair_key}_pred"] / n_days, 3
            ),
        }

    return {
        "diagonal": component("d", "se_diag", "diag"),
        "offdiagonal": component("o", "se_off", "off"),
    }


def anchor_config(anchors):
    """曜日アンカーをmetrics.jsonへ保存できる形にする。"""
    return {
        f"{wd}_{WD[wd]}": {
            "anchor_a": anchors[wd][0],
            "anchor_b": anchors[wd][1],
            "span_days": (to_date(anchors[wd][1]) - to_date(anchors[wd][0])).days,
        }
        for wd in range(7)
    }


def update_metrics(validation_anchors, submission_anchors, records):
    """検証結果をEXP01/metrics.jsonへ原子的に書き戻す。"""
    old = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}

    def select(predicate):
        selected = [record for record in records if predicate(record)]
        return (
            [record["error"] for record in selected],
            [record["zero"] for record in selected],
        )

    all_errors, all_zero = select(lambda _: True)
    windows = {
        name: score_entry(*select(lambda record, name=name: record["window"] == name))
        for name, _, _ in VAL_WINDOWS
    }
    groups = {
        "weekday": score_entry(*select(lambda record: record["weekday"] < 5)),
        "weekend": score_entry(*select(lambda record: record["weekday"] >= 5)),
    }
    weekdays = {
        f"{wd}_{WD[wd]}": score_entry(
            *select(lambda record, wd=wd: record["weekday"] == wd)
        )
        for wd in range(7)
    }
    payload = {
        "experiment": EXP_ID,
        "date": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "compared_to": old.get("compared_to") or "EXP00",
        "status": "評価済み",
        "config": {
            "method": "weekday-matched per-pair linear interpolation",
            "hyperparameters": None,
            "validation_windows": {
                name: [to_int(start), to_int(end)]
                for name, start, end in VAL_WINDOWS
            },
            "validation_anchors": anchor_config(validation_anchors),
            "submission_anchors": anchor_config(submission_anchors),
            "anchor_rule": "境界外の同曜日。NAなら曜日を保って7日刻みで外へ移動",
        },
        "results": {
            "overall": score_entry(all_errors, all_zero),
            "validation": windows,
            "groups": groups,
            "weekdays": weekdays,
            "error_breakdown": error_breakdown(all_errors),
        },
    }
    write_metrics(METRICS, payload)
    print(f"[write] {METRICS}")
    return payload


# ---------------------------------------------------------------- 検証

def validate(tsv, log=None):
    """EXP01を14日で評価し、予測・図・metrics・追記ログを保存する。"""
    # matplotlib依存は検証時だけ読み込む。
    from common.local_error_map import save_local_error_map
    from common.municipality_error import write_municipality_errors

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:validate", actual_command()
    )
    with context as log:
        log.stage("validate-load", "観測データと曜日アンカーの読み込み")
        rows, _ = read_days(tsv, lambda _: True)
        anchors = build_anchors(rows, VAL_BOUND)
        submission_anchors = build_anchors(rows, SUB_BOUND)

        log.stage("validate-days", "検証14日の抽出とリーク検査")
        expected_days = {
            day for _, start, end in VAL_WINDOWS for day in daterange(start, end)
        }
        truths, _ = read_days(tsv, expected_days.__contains__)
        missing_truth = sorted(expected_days - set(truths))
        if missing_truth:
            raise RuntimeError(f"検証正解が不足しています: {missing_truth}")
        used_anchors = {
            anchors[weekday][index]
            for weekday in anchors
            for index in (0, 1)
        }
        leaked = sorted(used_anchors & expected_days)
        if leaked:
            raise RuntimeError(f"アンカーが検証窓に触れています: {leaked}")

        print("=== EXP01 ローカル検証 ===")
        show_anchors(anchors, VAL_BOUND, "検証用")
        print(f"\nリーク検査  : [OK] 使用アンカー{len(used_anchors)}日は検証窓の外")
        print("ハイパラ    : なし（曜日、境界、NA時の7日刻み規則だけで決まる）")
        print(f"評価境界    : x={EVAL_X[0]}〜{EVAL_X[1]}, y={EVAL_Y[0]}〜{EVAL_Y[1]}"
              f" → {N_CELL:,}セル / 母数 diag {N_DIAG:,}・offdiag {N_OFF:,} ペア")
        print("指標        : Combined NRMSE（全ゼロ予測が約1.0。小さいほど良い）")

        log.stage("validate-score", "曜日別補間とスコア計算")
        predictions = {}
        records = []
        window_summaries = {}
        for window, start, end in VAL_WINDOWS:
            target_days = [day for day in daterange(start, end) if day in truths]
            print(f"\n--- 検証窓 {window}: {start}〜{end} ---")
            print(f"  検証日 ({len(target_days)}日): {target_days}")
            print(f"\n  {rjust('日付', 10)} {rjust('曜', 3)} {'w':>7} "
                  f"{'RMSE_diag':>11} {'NRMSE_diag':>11} "
                  f"{'RMSE_off':>10} {'NRMSE_off':>10} {'Combined':>9}")

            window_errors, window_zero = [], []
            for day in target_days:
                prediction, interpolation_weight = predict(anchors, day)
                error = day_errors(prediction, truths[day])
                zero = day_errors({}, truths[day])
                one = score([error])
                predictions[day] = prediction
                window_errors.append(error)
                window_zero.append(zero)
                records.append({
                    "day": day,
                    "window": window,
                    "weekday": to_date(day).weekday(),
                    "error": error,
                    "zero": zero,
                })
                print(f"  {day:>10} {rjust(WD[to_date(day).weekday()], 3)} "
                      f"{interpolation_weight:>7.4f} {one['rmse_diag']:>11.4f} "
                      f"{one['nrmse_diag']:>11.4f} {one['rmse_off']:>10.6f} "
                      f"{one['nrmse_off']:>10.4f} {one['combined']:>9.4f}")

            window_score, totals = aggregate_errors(window_errors)
            zero_score = score(window_zero)
            window_summaries[window] = (window_score, zero_score)
            print(f"  {rjust('平均', 10)} {'':>3} {'':>7} "
                  f"{window_score['rmse_diag']:>11.4f} "
                  f"{window_score['nrmse_diag']:>11.4f} "
                  f"{window_score['rmse_off']:>10.6f} "
                  f"{window_score['nrmse_off']:>10.4f} "
                  f"{window_score['combined']:>9.4f}")
            print(f"  {rjust('全ゼロ予測', 10)} {'':>3} {'':>7} "
                  f"{zero_score['rmse_diag']:>11.4f} "
                  f"{zero_score['nrmse_diag']:>11.4f} "
                  f"{zero_score['rmse_off']:>10.6f} "
                  f"{zero_score['nrmse_off']:>10.4f} "
                  f"{zero_score['combined']:>9.4f}")

            print(f"\n  誤差の内訳（{len(window_errors)}日合計SSEに占める割合）")
            for label, suffix, se_key, pair_key in (
                ("diagonal   ", "d", "se_diag", "diag"),
                ("offdiagonal", "o", "se_off", "off"),
            ):
                print(
                    f"    {label} hit {pct(totals[f'hit_{suffix}'], totals[se_key]):5.1f}%  "
                    f"miss {pct(totals[f'miss_{suffix}'], totals[se_key]):5.1f}%  "
                    f"extra {pct(totals[f'extra_{suffix}'], totals[se_key]):5.1f}%   "
                    f"observed {totals[f'n_{pair_key}_obs']/len(window_errors):,.0f}ペア/日 vs "
                    f"predicted {totals[f'n_{pair_key}_pred']/len(window_errors):,.0f}"
                )

        all_errors = [record["error"] for record in records]
        all_zero = [record["zero"] for record in records]
        overall = score(all_errors)
        overall_zero = score(all_zero)

        def subset_score(predicate):
            selected = [record["error"] for record in records if predicate(record)]
            return score(selected)

        weekday_score = subset_score(lambda record: record["weekday"] < 5)
        weekend_score = subset_score(lambda record: record["weekday"] >= 5)
        by_weekday = {
            weekday: subset_score(lambda record, weekday=weekday: record["weekday"] == weekday)
            for weekday in range(7)
        }

        print("\n=== まとめ ===")
        print(f"  {rjust('区分', 8)} {rjust('日数', 4)} {'NRMSE_diag':>11} "
              f"{'NRMSE_off':>10} {rjust('Combined', 10)} {rjust('全ゼロ比', 10)}")

        def summary_line(label, n_days, result, zero=None):
            ratio = result["combined"] / zero["combined"] if zero else float("nan")
            ratio_text = f"{ratio:>10.4f}" if zero else f"{'-':>10}"
            print(f"  {rjust(label, 8)} {n_days:>4} {result['nrmse_diag']:>11.4f} "
                  f"{result['nrmse_off']:>10.4f} {result['combined']:>10.4f} {ratio_text}")

        for window, start, end in VAL_WINDOWS:
            result, zero = window_summaries[window]
            summary_line(window, len(daterange(start, end)), result, zero)
        summary_line("統合", len(records), overall, overall_zero)
        summary_line("平日", sum(record["weekday"] < 5 for record in records), weekday_score)
        summary_line("週末", sum(record["weekday"] >= 5 for record in records), weekend_score)

        print("\n  曜日別Combined（各2日）")
        print("  " + "  ".join(f"{WD[weekday]}={by_weekday[weekday]['combined']:.4f}"
                              for weekday in range(7)))
        if BASE_METRICS.exists():
            base = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
            base_score = base["results"]["overall"]["combined_nrmse"]
            print(f"\n  EXP00比    : {overall['combined'] - base_score:+.4f} "
                  f"({base_score:.4f} → {overall['combined']:.4f})")

        log.stage("validate-artifacts", "誤差マップと市町別誤差CSVの生成")
        save_local_error_map(predictions, truths, ERROR_MAP, experiment=EXP_ID)
        municipality_error_csv = write_municipality_errors(
            predictions, truths, EXP_ID
        )

        log.stage("validate-metrics", "metrics.jsonを更新")
        update_metrics(anchors, submission_anchors, records)

        log.result(
            combined=overall["combined"],
            jan=window_summaries["jan"][0]["combined"],
            apr=window_summaries["apr"][0]["combined"],
            weekday=weekday_score["combined"],
            weekend=weekend_score["combined"],
        )
        log.note(f"→ {METRICS.relative_to(ROOT)}")
        log.note(f"→ {ERROR_MAP.relative_to(ROOT)}")
        log.note(f"→ {municipality_error_csv.relative_to(ROOT)}")

    return predictions


# ---------------------------------------------------------------- 提出物

def build(tsv, out, validation_predictions, log=None):
    """提出物を生成し、形式検査と日次推移図まで実行する。"""
    from common.pred_dist import save_pred_dist

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:submit", actual_command()
    )
    with context as log:
        log.stage("submit-load", "観測データと曜日アンカーの読み込み")
        rows, _ = read_days(tsv, lambda _: True)
        anchors = build_anchors(rows, SUB_BOUND)
        show_anchors(anchors, SUB_BOUND, "提出用")

        log.stage("submit-write", "提出58日の補間と書き出し")
        days = submission_days()
        predictions = {day: predict(anchors, day)[0] for day in days}
        output = write_od_tsv(predictions, out)

        log.stage("submit-check", "形式検査")
        print("\n=== 形式検査 ===")
        check_od_tsv(output, submission_days(), DAILY_TOTAL)

        log.stage("submit-figure", "日次推移図")
        save_pred_dist(EXP_ID, tsv=output,
                       validation_predictions=validation_predictions)

        log.result(
            days=len(days),
            size=f"{output.stat().st_size / 1e6:.1f}MB",
            check="PASS",
        )
        log.note(f"→ {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    return output


def run():
    """EXP01を検証し、採用モデルの提出物まで一度に再生成する。"""
    with run_log(EXP_ID, "run", actual_command()) as log:
        validation_predictions = validate(TSV, log)
        build(TSV, OUT, validation_predictions, log)


if __name__ == "__main__":
    run()
