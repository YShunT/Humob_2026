"""EXP00: 2観測日の線形補間ベースラインを検証し、提出物を生成する。

実行: python3 src/EXP00_run.py
入力: data/raw/humob2026-dataset.tsv
出力: experiments/EXP00/{metrics.json,.log,outputs/,figures/}
依存: common/{competition_metric,experiment_io,local_error_map,municipality_error,pred_dist,run_log}.py
"""

import datetime as dt
import json
import unicodedata
from contextlib import nullcontext
from pathlib import Path

from common.competition_metric import (
    EVAL_X, EVAL_Y, N_CELL, N_DIAG, N_OFF, NORM_DIAG, NORM_OFF,
    day_errors, pct, score,
)
from common.experiment_io import (
    DAILY_TOTAL, check_od_tsv, date_range as daterange,
    read_od_days as read_days, submission_days, to_date, to_int,
    write_metrics, write_od_tsv,
)
from common.run_log import run_log

EXP_ID = "EXP00"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TSV = ROOT / "data" / "raw" / "humob2026-dataset.tsv"
EXP = ROOT / "experiments" / EXP_ID
OUT = EXP / "outputs" / "submission.tsv"
METRICS = EXP / "metrics.json"
ERROR_MAP = EXP / "figures" / "local_error_map.png"

# ローカル検証窓: テスト期間を前後から挟む各1週間
VAL_WINDOWS = [
    ("jan", dt.date(2024, 1, 25), dt.date(2024, 1, 31)),
    ("apr", dt.date(2024, 4, 1), dt.date(2024, 4, 7)),
]

# アンカーは検証窓を外側から挟む単日。窓の内側を1日も含まないので検証にリークしない。
# この2点はテスト期間58日も覆うため、検証と提出で同じ補間直線を使える。
# 目標日（1/24・4/8）はどちらも NA なので、実行時に外側の直近観測日へずらす。
ANCHOR_A = min(d0 for _, d0, _ in VAL_WINDOWS) - dt.timedelta(days=1)   # 2024-01-24
ANCHOR_B = max(d1 for _, _, d1 in VAL_WINDOWS) + dt.timedelta(days=1)   # 2024-04-08


# ---------------------------------------------------------------- 共通ユーティリティ

def rjust(s, n):
    """全角文字を2桁として数えて右詰めする（表の桁ずれ防止）。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return " " * max(0, n - w) + s


def npairs(od):
    return sum(len(dm) for dm in od.values())


# ---------------------------------------------------------------- 補間

def resolve_anchor(tsv, target, step):
    """target から step 方向で最初に見つかる単一の観測日を返す。

    step=-1 なら過去方向（＝検証窓から離れる向き）、+1 なら未来方向へ探す。
    返り値は (使用日int, その日のOD, targetからのずれ日数)。
    """
    for k in range(0, 120):
        d = to_int(target + dt.timedelta(step * k))
        rows, _ = read_days(tsv, lambda x, d=d: x == d)
        if rows:
            return d, rows[d], k
    raise ValueError(f"{target} 付近に観測日が見つかりません: {tsv}")


def build_union(od_a, od_b):
    """2アンカーの OD を {origin: {dest: (a, b)}} に畳む。片方に無い側は 0。"""
    u = {}
    for o, dm in od_a.items():
        u.setdefault(o, {}).update({d: [v, 0.0] for d, v in dm.items()})
    for o, dm in od_b.items():
        t = u.setdefault(o, {})
        for d, v in dm.items():
            if d in t:
                t[d][1] = v
            else:
                t[d] = [0.0, v]
    return u


def interp(union, w):
    """重み w（0でアンカーA、1でアンカーB）の OD 行列を作る。"""
    return {o: {d: a + (b - a) * w for d, (a, b) in dm.items()}
            for o, dm in union.items()}


def weight(day, da, db):
    """日付 day のアンカー間位置。da で 0、db で 1。"""
    return (to_date(day) - to_date(da)).days / (to_date(db) - to_date(da)).days


def load_anchors(tsv):
    """両アンカーを解決して (da, db, union) を返し、内容を表示する。"""
    da, od_a, ka = resolve_anchor(tsv, ANCHOR_A, -1)
    db, od_b, kb = resolve_anchor(tsv, ANCHOR_B, +1)
    union = build_union(od_a, od_b)
    wd = "月火水木金土日"

    def show(tag, day, target, k, od):
        note = ""
        if k:
            note = (f" ※目標 {to_int(target)} が NA のため "
                    f"{k}日{'さかのぼった' if tag == 'A' else '進めた'}")
        print(f"[anchor] {tag} {day}（{wd[to_date(day).weekday()]}）{note} / "
              f"{npairs(od):,} ペア")

    show("A", da, ANCHOR_A, ka, od_a)
    show("B", db, ANCHOR_B, kb, od_b)
    print(f"[anchor] 区間 {(to_date(db) - to_date(da)).days}日 / "
          f"和集合 {npairs(union):,} ペア")
    return da, db, union


# ---------------------------------------------------------------- metrics.json

def update_metrics(da, db, windows, overall):
    """検証結果を metrics.json へ書き戻す。

    ``windows`` は [(名前, 日数, score, 全ゼロscore)]、``overall`` は同じ形の統合1件。
    status と compared_to は人が決める項目なので、既存の値を残す。
    """
    old = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}

    def entry(n_days, sc, sz):
        return {
            "n_days": n_days,
            "combined_nrmse": round(sc["combined"], 4),
            "nrmse_diag": round(sc["nrmse_diag"], 4),
            "nrmse_offdiag": round(sc["nrmse_off"], 4),
            "vs_zero": round(sc["combined"] / sz["combined"], 4),
        }

    m = {
        "experiment": EXP_ID,
        "date": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "compared_to": old.get("compared_to"),
        "status": old.get("status", "基準"),
        "config": {
            "method": "per-pair linear interpolation",
            "anchor_a": da,
            "anchor_b": db,
            "anchor_rule": "検証窓を外側から挟む直近の観測日（目標 1/24・4/8 は NA）",
            "val_windows": {name: [to_int(d0), to_int(d1)]
                            for name, d0, d1 in VAL_WINDOWS},
        },
        "results": {
            "overall": entry(*overall),
            "validation": {name: entry(nd, sc, sz) for name, nd, sc, sz in windows},
        },
    }
    write_metrics(METRICS, m)
    print(f"[write] {METRICS}")
    return m


# ---------------------------------------------------------------- 検証

def validate(tsv, log=None):
    """2つの検証窓で B0 を評価し、metrics.json と .log を更新する。"""
    # 可視化依存（numpy / matplotlib）は検証時だけ読み込む。
    from common.local_error_map import save_local_error_map
    from common.municipality_error import write_municipality_errors

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:validate", "python3 src/EXP00_run.py"
    )
    with context as log:
        log.stage("validate-load", "アンカー2日の読み込み")
        da, db, union = load_anchors(tsv)

        log.stage("validate-days", "検証14日の抽出")
        win_days = {name: set(daterange(d0, d1)) for name, d0, d1 in VAL_WINDOWS}
        expected_days = set().union(*win_days.values())
        rows, _ = read_days(tsv, expected_days.__contains__)

        print("\n=== B0 ローカル検証 ===")
        print(f"手法        : アンカー {da} と {db} の per-pair 線形補間")
        print(f"評価境界    : x={EVAL_X[0]}〜{EVAL_X[1]}, y={EVAL_Y[0]}〜{EVAL_Y[1]}"
              f" → {N_CELL:,}セル / 母数 diag {N_DIAG:,}・offdiag {N_OFF:,} ペア")
        print(f"指標        : Combined NRMSE = (NRMSE_diag + NRMSE_offdiag) / 2")
        print(f"正規化定数  : diag {NORM_DIAG} / offdiag {NORM_OFF}（主催者提供）")
        print("              全ゼロ予測が約 1.0。小さいほど良い。")
        print("リーク防止  : 検証窓14日はどちらもアンカーの内側にあり、アンカー自身は")
        print("              窓に含まれない。予測は全窓で同一の補間直線から作る。")

        log.stage("validate-score", "補間とスコア計算")
        results, predictions = [], {}
        all_days, all_zero = [], []
        for name, d0, d1 in VAL_WINDOWS:
            truth_days = sorted(d for d in win_days[name] if d in rows)
            missing = sorted(d for d in win_days[name] if d not in rows)

            print(f"\n--- 検証窓 {name}: {d0}〜{d1} ---")
            print(f"  検証日 ({len(truth_days)}日): {truth_days}"
                  + (f"  ※データ無し {missing}" if missing else ""))

            per_day, zero_day = [], []
            print(f"\n  {rjust('日付', 10)} {'w':>7} {'RMSE_diag':>11} {'NRMSE_diag':>11} "
                  f"{'RMSE_off':>10} {'NRMSE_off':>10} {'Combined':>9}")
            for d in truth_days:
                w = weight(d, da, db)
                pred = interp(union, w)
                predictions[d] = pred
                x = day_errors(pred, rows[d])
                per_day.append(x)
                zero_day.append(day_errors({}, rows[d]))
                one = score([x])
                print(f"  {d:>10} {w:>7.4f} {one['rmse_diag']:>11.4f} "
                      f"{one['nrmse_diag']:>11.4f} {one['rmse_off']:>10.6f} "
                      f"{one['nrmse_off']:>10.4f} {one['combined']:>9.4f}")

            sc, sz = score(per_day), score(zero_day)
            all_days += per_day
            all_zero += zero_day
            print(f"  {rjust('平均', 10)} {'':>7} {sc['rmse_diag']:>11.4f} "
                  f"{sc['nrmse_diag']:>11.4f} {sc['rmse_off']:>10.6f} "
                  f"{sc['nrmse_off']:>10.4f} {sc['combined']:>9.4f}")
            print(f"  {rjust('全ゼロ予測', 10)} {'':>7} {sz['rmse_diag']:>11.4f} "
                  f"{sz['nrmse_diag']:>11.4f} {sz['rmse_off']:>10.6f} "
                  f"{sz['nrmse_off']:>10.4f} {sz['combined']:>9.4f}")

            agg = {k: sum(x[k] for x in per_day) for k in
                   ("se_diag", "se_off", "hit_d", "miss_d", "extra_d",
                    "hit_o", "miss_o", "extra_o",
                    "n_diag_obs", "n_off_obs", "n_diag_pred", "n_off_pred")}
            nd = len(per_day)
            print(f"\n  誤差の内訳（{nd}日合計の二乗和に占める割合）")
            for lab, sfx, key in [("diagonal   ", "d", "se_diag"),
                                  ("offdiagonal", "o", "se_off")]:
                obs = agg[f"n_{'diag' if sfx == 'd' else 'off'}_obs"] / nd
                prd = agg[f"n_{'diag' if sfx == 'd' else 'off'}_pred"] / nd
                print(f"    {lab} hit {pct(agg[f'hit_{sfx}'], agg[key]):5.1f}%  "
                      f"miss {pct(agg[f'miss_{sfx}'], agg[key]):5.1f}%  "
                      f"extra {pct(agg[f'extra_{sfx}'], agg[key]):5.1f}%   "
                      f"observed {obs:,.0f}ペア/日 vs predicted {prd:,.0f}")
            results.append((name, nd, sc, sz))

        print("\n=== まとめ ===")
        print(f"  {rjust('窓', 6)} {rjust('日数', 4)} {'NRMSE_diag':>11} "
              f"{'NRMSE_off':>10} {rjust('Combined NRMSE', 15)} {rjust('全ゼロ比', 10)}")

        def line(label, nd, sc, sz):
            print(f"  {rjust(label, 6)} {nd:>4} {sc['nrmse_diag']:>11.4f} "
                  f"{sc['nrmse_off']:>10.4f} {sc['combined']:>15.4f} "
                  f"{sc['combined']/sz['combined']:>10.4f}")

        for name, nd, sc, sz in results:
            line(name, nd, sc, sz)
        sc_all, sz_all = score(all_days), score(all_zero)
        line("統合", len(all_days), sc_all, sz_all)

        print("\n  jan 窓 = アンカーAに近い側 / apr 窓 = アンカーBに近い側。")
        print("  統合 = 14日を1本にまとめたスコア。実験間の比較にはこれを主に使う。")

        log.stage("validate-artifacts", "誤差マップと市町別誤差CSVの生成")
        save_local_error_map(predictions, rows, ERROR_MAP, experiment=EXP_ID)
        municipality_error_csv = write_municipality_errors(
            predictions, rows, EXP_ID
        )

        log.stage("validate-metrics", "metrics.jsonを更新")
        update_metrics(da, db, results, (len(all_days), sc_all, sz_all))

        log.result(combined=sc_all["combined"],
                   **{name: sc["combined"] for name, _, sc, _ in results})
        log.note(f"→ {METRICS.relative_to(ROOT)}")
        log.note(f"→ {ERROR_MAP.relative_to(ROOT)}")
        log.note(f"→ {municipality_error_csv.relative_to(ROOT)}")

    return predictions


# ---------------------------------------------------------------- 提出物の生成・検査

def write_submission(da, db, union, out):
    """提出58日を1日ずつ補間して書き出す。行形式は配布データと同一。"""
    days = submission_days()
    predictions = {day: interp(union, weight(day, da, db)) for day in days}
    out = write_od_tsv(predictions, out)
    w0, w1 = weight(days[0], da, db), weight(days[-1], da, db)
    print(f"[write] {out}  ({len(days)}日, {out.stat().st_size/1e6:.1f}MB) "
          f"w={w0:.4f}〜{w1:.4f}")
    return out


def build(tsv, validation_predictions, log=None):
    """提出物を作り、形式検査と日次推移図まで通す。"""
    from common.pred_dist import save_pred_dist

    context = nullcontext(log) if log else run_log(
        EXP_ID, "run:submit", "python3 src/EXP00_run.py"
    )
    with context as log:
        log.stage("submit-load", "アンカー2日の読み込み")
        da, db, union = load_anchors(tsv)

        log.stage("submit-write", "提出58日の補間と書き出し")
        out = write_submission(da, db, union, OUT)

        log.stage("submit-check", "形式検査")
        print("\n=== 形式検査 ===")
        check_od_tsv(out, submission_days(), DAILY_TOTAL)

        log.stage("submit-figure", "日次推移図")
        save_pred_dist(EXP_ID, validation_predictions=validation_predictions)

        log.result(days=len(submission_days()),
                   size=f"{out.stat().st_size / 1e6:.1f}MB",
                   check="PASS")
        log.note(f"→ {out.relative_to(ROOT)}")
    return out


def run():
    """EXP00を検証し、採用モデルの提出物まで一度に再生成する。"""
    with run_log(EXP_ID, "run", "python3 src/EXP00_run.py") as log:
        validation_predictions = validate(TSV, log)
        build(TSV, validation_predictions, log)


if __name__ == "__main__":
    run()
