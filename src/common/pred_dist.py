"""提出物の評価グリッド内予測を「日次推移の折れ線1本」に要約する。

使い方:
  python3 src/common/pred_dist.py EXP00     # 単体実行（実験IDを渡す）

origin・destinationがともに評価グリッド内の値だけを対象に、対角（自セル滞留）と
非対角（セル間移動）を分けて縦に2段で描く。時間軸はローカル
検証期間を前後1週間ずつ覆う範囲（2024-01-18〜04-14）。背景色で CV期間（検証2窓）と
テスト期間を区別する。予測不要日は線を切って表す。表示範囲に観測日がある区間は、
配布データの正解フローを黒線で重ねる（テスト期間58日には正解が無いので線は切れる）。

入力 : experiments/<ID>/outputs/submission.tsv、または呼び出し側の予測辞書
       data/raw/humob2026-dataset.tsv（正解フロー）
出力 : experiments/<ID>/figures/<ID>_pred_dist.png
依存 : common/local_error_map.py（検証窓・配布TSVの定義）, matplotlib
"""

import ast
import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from common.experiment_io import RAW_TSV  # noqa: E402
from common.local_error_map import VAL_WINDOWS  # noqa: E402  定義はあちらが正本
from common.competition_metric import in_eval_box  # noqa: E402

# 日本語フォント。rcParams への代入はフォント名を検証しない
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro",        # macOS
    "Yu Gothic", "Meiryo",                              # Windows
    "Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic",  # Linux
    "DejaVu Sans",                                      # 最終手段（日本語は出ない）
]
matplotlib.rcParams["axes.unicode_minus"] = False

EXP = ROOT / "experiments"

TEST_SPAN = (dt.date(2024, 2, 1), dt.date(2024, 3, 31))
MARGIN = dt.timedelta(days=7)   # 表示範囲は検証期間の前後1週間まで

CV_COLOR = "#e8f0fb"     # 薄い青 = CV期間
TEST_COLOR = "#c3d9f0"   # やや濃い青 = テスト期間
PRED_COLOR = "#e8590c"   # 青系の背景に埋もれない橙で予測線を描く


def submission_path(exp_id):
    return EXP / exp_id / "outputs" / "submission.tsv"


def figure_path(exp_id):
    return EXP / exp_id / "figures" / f"{exp_id}_pred_dist.png"


def load_daily_stats(tsv, keep=None, evaluation_only=True):
    """TSVを1行ずつ読み、日付→{total, n_pairs, diag} を返す。

    ``keep`` に日付の集合を渡すと、その日だけをパースする（配布データ全306日を
    読むと重いので、表示範囲だけに絞るために使う）。
    ``evaluation_only`` はorigin・destinationがともに評価グリッド内の値へ限定する。
    """
    stats = {}
    with Path(tsv).open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            ds, payload = line.split("\t", 1)
            if payload.strip() == "NA":
                continue
            d = dt.date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
            if keep is not None and d not in keep:
                continue
            total = diag = 0.0
            n_pairs = 0
            for o_gid, dest_map in ast.literal_eval(payload).items():
                for d_gid, v in dest_map.items():
                    if evaluation_only and not (
                        in_eval_box(o_gid) and in_eval_box(d_gid)
                    ):
                        continue
                    total += v
                    n_pairs += 1
                    if d_gid == o_gid:
                        diag += v
            stats[d] = {"total": total, "n_pairs": n_pairs, "diag": diag}
    return stats


def daily_stats(predictions, keep=None, evaluation_only=True):
    """日付→OD辞書を、描画用の日別統計へ変換する。"""
    stats = {}
    for day, od in predictions.items():
        date = dt.date(day // 10000, day // 100 % 100, day % 100)
        if keep is not None and date not in keep:
            continue
        total = diag = 0.0
        n_pairs = 0
        for origin, destinations in od.items():
            for destination, value in destinations.items():
                if evaluation_only and not (
                    in_eval_box(origin) and in_eval_box(destination)
                ):
                    continue
                total += value
                n_pairs += 1
                if destination == origin:
                    diag += value
        stats[date] = {"total": total, "n_pairs": n_pairs, "diag": diag}
    return stats


def _span(ax, d0, d1, color, label=None):
    """日単位の背景帯。date 同士の演算は日に丸まるので datetime で半日ずらす。"""
    a = dt.datetime.combine(d0, dt.time()) - dt.timedelta(hours=12)
    b = dt.datetime.combine(d1, dt.time()) + dt.timedelta(hours=12)
    ax.axvspan(a, b, color=color, lw=0, zorder=0, label=label)


def save_pred_dist(exp_id, tsv=None, out=None, predictions=None,
                   validation_predictions=None, language="ja"):
    """予測の日次推移を描く。``language`` は ``ja`` または ``en``。"""
    if language not in ("ja", "en"):
        raise ValueError("language must be 'ja' or 'en'")
    tsv = Path(tsv) if tsv else submission_path(exp_id)
    out = Path(out) if out else figure_path(exp_id)
    if predictions is None and not tsv.exists():
        print(f"[skip] submission.tsv がありません: {tsv}")
        return None

    stats = daily_stats(predictions) if predictions is not None else load_daily_stats(tsv)
    if not stats:
        print("[skip] 予測日が1日もありません")
        return None

    # 検証期間を前後1週間ずつ覆う範囲を軸に取り、予測不要日は線の切れ目にする
    x0 = min(d0 for _, d0, _ in VAL_WINDOWS) - MARGIN
    x1 = max(d1 for _, _, d1 in VAL_WINDOWS) + MARGIN
    days = [x0 + dt.timedelta(i) for i in range((x1 - x0).days + 1)]
    truth = load_daily_stats(RAW_TSV, keep=set(days)) if Path(RAW_TSV).exists() else {}
    if validation_predictions:
        stats.update(daily_stats(validation_predictions, keep=set(days)))

    def series(src, key):
        return [(src[d]["diag"] if key == "diag" else
                 src[d]["total"] - src[d]["diag"]) if d in src
                else float("nan") for d in days]

    have = sorted(stats)
    diag_avg = sum(stats[d]["diag"] for d in have) / len(have)
    off_avg = sum(stats[d]["total"] - stats[d]["diag"] for d in have) / len(have)
    print(f"[pred_dist] {exp_id}: {len(have)}日 {have[0]}〜{have[-1]} / "
          f"日次フロー平均 対角 {diag_avg:,.2f} / 非対角 {off_avg:,.2f}")
    print(f"  正解フロー: 表示範囲に観測 {len(truth)}日")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # y軸ラベルは縦書き（上から読む）
    panels = [
        (axes[0], "diag", "対\n角\n成\n分"),
        (axes[1], "off", "非\n対\n角\n成\n分"),
    ]
    if language == "en":
        panels = [
            (axes[0], "diag", "Diagonal flow\n(normalized total)"),
            (axes[1], "off", "Off-diagonal flow\n(normalized total)"),
        ]

    for ax, key, ylabel in panels:
        _span(ax, *TEST_SPAN, TEST_COLOR)
        for _, d0, d1 in VAL_WINDOWS:
            _span(ax, d0, d1, CV_COLOR)

        pred = series(stats, key)
        obs = series(truth, key)
        ax.plot(days, obs, "o-", color="black", lw=1.4, ms=3.0, zorder=3)
        ax.plot(days, pred, "o-", color=PRED_COLOR, lw=1.6, ms=3.5, zorder=2)

        v = [x for x in pred + obs if x == x]
        lo, hi = min(v), max(v)
        pad = max((hi - lo) * 0.15, sum(v) / len(v) * 0.004)
        ax.set_ylim(lo - pad, hi + pad)
        if language == "en":
            ax.set_ylabel(ylabel, labelpad=10)
        else:
            ax.set_ylabel(ylabel, rotation=0, va="center", ha="right",
                          linespacing=1.1, labelpad=12)
        ax.grid(alpha=0.3)

    axes[0].set_xlim(days[0], days[-1])
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    labels = ("Predicted", "Observed", "Validation period", "Test period", "Date (2024)") \
        if language == "en" else ("予測", "正解（配布データ）", "CV期間", "テスト期間", "日付")
    title = (f"{exp_id} - Daily Flow Totals within the Evaluation Area" if language == "en"
             else f"{exp_id} - 評価グリッド内の日次推移")
    axes[0].set_title(title, fontsize=13)
    axes[0].legend(handles=[
        plt.Line2D([0], [0], marker="o", color=PRED_COLOR, lw=1.6, ms=3.5, label=labels[0]),
        plt.Line2D([0], [0], marker="o", color="black", lw=1.4, ms=3.0,
                   label=labels[1]),
        Patch(facecolor=CV_COLOR, label=labels[2]),
        Patch(facecolor=TEST_COLOR, label=labels[3]),
    ], fontsize=9, loc="best", framealpha=0.9)
    axes[1].set_xlabel(labels[4])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[write] {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("使い方: python3 src/common/pred_dist.py <ID>   （例: EXP00）")
    save_pred_dist(sys.argv[1])
