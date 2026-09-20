"""実験ごとのスコア推移を experiments/*/metrics.json から作図する。

使い方:
  python3 src/plot_score_history.py

出力 : experiments/score_history.png
依存 : matplotlib
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "experiments"
OUT = EXP / "score_history.png"

# 日本語フォント。rcParams への代入はフォント名を検証しない
# （src/common/grid_geo_map.py と同じ指定）。
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro",        # macOS
    "Yu Gothic", "Meiryo",                              # Windows
    "Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic",  # Linux
    "DejaVu Sans",                                      # 最終手段（日本語は出ない）
]
matplotlib.rcParams["axes.unicode_minus"] = False


def load():
    out = []
    for d in sorted(EXP.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        f = d / "metrics.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    out.sort(key=lambda m: m["experiment"])
    return out


def main():
    ms = load()
    if not ms:
        sys.exit("[error] experiments/ に metrics.json が1つもありません")

    ids = [m["experiment"] for m in ms]
    x = range(len(ms))

    def series(name):
        return [m.get("results", {}).get("validation", {})
                 .get(name, {}).get("combined_nrmse") for m in ms]

    overall = [(m.get("results", {}).get("overall") or {}).get("combined_nrmse")
               for m in ms]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(ms) + 3), 4.5))
    for ys, lab, style, lw, ms_ in [
            (overall, "Validation", "o-", 1.5, 6),
            (series("jan"), "Jan", "^--", 0.8, 5),
            (series("apr"), "Apr", "s--", 0.8, 5)]:
        xs = [i for i, y in zip(x, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        if vs:
            ax.plot(xs, vs, style, label=lab, linewidth=lw, markersize=ms_)

    ax.set_xticks(list(x))
    ax.set_xticklabels(ids)
    ax.set_ylabel("Combined NRMSE")
    ax.set_title("実験ごとの Combined NRMSE の推移")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"[write] {OUT}  ({len(ms)}実験)")


if __name__ == "__main__":
    main()
