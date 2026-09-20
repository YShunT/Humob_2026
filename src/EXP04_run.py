"""EXP04: 出発側・到着側の非対角予測を比較し、提出物を生成する。

実行: python3 src/EXP04_run.py
入力: EXP03のmetrics.json、配布OD、data/processed/の外部特徴CSV
出力: experiments/EXP04/{metrics.json,.log,outputs/,figures/}
依存: EXP03_run.py、common/*.py、numpy、scipy、pandas、matplotlib
"""

from __future__ import annotations

import datetime as dt
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys

import numpy as np

import EXP03_run as base
from common.competition_metric import day_errors, score
from common.experiment_io import (
    check_od_tsv, write_metrics, write_od_tsv, write_records_csv,
)
from common.local_error_map import save_local_error_map
from common.municipality_error import municipality_error_rows, write_municipality_errors
from common.pred_dist import save_pred_dist
from common.run_log import run_log

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/EXP04"
OUT = EXP / "outputs"
FIG = EXP / "figures"
REFERENCE = ROOT / "experiments/EXP03/metrics.json"
DIAG_GROUPS = ("evac", "water", "rain")
OFF_GROUPS = ("evac", "water", "rain", "snow", "shelter")
CANDIDATES = {"N-F": 0.0, "N-R": 1.0, "N-B": 0.5}
# 共通モジュールは採点・描画だけを再利用。予測はEXP03の関数を呼ぶ。
# base.OUT等は変更せず、EXP03の保存関数を呼ばない。


def specification():
    return dict(diag_groups=list(DIAG_GROUPS), off_groups=list(OFF_GROUPS),
                state_settings=base.SETTINGS, candidates=CANDIDATES,
                reverse_features="mirrored origin/destination roles",
                catboost=False, fixed=False)


def check_reference():
    ref = json.loads(REFERENCE.read_text())
    config = ref["config"]
    for kind, groups in (("diag", DIAG_GROUPS), ("off", OFF_GROUPS)):
        selected = config["selected"][kind]
        if tuple(selected["groups"]) != groups or selected["hybrid"] or selected["fixed"]:
            raise ValueError("EXP03 selected specification differs; review EXP04 strategy")
    if config["settings"] != base.SETTINGS:
        raise ValueError("EXP03 settings changed")
    return ref


class IncomingDataset(base.Dataset):
    """列合計で再分解する読み取り用ビュー。配列の列pは元の有向ODを維持する。"""

    def __init__(self, source):
        self.__dict__ = source.__dict__.copy()
        self.pairs = [(d, o) for o, d in source.pairs]
        self.pi = {pair: p for p, pair in enumerate(self.pairs)}
        self.origin = source.dest.copy()
        self.dest = source.origin.copy()
        # flow[:, p]は元y[o,d]。y[d,o]を参照したり対称化したりしない。
        self.total = np.zeros_like(source.total)
        for p, destination in enumerate(self.origin):
            self.total[:, destination] += self.flow[:, p]
        den = self.total[:, self.origin]
        self.ratio = np.divide(self.flow, den, out=np.zeros_like(self.flow), where=den > 0)


def reconstruct(data, times, diagonal, flow, outside=False):
    """元の有向OD順序で復元。全域総量・行合計・列合計への追加補正はしない。"""
    if flow.shape != (len(times), len(data.pairs)):
        raise ValueError("Unexpected flow shape")
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Invalid flow")
    result = data.reconstruct(times, diagonal, np.zeros_like(diagonal),
                              np.zeros_like(flow), outside=outside)
    for k, t in enumerate(times):
        od = result[int(base.DAYS[t])]
        for p, (origin, destination) in enumerate(data.pairs):
            if flow[k, p] > 0:
                od.setdefault(origin, {})[destination] = float(flow[k, p])
    return result


def fit(data, times, log, phase):
    fits = []
    def estimate(view, kind, groups, model, stage):
        log.stage(f"{phase}-{stage}", f"{model}: 状態空間モデルを推定")
        print(f"=== {model} / {kind} / {','.join(groups)} ===", flush=True)
        prediction = base.fit_component(view, kind, groups, False, data.observed)
        fits.append(dict(model=model, **prediction.diagnostics))
        print(prediction.diagnostics, flush=True)
        return prediction.mean[times]
    diagonal = estimate(data, "D", DIAG_GROUPS, "stage0-D5", "D5")
    flows = {}
    for name, view in (("F", data), ("R", IncomingDataset(data))):
        number = "1" if name == "F" else "2"
        total = estimate(view, "O", OFF_GROUPS,
                         f"stage{number}-{name}-total", f"{name}-total")
        share = estimate(view, "r", OFF_GROUPS,
                         f"stage{number}-{name}-share", f"{name}-share")
        share, _ = base.normalize(view, share, data.observed)
        flows[name] = total[:, view.origin] * share
    return diagonal, flows, fits


def standard_score(rows):
    result = score(rows)
    return dict(n_days=len(rows), combined_nrmse=result["combined"],
                nrmse_diag=result["nrmse_diag"], nrmse_offdiag=result["nrmse_off"])


def evaluate(predictions, truths):
    daily = {d: day_errors(predictions[d], truths[d]) for d in sorted(truths)}
    periods = {"overall": list(daily),
               "jan": [d for d in daily if str(d).startswith("202401")],
               "apr": [d for d in daily if str(d).startswith("202404")]}
    values = {k: standard_score([daily[d] for d in days]) for k, days in periods.items()}
    se = sum(r["se_off"] for r in daily.values())
    values["error_breakdown"] = {
        kind + "_sse": sum(r[kind + "_o"] for r in daily.values())
        for kind in ("hit", "miss", "extra")
    }
    values["error_breakdown"].update({
        kind + "_pct": 100 * values["error_breakdown"][kind + "_sse"] / se if se else 0.
        for kind in ("hit", "miss", "extra")
    })
    return values, daily


def select(results):
    winner = "N-F"
    for name in ("N-R", "N-B"):
        if results[name]["overall"]["combined_nrmse"] < results[winner]["overall"]["combined_nrmse"] - 1e-12:
            winner = name
    return winner


def candidate_plot(results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("combined_nrmse", "nrmse_offdiag"),
                              ("Combined NRMSE", "非対角 NRMSE")):
        for period, label in (("overall", "14日全体"), ("jan", "1月7日"), ("apr", "4月7日")):
            ax.plot(list(CANDIDATES), [results[n][period][key] for n in CANDIDATES],
                    marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("N-F: 出発側 / N-R: 到着側 / N-B: 1:1結合")
        ax.set_ylabel("NRMSE")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("EXP04 出発側・到着側の比較")
    fig.tight_layout()
    fig.savefig(FIG / "direction_comparison.png", dpi=150)
    plt.close(fig)


def validate(log=None):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    context = nullcontext(log) if log else run_log(
        "EXP04", "run:validate", "python3 src/EXP04_run.py"
    )
    with context as log:
        log.stage("validate-load", "参照設定・入力確認、CV14日を除外")
        ref = check_reference()
        data = base.Dataset(True)
        times = np.flatnonzero(np.isin(base.DAYS, list(base.CV)))
        assert not data.observed[times].any()
        diagonal, flows, fits = fit(data, times, log, "validate")
        log.stage("validate-score", "順方向・逆方向・固定1:1結合の公式評価")
        results, municipalities = {}, []
        for name, weight in CANDIDATES.items():
            flow = (1-weight)*flows["F"] + weight*flows["R"]
            prediction = reconstruct(data, times, diagonal, flow)
            result, _ = evaluate(prediction, data.truth)
            results[name] = result
            municipalities.extend(dict(candidate=name, **row) for row in
                                   municipality_error_rows(prediction, data.truth, "EXP04"))
            print(f"{name}: {result['overall']}", flush=True)
        # 同じ入力・パラメータでEXP03を再現できたことを採用前に必ず検査する。
        # metrics.json は小数6桁が正本なので、同じ精度へ丸めて再現性を判定する。
        difference = {k: round(results["N-F"]["overall"][k], 6) - ref["results"]["overall"][k]
                      for k in ("combined_nrmse", "nrmse_diag", "nrmse_offdiag")}
        if max(abs(v) for v in difference.values()) > 1e-12:
            raise ValueError(f"EXP03 baseline reproduction failed: {difference}")
        selected = select(results)
        weight = CANDIDATES[selected]
        predictions = reconstruct(data, times, diagonal,
                                  (1-weight)*flows["F"] + weight*flows["R"], outside=True)
        log.stage("validate-artifacts", "採用候補の図・地域誤差・metricsを保存")
        write_records_csv(OUT / "candidate_municipality_errors.csv", municipalities)
        # 方向間の差と誤差相関。未観測候補外の真値は採点側のmissに含まれる。
        truth = np.array([[data.truth[int(base.DAYS[t])].get(o, {}).get(d, 0.)
                           for o, d in data.pairs] for t in times])
        ef, er = (flows["F"]-truth).ravel(), (flows["R"]-truth).ravel()
        direction = dict(error_correlation=float(np.corrcoef(ef, er)[0, 1]),
                         mean_abs_prediction_difference=float(np.abs(flows["F"]-flows["R"]).mean()),
                         candidate_pair_days=int(truth.size))
        compact_candidates = {
            name: dict(
                combined_nrmse=value["overall"]["combined_nrmse"],
                nrmse_offdiag=value["overall"]["nrmse_offdiag"],
                jan_combined_nrmse=value["jan"]["combined_nrmse"],
                apr_combined_nrmse=value["apr"]["combined_nrmse"],
                **value["error_breakdown"],
            )
            for name, value in results.items()
        }
        metadata = dict(experiment="EXP04", date=dt.datetime.now().astimezone().isoformat(timespec="minutes"),
                        compared_to="EXP03", status="評価済み",
                        config=dict(**specification(), selected=selected,
                                    observed_days=int(data.observed.sum()), candidate_pairs=len(data.pairs)),
                        results=dict(overall=results[selected]["overall"],
                                     validation={p: results[selected][p] for p in ("jan", "apr")},
                                     candidates=compact_candidates,
                                     baseline_reproduction_difference=difference,
                                     direction_diagnostics=direction, fit_diagnostics=fits))
        write_metrics(EXP / "metrics.json", metadata)
        write_municipality_errors(predictions, data.truth, "EXP04")
        save_local_error_map(predictions, data.truth, FIG / "local_error_map.png", experiment="EXP04")
        candidate_plot(results)
        log.result(selected=selected, **results[selected]["overall"])
        log.note("→ experiments/EXP04/metrics.json")
        log.note("→ experiments/EXP04/outputs/candidate_municipality_errors.csv")
        log.note("→ experiments/EXP04/figures/")
    return predictions


def submit(validation_predictions, log=None):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    context = nullcontext(log) if log else run_log(
        "EXP04", "run:submit", "python3 src/EXP04_run.py"
    )
    with context as log:
        log.stage("submit-check", "検証時の採用設定を確認")
        metadata = json.loads((EXP / "metrics.json").read_text())
        for key, value in specification().items():
            if metadata["config"][key] != value:
                raise ValueError("Specification changed; rerun EXP04")
        check_reference()
        data = base.Dataset(False)
        times = np.flatnonzero(np.isin(base.DAYS, base.TEST))
        # テスト日は観測に入れない。CV日は提出用再学習に含める。
        assert not data.observed[times].any()
        assert data.observed[np.isin(base.DAYS, list(base.CV))].all()
        diagonal, flows, _ = fit(data, times, log, "submit")
        weight = CANDIDATES[metadata["config"]["selected"]]
        predictions = reconstruct(data, times, diagonal,
                                  (1-weight)*flows["F"] + weight*flows["R"], outside=True)
        log.stage("submit-write", "提出58日復元・形式検査・日次推移")
        path = OUT / "submission.tsv"
        write_od_tsv(predictions, path)
        check_od_tsv(path, base.TEST)
        official_validator = ROOT / "src/common/humob2026_validator.py"
        validator_status = "Internal validation passed"
        if official_validator.exists():
            checked = subprocess.run([sys.executable, str(official_validator), str(path)],
                                     check=True, capture_output=True, text=True)
            validator_status = checked.stdout.strip()
            print(checked.stdout, flush=True)
        else:
            print("[check] official validator is not bundled; internal validation passed",
                  flush=True)
        save_pred_dist("EXP04", validation_predictions=validation_predictions)
        log.result(selected=metadata["config"]["selected"], days=len(predictions),
                   validator=validator_status)
        log.note("→ experiments/EXP04/outputs/submission.tsv")
    return predictions


def self_test():
    from types import SimpleNamespace
    source = SimpleNamespace(
        pairs=[("a", "b"), ("c", "b"), ("b", "c")],
        origin=np.array([0, 2, 1]), dest=np.array([1, 1, 2]),
        total=np.array([[2., 3., 4.], [0., 0., 0.]]),
        flow=np.array([[2., 4., 3.], [0., 0., 0.]]))
    view = IncomingDataset(source)
    assert view.pairs == [("b", "a"), ("b", "c"), ("c", "b")]
    assert np.array_equal(view.total, [[0., 6., 3.], [0., 0., 0.]])
    assert np.allclose(view.total[:, view.origin]*view.ratio, source.flow)
    assert np.array_equal(source.total, [[2., 3., 4.], [0., 0., 0.]])
    twice = IncomingDataset(view)
    assert twice.pairs == source.pairs
    assert np.array_equal(twice.total, source.total)
    view.cells = ["a", "b", "c"]
    normalized, _ = base.normalize(view, np.zeros_like(view.ratio), np.array([True, False]))
    assert np.allclose(normalized[1], [1/3, 2/3, 1])
    print("[self-test] PASS asymmetric transpose / zero inflow / fallback / source preservation")


def run():
    """EXP04を自己検査・比較・採用し、提出物まで一度に再生成する。"""
    with run_log("EXP04", "run", "python3 src/EXP04_run.py") as log:
        log.stage("self-test", "数値実装の不変条件を検査")
        self_test()
        validation_predictions = validate(log)
        submit(validation_predictions, log)


if __name__ == "__main__":
    run()
