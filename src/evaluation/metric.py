"""公式 Combined NRMSE（日ごとに対角・非対角のRMSE→期間平均→固定RMSで正規化して平均）を計算する（正規化定数の正本）。

使い方 : from evaluation.metric import score_periods
入力   : なし（日付→予測OD・正解ODの辞書を受け取る）
出力   : なし
依存   : dataset/{grid,periods}.py、utils/od_io.py
"""

from __future__ import annotations

from dataset.grid import EVAL_X, EVAL_Y, in_eval_box
from dataset.periods import CV_WINDOW_DAYS
from utils.od_io import OD

N_CELL = (EVAL_X[1] - EVAL_X[0] + 1) * (EVAL_Y[1] - EVAL_Y[0] + 1)
N_DIAG = N_CELL
N_OFF = N_CELL * N_CELL - N_CELL
NORM_DIAG = 26.57
NORM_OFF = 0.0176


def day_errors(pred: OD, truth: OD) -> dict[str, float]:
    """1日分の対角・非対角の二乗誤差和を求め、未記録ペアを0とした固定母数でRMSEにする。"""
    se_diag = 0.0
    se_off = 0.0
    for origin, truth_dests in truth.items():
        if not in_eval_box(origin):
            continue
        pred_dests = pred.get(origin) or {}
        for dest, actual in truth_dests.items():
            if not in_eval_box(dest):
                continue
            predicted = pred_dests.get(dest)
            error2 = actual * actual if predicted is None else (predicted - actual) ** 2
            if origin == dest:
                se_diag += error2
            else:
                se_off += error2
    for origin, pred_dests in pred.items():
        if not in_eval_box(origin):
            continue
        truth_dests = truth.get(origin) or {}
        for dest, predicted in pred_dests.items():
            if not in_eval_box(dest) or dest in truth_dests:
                continue
            if origin == dest:
                se_diag += predicted * predicted
            else:
                se_off += predicted * predicted
    return {"rmse_diag": (se_diag / N_DIAG) ** 0.5, "rmse_off": (se_off / N_OFF) ** 0.5}


def summarize(day_results: list[dict[str, float]]) -> dict[str, float]:
    """日次RMSEを期間平均・正規化し、metrics.json の共通キー（n_days・combined_nrmse など）にまとめる。"""
    n_days = len(day_results)
    nrmse_diag = sum(item["rmse_diag"] for item in day_results) / n_days / NORM_DIAG
    nrmse_off = sum(item["rmse_off"] for item in day_results) / n_days / NORM_OFF
    return {"n_days": n_days, "combined_nrmse": (nrmse_diag + nrmse_off) / 2,
            "nrmse_diag": nrmse_diag, "nrmse_offdiag": nrmse_off}


def score_periods(predictions: dict[int, OD], truths: dict[int, OD]) -> dict[str, dict[str, float]]:
    """正解のある全日（overall）と、CV窓（jan・apr）ごとの公式スコアを返す。"""
    daily = {day: day_errors(predictions[day], truths[day]) for day in sorted(truths)}
    periods = {"overall": summarize(list(daily.values()))}
    periods.update({name: summarize([daily[day] for day in days if day in daily])
                    for name, days in CV_WINDOW_DAYS.items()})
    return periods
