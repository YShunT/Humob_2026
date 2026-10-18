"""1つの成分（対角・総量・配分）を、状態空間モデルとリッジ回帰のEM交互推定で全日について平滑化推定する。

使い方 : from models.state_space_em import fit_component
入力   : なし（目的系列・観測マスク・有効系列・標準化済み特徴を受け取る）
出力   : なし（ComponentFit を返す）
依存   : config.py、dataset/periods.py、models/{kalman_smoother,ridge_regression}.py、numpy
形状   : T = 暦日数、S = 系列数、K = 外部特徴列数
"""


from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import EM_MAX_ITER, EM_TOL, INITIAL_Q_RATIO, MAX_LOG_MEAN, VARIANCE_FLOOR
from dataset.periods import QUAKE_INDEX
from models.kalman_smoother import smooth_local_level
from models.ridge_regression import CalendarRidge


@dataclass(frozen=True)
class ComponentFit:
    """成分1つの全日の平滑化推定値と、EMの収束診断。"""

    mean: np.ndarray  # (T, S)
    diagnostics: dict[str, object]


def initial_level(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """震災前・震災後それぞれの観測平均を、EMの初期水準として系列ごとに置く。"""
    level = np.zeros_like(y)
    for lo, hi in ((0, QUAKE_INDEX), (QUAKE_INDEX, len(y))):
        count = mask[lo:hi].sum(axis=0)
        level[lo:hi] = np.divide(y[lo:hi].sum(axis=0), count, out=np.zeros(y.shape[1]), where=count > 0)
    return level  # (T, S)


def fit_component(name: str, target: np.ndarray, mask: np.ndarray, active: np.ndarray, features: np.ndarray,
                  log_scale: bool, local_features: bool) -> ComponentFit:
    """観測マスク上でEMを最大200回回し、状態＋回帰の平均を元の尺度へ戻して返す（非観測値は参照しない）。"""
    started = time.monotonic()
    y = np.where(mask, np.log1p(target) if log_scale else target, 0.0)
    regression = CalendarRidge(features, mask, local_features)
    level = initial_level(y, mask)
    offset = regression.fit(y - level)
    r = max(float(np.mean((y - level - offset)[mask] ** 2)), VARIANCE_FLOOR)
    q = max(INITIAL_Q_RATIO * r, VARIANCE_FLOOR)
    continuous = np.ones(len(y) - 1, bool)
    continuous[QUAKE_INDEX - 1] = False
    converged, last = False, None
    for iterations in range(1, EM_MAX_ITER + 1):
        state, variance, lag, nll = smooth_local_level(y - offset, mask, q, r)
        objective = nll + 0.5 * regression.penalty_value()
        offset_new = regression.fit(y - state, r)
        r_new = max(float(np.mean(((y - state - offset_new) ** 2 + variance)[mask])), VARIANCE_FLOOR)
        step = np.diff(state, axis=0) ** 2 + variance[1:] + variance[:-1] - 2 * lag[1:]
        q_new = max(float(step[continuous][:, active].mean()), VARIANCE_FLOOR)
        offset, q, r = offset_new, q_new, r_new
        if last is not None and abs(objective - last) <= EM_TOL * (1 + abs(last)):
            converged = True
            break
        last = objective
    state, variance, _, nll = smooth_local_level(y - offset, mask, q, r)
    mean = state + offset
    objective = nll + 0.5 * regression.penalty_value()
    if log_scale:
        if np.max(mean[:, active]) > MAX_LOG_MEAN:
            raise FloatingPointError(f"{name}: log1p 尺度の平均が発散しました")
        mean = np.expm1(mean)
    mean[:, ~active] = 0
    variance[:, ~active] = 0
    diagnostics = {
        "model": name, "active": int(active.sum()), "observations": int(mask.sum()), "q": q, "r": r,
        "iterations": iterations, "converged": converged, "objective": objective,
        "negative_fraction": float(np.mean(mean < 0)),
        "max_state_variance": float(variance[:, active].max()),
        "seconds": time.monotonic() - started,
    }
    return ComponentFit(np.maximum(mean, 0) if log_scale else mean, diagnostics)
