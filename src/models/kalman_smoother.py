"""系列ごとに独立なローカルレベル模型を、カルマンフィルター＋RTSスムーザーで平滑化する（震災日で状態をリセット）。

使い方 : from models.kalman_smoother import smooth_local_level
入力   : なし（残差系列と観測マスクを受け取る）
出力   : なし
依存   : config.py、dataset/periods.py、numpy
形状   : T = 暦日数、S = 系列数
"""


from __future__ import annotations

import numpy as np

from config import INITIAL_VARIANCE
from dataset.periods import QUAKE_INDEX


def smooth_local_level(y: np.ndarray, mask: np.ndarray, q: float, r: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """状態の平滑化平均・分散・1期ラグ共分散と、観測の負対数尤度を返す（欠損日は観測更新しない）。"""
    n_days, n_series = y.shape
    filtered_mean, filtered_var, predicted_var = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    m, p = np.zeros(n_series), np.full(n_series, INITIAL_VARIANCE)
    nll = 0.0
    for t in range(n_days):
        if t == QUAKE_INDEX:
            m, p = np.zeros(n_series), np.full(n_series, INITIAL_VARIANCE)
        if t != 0 and t != QUAKE_INDEX:
            p = p + q
        predicted_var[t] = p
        innovation = np.where(mask[t], y[t] - m, 0.0)
        f = p + r
        nll += 0.5 * np.sum(np.where(mask[t], np.log(2 * np.pi * f) + innovation * innovation / f, 0.0))
        gain = np.where(mask[t], p / f, 0.0)
        m = m + gain * innovation
        p = (1 - gain) * p
        filtered_mean[t], filtered_var[t] = m, p
    mean, var = filtered_mean.copy(), filtered_var.copy()
    lag = np.zeros_like(y)
    for t in range(n_days - 2, -1, -1):
        if t + 1 == QUAKE_INDEX:
            continue
        j = filtered_var[t] / predicted_var[t + 1]
        mean[t] += j * (mean[t + 1] - filtered_mean[t])
        var[t] = np.maximum(0, filtered_var[t] + j * j * (var[t + 1] - predicted_var[t + 1]))
        lag[t + 1] = j * var[t + 1]
    return mean, var, lag, float(nll)  # (T, S), (T, S), (T, S), ()
