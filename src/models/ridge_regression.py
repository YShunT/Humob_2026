"""系列別の曜日・年末年始効果（対角は外部特徴の偏差も）と全系列共通の外部係数を、L2罰則つきで同時推定する。

使い方 : from models.ridge_regression import CalendarRidge
入力   : なし（標準化済み特徴と観測マスクを受け取る）
出力   : なし
依存   : config.py、dataset/periods.py、numpy、scipy
形状   : T = 暦日数、S = 系列数、K = 外部特徴列数
"""


from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from config import DEVIATION_RIDGE, GLOBAL_RIDGE, WEEKDAY_RIDGE
from dataset.periods import WEEKDAY, YEAR_END


def calendar_regressors() -> np.ndarray:
    """火〜日曜のダミー6列と年末年始（12/29〜1/3）フラグ1列を並べる。"""
    columns = [(WEEKDAY == weekday).astype(float) for weekday in range(1, 7)]
    columns.append(YEAR_END.astype(float))
    return np.stack(columns, axis=-1)  # (T, 7)


def symmetric_inverse(matrix: np.ndarray) -> np.ndarray:
    """対称行列の逆行列を固有値分解で求め、ほぼ0の固有値方向は捨てる（不定値なら例外）。"""
    values, vectors = eigh((matrix + matrix.T) / 2)
    floor = max(float(np.max(np.abs(values))) * 1e-12, 1e-12)
    if values.min() < -floor * 10:
        raise np.linalg.LinAlgError("Indefinite regression normal matrix")
    reciprocals = np.divide(1.0, values, out=np.zeros_like(values), where=values > floor)
    return np.einsum("ik,k,jk->ij", vectors, reciprocals, vectors)  # (K, K)


class CalendarRidge:
    """系列別の局所係数と全系列共通の外部係数を持つ線形回帰を、正規方程式のシューア補完で解く。"""

    def __init__(self, features: np.ndarray, mask: np.ndarray, local_features: bool) -> None:
        """設計行列と、観測マスクで重みづけたグラム行列を前計算する。"""
        calendar = calendar_regressors()
        self.features = features
        self.mask = mask
        self.design = np.broadcast_to(calendar[:, None, :], (*mask.shape, calendar.shape[1]))
        if local_features:
            self.design = np.concatenate([self.design, features], axis=-1)
        self.penalty = np.full(self.design.shape[-1], WEEKDAY_RIDGE)
        self.penalty[calendar.shape[1]:] = DEVIATION_RIDGE
        self.gram = np.einsum("tnk,tnl,tn->nkl", self.design, self.design, mask, optimize=True)
        self.zx = np.einsum("tnk,tnl,tn->nkl", self.design, features, mask, optimize=True)
        self.xx = np.einsum("tnk,tnl,tn->kl", features, features, mask, optimize=True)
        self.local = np.zeros((mask.shape[1], self.design.shape[-1]))
        self.beta = np.zeros(features.shape[-1])

    def fit(self, target: np.ndarray, noise_variance: float = 1.0) -> np.ndarray:
        """観測負対数尤度と同じ尺度の罰則で係数を解き直し、全日・全系列の回帰値を返す。"""
        inverse = np.linalg.inv(self.gram + noise_variance * np.diag(self.penalty)[None])
        izx = np.einsum("nkl,nlj->nkj", inverse, self.zx)
        masked_target = np.where(self.mask, target, 0)
        rhs = np.einsum("tnk,tn->nk", self.design, masked_target, optimize=True)
        irhs = np.einsum("nkl,nl->nk", inverse, rhs)
        schur = self.xx - np.einsum("nki,nkj->ij", self.zx, izx)
        global_inverse = symmetric_inverse(schur + np.eye(self.features.shape[-1]) * noise_variance * GLOBAL_RIDGE)
        xy = np.einsum("tnk,tn->k", self.features, masked_target, optimize=True)
        global_rhs = xy - np.einsum("nki,nk->i", self.zx, irhs)
        self.beta = np.einsum("ij,j->i", global_inverse, global_rhs)
        self.local = irhs - np.einsum("nki,i->nk", izx, self.beta)
        return (np.einsum("tnk,nk->tn", self.design, self.local, optimize=True)
                + np.einsum("tnk,k->tn", self.features, self.beta, optimize=True))  # (T, S)

    def penalty_value(self) -> float:
        """現在の係数に対するL2罰則の値を返す。"""
        return float(np.sum(self.local ** 2 * self.penalty) + GLOBAL_RIDGE * np.sum(self.beta ** 2))
