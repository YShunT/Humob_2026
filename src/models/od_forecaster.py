"""最終モデル（EXP04 N-B）：対角D5と、出発側・到着側それぞれの総量×配分を1:1で平均した非対角からODを予測する。

使い方 : from models.od_forecaster import predict, to_od_dicts
入力   : なし（ODMatrix と ExternalFeatures を受け取る）
出力   : なし（Forecast と日付→OD辞書を返す）
依存   : config.py、dataset/{external_features,od_matrix,periods}.py、models/state_space_em.py、
         utils/{od_io,run_log}.py、numpy
形状   : T = 暦日数、C = 評価セル数、P = 候補OD対数、N = 予測対象日数（CV 14 / テスト 58）
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DIAGONAL_GROUPS, INCOMING_WEIGHT, OFFDIAGONAL_GROUPS
from dataset.external_features import ExternalFeatures, cell_features, pair_features, standardize
from dataset.od_matrix import FlowView, ODMatrix
from dataset.periods import DAYS, WEEKDAY
from models.state_space_em import ComponentFit, fit_component
from utils.od_io import OD
from utils.run_log import RunLog


@dataclass(frozen=True)
class Forecast:
    """予測対象日の対角・非対角の予測値と、5成分の収束診断。"""

    target_index: np.ndarray  # (N,)
    diagonal: np.ndarray  # (N, C)
    offdiagonal: np.ndarray  # (N, P)
    diagnostics: list[dict[str, object]]


def fit_diagonal(od: ODMatrix, external: ExternalFeatures, train: np.ndarray) -> ComponentFit:
    """対角（セル内滞留）を log1p 尺度で、外部特徴の局所偏差つきで推定する。"""
    active = (od.diag[train] > 0).any(axis=0)
    mask = train[:, None] & active[None]
    features = standardize(cell_features(external, DIAGONAL_GROUPS), mask)
    return fit_component("D5", od.diag, mask, active, features, log_scale=True, local_features=True)


def fit_total(view: FlowView, raw_features: np.ndarray, train: np.ndarray, name: str) -> ComponentFit:
    """anchor 側セルの非対角総量を log1p 尺度で推定する。"""
    active = (view.total[train] > 0).any(axis=0)
    mask = train[:, None] & active[None]
    return fit_component(name, view.total, mask, active, standardize(raw_features, mask),
                         log_scale=True, local_features=False)


def fit_share(view: FlowView, external: ExternalFeatures, train: np.ndarray, support: np.ndarray,
              name: str) -> ComponentFit:
    """anchor 側総量が正の日だけを観測として、各ODの配分比を元の尺度で推定する。"""
    mask = train[:, None] & (view.total[:, view.anchor] > 0) & support[None]
    features = standardize(pair_features(external, OFFDIAGONAL_GROUPS, view), mask)
    return fit_component(name, view.ratio, mask, support, features, log_scale=False, local_features=False)


def normalize_share(view: FlowView, share: np.ndarray, train: np.ndarray, support: np.ndarray) -> np.ndarray:
    """配分比を非負化して anchor ごとに合計1へ正規化し、合計0の日は学習期間の流量比率で埋める。"""
    positive = np.maximum(share, 0) * support[None]
    sums = np.zeros((len(share), view.total.shape[1]))
    for p, cell in enumerate(view.anchor):
        sums[:, cell] += positive[:, p]
    anchor_sums = sums[:, view.anchor]
    flow_sum = view.flow[train].sum(axis=0)
    total_sum = view.total[train].sum(axis=0)[view.anchor]
    fallback = np.divide(flow_sum, total_sum, out=np.zeros_like(flow_sum), where=total_sum > 0)
    normalized = np.divide(positive, anchor_sums, out=np.zeros_like(positive), where=anchor_sums > 0)
    return np.where(anchor_sums > 0, normalized, fallback[None])  # (N, P)


def directional_flow(view: FlowView, external: ExternalFeatures, raw_cell_features: np.ndarray, train: np.ndarray,
                     target_index: np.ndarray, label: str, log: RunLog) -> tuple[np.ndarray, list[dict[str, object]]]:
    """1方向の分解（総量×配分）で予測対象日の非対角ODを推定し、2成分の診断とともに返す。"""
    log.stage(f"fit-{label}-total", f"{label}側の非対角総量を状態空間モデルで推定")
    total = fit_total(view, raw_cell_features, train, f"{label}-total")
    log.stage(f"fit-{label}-share", f"{label}側の配分比を状態空間モデルで推定")
    support = view.flow[train].sum(axis=0) > 0
    share = fit_share(view, external, train, support, f"{label}-share")
    share_target = normalize_share(view, share.mean[target_index], train, support)
    flow = total.mean[target_index][:, view.anchor] * share_target
    return flow, [total.diagnostics, share.diagnostics]  # (N, P), list


def predict(od: ODMatrix, external: ExternalFeatures, target_days: list[int], log: RunLog) -> Forecast:
    """観測日すべてで5成分を学習し、target_days の対角と非対角（出発側・到着側の1:1平均）を予測する。"""
    target_index = np.flatnonzero(np.isin(DAYS, target_days))
    if od.observed[target_index].any():
        raise ValueError("予測対象日が学習データに含まれています")
    train = od.observed
    log.stage("fit-D5", "対角を状態空間モデルで推定（避難者率・断水率・降水量）")
    diagonal = fit_diagonal(od, external, train)
    raw_cell_features = cell_features(external, OFFDIAGONAL_GROUPS)
    outgoing, outgoing_diagnostics = directional_flow(od.outgoing, external, raw_cell_features, train,
                                                      target_index, "F", log)
    incoming, incoming_diagnostics = directional_flow(od.incoming, external, raw_cell_features, train,
                                                      target_index, "R", log)
    offdiagonal = (1 - INCOMING_WEIGHT) * outgoing + INCOMING_WEIGHT * incoming
    diagonal_target = diagonal.mean[target_index]
    if not (np.isfinite(diagonal_target).all() and np.isfinite(offdiagonal).all()):
        raise FloatingPointError("予測値に有限でない値があります")
    return Forecast(target_index, diagonal_target, offdiagonal,
                           [diagonal.diagnostics, *outgoing_diagnostics, *incoming_diagnostics])


def to_od_dicts(od: ODMatrix, forecast: Forecast) -> dict[int, OD]:
    """予測配列を日付→{origin: {destination: value}} に戻し、評価外ODは学習期間の曜日平均で埋める。"""
    result = {}
    for row, t in enumerate(forecast.target_index):
        template = od.outside_mean[WEEKDAY[t]]
        if not template:
            raise ValueError(f"{DAYS[t]}: 評価外ODの曜日平均がありません")
        day_od: OD = {}
        for (origin, destination), value in template.items():
            day_od.setdefault(origin, {})[destination] = value
        for i in np.flatnonzero(forecast.diagonal[row] > 0):
            cell = str(od.cells[i])
            day_od.setdefault(cell, {})[cell] = float(forecast.diagonal[row, i])
        for p in np.flatnonzero(forecast.offdiagonal[row] > 0):
            origin, destination = od.pairs[p]
            day_od.setdefault(origin, {})[destination] = float(forecast.offdiagonal[row, p])
        result[int(DAYS[t])] = day_od
    return result
