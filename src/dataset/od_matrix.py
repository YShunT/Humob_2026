"""配布ODを1回読み、評価セルの対角・候補非対角の日次配列、出発側/到着側の分解、評価外ODの曜日平均を作る。

使い方 : from dataset.od_matrix import load_od_matrix
入力   : data/raw/humob2026-dataset.tsv
出力   : なし（ODMatrix を返す）
依存   : config.py、dataset/{grid,periods}.py、utils/od_io.py、numpy
形状   : T = 暦日数（366）、C = 評価セル数（1,476）、P = 候補OD対数（学習観測で正の値をとった有向非対角OD）
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RAW_OD_TSV
from dataset.grid import eval_cell_ids, in_eval_box
from dataset.periods import DAY_INDEX, DAYS, QUAKE_DAY, WEEKDAY
from utils.od_io import OD, iter_od_days


@dataclass(frozen=True)
class FlowView:
    """非対角流量を、集約側セル（anchor）の総量と相手側セル（partner）への配分比に分けた表現。"""

    flow: np.ndarray  # (T, P)
    anchor: np.ndarray  # (P,)
    partner: np.ndarray  # (P,)
    total: np.ndarray  # (T, C)
    ratio: np.ndarray  # (T, P)


@dataclass(frozen=True)
class ODMatrix:
    """学習に使う観測ODの配列と、採点のために学習から隠した正解OD。"""

    cells: np.ndarray  # (C,)
    pairs: list[tuple[str, str]]
    observed: np.ndarray  # (T,)
    diag: np.ndarray  # (T, C)
    outgoing: FlowView
    incoming: FlowView
    outside_mean: list[dict[tuple[str, str], float]]
    truth: dict[int, OD]


def make_flow_view(flow: np.ndarray, anchor: np.ndarray, partner: np.ndarray, n_cells: int) -> FlowView:
    """anchor 側セルごとに流量を合計し、各ODの配分比 = 流量 / anchor 側総量 を求める。"""
    total = np.zeros((flow.shape[0], n_cells))
    for p, cell in enumerate(anchor):
        total[:, cell] += flow[:, p]
    denominator = total[:, anchor]
    ratio = np.divide(flow, denominator, out=np.zeros_like(flow), where=denominator > 0)
    return FlowView(flow, anchor, partner, total, ratio)


def load_od_matrix(holdout_days: set[int]) -> ODMatrix:
    """配布ODを読み、holdout_days を正解として隠したうえで学習用の配列を組み立てる。"""
    cells = eval_cell_ids()
    index = {cell: i for i, cell in enumerate(cells)}
    observed = np.zeros(len(DAYS), bool)
    inside_by_t: dict[int, OD] = {}
    truth: dict[int, OD] = {}
    outside_sum: list[dict[tuple[str, str], float]] = [{} for _ in range(7)]
    outside_days = np.zeros(7)
    for day, full in iter_od_days(RAW_OD_TSV):
        inside = {o: {d: v for d, v in dests.items() if in_eval_box(d)}
                  for o, dests in full.items() if in_eval_box(o)}
        if day in holdout_days:
            truth[day] = inside
            continue
        t = DAY_INDEX[day]
        observed[t] = True
        inside_by_t[t] = inside
        if day >= QUAKE_DAY:
            weekday = WEEKDAY[t]
            outside_days[weekday] += 1
            for o, dests in full.items():
                for d, v in dests.items():
                    if not (in_eval_box(o) and in_eval_box(d)):
                        outside_sum[weekday][(o, d)] = outside_sum[weekday].get((o, d), 0.0) + v

    pairs = sorted({(o, d) for od in inside_by_t.values() for o, dests in od.items()
                    for d, v in dests.items() if o != d and v > 0})
    pair_index = {pair: p for p, pair in enumerate(pairs)}
    diag = np.zeros((len(DAYS), len(cells)))
    flow = np.zeros((len(DAYS), len(pairs)))
    for t, od in inside_by_t.items():
        for o, dests in od.items():
            for d, v in dests.items():
                if o == d:
                    diag[t, index[o]] = v
                elif (o, d) in pair_index:
                    flow[t, pair_index[o, d]] = v

    origin = np.array([index[o] for o, _ in pairs], dtype=int)
    dest = np.array([index[d] for _, d in pairs], dtype=int)
    outside_mean = [{key: float(value / outside_days[w]) for key, value in outside_sum[w].items()}
                    if outside_days[w] else {} for w in range(7)]
    return ODMatrix(cells, pairs, observed, diag,
                  make_flow_view(flow, origin, dest, len(cells)),
                  make_flow_view(flow, dest, origin, len(cells)),
                  outside_mean, truth)
