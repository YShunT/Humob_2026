"""HuMob グリッドの評価境界を定義し、セルIDの解釈・評価境界内判定・評価セル一覧を提供する（評価境界の正本）。

使い方 : from dataset.grid import EVAL_X, EVAL_Y, in_eval_box, eval_cell_ids
入力   : なし
出力   : なし
依存   : numpy
形状   : C = 評価セル数（x 30〜70 × y 35〜70 の1,476）
"""

from __future__ import annotations

import numpy as np

NX, NY = 100, 70
OUTSIDE_ID = "-1_-1"
EVAL_X = (30, 70)
EVAL_Y = (35, 70)


def parse_cell_id(cell_id: str) -> tuple[int, int] | None:
    """y_x 形式のセルIDを (y, x) に変換し、域外・不正IDなら None を返す。"""
    if cell_id == OUTSIDE_ID:
        return None
    parts = cell_id.split("_")
    if len(parts) != 2:
        return None
    try:
        y, x = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return (y, x) if 1 <= x <= NX and 1 <= y <= NY else None


def in_eval_box(cell_id: str) -> bool:
    """セルIDが公式評価境界の内側にあるかを返す。"""
    yx = parse_cell_id(cell_id)
    return yx is not None and EVAL_X[0] <= yx[1] <= EVAL_X[1] and EVAL_Y[0] <= yx[0] <= EVAL_Y[1]


def eval_cell_ids() -> np.ndarray:
    """評価境界内のセルIDを y 優先・x 昇順で返す。"""
    return np.array([f"{y}_{x}" for y in range(EVAL_Y[0], EVAL_Y[1] + 1)
                     for x in range(EVAL_X[0], EVAL_X[1] + 1)])  # (C,)
