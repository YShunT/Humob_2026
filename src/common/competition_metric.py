"""HuMob Challenge 2026 の公式評価指標を共通化したモジュール。

一般的な ``RMSE / mean(y)`` ではなく、次の公式手順を実装する。

1. 評価境界内の全ODペアを母数とし、未記録ペアは0とする。
2. 日ごとに diagonal / off-diagonal のRMSEを別々に計算する。
3. 日次RMSEを期間内で平均する。
4. 主催者提供の固定RMS（26.57 / 0.0176）で正規化して単純平均する。

各実験は ``day_errors`` と ``score``、またはまとめて
``calculate_competition_nrmse`` を呼ぶ。
"""


NX, NY = 100, 70
OOB = "-1_-1"

EVAL_X = (30, 70)
EVAL_Y = (35, 70)
N_CELL = (EVAL_X[1] - EVAL_X[0] + 1) * (EVAL_Y[1] - EVAL_Y[0] + 1)
N_DIAG = N_CELL
N_OFF = N_CELL * N_CELL - N_CELL

NORM_DIAG = 26.57
NORM_OFF = 0.0176


def parse_gid(gid):
    """通常グリッドIDを ``(y, x)`` で返す。域外・不正IDは ``None``。"""
    if gid == OOB:
        return None
    parts = gid.split("_")
    if len(parts) != 2:
        return None
    try:
        y, x = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return (y, x) if 1 <= x <= NX and 1 <= y <= NY else None


def valid_gid(gid):
    """提出形式として有効なIDか。特殊ID ``-1_-1`` も許可する。"""
    return gid == OOB or parse_gid(gid) is not None


def in_eval_box(gid):
    """公式評価境界内の通常グリッドIDか。``-1_-1`` は常にFalse。"""
    yx = parse_gid(gid)
    if yx is None:
        return False
    y, x = yx
    return EVAL_X[0] <= x <= EVAL_X[1] and EVAL_Y[0] <= y <= EVAL_Y[1]


def day_errors(pred, truth):
    """1日分の公式RMSEと誤差内訳を返す。

    ``pred`` / ``truth`` は ``{origin: {destination: value}}``。評価するのは
    origin・destinationの両方が評価境界内のペアだけ。疎な辞書に存在しない
    ペアは0だが、母数は常に ``N_DIAG`` / ``N_OFF`` で固定する。
    """
    result = {
        key: 0.0
        for key in (
            "se_diag", "se_off",
            "hit_d", "miss_d", "extra_d",
            "hit_o", "miss_o", "extra_o",
        )
    }
    counts = {
        key: 0
        for key in ("n_diag_obs", "n_off_obs", "n_diag_pred", "n_off_pred")
    }

    for origin, truth_dests in truth.items():
        if not in_eval_box(origin):
            continue
        pred_dests = pred.get(origin)
        for dest, actual in truth_dests.items():
            if not in_eval_box(dest):
                continue
            diag = origin == dest
            counts["n_diag_obs" if diag else "n_off_obs"] += 1
            predicted = pred_dests.get(dest) if pred_dests else None
            if predicted is None:
                error2, kind = actual * actual, "miss"
            else:
                error2, kind = (predicted - actual) ** 2, "hit"
            suffix = "d" if diag else "o"
            result["se_diag" if diag else "se_off"] += error2
            result[f"{kind}_{suffix}"] += error2

    for origin, pred_dests in pred.items():
        if not in_eval_box(origin):
            continue
        truth_dests = truth.get(origin)
        for dest, predicted in pred_dests.items():
            if not in_eval_box(dest):
                continue
            diag = origin == dest
            counts["n_diag_pred" if diag else "n_off_pred"] += 1
            if truth_dests is not None and dest in truth_dests:
                continue
            error2 = predicted * predicted
            suffix = "d" if diag else "o"
            result["se_diag" if diag else "se_off"] += error2
            result[f"extra_{suffix}"] += error2

    result.update(counts)
    result["rmse_diag"] = (result["se_diag"] / N_DIAG) ** 0.5
    result["rmse_off"] = (result["se_off"] / N_OFF) ** 0.5
    return result


def score(day_results):
    """日次RMSEを平均し、公式のCombined NRMSEを返す。"""
    if not day_results:
        raise ValueError("day_results must contain at least one day")
    n_days = len(day_results)
    rmse_diag = sum(item["rmse_diag"] for item in day_results) / n_days
    rmse_off = sum(item["rmse_off"] for item in day_results) / n_days
    nrmse_diag = rmse_diag / NORM_DIAG
    nrmse_off = rmse_off / NORM_OFF
    return {
        "rmse_diag": rmse_diag,
        "rmse_off": rmse_off,
        "nrmse_diag": nrmse_diag,
        "nrmse_off": nrmse_off,
        "combined": (nrmse_diag + nrmse_off) / 2,
    }


def calculate_competition_nrmse(predictions, truths, days=None):
    """日別OD辞書から公式Combined NRMSEを一度に計算する。

    ``predictions`` / ``truths`` は ``{YYYYMMDD: nested_od}``。``days`` を省略
    した場合は両方に存在する日付の共通部分を使う。
    """
    if days is None:
        days = sorted(set(predictions) & set(truths))
    else:
        days = list(days)
    missing_pred = [day for day in days if day not in predictions]
    missing_truth = [day for day in days if day not in truths]
    if missing_pred or missing_truth:
        raise ValueError(
            f"missing dates: predictions={missing_pred}, truths={missing_truth}"
        )
    return score([day_errors(predictions[day], truths[day]) for day in days])


def observed_rms(rows):
    """観測ODから公式正規化定数と同じ定義のRMSを再計算する。"""
    if not rows:
        raise ValueError("rows must contain at least one observed day")
    squared_diag = 0.0
    squared_off = 0.0
    for od in rows.values():
        for origin, dests in od.items():
            if not in_eval_box(origin):
                continue
            for dest, value in dests.items():
                if not in_eval_box(dest):
                    continue
                if origin == dest:
                    squared_diag += value * value
                else:
                    squared_off += value * value
    n_days = len(rows)
    return {
        "n_days": n_days,
        "rms_diag": (squared_diag / (N_DIAG * n_days)) ** 0.5,
        "rms_off": (squared_off / (N_OFF * n_days)) ** 0.5,
    }


def pct(part, whole):
    """誤差内訳表示用の百分率。分母0なら0を返す。"""
    return 100.0 * part / whole if whole else 0.0

