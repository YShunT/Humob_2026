"""EXP04のローカル検証予測を1月側・4月側のTSVに分けて保存する。

実行: python3 src/preprocessing/exp04_validation_tsv.py
入力: EXP03/EXP04のmetrics.json、配布OD、EXP03で定義された外部特徴CSV
出力: experiments/EXP04/outputs/validation_jan.tsv, validation_apr.tsv
依存: EXP04_run.py、EXP03_run.py、common/experiment_io.py、numpy、scipy、pandas、matplotlib

両窓14日を同時に学習から除外して再学習し、保存済み採用モデルだけを復元する。
モデル選択・提出用再学習・既存metrics/図/submissionの更新は行わない。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

# ファイルの位置を基準に解決し、実行時のカレントディレクトリに依存させない。
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import EXP04_run as experiment
from common.experiment_io import check_od_tsv, write_od_tsv

OUTPUT_DIR = experiment.OUT
METRICS_PATH = experiment.EXP / "metrics.json"
WINDOWS = {
    "jan": (202401, "validation_jan.tsv"),
    "apr": (202404, "validation_apr.tsv"),
}
SCORE_KEYS = ("combined_nrmse", "nrmse_diag", "nrmse_offdiag")


class ConsoleLog:
    """既存のfitが利用するstageインターフェース。実験の.logは更新しない。"""

    def stage(self, name, description):
        print(f"\n{'=' * 64}\n{name}: {description}\n{'=' * 64}", flush=True)


def load_metadata():
    """保存済みの採用候補と、現在の実装の設定が一致するか確認する。"""
    experiment.check_reference()
    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metadata["experiment"] != "EXP04":
        raise ValueError("metrics.jsonがEXP04の記録ではありません")
    config = metadata["config"]
    for key, value in experiment.specification().items():
        if config.get(key) != value:
            raise ValueError(f"保存時と実装の設定が異なります: {key}")
    if config["selected"] not in experiment.CANDIDATES:
        raise ValueError(f"未対応の採用候補: {config['selected']}")
    return metadata


def validation_windows():
    """日付の正本は既存CV定義とし、保存対象だけ月別に分割する。"""
    days = sorted(experiment.base.CV)
    windows = {
        name: [day for day in days if day // 100 == month]
        for name, (month, _) in WINDOWS.items()
    }
    if any(len(window) != 7 for window in windows.values()):
        raise ValueError("1月側・4月側のCVはそれぞれ7日である必要があります")
    if sorted(day for window in windows.values() for day in window) != days:
        raise ValueError("CV定義に1月・4月以外の日付または重複があります")
    return windows


def predict_validation(metadata, log):
    """CVを隠した1回の学習で両窓を推定する。正解は採点だけに使う。"""
    data = experiment.base.Dataset(validation=True)
    times = np.flatnonzero(np.isin(experiment.base.DAYS, sorted(experiment.base.CV)))
    if data.observed[times].any():
        raise ValueError("CV日が学習対象に含まれています")
    if set(data.truth) != experiment.base.CV:
        raise ValueError("検証14日分の正解データが揃っていません")
    for key, actual in (
        ("observed_days", int(data.observed.sum())),
        ("candidate_pairs", len(data.pairs)),
    ):
        if metadata["config"][key] != actual:
            raise ValueError(f"保存時と入力データの件数が異なります: {key}")

    diagonal, flows, _ = experiment.fit(data, times, log, "export-cv")
    selected = metadata["config"]["selected"]
    reverse_weight = experiment.CANDIDATES[selected]
    flow = (1 - reverse_weight) * flows["F"] + reverse_weight * flows["R"]

    # EXP04と同じ全域形式。評価外はCVを除いた震災後の曜日平均で補完。
    predictions = experiment.reconstruct(data, times, diagonal, flow, outside=True)
    return predictions, data.truth


def verify_scores(predictions, truths, metadata):
    """全体・各窓の3指標をmetrics.jsonの保存精度（小数6桁）で照合する。"""
    if set(predictions) != set(truths):
        raise ValueError("予測と正解の日付集合が一致しません")
    results, _ = experiment.evaluate(predictions, truths)
    references = {
        "overall": metadata["results"]["overall"],
        **metadata["results"]["validation"],
    }
    for period in ("overall", "jan", "apr"):
        actual, expected = results[period], references[period]
        if actual["n_days"] != expected["n_days"]:
            raise ValueError(f"{period}: 保存済み記録と検証日数が異なります")
        for key in SCORE_KEYS:
            value = actual[key]
            if not np.isfinite(value) or abs(round(value, 6) - expected[key]) > 1e-12:
                raise ValueError(
                    f"{period}/{key}が再現しません: {value} != {expected[key]}"
                )
        print(f"[score] {period}: " + ", ".join(
            f"{key}={actual[key]:.6f}" for key in SCORE_KEYS
        ), flush=True)


def save_windows(predictions, truths, metadata, windows):
    """両ファイルを一時保存・再読込検査してから出力先へ置き換える。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".validation-tsv-", dir=OUTPUT_DIR) as temporary:
        restored, staged = {}, []
        for name, days in windows.items():
            filename = WINDOWS[name][1]
            path = Path(temporary) / filename
            subset = {day: predictions[day] for day in days}
            write_od_tsv(subset, path)
            reloaded = check_od_tsv(path, days)
            if reloaded != subset:
                raise ValueError(f"{name}: 保存前後でOD値が変わりました")
            restored.update(reloaded)
            staged.append((path, OUTPUT_DIR / filename))

        # メモリ上だけでなく、実際に保存したTSVでも公式指標を照合する。
        verify_scores(restored, truths, metadata)
        for temporary_path, output_path in staged:
            temporary_path.replace(output_path)
            print(f"[saved] {output_path}", flush=True)


def main():
    log = ConsoleLog()
    log.stage("check", "保存済み採用モデルと検証期間を確認")
    metadata = load_metadata()
    windows = validation_windows()
    print(f"selected={metadata['config']['selected']}", flush=True)
    for name, days in windows.items():
        print(f"{name}: {days[0]}〜{days[-1]} ({len(days)}日)", flush=True)

    predictions, truths = predict_validation(metadata, log)
    log.stage("verify", "保存済みの全体・1月・4月スコアを再現確認")
    verify_scores(predictions, truths, metadata)
    log.stage("save", "月別TSVの保存・再読込・形式とスコアの検査")
    save_windows(predictions, truths, metadata, windows)
    print("\n完了: 既存のmetrics.json・.log・図・submission.tsvは変更していません。",
          flush=True)


if __name__ == "__main__":
    main()
