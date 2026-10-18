"""最終モデル（EXP04 N-B）を学習し、CV14日の検証・テスト58日の提出物生成・設定表示・提出形式検査を行う。

使い方（モードの指定は必須）:
  python3 src/main.py --validate       # CV14日を隠して学習・採点（metrics.json・CV予測TSV）
  python3 src/main.py --submit         # 全観測日で学習し、テスト58日の提出TSVを生成して公式validatorで検査
  python3 src/main.py --inspect        # 設定と学習データの概要を表示するだけ（ファイルを書かない）
  python3 src/main.py --check [PATH]   # 既存の提出TSVを公式validatorで検査（既定は experiments_final/outputs/submission.tsv）

入力 : data/raw/humob2026-dataset.tsv、data/processed/ の前処理済みCSV5本（config.py に列挙）
出力 : experiments_final/{metrics.json,.log}、experiments_final/outputs/{submission.tsv,validation_jan.tsv,validation_apr.tsv}
依存 : config.py、dataset/*.py、models/*.py、evaluation/*.py、utils/*.py、numpy、scipy、pandas
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from config import (EXPERIMENT_ID, METRICS_JSON, ROOT, SOURCE_EXPERIMENT, SUBMISSION_TSV, VALIDATION_TSV,
                    model_settings)
from dataset.external_features import load_external_features
from dataset.od_matrix import load_od_matrix
from dataset.periods import CV_DAYS, CV_WINDOW_DAYS, DAYS, TEST_DAYS
from evaluation.metric import score_periods
from evaluation.submission_check import run_official_validator
from models.od_forecaster import predict, to_od_dicts
from utils.od_io import write_metrics, write_od_tsv
from utils.run_log import run_log


def command(flag: str) -> str:
    """.log に残す再現コマンドを返す。"""
    return f"python3 src/main.py {flag}"


def relative(path: Path) -> str:
    """リポジトリルートからの相対パスを文字列で返す。"""
    return str(path.relative_to(ROOT))


def validate() -> None:
    """CV14日を学習から外して5成分を学習し、公式指標で採点して metrics.json と CV予測TSV を保存する。"""
    with run_log(EXPERIMENT_ID, "--validate", command("--validate")) as log:
        log.stage("load", "配布OD・外部特徴を読み込み、CV14日を学習から除外")
        od = load_od_matrix(set(CV_DAYS))
        external = load_external_features(od.cells)
        forecast = predict(od, external, CV_DAYS, log)
        log.stage("score", "公式指標で14日・1月・4月を採点し、metrics.json と CV予測TSV を保存")
        predictions = to_od_dicts(od, forecast)
        scores = score_periods(predictions, od.truth)
        write_metrics(METRICS_JSON, {
            "experiment": EXPERIMENT_ID,
            "date": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
            "status": "評価済み",
            "compared_to": SOURCE_EXPERIMENT,
            "results": {"overall": scores["overall"],
                        "validation": {name: scores[name] for name in CV_WINDOW_DAYS},
                        "fit_diagnostics": forecast.diagnostics},
            "config": {**model_settings(), "observed_days": int(od.observed.sum()), "candidate_pairs": len(od.pairs)},
        })
        for name, days in CV_WINDOW_DAYS.items():
            write_od_tsv({day: predictions[day] for day in days}, VALIDATION_TSV[name])
        log.result(**{f"{period}_{key}": value for period, summary in scores.items()
                      for key, value in summary.items() if key != "n_days"})
        for path in (METRICS_JSON, *VALIDATION_TSV.values()):
            log.note(relative(path))
        print(f"[score] {scores['overall']}")


def submit() -> None:
    """全観測日で5成分を学習し、テスト58日の提出TSVを保存して公式validatorで検査する。"""
    with run_log(EXPERIMENT_ID, "--submit", command("--submit")) as log:
        log.stage("load", "配布OD・外部特徴を読み込み（全観測日を学習に使用）")
        od = load_od_matrix(set())
        external = load_external_features(od.cells)
        forecast = predict(od, external, TEST_DAYS, log)
        log.stage("write", "提出58日の書き出しと公式validator")
        predictions = to_od_dicts(od, forecast)
        write_od_tsv(predictions, SUBMISSION_TSV)
        validator = run_official_validator(SUBMISSION_TSV)
        log.result(days=len(predictions), observed_days=int(od.observed.sum()), validator=validator)
        log.note(relative(SUBMISSION_TSV))


def inspect_model() -> None:
    """モデル設定・期間・CV時の学習データ規模を表示する（ファイルは書かない）。"""
    for key, value in model_settings().items():
        print(f"{key:<20}: {value}")
    print(f"{'calendar':<20}: {DAYS[0]}〜{DAYS[-1]} ({len(DAYS)}日)")
    for name, days in CV_WINDOW_DAYS.items():
        print(f"{'cv_' + name:<20}: {days[0]}〜{days[-1]} ({len(days)}日)")
    print(f"{'test':<20}: {TEST_DAYS[0]}〜{TEST_DAYS[-1]} ({len(TEST_DAYS)}日)")
    od = load_od_matrix(set(CV_DAYS))
    print(f"{'observed_days(cv)':<20}: {int(od.observed.sum())}（--submit ではCV {len(od.truth)}日も学習に加える）")
    print(f"{'candidate_pairs(cv)':<20}: {len(od.pairs)}")
    print(f"{'outside_pairs':<20}: " + ", ".join(f"曜日{w}={len(t)}" for w, t in enumerate(od.outside_mean)))


def main() -> None:
    """コマンドライン引数のモードを1つだけ受け取り、対応する処理を実行する。"""
    parser = argparse.ArgumentParser(description="最終モデル（EXP04 N-B）の検証・提出")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="CV14日のローカル検証")
    mode.add_argument("--submit", action="store_true", help="テスト58日の提出物を生成")
    mode.add_argument("--inspect", action="store_true", help="設定と学習データの概要を表示するだけ")
    mode.add_argument("--check", nargs="?", const=SUBMISSION_TSV, type=Path, metavar="PATH",
                      help="既存の提出TSVを公式validatorで検査")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.submit:
        submit()
    elif args.inspect:
        inspect_model()
    else:
        run_official_validator(args.check)


if __name__ == "__main__":
    main()
