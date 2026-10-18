"""1回の実行を stage ごとの所要時間つきで experiments_final/.log へ追記する。

使い方 :
  with run_log("FINAL", "--validate", "python3 src/main.py --validate") as log:
      log.stage("load", "配布ODの読み込み")
      log.result(combined_nrmse=0.2008)
      log.note("experiments_final/metrics.json")
入力   : なし
出力   : experiments_final/.log（追記のみ。例外で落ちても FAIL 行を残す）
依存   : config.py
"""


from __future__ import annotations

import datetime as dt
import time
from contextlib import contextmanager
from typing import Any, Iterator

from config import LOG_FILE

RULE = "=" * 80
THIN = "-" * 80


def format_value(value: Any) -> str:
    """ログに載せる値を文字列にし、float は小数4桁にそろえる。"""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


class RunLog:
    """1実行分のログブロックを組み立てる（直接作らず run_log() から使う）。"""

    def __init__(self, exp_id: str, mode: str, command: str) -> None:
        """ログの書き込み先と実行情報を保持し、計時を開始する。"""
        self.path = LOG_FILE
        self.exp_id, self.mode, self.command = exp_id, mode, command
        self.started = dt.datetime.now()
        self.t0 = time.perf_counter()
        self.stages: list[list[Any]] = []
        self.results: list[tuple[str, Any]] = []
        self.artifacts: list[str] = []
        self.failed: BaseException | None = None

    def stage(self, name: str, label: str) -> None:
        """直前の stage を閉じて所要時間を確定し、新しい stage を開始する。"""
        self.close_stage()
        print(f"[stage] {name}: {label}", flush=True)
        self.stages.append([dt.datetime.now(), name, label, None])

    def result(self, **values: Any) -> None:
        """主要な結果をキーと値で記録する。"""
        self.results.extend(values.items())

    def note(self, path: str) -> None:
        """生成物のパスを記録する。"""
        self.artifacts.append(path)

    def close_stage(self, failed: bool = False) -> None:
        """実行中の stage があれば所要秒数（失敗時は FAIL）を記入する。"""
        if self.stages and self.stages[-1][3] is None:
            started = self.stages[-1][0]
            self.stages[-1][3] = "FAIL" if failed else (dt.datetime.now() - started).total_seconds()

    def render(self) -> str:
        """ログブロック全体を .log の書式の文字列にする。"""
        status = "FAILED" if self.failed else "SUCCESS"
        lines = ["", RULE,
                 f"{self.started:%Y-%m-%d %H:%M:%S} | {self.exp_id} | {self.mode} | {status}",
                 f"command: {self.command}", THIN, "stages:"]
        for started, name, label, seconds in self.stages:
            duration = "FAIL" if seconds == "FAIL" else f"{seconds:.1f}s"
            lines.append(f"  - {name}: {label} ({duration}, {started:%H:%M:%S})")
        if self.results:
            lines.append("results:")
            lines.extend(f"  - {key}: {format_value(value)}" for key, value in self.results)
        if self.artifacts:
            lines.append("artifacts:")
            lines.extend(f"  - {path}" for path in self.artifacts)
        if self.failed:
            lines.append(f"error: {type(self.failed).__name__}: {self.failed}")
        lines.append(f"duration: {time.perf_counter() - self.t0:.1f}s")
        return "\n".join(lines) + "\n"

    def flush(self) -> None:
        """組み立てたブロックを .log の末尾へ追記する。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as destination:
            destination.write(self.render())


@contextmanager
def run_log(exp_id: str, mode: str, command: str) -> Iterator[RunLog]:
    """with ブロック1回を1ログブロックとして experiments_final/.log へ追記する。"""
    log = RunLog(exp_id, mode, command)
    try:
        yield log
    except BaseException as error:
        log.close_stage(failed=True)
        log.failed = error
        log.flush()
        raise
    log.close_stage()
    log.flush()
