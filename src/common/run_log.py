"""実験の実行内容を読みやすいブロックで.logへ追記する。

使い方:
  from common.run_log import run_log
  with run_log("EXP00", "run", "python3 src/EXP00_run.py") as log:
      log.stage("stage0", "アンカー2日の読み込み")
      ...
      log.result(combined=0.8421, jan=0.7912, apr=0.8930)
      log.note("→ experiments/EXP00/figures/local_error_map.png")

stage() を呼ぶたびに直前の stage が閉じ、所要時間が確定する。with を抜けるときに
ブロック全体を1回で追記するので、例外で落ちても FAIL 行を含む記録が残る。

書式はこのモジュールの stage / result / error 出力を正とする。

入力 : なし
出力 : experiments/<ID>/.log（追記のみ）
依存 : 標準ライブラリのみ
"""

import datetime as dt
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXP = ROOT / "experiments"

RULE = "=" * 80
THIN = "-" * 80


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


class RunLog:
    """1実行分のログブロックを組み立てる。直接作らず run_log() を使う。"""

    def __init__(self, exp_id, mode, command):
        self.path = EXP / exp_id / ".log"
        self.exp_id, self.mode, self.command = exp_id, mode, command
        self.started = dt.datetime.now()
        self.t0 = time.perf_counter()
        self.stages = []          # [(開始時刻, 名前, 内容, 所要秒 or None)]
        self.lines = []           # (種別, 内容)
        self.failed = None

    def stage(self, name, label):
        """直前の stage を閉じ、新しい stage を開始する。"""
        self._close_stage()
        self.stages.append([dt.datetime.now(), name, label, None])

    def result(self, **kv):
        """結果をキー・値で記録する。"""
        self.lines.append(("result", kv))

    def note(self, text):
        """生成物のパスなどを artifacts に記録する。"""
        self.lines.append(("artifact", text))

    def _close_stage(self, failed=False):
        if self.stages and self.stages[-1][3] is None:
            s = self.stages[-1]
            s[3] = "FAIL" if failed else (dt.datetime.now() - s[0]).total_seconds()

    def render(self):
        total = time.perf_counter() - self.t0
        status = "FAILED" if self.failed else "SUCCESS"
        out = ["", RULE,
               f"{self.started:%Y-%m-%d %H:%M:%S} | {self.exp_id} | "
               f"{self.mode} | {status}",
               f"command: {self.command}", THIN, "stages:"]
        for at, name, label, sec in self.stages:
            dur = "FAIL" if sec == "FAIL" else f"{sec:.1f}s"
            out.append(f"  - {name}: {label} ({dur}, {at:%H:%M:%S})")
        results = [value for kind, value in self.lines if kind == "result"]
        artifacts = [value for kind, value in self.lines if kind == "artifact"]
        if results:
            out.append("results:")
            for values in results:
                out.extend(f"  - {key}: {_fmt(value)}" for key, value in values.items())
        if artifacts:
            out.append("artifacts:")
            out.extend(f"  - {value.removeprefix('→ ').strip()}" for value in artifacts)
        if self.failed:
            out.append(f"error: {type(self.failed).__name__}: {self.failed}")
        out.append(f"duration: {total:.1f}s")
        return "\n".join(out) + "\n"

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(self.render())
        return self.path


@contextmanager
def run_log(exp_id, mode, command):
    """1実行＝1ブロックを experiments/<ID>/.log へ追記する。"""
    log = RunLog(exp_id, mode, command)
    try:
        yield log
    except BaseException as e:
        log._close_stage(failed=True)
        log.failed = e
        log.flush()
        raise
    log._close_stage()
    log.flush()
