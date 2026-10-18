"""主催者配布の公式validatorを別プロセスで実行し、提出TSVの形式を検査する（validator が無ければ検査を飛ばす）。

使い方 : from evaluation.submission_check import run_official_validator
入力   : 提出TSV、src/evaluation/humob2026_validator.py（主催者配布・git管理外）
出力   : なし（結果の1行を返す）
依存   : config.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import OFFICIAL_VALIDATOR


def run_official_validator(path: Path) -> str:
    """公式validatorで提出TSVを検査して結果の1行を返し、不合格なら例外を投げる。"""
    if not OFFICIAL_VALIDATOR.exists():
        print(f"[skip] 公式validatorがありません: {OFFICIAL_VALIDATOR}")
        return "skipped (validator not found)"
    checked = subprocess.run([sys.executable, str(OFFICIAL_VALIDATOR), str(path)], capture_output=True, text=True)
    message = checked.stdout.strip()
    if checked.returncode != 0:
        raise ValueError(f"公式validatorで不合格: {message}")
    print(f"[check] {path}: {message}")
    return message
