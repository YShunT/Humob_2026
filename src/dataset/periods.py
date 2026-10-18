"""学習カレンダー・CV14日・テスト58日と、震災境界・曜日・年末年始の暦配列を定義する（日付の正本）。

使い方 : from dataset.periods import DAYS, CV_DAYS, TEST_DAYS, QUAKE_INDEX
入力   : なし
出力   : なし
依存   : numpy
形状   : T = 学習カレンダーの日数（2023-11-01〜2024-10-31 の366日）
"""

from __future__ import annotations

import datetime as dt

import numpy as np


def to_int(day: dt.date) -> int:
    """日付を YYYYMMDD 形式の整数へ変換する。"""
    return int(day.strftime("%Y%m%d"))


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """start から end までの日付を両端を含めて返す。"""
    return [start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)]


CALENDAR = date_range(dt.date(2023, 11, 1), dt.date(2024, 10, 31))
DAYS = np.array([to_int(day) for day in CALENDAR])  # (T,)
DAY_INDEX = {int(day): t for t, day in enumerate(DAYS)}
WEEKDAY = np.array([day.weekday() for day in CALENDAR])  # (T,)
YEAR_END = np.array([(day.month == 12 and day.day >= 29) or (day.month == 1 and day.day <= 3)
                     for day in CALENDAR])  # (T,)
QUAKE_DAY = 20240101
QUAKE_INDEX = DAY_INDEX[QUAKE_DAY]

CV_WINDOWS = {
    "jan": (dt.date(2024, 1, 25), dt.date(2024, 1, 31)),
    "apr": (dt.date(2024, 4, 1), dt.date(2024, 4, 7)),
}
CV_WINDOW_DAYS = {name: [to_int(day) for day in date_range(start, end)] for name, (start, end) in CV_WINDOWS.items()}
CV_DAYS = [day for days in CV_WINDOW_DAYS.values() for day in days]

TEST_SPAN = (dt.date(2024, 2, 1), dt.date(2024, 3, 31))
TEST_EXCLUDED_DAYS = {20240202, 20240305}
TEST_DAYS = [to_int(day) for day in date_range(*TEST_SPAN) if to_int(day) not in TEST_EXCLUDED_DAYS]
