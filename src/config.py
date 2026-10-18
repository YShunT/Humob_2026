"""入出力パスとモデルのハイパーパラメータ（特徴群・結合重み・EM・リッジ・気象観測所）を1か所で定義する。

使い方 : from config import RAW_OD_TSV, SUBMISSION_TSV, DIAGONAL_GROUPS, EM_MAX_ITER
入力   : なし
出力   : なし
依存   : 標準ライブラリのみ
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_OD_TSV = ROOT / "data/raw/humob2026-dataset.tsv"
PROCESSED = ROOT / "data/processed"
CELL_MAP_CSV = PROCESSED / "events/population_cell_map.csv"
EVACUATION_RATE_CSV = PROCESSED / "避難者率/市町別避難者率.csv"
WATER_OUTAGE_RATE_CSV = PROCESSED / "断水率/市町別断水率.csv"
SHELTER_CELL_CSV = PROCESSED / "避難所/避難所_セル別集計.csv"
WEATHER_CSV = PROCESSED / "weather/weather.csv"

EXPERIMENT_ID = "FINAL"
OUTPUT_DIR = ROOT / "experiments_final"
LOG_FILE = OUTPUT_DIR / ".log"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
SUBMISSION_TSV = OUTPUT_DIR / "outputs/submission.tsv"
VALIDATION_TSV = {
    "jan": OUTPUT_DIR / "outputs/validation_jan.tsv",
    "apr": OUTPUT_DIR / "outputs/validation_apr.tsv",
}
OFFICIAL_VALIDATOR = ROOT / "src/evaluation/humob2026_validator.py"

SOURCE_EXPERIMENT = "EXP04"
CANDIDATE = "N-B"
DIAGONAL_GROUPS = ("evac", "water", "rain")
OFFDIAGONAL_GROUPS = ("evac", "water", "rain", "snow", "shelter")
INCOMING_WEIGHT = 0.5

EM_MAX_ITER = 200
EM_TOL = 1e-5
VARIANCE_FLOOR = 1e-8
INITIAL_VARIANCE = 1e4
INITIAL_Q_RATIO = 0.03
MAX_LOG_MEAN = 50.0

WEEKDAY_RIDGE = 1.0
DEVIATION_RIDGE = 20.0
GLOBAL_RIDGE = 1.0
CONSTANT_SD = 1e-10

STATIONS = {
    "珠洲": (37 + 26.8 / 60, 137 + 17.2 / 60),
    "輪島": (37 + 23.4 / 60, 136 + 53.7 / 60),
    "門前": (37 + 15.7 / 60, 136 + 43.7 / 60),
    "三井": (37 + 17.6 / 60, 136 + 57.7 / 60),
    "志賀": (37 + 8.6 / 60, 136 + 43.5 / 60),
    "七尾": (37 + 1.8 / 60, 136 + 59.5 / 60),
}


def model_settings() -> dict[str, object]:
    """metrics.json の config と --inspect に載せるモデル設定一式を辞書で返す。"""
    return {
        "source_experiment": SOURCE_EXPERIMENT,
        "candidate": CANDIDATE,
        "diagonal_groups": list(DIAGONAL_GROUPS),
        "offdiagonal_groups": list(OFFDIAGONAL_GROUPS),
        "incoming_weight": INCOMING_WEIGHT,
        "em": {"max_iter": EM_MAX_ITER, "tol": EM_TOL, "variance_floor": VARIANCE_FLOOR,
               "initial_variance": INITIAL_VARIANCE, "initial_q_ratio": INITIAL_Q_RATIO},
        "ridge": {"weekday": WEEKDAY_RIDGE, "deviation": DEVIATION_RIDGE, "global": GLOBAL_RIDGE},
    }
