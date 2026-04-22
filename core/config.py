# core/config.py
from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRAMES_DIR = BASE_DIR / "frames"

YOLO_WEIGHTS = MODELS_DIR / "yolo" / "best.pt"

# ── Model input ───────────────────────────────────────────────────────────────
INPUT_SIZE = (640, 640)

# ── YOLO 9-class map ──────────────────────────────────────────────────────────
#
#  class 0 : mouse
#
#  classes 1–4 : water container at different fill levels
#  class 1 : water_critical   (0–15%)
#  class 2 : water_low        (15–35%)
#  class 3 : water_ok         (35–80%)
#  class 4 : water_full       (80–100%)
#
#  classes 5–8 : food area at different fill levels
#  class 5 : food_critical    (0–15%)
#  class 6 : food_low         (15–35%)
#  class 7 : food_ok          (35–80%)
#  class 8 : food_full        (80–100%)

YOLO_CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
]

YOLO_CLASS_MAP: dict[int, str] = {i: n for i, n in enumerate(YOLO_CLASS_NAMES)}

# Maps class ID → (type, status, representative pct)
CLASS_TO_LEVEL: dict[int, tuple[str, str, float]] = {
    1: ("water", "CRITICAL",  7.5),
    2: ("water", "LOW",      25.0),
    3: ("water", "OK",       57.5),
    4: ("water", "OK",       90.0),
    5: ("food",  "CRITICAL",  7.5),
    6: ("food",  "LOW",      25.0),
    7: ("food",  "OK",       57.5),
    8: ("food",  "OK",       90.0),
}

# Fill fraction → class ID used by augment.py
WATER_CLASS_BOUNDARIES = [
    (0.00, 0.15, 1),
    (0.15, 0.35, 2),
    (0.35, 0.80, 3),
    (0.80, 1.01, 4),
]
FOOD_CLASS_BOUNDARIES = [
    (0.00, 0.15, 5),
    (0.15, 0.35, 6),
    (0.35, 0.80, 7),
    (0.80, 1.01, 8),
]

# ── Detection thresholds ──────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD: float = 0.35
YOLO_IOU_THRESHOLD:  float = 0.45

# ── ROI zones (fallback only) ─────────────────────────────────────────────────
ROI_ZONES = {
    "default": {
        "jug":    (480, 80,  140, 300),
        "hopper": (20,  80,  160, 200),
        "floor":  (0,   400, 640, 240),
    },
    "type_b": {
        "jug":    (460, 60,  160, 320),
        "hopper": (10,  60,  180, 220),
        "floor":  (0,   400, 640, 240),
    },
}

# ── Level thresholds ──────────────────────────────────────────────────────────
LEVEL_THRESHOLDS = {
    "CRITICAL": 15.0,
    "LOW":      35.0,
}

# ── Inference scheduling ──────────────────────────────────────────────────────
INFERENCE_INTERVAL_SEC = 300
MOTION_PIXEL_THRESHOLD = 0.05
MOTION_CHECK_INTERVAL  = 30

# ── Frame saving ──────────────────────────────────────────────────────────────
SAVE_FLAGGED_FRAMES    = True
STALE_READING_MIN      = 15
ALERT_COOLDOWN_SECONDS = 600