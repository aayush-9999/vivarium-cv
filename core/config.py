# core/config.py
from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRAMES_DIR = BASE_DIR / "frames"   # flagged frames saved here

YOLO_WEIGHTS = MODELS_DIR / "yolo" / "best.pt"

# ── Model input ───────────────────────────────────────────────────────────────
INPUT_SIZE = (640, 640)   # (width, height)

# ── YOLO class mapping ────────────────────────────────────────────────────────
YOLO_CLASS_NAMES = ["mouse", "water_container", "food_area"]

YOLO_CLASS_MAP: dict[int, str] = {
    0: "mouse",
    1: "water_container",
    2: "food_area",
}

# ── Confidence / NMS thresholds ───────────────────────────────────────────────
YOLO_CONF_THRESHOLD: float = 0.35
YOLO_IOU_THRESHOLD:  float = 0.45

# ── ROI zones per cage (x, y, w, h) in 640×640 pixel space ───────────────────
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

# ── Level thresholds (%) ──────────────────────────────────────────────────────
LEVEL_THRESHOLDS = {
    "CRITICAL": 15.0,   # below 15% → CRITICAL
    "LOW":      35.0,   # below 35% → LOW
}

# ── Inference scheduling ──────────────────────────────────────────────────────
INFERENCE_INTERVAL_SEC = 300   # 5 min baseline polling
MOTION_PIXEL_THRESHOLD = 0.05  # 5% of pixels changed = motion
MOTION_CHECK_INTERVAL  = 30    # seconds between motion checks

# ── Frame saving ──────────────────────────────────────────────────────────────
SAVE_FLAGGED_FRAMES = True     # save frame when status is LOW or CRITICAL
STALE_READING_MIN   = 15       # minutes before CageStatus.is_stale = True

# ── Alert cooldown ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS: int = 600   # 10 min between repeated alerts per cage