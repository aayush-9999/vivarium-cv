# core/config.py
from pathlib import Path
import os
# ── Paths ────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "models"
FRAMES_DIR   = BASE_DIR / "frames"   # flagged frames saved here

YOLO_WEIGHTS = MODELS_DIR / "yolo" / "best.pt"

# ── Model input ──────────────────────────────────────────────────
INPUT_SIZE = (640, 640)   # (width, height)

# ── YOLO class map ───────────────────────────────────────────────
# Must match the order classes were labelled in your training dataset
YOLO_CLASS_NAMES = ["mouse", "water_container", "food_area"]

YOLO_CLASS_MAP = {
    0: "mouse",
    1: "water_container",
    2: "food_area",
}

# ── Confidence thresholds ────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.35   # lower a bit — containers are easy to miss at 0.45
YOLO_IOU_THRESHOLD  = 0.45

# ── ROI zones per cage (x, y, w, h) in pixels ───────────────────
# These assume 640x640 input. Recalibrate after first camera mount.
ROI_ZONES = {
    "default": {
        "jug":    (480, 40,  120, 320),  # right side, tall vertical zone
        "hopper": (40,  40,  160, 220),  # left side, food hopper
        "floor":  (40,  280, 560, 320),  # bottom strip, bedding/floor
    }
}

# ── Inference scheduling ─────────────────────────────────────────
INFERENCE_INTERVAL_SEC  = 300   # 5 minutes baseline polling
MOTION_PIXEL_THRESHOLD  = 0.05  # 5% of frame pixels changed = motion detected
MOTION_CHECK_INTERVAL   = 30    # seconds between motion checks

# ── Frame saving ─────────────────────────────────────────────────
SAVE_FLAGGED_FRAMES = True      # save frame to disk when status is LOW or CRITICAL
STALE_READING_MIN   = 15        # minutes before CageStatus.is_stale = True



SIZE: tuple[int, int] = (640, 640)   # (width, height) — YOLOv8-nano expects this

# ── Level thresholds (%) ──────────────────────────────────────────
LEVEL_THRESHOLDS = {
    "CRITICAL": 15.0,   # below 15% → CRITICAL alert
    "LOW":      35.0,   # below 35% → LOW alert
}

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

# ── YOLO class mapping ────────────────────────────────────────────
# Maps YOLO class index → human label.
# Must match the order in your dataset's data.yaml.
YOLO_CLASS_MAP: dict[int, str] = {
    0: "mouse",
}

# ── Inference confidence / NMS ────────────────────────────────────
YOLO_CONF_THRESHOLD: float = 0.35
YOLO_IOU_THRESHOLD:  float = 0.45

# ── Alert cooldown ────────────────────────────────────────────────
# Minimum seconds between repeated alerts for the same cage + type
ALERT_COOLDOWN_SECONDS: int = 600   # 10 minutes

# ── Stale reading threshold ───────────────────────────────────────
STALE_READING_MINUTES: int = 15
