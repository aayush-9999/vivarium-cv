# core/config.py
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "models"
FRAMES_DIR   = BASE_DIR / "frames"   # flagged frames saved here

YOLO_WEIGHTS = MODELS_DIR / "yolo" / "best.pt"

# ── Model input ──────────────────────────────────────────────────
INPUT_SIZE = (640, 640)   # (width, height)

# ── YOLO class map ───────────────────────────────────────────────
# Must match the order classes were labelled in your training dataset
CLASS_NAMES = {
    0: "mouse",
    1: "water",
    2: "food",
}

# ── Confidence thresholds ────────────────────────────────────────
CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.40   # NMS IOU

# ── Level status thresholds (%) ──────────────────────────────────
LEVEL_THRESHOLDS = {
    "CRITICAL": 10,
    "LOW":      20,
}

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
MOTION_PIXEL_THRESHOLD  = 0.03  # 3% of frame pixels changed = motion detected
MOTION_CHECK_INTERVAL   = 30    # seconds between motion checks

# ── Frame saving ─────────────────────────────────────────────────
SAVE_FLAGGED_FRAMES = True      # save frame to disk when status is LOW or CRITICAL
STALE_READING_MIN   = 15        # minutes before CageStatus.is_stale = True