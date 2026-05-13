# core/config_loader.py  — BEDDING PATCH
"""
Central configuration loader.

Changes vs original:
  - num_classes bumped to 10 (added class 9 = bedding)
  - YOLO_CLASS_MAP now includes {9: "bedding"}
  - BEDDING_AREA_THRESHOLD added (50 %) — bedding bbox area >= this → BAD

Usage
-----
    from core.config_loader import CONFIG

    backend   = CONFIG["backend"]
    weights   = CONFIG["yolox"]["weights"]
    device    = CONFIG["device"]
    roi_zones = CONFIG["roi_zones"]

    from core.config_loader import (
        YOLOX_EXP_FILE, YOLOX_INPUT_SIZE,
        YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
        BEDDING_AREA_THRESHOLD,
    )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pull hard-coded constants from core/config.py
# ---------------------------------------------------------------------------
try:
    from core.config import (          # type: ignore[import]
        INPUT_SIZE,
        ROI_ZONES,
        YOLO_CLASS_MAP,
        CLASS_TO_LEVEL,
        WATER_CLASS_BOUNDARIES,
        FOOD_CLASS_BOUNDARIES,
        MOTION_PIXEL_THRESHOLD,
    )
except ImportError:
    INPUT_SIZE               = (640, 640)
    ROI_ZONES                = {"default": {}}
    YOLO_CLASS_MAP           = {0: "mouse"}
    CLASS_TO_LEVEL           = {}
    WATER_CLASS_BOUNDARIES   = [
        (0.00, 0.15, 1),
        (0.15, 0.35, 2),
        (0.35, 0.80, 3),
        (0.80, 1.01, 4),
    ]
    FOOD_CLASS_BOUNDARIES    = [
        (0.00, 0.15, 5),
        (0.15, 0.35, 6),
        (0.35, 0.80, 7),
        (0.80, 1.01, 8),
    ]
    MOTION_PIXEL_THRESHOLD   = 0.02

for cls_id, name in {
    9:  "bedding_worst",
    10: "bedding_bad",
    11: "bedding_ok",
    12: "bedding_perfect",
}.items():
    if cls_id not in YOLO_CLASS_MAP:
        YOLO_CLASS_MAP = dict(YOLO_CLASS_MAP)
        YOLO_CLASS_MAP[cls_id] = name


# ---------------------------------------------------------------------------
# YOLOX experiment file + inference constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT       = Path(__file__).resolve().parent.parent
_DEFAULT_EXP        = str(_PROJECT_ROOT / "exps" / "yolox_vivarium_tiny.py")

YOLOX_EXP_FILE      = Path(os.getenv("YOLOX_EXP_FILE", _DEFAULT_EXP))
YOLOX_INPUT_SIZE: tuple[int, int] = tuple(INPUT_SIZE)  # type: ignore[assignment]

YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU_THRESHOLD  = float(os.getenv("YOLO_IOU",  "0.45"))

# Bedding: bbox area fraction of frame >= this → condition = "BAD"
BEDDING_AREA_THRESHOLD = float(os.getenv("BEDDING_AREA_THRESHOLD", "50.0"))


# ---------------------------------------------------------------------------
# Master CONFIG dict
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # ── Runtime ──────────────────────────────────────────────────────────
    "backend": os.getenv("BACKEND", "yolo").lower(),
    "device":  os.getenv("YOLO_DEVICE", "cpu"),

    # ── YOLOX ────────────────────────────────────────────────────────────
     "yolox": {
        "weights":     os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt"),
        "exp_file":    str(YOLOX_EXP_FILE),
        "input_size":  YOLOX_INPUT_SIZE,
        "conf_thre":   YOLO_CONF_THRESHOLD,
        "nms_thre":    YOLO_IOU_THRESHOLD,
        "num_classes": int(os.getenv("YOLOX_NUM_CLASSES", "13")),
    },                                                         

    # ── SSD (legacy) ─────────────────────────────────────────────────────
    "ssd": {
        "weights": os.getenv("SSD_WEIGHTS", "models/ssd/ssd_mobilenet.onnx"),
    },
    # ── PSPNet ────────────────────────────────────────────────────────────
    "pspnet": {
        "water_weights":     os.getenv("PSP_WATER_WEIGHTS"),
        "food_weights":      os.getenv("PSP_FOOD_WEIGHTS"),
        "backbone":          os.getenv("PSP_BACKBONE", "resnet50"),
        "fallback_to_yolox": True,
    },

    # ── Status thresholds ─────────────────────────────────────────────────
    "thresholds": {
        "water": {"CRITICAL": (0.0, 15.0), "LOW": (15.0, 35.0), "OK": (35.0, 100.1)},
        "food":  {"CRITICAL": (0.0, 15.0), "LOW": (15.0, 35.0), "OK": (35.0, 100.1)},
    },

    # ── Bedding ───────────────────────────────────────────────────────────
    "bedding": {
        # bbox_area_pct >= this threshold → BAD, else GOOD
        "area_threshold": BEDDING_AREA_THRESHOLD,
            "use_clip":            True,                      # ADD: set False to disable CLIP
            "clip_dirty_threshold": 0.55, 
    },

    # ── Camera / motion ───────────────────────────────────────────────────
    "camera": {
        "rtsp_url":         os.getenv("CAMERA_RTSP_URL", ""),
        "motion_threshold": MOTION_PIXEL_THRESHOLD,
    },

    # ── Alerts ────────────────────────────────────────────────────────────
    "alerts": {
        "webhook_url": os.getenv("ALERT_WEBHOOK_URL", ""),
    },

    # ── Static constants (from core/config.py) ────────────────────────────
    "input_size":             INPUT_SIZE,
    "roi_zones":              ROI_ZONES,
    "yolo_class_map":         YOLO_CLASS_MAP,
    "class_to_level":         CLASS_TO_LEVEL,
    "water_class_boundaries": WATER_CLASS_BOUNDARIES,
    "food_class_boundaries":  FOOD_CLASS_BOUNDARIES,
}