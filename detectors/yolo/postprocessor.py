# detectors/yolo/postprocessor.py
"""
Converts raw YOLOv8 Results → DetectionResult.

9-class model:
    0            → mouse  (count only)
    1,2,3,4      → water_critical / low / ok / full
    5,6,7,8      → food_critical  / low / ok / full

Level reading is derived directly from the detected class ID.
No HSV estimation, no colour math.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.config import YOLO_CLASS_MAP, CLASS_TO_LEVEL
from core.schemas import DetectionResult, LevelReading, BoundingBox
from core.exceptions import InferenceError

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

CLASS_MOUSE       = 0
WATER_CLASS_IDS   = {1, 2, 3, 4}
FOOD_CLASS_IDS    = {5, 6, 7, 8}


def parse_yolo_results(
    results: list["Results"],
    cage_id: str,
    inference_start_ns: int,
) -> DetectionResult:
    """
    Build a DetectionResult from Ultralytics Results.
    Level readings come from the detected class ID — no external estimator needed.
    """
    if not results:
        raise InferenceError("YOLO returned empty results list.")

    r = results[0]
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)
    mouse_count  = _count_class(r, CLASS_MOUSE)

    water_reading, water_bbox = _best_level_reading(r, WATER_CLASS_IDS)
    food_reading,  food_bbox  = _best_level_reading(r, FOOD_CLASS_IDS)

    return DetectionResult(
        cage_id=cage_id,
        timestamp=datetime.now(tz=timezone.utc),
        mouse_count=mouse_count,
        water=water_reading,
        food=food_reading,
        inference_ms=inference_ms,
        image_path=None,
        water_bbox=water_bbox,
        food_bbox=food_bbox,
    )


def _best_level_reading(
    r: "Results",
    target_classes: set[int],
) -> tuple[LevelReading, Optional[BoundingBox]]:
    """
    Among all detections matching target_classes, pick the highest-confidence one.
    Returns (LevelReading derived from class ID, BoundingBox).
    Falls back to LevelReading.unknown() if nothing detected.
    """
    if r.boxes is None or len(r.boxes) == 0:
        return LevelReading.unknown(), None

    try:
        boxes   = r.boxes.xyxy.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        raise InferenceError(f"Failed to parse YOLO boxes: {e}") from e

    best_conf  = -1.0
    best_cls   = None
    best_box   = None

    for box, conf, cls in zip(boxes, confs, classes):
        if int(cls) in target_classes and float(conf) > best_conf:
            best_conf = float(conf)
            best_cls  = int(cls)
            best_box  = box

    if best_cls is None:
        return LevelReading.unknown(), None

    reading = LevelReading.from_class_id(best_cls)
    bbox    = BoundingBox(
        x1=float(best_box[0]), y1=float(best_box[1]),
        x2=float(best_box[2]), y2=float(best_box[3]),
        conf=best_conf,
    )
    return reading, bbox


def _count_class(r: "Results", target_class: int) -> int:
    if r.boxes is None or len(r.boxes) == 0:
        return 0
    try:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        return int(np.sum(classes == target_class))
    except Exception as e:
        raise InferenceError(f"Failed to parse YOLO box classes: {e}") from e


def extract_boxes(r: "Results") -> list[dict]:
    """Extract all boxes as dicts for logging/debug."""
    if r.boxes is None or len(r.boxes) == 0:
        return []
    out = []
    try:
        boxes   = r.boxes.xyxy.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        raise InferenceError(f"Failed to extract YOLO boxes: {e}") from e

    for box, conf, cls in zip(boxes, confs, classes):
        out.append({
            "class_id": int(cls),
            "label":    YOLO_CLASS_MAP.get(int(cls), f"class_{cls}"),
            "conf":     float(round(conf, 3)),
            "xyxy":     tuple(float(v) for v in box),
        })
    return out