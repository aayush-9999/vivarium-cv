# detectors/yolo/postprocessor.py
"""
Converts raw YOLOv8 Results object → DetectionResult.
Kept separate from the detector so it can be unit-tested
without a GPU or model weights.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from core.config import YOLO_CLASS_MAP
from core.schemas import DetectionResult, LevelReading
from core.exceptions import InferenceError

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


def parse_yolo_results(
    results: list["Results"],
    cage_id: str,
    water_reading: LevelReading,
    food_reading: LevelReading,
    inference_start_ns: int,
) -> DetectionResult:
    """
    Build a DetectionResult from Ultralytics Results + pre-computed level readings.

    Args:
        results:            Output of model.predict() — list with one Results object.
        cage_id:            Cage identifier string.
        water_reading:      Already-computed LevelReading for water.
        food_reading:       Already-computed LevelReading for food.
        inference_start_ns: time.perf_counter_ns() captured before model.predict().

    Returns:
        Fully populated DetectionResult.
    """
    if not results:
        raise InferenceError("YOLO returned empty results list.")

    r = results[0]   # single-image inference → always one Results object

    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)

    mouse_count = _count_class(r, target_class=0)   # class 0 = mouse

    return DetectionResult(
        cage_id=cage_id,
        timestamp=datetime.now(tz=timezone.utc),
        mouse_count=mouse_count,
        water=water_reading,
        food=food_reading,
        inference_ms=inference_ms,
        image_path=None,   # set by pipeline if frame is flagged for saving
    )


def _count_class(r: "Results", target_class: int) -> int:
    """Count detections of a given class index in a Results object."""
    if r.boxes is None or len(r.boxes) == 0:
        return 0
    try:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        return int(np.sum(classes == target_class))
    except Exception as e:
        raise InferenceError(f"Failed to parse YOLO box classes: {e}") from e


def extract_boxes(
    r: "Results",
    frame_shape: tuple[int, int],
) -> list[dict]:
    """
    Extract all bounding boxes as dicts for optional saving / downstream use.

    Returns list of:
        {
            "class_id":    int,
            "label":       str,
            "conf":        float,
            "xyxy":        (x1, y1, x2, y2) in frame pixel coords,
        }
    """
    if r.boxes is None or len(r.boxes) == 0:
        return []

    out = []
    try:
        boxes   = r.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs   = r.boxes.conf.cpu().numpy()   # (N,)
        classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
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