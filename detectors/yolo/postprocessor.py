# detectors/yolo/postprocessor.py
"""
Converts raw YOLOv8 Results object → DetectionResult.

Class map:
    0 = mouse            → counted, not measured
    1 = water_container  → bbox passed to WaterLevelEstimator as dynamic ROI
    2 = food_area        → bbox passed to FoodLevelEstimator  as dynamic ROI
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.config import YOLO_CLASS_MAP
from core.schemas import DetectionResult, LevelReading, BoundingBox
from core.exceptions import InferenceError

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# Class indices — keep in sync with core/config.py and vivarium.yaml
CLASS_MOUSE           = 0
CLASS_WATER_CONTAINER = 1
CLASS_FOOD_AREA       = 2


def parse_yolo_results(
    results: list["Results"],
    cage_id: str,
    water_reading: LevelReading,
    food_reading: LevelReading,
    inference_start_ns: int,
    water_bbox: Optional[BoundingBox] = None,
    food_bbox:  Optional[BoundingBox] = None,
) -> DetectionResult:
    """
    Build a DetectionResult from Ultralytics Results + pre-computed level readings.

    Args:
        results:            Output of model.predict() — list with one Results object.
        cage_id:            Cage identifier string.
        water_reading:      Already-computed LevelReading for water.
        food_reading:       Already-computed LevelReading for food.
        inference_start_ns: time.perf_counter_ns() captured before model.predict().
        water_bbox:         BoundingBox of the detected water container (or None).
        food_bbox:          BoundingBox of the detected food area (or None).

    Returns:
        Fully populated DetectionResult.
    """
    if not results:
        raise InferenceError("YOLO returned empty results list.")

    r = results[0]   # single-image inference → always one Results object
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)
    mouse_count  = _count_class(r, target_class=CLASS_MOUSE)

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


def extract_container_bboxes(
    results: list["Results"],
) -> tuple[Optional[BoundingBox], Optional[BoundingBox]]:
    """
    Pull the highest-confidence water_container and food_area detections
    from YOLO results and return them as BoundingBox objects.

    Returns:
        (water_bbox, food_bbox) — either can be None if not detected.
    """
    if not results:
        return None, None

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None

    try:
        boxes   = r.boxes.xyxy.cpu().numpy()              # (N, 4)
        confs   = r.boxes.conf.cpu().numpy()              # (N,)
        classes = r.boxes.cls.cpu().numpy().astype(int)   # (N,)
    except Exception as e:
        raise InferenceError(f"Failed to extract YOLO boxes: {e}") from e

    water_bbox = _best_box_for_class(boxes, confs, classes, CLASS_WATER_CONTAINER)
    food_bbox  = _best_box_for_class(boxes, confs, classes, CLASS_FOOD_AREA)

    return water_bbox, food_bbox


def _best_box_for_class(
    boxes: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    target_class: int,
) -> Optional[BoundingBox]:
    """Return the highest-confidence detection for a given class, or None."""
    mask = classes == target_class
    if not np.any(mask):
        return None

    idx  = int(np.argmax(np.where(mask, confs, -1)))
    box  = boxes[idx]
    conf = float(confs[idx])

    return BoundingBox(
        x1=float(box[0]),
        y1=float(box[1]),
        x2=float(box[2]),
        y2=float(box[3]),
        conf=conf,
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
    """
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