# pipeline/detectors/yolo/postprocessor.py
"""
Post-processing helpers for YOLOX raw output.

parse_yolox_results() is the main entry-point called by YOLODetector.
Everything here is stateless — pure functions only.

BEDDING PATCH
─────────────
Class 9 (bedding) is now parsed alongside mice, water, and food.
Bedding condition is determined by comparing the total bbox area of all
bedding detections (as a % of the 640×640 frame) against the threshold
from CONFIG["bedding"]["area_threshold"] (default 50 %).

  total_bedding_area_pct >= 50 % → condition = "BAD"   (needs changing)
  total_bedding_area_pct <  50 % → condition = "GOOD"  (still clean)

If no bedding box is detected, BeddingReading.not_detected() is returned
(area_pct=0, condition="GOOD").
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.config_loader import CONFIG
from core.schemas import BoundingBox, BeddingReading, DetectionResult, LevelReading
from core.exceptions import InferenceError

logger = logging.getLogger("vivarium.postprocessor")

# Class ID groups
WATER_CLASS_IDS   = {1, 2, 3, 4}
FOOD_CLASS_IDS    = {5, 6, 7, 8}
BEDDING_CLASS_ID  = 9

# Frame size used during YOLOX training (used for bedding area calculation)
_FRAME_AREA = 640 * 640


# ---------------------------------------------------------------------------
# YOLOX post-processing
# ---------------------------------------------------------------------------

def parse_yolox_results(
    outputs,
    ratio,
    cage_id: str,
    inference_start_ns: int,
) -> DetectionResult:
    """
    Convert raw YOLOX model output to a DetectionResult.

    Parameters
    ----------
    outputs : list[Tensor | None]
        Direct output of yolox.utils.postprocess — one element per image.
    ratio : float | tuple | ndarray
        Scale factor from ValTransform.
    cage_id : str
    inference_start_ns : int
        Value of time.perf_counter_ns() captured before the forward pass.
    """
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)

    if outputs[0] is None:
        return DetectionResult(
            cage_id      = cage_id,
            timestamp    = datetime.now(tz=timezone.utc),
            mouse_count  = 0,
            water        = LevelReading.unknown(),
            food         = LevelReading.unknown(),
            bedding      = BeddingReading.not_detected(),
            inference_ms = inference_ms,
        )

    detections = outputs[0].cpu().numpy()

    # Normalise ratio safely
    if isinstance(ratio, (tuple, list, np.ndarray)):
        ratio = float(np.array(ratio).flatten()[0])
    if ratio is None or not np.isfinite(ratio) or ratio == 0:
        ratio = 1.0
    ratio = float(ratio)

    # YOLOX output: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
    boxes   = np.nan_to_num(detections[:, :4] / ratio, nan=0.0, posinf=0.0, neginf=0.0)
    scores  = detections[:, 4] * detections[:, 5]
    classes = detections[:, 6].astype(int)

    yolo_class_map = CONFIG["yolo_class_map"]
    mouse_id = next((k for k, v in yolo_class_map.items() if v == "mouse"), 0)

    mouse_count                = int(np.sum(classes == mouse_id))
    water_reading, water_bbox  = _best_level(boxes, scores, classes, WATER_CLASS_IDS)
    food_reading,  food_bbox   = _best_level(boxes, scores, classes, FOOD_CLASS_IDS)
    bedding_reading, bedding_bbox = _bedding_result(boxes, scores, classes)

    return DetectionResult(
        cage_id      = cage_id,
        timestamp    = datetime.now(tz=timezone.utc),
        mouse_count  = mouse_count,
        water        = water_reading,
        food         = food_reading,
        bedding      = bedding_reading,
        inference_ms = inference_ms,
        water_bbox   = water_bbox,
        food_bbox    = food_bbox,
        bedding_bbox = bedding_bbox,
    )


# ---------------------------------------------------------------------------
# Bedding helper
# ---------------------------------------------------------------------------

def _bedding_result(
    boxes:   np.ndarray,
    scores:  np.ndarray,
    classes: np.ndarray,
) -> tuple[BeddingReading, Optional[BoundingBox]]:
    """
    Aggregate all bedding detections (class 9).

    area_pct = sum(bbox_areas) / frame_area * 100
    condition = BAD if area_pct >= threshold, else GOOD

    Returns the highest-confidence single bbox for visualisation.
    """
    threshold = CONFIG["bedding"]["area_threshold"]

    bedding_boxes  = []
    best_conf      = -1.0
    best_box       = None

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) != BEDDING_CLASS_ID:
            continue
        x1, y1, x2, y2 = box
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        bedding_boxes.append(w * h)

        if float(score) > best_conf:
            best_conf = float(score)
            best_box  = box

    if not bedding_boxes:
        return BeddingReading.not_detected(), None

    total_area_pct = min(100.0, sum(bedding_boxes) / _FRAME_AREA * 100.0)
    reading = BeddingReading.from_area_pct(total_area_pct)

    bbox = BoundingBox(
        x1   = float(best_box[0]),
        y1   = float(best_box[1]),
        x2   = float(best_box[2]),
        y2   = float(best_box[3]),
        conf = best_conf,
    )

    logger.debug(
        "Bedding: total_area_pct=%.1f%%  condition=%s  threshold=%.0f%%",
        total_area_pct, reading.condition, threshold,
    )
    return reading, bbox


# ---------------------------------------------------------------------------
# Water / food level helper (unchanged)
# ---------------------------------------------------------------------------

def _best_level(
    boxes:     np.ndarray,
    scores:    np.ndarray,
    classes:   np.ndarray,
    class_ids: set[int],
) -> tuple[LevelReading, Optional[BoundingBox]]:
    """
    Among all detections belonging to class_ids, pick the highest-confidence
    one and build a LevelReading from its class ID.

    Additional spatial filter for food: reject detections in the top 30 % of
    the frame (cy < 0.30) — those are almost always false positives from cage
    wire or background.
    """
    IMG_SIZE    = 640.0
    is_food     = bool(class_ids & FOOD_CLASS_IDS)

    best_conf   = -1.0
    best_cls    = None
    best_box    = None

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in class_ids:
            continue

        if is_food:
            cy = (box[1] + box[3]) / 2 / IMG_SIZE
            if cy < 0.30:
                logger.debug("Rejected food detection: cy=%.3f (too high in frame)", cy)
                continue

        if float(score) > best_conf:
            best_conf = float(score)
            best_cls  = int(cls)
            best_box  = box

    if best_cls is None:
        return LevelReading.unknown(), None

    reading = LevelReading.from_class_id(best_cls)
    bbox    = BoundingBox(
        x1   = float(best_box[0]),
        y1   = float(best_box[1]),
        x2   = float(best_box[2]),
        y2   = float(best_box[3]),
        conf = best_conf,
    )
    return reading, bbox