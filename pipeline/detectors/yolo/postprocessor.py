# pipeline/detectors/yolo/postprocessor.py
"""
Post-processing helpers for YOLOX raw output.

parse_yolox_results() is the main entry-point called by YOLODetector.
Everything here is stateless — pure functions only.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.config_loader import CONFIG
from core.schemas import BoundingBox, DetectionResult, LevelReading
from core.exceptions import InferenceError

logger = logging.getLogger("vivarium.postprocessor")

# Class ID groups — derived from YOLO_CLASS_MAP in config
WATER_CLASS_IDS = {1, 2, 3, 4}
FOOD_CLASS_IDS  = {5, 6, 7, 8}


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
        Direct output of yolox.utils.postprocess — one element per image in batch.
    ratio : float | tuple | ndarray
        Scale factor from ValTransform used to map coords back to frame space.
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
    # Divide by ratio to recover original-frame pixel coordinates.
    boxes   = np.nan_to_num(detections[:, :4] / ratio, nan=0.0, posinf=0.0, neginf=0.0)
    scores  = detections[:, 4] * detections[:, 5]   # obj_conf × cls_conf
    classes = detections[:, 6].astype(int)

    yolo_class_map = CONFIG["yolo_class_map"]
    # Resolve the mouse class ID dynamically from config
    mouse_id = next((k for k, v in yolo_class_map.items() if v == "mouse"), 0)

    mouse_count                = int(np.sum(classes == mouse_id))
    water_reading, water_bbox  = _best_level(boxes, scores, classes, WATER_CLASS_IDS)
    food_reading,  food_bbox   = _best_level(boxes, scores, classes, FOOD_CLASS_IDS)

    return DetectionResult(
        cage_id      = cage_id,
        timestamp    = datetime.now(tz=timezone.utc),
        mouse_count  = mouse_count,
        water        = water_reading,
        food         = food_reading,
        inference_ms = inference_ms,
        water_bbox   = water_bbox,
        food_bbox    = food_bbox,
    )


# ---------------------------------------------------------------------------
# Internal helpers
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