# pipeline/detectors/yolo/postprocessor.py

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.config_loader import CONFIG
from core.schemas import BoundingBox, BeddingReading, DetectionResult, LevelReading

logger = logging.getLogger("vivarium.postprocessor")

# ── Class ID schema (13 classes total) ───────────────────────────────────────
MOUSE_CLASS_ID    = 0
WATER_CLASS_IDS   = {1, 2, 3, 4}
FOOD_CLASS_IDS    = {5, 6, 7, 8}
BEDDING_CLASS_IDS = {9, 10, 11, 12}

_FRAME_AREA       = 640 * 640
_MIN_BEDDING_CONF = 0.25


def parse_yolox_results(
    outputs,
    ratio,
    cage_id: str,
    inference_start_ns: int,
) -> DetectionResult:
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)

    if outputs[0] is None:
        return DetectionResult(
            cage_id      = cage_id,
            timestamp    = datetime.now(tz=timezone.utc),
            mouse_count  = 0,
            mouse_bboxes = [],
            water        = LevelReading.unknown(),
            food         = LevelReading.unknown(),
            bedding      = BeddingReading.not_detected(),
            inference_ms = inference_ms,
        )

    detections = outputs[0].cpu().numpy()

    if isinstance(ratio, (tuple, list, np.ndarray)):
        ratio = float(np.array(ratio).flatten()[0])
    if ratio is None or not np.isfinite(ratio) or ratio == 0:
        ratio = 1.0
    ratio = float(ratio)

    boxes   = np.nan_to_num(detections[:, :4] / ratio, nan=0.0, posinf=0.0, neginf=0.0)
    scores  = detections[:, 4] * detections[:, 5]
    classes = detections[:, 6].astype(int)

    # ── Mouse ─────────────────────────────────────────────────────────────────
    mouse_bboxes: list[BoundingBox] = []
    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) == MOUSE_CLASS_ID:
            mouse_bboxes.append(BoundingBox(
                x1=float(box[0]), y1=float(box[1]),
                x2=float(box[2]), y2=float(box[3]),
                conf=float(score),
            ))

    # ── Water / food — pass raw class ID, schema owns the mapping ────────────
    water_reading, water_bbox = _best_level(boxes, scores, classes, WATER_CLASS_IDS)
    food_reading,  food_bbox  = _best_level(boxes, scores, classes, FOOD_CLASS_IDS)

    # ── Bedding ───────────────────────────────────────────────────────────────
    bedding_reading, bedding_bbox = _bedding_result(boxes, scores, classes)

    return DetectionResult(
        cage_id      = cage_id,
        timestamp    = datetime.now(tz=timezone.utc),
        mouse_count  = len(mouse_bboxes),
        mouse_bboxes = mouse_bboxes,
        water        = water_reading,
        food         = food_reading,
        bedding      = bedding_reading,
        inference_ms = inference_ms,
        water_bbox   = water_bbox,
        food_bbox    = food_bbox,
        bedding_bbox = bedding_bbox,
    )


def _best_level(
    boxes:     np.ndarray,
    scores:    np.ndarray,
    classes:   np.ndarray,
    class_ids: set[int],
) -> tuple[LevelReading, Optional[BoundingBox]]:
    """
    Pick the highest-confidence detection from class_ids and return a
    LevelReading. Raw class ID is passed directly to LevelReading.from_class_id()
    because CLASS_TO_LEVEL in core/config.py is keyed on raw IDs (1-8).
    """
    IMG_SIZE  = 640.0
    is_food   = bool(class_ids & FOOD_CLASS_IDS)
    best_conf = -1.0
    best_cls  = None
    best_box  = None

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in class_ids:
            continue
        if is_food:
            cy = (box[1] + box[3]) / 2 / IMG_SIZE
            if cy < 0.30:
                logger.debug("Rejected food detection cy=%.3f (too high in frame)", cy)
                continue
        if float(score) > best_conf:
            best_conf = float(score)
            best_cls  = int(cls)
            best_box  = box

    if best_cls is None:
        return LevelReading.unknown(), None

    # Pass raw class ID — LevelReading.from_class_id() expects 1-8
    reading = LevelReading.from_class_id(best_cls)
    bbox    = BoundingBox(
        x1=float(best_box[0]), y1=float(best_box[1]),
        x2=float(best_box[2]), y2=float(best_box[3]),
        conf=best_conf,
    )
    return reading, bbox


def _bedding_result(
    boxes:   np.ndarray,
    scores:  np.ndarray,
    classes: np.ndarray,
) -> tuple[BeddingReading, Optional[BoundingBox]]:
    """
    Among all bedding detections, pick:
      - worst_cls: highest class ID seen (9=WORST → 12=PERFECT), used for condition
      - best bbox: highest-confidence detection, used for annotation only

    Raw class ID is passed directly to BeddingReading.from_class_id()
    because its mapping is keyed on raw IDs (9-12).
    """
    best_conf  = -1.0
    best_box   = None
    best_cls   = -1   # only used for bbox, initialised to avoid NameError
    worst_cls  = None  # highest raw class ID = worst condition
    total_area = 0.0

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in BEDDING_CLASS_IDS:
            continue
        if float(score) < _MIN_BEDDING_CONF:
            continue

        x1, y1, x2, y2 = box
        total_area += max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        # Worst condition = lowest raw class ID (9=WORST beats 12=PERFECT)
        if worst_cls is None or int(cls) < worst_cls:
            worst_cls = int(cls)

        if float(score) > best_conf:
            best_conf = float(score)
            best_cls  = int(cls)
            best_box  = box

    if worst_cls is None:
        return BeddingReading.not_detected(), None

    area_pct = min(100.0, total_area / _FRAME_AREA * 100.0)

    # Pass raw class ID — BeddingReading.from_class_id() expects 9-12
    reading = BeddingReading.from_class_id(worst_cls, area_pct)
    bbox    = BoundingBox(
        x1=float(best_box[0]), y1=float(best_box[1]),
        x2=float(best_box[2]), y2=float(best_box[3]),
        conf=best_conf,
    )
    return reading, bbox