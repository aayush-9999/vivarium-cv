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

_MIN_BEDDING_CONF = 0.25


def parse_yolox_results(
    outputs,
    ratio,
    cage_id: str,
    inference_start_ns: int,
    input_size: tuple[int, int] = (416, 416),
    orig_size: tuple[int, int] = None,
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

    # ── Letterbox inverse ─────────────────────────────────────────────────────
    if orig_size is not None:
        orig_h, orig_w = orig_size
        ih, iw         = input_size
        ratio          = min(iw / orig_w, ih / orig_h)   # recompute correctly
        pad_left       = (iw - orig_w * ratio) / 2
        pad_top        = (ih - orig_h * ratio) / 2
    else:
        if isinstance(ratio, (tuple, list, np.ndarray)):
            ratio = float(np.array(ratio).flatten()[0])
        if ratio is None or not np.isfinite(ratio) or ratio == 0:
            ratio = 1.0
        ratio    = float(ratio)
        pad_top  = 0.0
        pad_left = 0.0

    raw_boxes = detections[:, :4].copy()
    raw_boxes[:, 0] -= pad_left   # x1
    raw_boxes[:, 1] -= pad_top    # y1
    raw_boxes[:, 2] -= pad_left   # x2
    raw_boxes[:, 3] -= pad_top    # y2
    boxes = np.nan_to_num(raw_boxes / ratio, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Clip to original frame bounds ─────────────────────────────────────────
    if orig_size is not None:
        orig_h, orig_w = orig_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)  # y2
    # ─────────────────────────────────────────────────────────────────────────

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

    # ── Water / food ──────────────────────────────────────────────────────────
    water_reading, water_bbox = _best_level(boxes, scores, classes, WATER_CLASS_IDS)
    food_reading,  food_bbox  = _best_level(boxes, scores, classes, FOOD_CLASS_IDS)

    # ── Bedding ───────────────────────────────────────────────────────────────
    bedding_reading, bedding_bbox = _bedding_result(boxes, scores, classes, orig_size)

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
    is_food   = bool(class_ids & FOOD_CLASS_IDS)
    best_conf = -1.0
    best_cls  = None
    best_box  = None

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in class_ids:
            continue
        if is_food:
            # use normalised cy relative to box array (already in orig coords)
            # skip food detections in top 20% of frame
            pass   # removed hardcoded IMG_SIZE=640 filter — coords now in orig space
        if float(score) > best_conf:
            best_conf = float(score)
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


def _bedding_result(
    boxes:     np.ndarray,
    scores:    np.ndarray,
    classes:   np.ndarray,
    orig_size: tuple[int, int] = None,
) -> tuple[BeddingReading, Optional[BoundingBox]]:
    best_conf  = -1.0
    best_box   = None
    worst_cls  = None
    total_area = 0.0

    # Frame area in original coords for area_pct calculation
    if orig_size is not None:
        frame_area = float(orig_size[0] * orig_size[1])
    else:
        frame_area = float(640 * 640)

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in BEDDING_CLASS_IDS:
            continue
        if float(score) < _MIN_BEDDING_CONF:
            continue

        x1, y1, x2, y2 = box
        total_area += max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        if worst_cls is None or int(cls) < worst_cls:
            worst_cls = int(cls)

        if float(score) > best_conf:
            best_conf = float(score)
            best_box  = box

    if worst_cls is None:
        return BeddingReading.not_detected(), None

    area_pct = min(100.0, total_area / frame_area * 100.0)
    reading  = BeddingReading.from_class_id(worst_cls, area_pct)
    bbox     = BoundingBox(
        x1=float(best_box[0]), y1=float(best_box[1]),
        x2=float(best_box[2]), y2=float(best_box[3]),
        conf=best_conf,
    )
    return reading, bbox