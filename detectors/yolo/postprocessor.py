# detectors/yolo/postprocessor.py
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

CLASS_MOUSE     = 0
WATER_CLASS_IDS = {1, 2, 3, 4}
FOOD_CLASS_IDS  = {5, 6, 7, 8}

# detectors/yolo/postprocessor.py — key differences

def parse_yolox_results(outputs, ratio, cage_id, inference_start_ns):
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)

    # outputs[0] is None when nothing is detected (YOLOx returns list of per-image results)
    if outputs[0] is None:
        return DetectionResult(
            cage_id=cage_id,
            timestamp=datetime.now(tz=timezone.utc),
            mouse_count=0,
            water=LevelReading.unknown(),
            food=LevelReading.unknown(),
            inference_ms=inference_ms,
        )

    detections = outputs[0].cpu().numpy()

    # YOLOx output: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
    # Coordinates are in padded-input space — divide by ratio to get original frame coords
# normalize ratio safely
# FIX ratio safely
# normalize ratio safely
    if isinstance(ratio, (tuple, list, np.ndarray)):
        ratio = np.array(ratio).flatten()[0]

    # 🔥 critical safety
    if ratio is None or not np.isfinite(ratio) or ratio == 0:
        ratio = 1.0

    ratio = float(ratio)

    boxes = detections[:, :4] / ratio

    # 🔥 sanitize boxes (prevents future crashes)
    boxes = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    scores  = detections[:, 4] * detections[:, 5]  # obj_conf × cls_conf = final confidence
    classes = detections[:, 6].astype(int)

    from core.config import YOLO_CLASS_MAP

    # find mouse class id dynamically
    MOUSE_ID = next(k for k, v in YOLO_CLASS_MAP.items() if v == "mouse")

    mouse_count = int(np.sum(classes == MOUSE_ID))
    water_reading, water_bbox = _best_level_reading_yolox(boxes, scores, classes, WATER_CLASS_IDS)
    food_reading, food_bbox   = _best_level_reading_yolox(boxes, scores, classes, FOOD_CLASS_IDS)

    return DetectionResult(
        cage_id=cage_id,
        timestamp=datetime.now(tz=timezone.utc),
        mouse_count=mouse_count,
        water=water_reading,
        food=food_reading,
        inference_ms=inference_ms,
        water_bbox=water_bbox,
        food_bbox=food_bbox,
    )

def parse_yolo_results(
    results: list["Results"],
    cage_id: str,
    inference_start_ns: int,
) -> DetectionResult:
    if not results:
        raise InferenceError("YOLO returned empty results list.")

    r            = results[0]
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
    if r.boxes is None or len(r.boxes) == 0:
        return LevelReading.unknown(), None

    try:
        boxes   = r.boxes.xyxy.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        raise InferenceError(f"Failed to parse YOLO boxes: {e}") from e

    best_conf = -1.0
    best_cls  = None
    best_box  = None

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

def _best_level_reading_yolox(boxes, scores, classes, class_ids):
    best_conf = -1.0
    best_cls = None
    best_box = None

    img_size = 640.0

    # Determine type by checking which set this is
    is_water = bool(class_ids & WATER_CLASS_IDS)
    is_food  = bool(class_ids & FOOD_CLASS_IDS)

    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) not in class_ids:
            continue

        x1, y1, x2, y2 = box
        bw   = (x2 - x1) / img_size
        bh   = (y2 - y1) / img_size
        area = bw * bh
        cx   = (x1 + x2) / 2 / img_size
        cy   = (y1 + y2) / 2 / img_size

        print(f"  cls={cls} score={score:.3f} cx={cx:.3f} cy={cy:.3f} area={area:.3f} is_water={is_water}")

        # Food must not be in top 30% of frame
        if is_food and cy < 0.30:
            print(f"  → REJECTED food too high cy={cy:.3f}")
            continue

        if float(score) > best_conf:
            best_conf = float(score)
            best_cls  = int(cls)
            best_box  = box

    if best_cls is None:
        return LevelReading.unknown(), None

    reading = LevelReading.from_class_id(best_cls)
    bbox = BoundingBox(
        x1=float(best_box[0]), y1=float(best_box[1]),
        x2=float(best_box[2]), y2=float(best_box[3]),
        conf=best_conf,
    )
    return reading, bbox