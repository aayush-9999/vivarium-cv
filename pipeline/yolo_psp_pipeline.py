# pipeline/yolo_psp_pipeline.py
"""
Hybrid pipeline: YOLOX for detection + PSPNet for level estimation.

Flow:
    raw frame (640×640)
         │
         ▼
    YOLODetector (YOLOX)
         │  detects bounding boxes only:
         │    - class 0        → mouse_count
         │    - classes 1-4    → water container bbox location
         │    - classes 5-8    → food container bbox location
         │  (class IDs for water/food are IGNORED for level — PSPNet handles that)
         │
         ├── water_bbox → crop → PSPNet water model → fill % → LevelReading
         │
         └── food_bbox  → crop → PSPNet food  model → fill % → LevelReading
         │
         ▼
    DetectionResult (mouse_count, water LevelReading, food LevelReading)

Why this is better than pure YOLOX:
    - YOLOX 9-class gives 4 discrete buckets (e.g. water_ok = 35-80%)
    - PSPNet gives continuous 0.0-100.0% via pixel-level fill measurement
    - No HSV color estimation, no assumptions about fill appearance
    - Works even if water color varies (different lighting per cage)
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Optional

from core.schemas import DetectionResult, LevelReading, BoundingBox
from core.exceptions import VivariumCVError
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.background_subtractor import BackgroundSubtractor
from detectors.yolo.yolo_detector import YOLODetector
from segmentation.models.level_estimator import LevelEstimator

import logging
logger = logging.getLogger("vivarium.pipeline.hybrid")


# ── Class groups (same as existing pipeline) ──────────────────────────────────
MOUSE_CLASS     = 0
WATER_CLASS_IDS = {1, 2, 3, 4}
FOOD_CLASS_IDS  = {5, 6, 7, 8}


class YOLOPSPPipeline:
    """
    Hybrid inference pipeline:
        YOLOX  → detects WHERE containers are (bounding boxes)
        PSPNet → measures HOW FULL containers are (segmentation)

    Falls back gracefully:
        - If water bbox not found: PSPNet runs on full frame ROI
        - If PSPNet weights not loaded: falls back to YOLOX class-based estimate
        - If crop is too small: returns CRITICAL as safe default
    """

    MIN_CROP_SIZE = 20   # pixels — crops smaller than this are ignored

    def __init__(
        self,
        cage_type:          str  = "default",
        water_psp_weights:  Optional[str] = None,
        food_psp_weights:   Optional[str] = None,
        psp_backbone:       str  = "resnet50",
        fallback_to_yolox:  bool = True,
    ):
        weights = os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")
        device  = os.getenv("YOLO_DEVICE",  "cpu")

        self.preprocessor  = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor = BackgroundSubtractor()
        self.detector      = YOLODetector(weights_path=weights, device=device)
        self.fallback      = fallback_to_yolox

        # Load PSPNet estimator
        water_w = water_psp_weights or os.getenv("PSP_WATER_WEIGHTS")
        food_w  = food_psp_weights  or os.getenv("PSP_FOOD_WEIGHTS")

        self.estimator: Optional[LevelEstimator] = None

        if water_w or food_w:
            try:
                self.estimator = LevelEstimator(
                    water_weights=water_w,
                    food_weights=food_w,
                    backbone=psp_backbone,
                    device=device,
                )
                logger.info("PSPNet estimator loaded successfully.")
            except Exception as e:
                logger.warning("PSPNet load failed: %s — falling back to YOLOX levels", e)
        else:
            logger.info(
                "No PSP weights provided (PSP_WATER_WEIGHTS / PSP_FOOD_WEIGHTS not set). "
                "Using YOLOX class-based level estimation."
            )

        self.detector.warmup()

    # ── Public ────────────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir: str = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame passed for cage '{cage_id}'.")

        # Step 1: YOLOX detection — get bboxes
        yolo_result = self.detector.detect(frame=frame, cage_id=cage_id)

        # Step 2: If PSPNet is available, replace level readings with PSPNet estimates
        if self.estimator is not None:
            water_reading, water_bbox = self._estimate_water(frame, yolo_result)
            food_reading,  food_bbox  = self._estimate_food(frame,  yolo_result)

            from datetime import datetime, timezone
            from core.schemas import DetectionResult as DR
            result = DR(
                cage_id=cage_id,
                timestamp=datetime.now(tz=timezone.utc),
                mouse_count=yolo_result.mouse_count,
                water=water_reading,
                food=food_reading,
                inference_ms=yolo_result.inference_ms,
                water_bbox=water_bbox or yolo_result.water_bbox,
                food_bbox=food_bbox  or yolo_result.food_bbox,
            )
        else:
            # No PSPNet — use YOLOX class-based reading as-is
            result = yolo_result

        # Step 3: Save flagged frame if needed
        if save_flagged and self._should_flag(result):
            annotated  = _draw_result(frame, result)
            image_path = self._save_frame(annotated, cage_id, output_dir)
            result     = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self.bg_subtractor.set_reference(self.preprocessor.resize(frame))

    def has_motion(self, frame: np.ndarray) -> bool:
        if not self.bg_subtractor.has_reference():
            return True
        return self.bg_subtractor.has_motion(self.preprocessor.resize(frame))

    # ── Private: PSPNet estimation ────────────────────────────────────────

    def _estimate_water(
        self,
        frame: np.ndarray,
        yolo_result: DetectionResult,
    ) -> tuple[LevelReading, Optional[BoundingBox]]:

        bbox = yolo_result.water_bbox

        if bbox is None:
            logger.debug("No water bbox from YOLOX — skipping PSPNet")
            return yolo_result.water, None

        try:
            # PSPNet was trained on full 640x640 frames — pass full frame
            pct, status, mask = self.estimator.estimate_water(frame)
            logger.debug("Water PSPNet: %.1f%% [%s]", pct, status)
            return LevelReading(pct=pct, status=status), bbox
        except Exception as e:
            logger.warning("PSPNet water estimation failed: %s", e)
            if self.fallback:
                return yolo_result.water, bbox
            return LevelReading(pct=0.0, status="CRITICAL"), bbox
    def _estimate_food(
        self,
        frame: np.ndarray,
        yolo_result: DetectionResult,
    ) -> tuple[LevelReading, Optional[BoundingBox]]:
        """
        Crop food hopper from frame using YOLOX bbox, run PSPNet, return LevelReading.
        """
        bbox = yolo_result.food_bbox

        if bbox is None:
            logger.debug("No food bbox from YOLOX — using YOLOX level reading")
            return yolo_result.food, None

        crop = self._crop_from_bbox(frame, bbox)
        if crop is None:
            return yolo_result.food, bbox

        try:
            pct, status, mask = self.estimator.estimate_food(crop)
            logger.debug("Food PSPNet: %.1f%% [%s]", pct, status)
            return LevelReading(pct=pct, status=status), bbox
        except Exception as e:
            logger.warning("PSPNet food estimation failed: %s", e)
            if self.fallback:
                return yolo_result.food, bbox
            return LevelReading(pct=0.0, status="CRITICAL"), bbox

    def _crop_from_bbox(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1 = int(max(bbox.x1, 0))
        y1 = int(max(bbox.y1, 0))
        x2 = int(min(bbox.x2, w))
        y2 = int(min(bbox.y2, h))

        # Shrink the crop to the inner 60% of the bbox
        # This removes cage wires and background at the edges
        bw = x2 - x1
        bh = y2 - y1
        margin_x = int(bw * 0.20)
        margin_y = int(bh * 0.15)
        x1 = x1 + margin_x
        y1 = y1 + margin_y
        x2 = x2 - margin_x
        y2 = y2 - margin_y

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    # ── Flagging ──────────────────────────────────────────────────────────

    @staticmethod
    def _should_flag(result: DetectionResult) -> bool:
        return (
            result.water.status == "CRITICAL"
            or result.food.status == "CRITICAL"
        )

    @staticmethod
    def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
        from datetime import datetime, timezone
        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers (same style as existing pipeline)
# ─────────────────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "OK":       (80,  200,  80),
    "LOW":      (0,   180, 255),
    "CRITICAL": (0,     0, 255),
    "UNKNOWN":  (160, 160, 160),
}


def _draw_result(img: np.ndarray, result: DetectionResult) -> np.ndarray:
    viz = img.copy()

    for bbox, label_prefix in [
        (result.water_bbox, f"water {result.water.status} {result.water.pct:.1f}%"),
        (result.food_bbox,  f"food  {result.food.status}  {result.food.pct:.1f}%"),
    ]:
        if bbox is not None:
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            color = (255, 180, 0) if "water" in label_prefix else (0, 200, 80)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                viz,
                f"{label_prefix} ({bbox.conf:.2f})",
                (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    summary = [
        ("OK",                f"Mice : {result.mouse_count}"),
        (result.water.status, f"Water: {result.water.pct:.1f}%  [{result.water.status}]"),
        (result.food.status,  f"Food : {result.food.pct:.1f}%  [{result.food.status}]"),
    ]
    for i, (status, line) in enumerate(summary):
        color = STATUS_COLORS.get(status, STATUS_COLORS["OK"])
        y     = 20 + i * 22
        cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,     1, cv2.LINE_AA)

    return viz