# pipeline/pipeline.py
"""
Single unified inference pipeline.

Backend matrix
──────────────
  "yolo"     → YOLOX detector only; discrete class-based readings
  "yolo_psp" → YOLOX for mouse count + bboxes; PSPNet for continuous levels
  "ssd"      → SSD MobileNet (legacy)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.config_loader import CONFIG
from core.exceptions import VivariumCVError
from core.schemas import DetectionResult, LevelReading
from pipeline.annotator.factory import get_annotator
from pipeline.detectors.factory import get_detector
from pipeline.preprocessors.background_subtractor import BackgroundSubtractor
from pipeline.preprocessors.frame_preprocessor import FramePreprocessor

logger = logging.getLogger("vivarium.pipeline")

_FOOD_ROI_FALLBACK = (0.40, 0.55, 0.70, 0.90)
_MIN_CROP_PX       = 20


class InferencePipeline:

    def __init__(
        self,
        cage_type: str = "default",
        backend: Optional[str] = None,
    ) -> None:
        self._backend    = (backend or CONFIG["backend"]).lower()
        self._cage_type  = cage_type
        self._annotator  = get_annotator()

        self._preprocessor  = FramePreprocessor(cage_type=cage_type)
        self._bg_subtractor = BackgroundSubtractor()
        self._detector      = get_detector(cage_type=cage_type, backend=self._backend)

        # PSPNet — only for yolo_psp backend
        self._estimator: Optional[object] = None
        if self._backend == "yolo_psp":
            self._estimator = self._load_pspnet()

        self._detector.warmup()
        logger.info("InferencePipeline ready — backend=%s  cage=%s", self._backend, cage_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir: str = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame for cage '{cage_id}'.")

        # Step 1 — YOLOX / SSD detection
        yolo_result = self._detector.detect(frame=frame, cage_id=cage_id)

        # Step 2 — level estimation
        if self._backend == "yolo_psp" and self._estimator is not None:
            water = self._water_level(frame, yolo_result)
            food  = self._food_level(frame,  yolo_result)
        else:
            if self._backend == "yolo_psp":
                logger.warning(
                    "PSPNet not loaded — using YOLOX discrete buckets. "
                    "Set PSP_WATER_WEIGHTS and PSP_FOOD_WEIGHTS in .env."
                )
            water = yolo_result.water
            food  = yolo_result.food

        result = DetectionResult(
            cage_id      = cage_id,
            timestamp    = datetime.now(tz=timezone.utc),
            mouse_count  = yolo_result.mouse_count,
            water        = water,
            food         = food,
            inference_ms = yolo_result.inference_ms,
            water_bbox   = yolo_result.water_bbox,
            food_bbox    = yolo_result.food_bbox,
        )

        if save_flagged and _is_critical(result):
            annotated  = self._annotator.draw(frame, result)
            image_path = _save_frame(annotated, cage_id, output_dir)
            result     = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self._bg_subtractor.set_reference(self._preprocessor.resize(frame))

    def has_motion(self, frame: np.ndarray) -> bool:
        if not self._bg_subtractor.has_reference():
            return True
        return self._bg_subtractor.has_motion(self._preprocessor.resize(frame))

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self._detector.detect(frame=frame, cage_id="__debug__")
        return self._annotator.draw(frame, result)

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # PSPNet helpers (yolo_psp only)
    # ------------------------------------------------------------------

    def _load_pspnet(self):
        from pipeline.measurers.pspnet_measurer import LevelEstimator
        cfg     = CONFIG["pspnet"]
        water_w = cfg["water_weights"]
        food_w  = cfg["food_weights"]

        if not water_w and not food_w:
            logger.warning(
                "PSP_WATER_WEIGHTS / PSP_FOOD_WEIGHTS not set — "
                "PSPNet disabled; falling back to YOLOX discrete buckets."
            )
            return None

        try:
            estimator = LevelEstimator(
                water_weights = water_w,
                food_weights  = food_w,
                backbone      = cfg["backbone"],
                device        = CONFIG["device"],
            )
            logger.info("PSPNet loaded — water=%s  food=%s  backbone=%s",
                        water_w, food_w, cfg["backbone"])
            return estimator
        except Exception as exc:
            logger.error("PSPNet failed to load: %s", exc)
            return None

    def _water_level(self, frame: np.ndarray, yolo_result: DetectionResult) -> LevelReading:
        from pipeline.measurers.pspnet_measurer import NO_CONTAINER_SENTINEL
        try:
            pct, status, _ = self._estimator.estimate_water(frame)
            if pct == NO_CONTAINER_SENTINEL:
                logger.warning("Water PSPNet: no container — YOLOX fallback")
                return yolo_result.water
            if pct >= 97.0 and yolo_result.water.status != "OK":
                logger.warning("Water PSPNet saturated — YOLOX fallback")
                return yolo_result.water
            return LevelReading(pct=round(pct, 2), status=status)
        except Exception as exc:
            logger.error("PSPNet water failed: %s", exc)
            return yolo_result.water if CONFIG["pspnet"]["fallback_to_yolox"] \
                else LevelReading(pct=0.0, status="CRITICAL")

    def _food_level(self, frame: np.ndarray, yolo_result: DetectionResult) -> LevelReading:
        from pipeline.measurers.pspnet_measurer import NO_CONTAINER_SENTINEL
        try:
            crop = self._get_food_crop(frame, yolo_result)
            if crop is None:
                return yolo_result.food
            pct, status, _ = self._estimator.estimate_food(crop)
            if pct == NO_CONTAINER_SENTINEL:
                logger.warning("Food PSPNet: no container — YOLOX fallback")
                return yolo_result.food
            if pct > 40.0 and yolo_result.food.status == "CRITICAL":
                logger.warning("Food PSPNet bedding confusion — YOLOX fallback")
                return yolo_result.food
            if pct >= 97.0 and yolo_result.food.status != "OK":
                logger.warning("Food PSPNet saturated — YOLOX fallback")
                return yolo_result.food
            return LevelReading(pct=round(pct, 2), status=status)
        except Exception as exc:
            logger.error("PSPNet food failed: %s", exc)
            return yolo_result.food if CONFIG["pspnet"]["fallback_to_yolox"] \
                else LevelReading(pct=0.0, status="CRITICAL")

    def _get_food_crop(self, frame: np.ndarray, yolo_result: DetectionResult) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        bbox = yolo_result.food_bbox
        if bbox is not None:
            x1 = max(0, int(bbox.x1)); y1 = max(0, int(bbox.y1))
            x2 = min(w, int(bbox.x2)); y2 = min(h, int(bbox.y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and (x2 - x1) >= _MIN_CROP_PX and (y2 - y1) >= _MIN_CROP_PX:
                return crop
        x1f, y1f, x2f, y2f = _FOOD_ROI_FALLBACK
        rx1, ry1 = int(x1f * w), int(y1f * h)
        rx2, ry2 = int(x2f * w), int(y2f * h)
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size > 0 and (rx2 - rx1) >= _MIN_CROP_PX and (ry2 - ry1) >= _MIN_CROP_PX:
            return crop
        return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_critical(result: DetectionResult) -> bool:
    return result.water.status == "CRITICAL" or result.food.status == "CRITICAL"


def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path