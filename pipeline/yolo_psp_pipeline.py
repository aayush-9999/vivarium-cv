# pipeline/yolo_psp_pipeline.py
"""
Hybrid pipeline: YOLOX bbox detection + PSPNet level estimation.

IMPORTANT — PSPNet was trained on FULL 640x640 frames, not crops.
So PSPNet always receives the full frame. YOLOX is only used to:
    - Count mice (class 0)
    - Confirm a container is present (bbox existence check)
    - Provide bbox location for the draw overlay

The level % comes entirely from PSPNet on the full frame.
from_class_id() / YOLO bucket midpoints are never used for level.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Optional
from datetime import datetime, timezone

from core.schemas import DetectionResult, LevelReading, BoundingBox
from core.exceptions import VivariumCVError
from pipeline.preprocessors.frame_preprocessor import FramePreprocessor
from pipeline.preprocessors.background_subtractor import BackgroundSubtractor
from pipeline.detectors.yolo.yolo_detector import YOLODetector
from pipeline.measurers.pspnet_measurer import (
    LevelEstimator, pct_to_status, NO_CONTAINER_SENTINEL
)

import logging
logger = logging.getLogger("vivarium.pipeline.hybrid")

MOUSE_CLASS     = 0
WATER_CLASS_IDS = {1, 2, 3, 4}
FOOD_CLASS_IDS  = {5, 6, 7, 8}


class YOLOPSPPipeline:
    """
    YOLOX detects WHERE containers are (bbox) and counts mice.
    PSPNet runs on the FULL FRAME to measure fill levels.

    PSPNet was trained on full 640x640 frames — passing a crop would
    produce wrong results because the model never saw crops during training.

    LevelReading is always built from PSPNet pct output.
    from_class_id() / YOLO discrete buckets are never used for level.
    """

    def __init__(
        self,
        cage_type:         str  = "default",
        water_psp_weights: Optional[str] = None,
        food_psp_weights:  Optional[str] = None,
        psp_backbone:      str  = "resnet50",
        fallback_to_yolox: bool = True,
    ):
        weights = os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")
        device  = os.getenv("YOLO_DEVICE",  "cpu")

        self.fallback_to_yolox = fallback_to_yolox
        self.preprocessor      = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor     = BackgroundSubtractor()
        self.detector          = YOLODetector(weights_path=weights, device=device)

        # Load PSPNet — weights from args or .env
        water_w = water_psp_weights or os.getenv("PSP_WATER_WEIGHTS")
        food_w  = food_psp_weights  or os.getenv("PSP_FOOD_WEIGHTS")
        backbone = psp_backbone or os.getenv("PSP_BACKBONE", "resnet50")

        self.estimator: Optional[LevelEstimator] = None

        if water_w or food_w:
            try:
                self.estimator = LevelEstimator(
                    water_weights=water_w,
                    food_weights=food_w,
                    backbone=backbone,
                    device=device,
                )
                logger.info(
                    "PSPNet loaded — water=%s  food=%s  backbone=%s",
                    water_w, food_w, backbone
                )
            except Exception as e:
                logger.error("PSPNet failed to load: %s", e)
                self.estimator = None
        else:
            logger.warning(
                "PSP_WATER_WEIGHTS / PSP_FOOD_WEIGHTS not set in .env — "
                "PSPNet will not run. Set these to use continuous level estimation."
            )

        self.detector.warmup()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        frame:        np.ndarray,
        cage_id:      str,
        save_flagged: bool = False,
        output_dir:   str  = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame for cage '{cage_id}'.")

        # Step 1: YOLOX — mouse count + bbox locations
        # Level class IDs (1-8) are intentionally ignored here
        yolo_result = self.detector.detect(frame=frame, cage_id=cage_id)

        # Step 2: PSPNet on FULL FRAME
        if self.estimator is not None:
            water_reading = self._run_water_psp(frame, yolo_result)
            food_reading  = self._run_food_psp(frame, yolo_result)
        else:
            # No PSPNet — fall back to YOLOX discrete buckets with clear warning
            logger.warning(
                "PSPNet not loaded — falling back to YOLOX class-based level "
                "(discrete buckets: 7.5/25.0/57.5/90.0). "
                "Set PSP_WATER_WEIGHTS and PSP_FOOD_WEIGHTS in .env."
            )
            water_reading = yolo_result.water
            food_reading  = yolo_result.food

        result = DetectionResult(
            cage_id=cage_id,
            timestamp=datetime.now(tz=timezone.utc),
            mouse_count=yolo_result.mouse_count,
            water=water_reading,
            food=food_reading,
            inference_ms=yolo_result.inference_ms,
            water_bbox=yolo_result.water_bbox,
            food_bbox=yolo_result.food_bbox,
        )

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

    # ── PSPNet on full frame ──────────────────────────────────────────────────

    def _run_water_psp(
        self,
        frame:       np.ndarray,
        yolo_result: DetectionResult,
    ) -> LevelReading:
        """
        Run PSPNet water model on the FULL frame.
        Falls back to YOLOX if sentinel returned (no container detected)
        or if PSPNet throws an exception.
        """
        try:
            pct, status, _ = self.estimator.estimate_water(frame)
            if pct == NO_CONTAINER_SENTINEL:
                logger.warning(
                    "Water PSPNet: no container detected in frame — "
                    "falling back to YOLOX reading"
                )
                return yolo_result.water

            # Saturated reading — PSPNet uncertain, trust YOLOX if it disagrees
            if pct >= 97.0 and yolo_result.water.status != "OK":
                logger.warning(
                    "Water PSPNet returned saturated 97%% but YOLOX says %s "
                    "— falling back to YOLOX reading.",
                    yolo_result.water.status,
                )
                return yolo_result.water

            logger.debug("Water PSPNet: %.2f%% [%s]", pct, status)
            return LevelReading(pct=round(pct, 2), status=status)

        except Exception as e:
            logger.error("PSPNet water failed: %s", e)
            if self.fallback_to_yolox:
                return yolo_result.water
            return LevelReading(pct=0.0, status="CRITICAL")

    # Tune these to your cage: (x1_frac, y1_frac, x2_frac, y2_frac) of 640x640 frame
    _FOOD_ROI_FALLBACK = (0.40, 0.55, 0.70, 0.90)
    _MIN_CROP_PX = 20

    def _run_food_psp(
        self,
        frame:       np.ndarray,
        yolo_result: DetectionResult,
    ) -> LevelReading:
        try:
            crop = None
            bbox = yolo_result.food_bbox

            # Step 1: try YOLOX bbox crop
            if bbox is not None:
                h, w = frame.shape[:2]
                x1 = max(0, int(bbox.x1)); y1 = max(0, int(bbox.y1))
                x2 = min(w, int(bbox.x2)); y2 = min(h, int(bbox.y2))
                candidate = frame[y1:y2, x1:x2]
                if candidate.size > 0 and (x2 - x1) >= self._MIN_CROP_PX and (y2 - y1) >= self._MIN_CROP_PX:
                    crop = candidate
                else:
                    logger.debug("Food bbox crop degenerate — trying fixed ROI fallback")

            # Step 2: fixed ROI fallback when YOLOX missed the hopper
            if crop is None:
                h, w = frame.shape[:2]
                x1f, y1f, x2f, y2f = self._FOOD_ROI_FALLBACK
                rx1, ry1 = int(x1f * w), int(y1f * h)
                rx2, ry2 = int(x2f * w), int(y2f * h)
                candidate = frame[ry1:ry2, rx1:rx2]
                if candidate.size > 0 and (rx2 - rx1) >= self._MIN_CROP_PX and (ry2 - ry1) >= self._MIN_CROP_PX:
                    logger.info(
                        "Food bbox missing — using fixed ROI fallback (%d,%d,%d,%d)",
                        rx1, ry1, rx2, ry2,
                    )
                    crop = candidate
                else:
                    logger.warning("Fixed ROI fallback also degenerate — using YOLOX reading")
                    return yolo_result.food

            pct, status, _ = self.estimator.estimate_food(crop)

            # Sentinel: PSPNet found no container pixels at all
            if pct == NO_CONTAINER_SENTINEL:
                logger.warning(
                    "Food PSPNet: no container detected — falling back to YOLOX reading"
                )
                return yolo_result.food

            # Sanity gate: PSPNet says >40% full but YOLOX says CRITICAL
            # → bedding/background confusion, trust YOLOX
            if pct > 40.0 and yolo_result.food.status == "CRITICAL":
                logger.warning(
                    "Food PSPNet returned %.1f%% [%s] but YOLOX says CRITICAL "
                    "— falling back to YOLOX (bedding confusion).",
                    pct, status,
                )
                return yolo_result.food

            # Saturated reading — uncertain, fall back to YOLOX if it disagrees
            if pct >= 97.0 and yolo_result.food.status != "OK":
                logger.warning(
                    "Food PSPNet returned saturated 97%% but YOLOX says %s "
                    "— falling back to YOLOX reading.",
                    yolo_result.food.status,
                )
                return yolo_result.food

            logger.debug("Food PSPNet: %.2f%% [%s]", pct, status)
            return LevelReading(pct=round(pct, 2), status=status)

        except Exception as e:
            logger.error("PSPNet food failed: %s", e)
            if self.fallback_to_yolox:
                return yolo_result.food
            return LevelReading(pct=0.0, status="CRITICAL")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _should_flag(result: DetectionResult) -> bool:
        return (result.water.status == "CRITICAL"
                or result.food.status == "CRITICAL")

    @staticmethod
    def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "OK":       (80,  200,  80),
    "LOW":      (0,   180, 255),
    "CRITICAL": (0,     0, 255),
}


def _draw_result(img: np.ndarray, result: DetectionResult) -> np.ndarray:
    viz = img.copy()

    for bbox, label in [
        (result.water_bbox, f"water {result.water.pct:.1f}% [{result.water.status}]"),
        (result.food_bbox,  f"food  {result.food.pct:.1f}%  [{result.food.status}]"),
    ]:
        if bbox is not None:
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            color = (255, 180, 0) if "water" in label else (0, 200, 80)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz, f"{label} ({bbox.conf:.2f})",
                        (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    summary = [
        ("OK",                f"Mice : {result.mouse_count}"),
        (result.water.status, f"Water: {result.water.pct:.1f}%  [{result.water.status}]"),
        (result.food.status,  f"Food : {result.food.pct:.1f}%  [{result.food.status}]"),
    ]
    for i, (status, line) in enumerate(summary):
        color = STATUS_COLORS.get(status, (160, 160, 160))
        y = 20 + i * 22
        cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)

    return viz