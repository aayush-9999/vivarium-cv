# pipeline/pipeline.py
"""
Single unified inference pipeline.

The backend (YOLOX-only vs YOLOX+PSPNet vs SSD) is chosen at construction
time from CONFIG["backend"] (or an explicit override).  All callers — the
API, the orchestrator, the camera loop — use this one class; they never
import a backend-specific file.

Backend matrix
──────────────
  "yolo"     → YOLOX detector only; discrete 4-bucket level readings
  "yolo_psp" → YOLOX for mouse count + bboxes; PSPNet for continuous levels
  "ssd"      → SSD MobileNet (legacy; detector must be available)

PSPNet fallback chain (yolo_psp only, per container per frame)
──────────────────────────────────────────────────────────────
    PSPNet result
        ├─ NO_CONTAINER_SENTINEL (-1.0)          → YOLOX reading
        ├─ pct ≥ 97 % AND YOLOX ≠ OK            → YOLOX reading
        ├─ food pct > 40 % AND YOLOX = CRITICAL  → YOLOX reading  (bedding noise)
        └─ any Exception                         → YOLOX reading  (if fallback=True)
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

# Fixed fallback ROI for food when YOLOX misses the hopper (normalised fractions)
_FOOD_ROI_FALLBACK = (0.40, 0.55, 0.70, 0.90)   # x1_frac, y1_frac, x2_frac, y2_frac
_MIN_CROP_PX       = 20


class InferencePipeline:
    """
    Backend-agnostic inference pipeline.

    Parameters
    ----------
    cage_type : str
        Key into ROI_ZONES config (default "default").
    backend : str | None
        Override CONFIG["backend"].  One of "yolo", "yolo_psp", "ssd".
        If None, reads CONFIG["backend"].
    """

    def __init__(
        self,
        cage_type: str = "default",
        backend: Optional[str] = None,
    ) -> None:
        self._backend    = (backend or CONFIG["backend"]).lower()
        self._cage_type  = cage_type
        self._annotator  = get_annotator()

        # Shared components
        self._preprocessor  = FramePreprocessor(cage_type=cage_type)
        self._bg_subtractor = BackgroundSubtractor()

        # Backend-specific detector
        self._detector = get_detector(cage_type=cage_type, backend=self._backend)

        # PSPNet estimator — only loaded for yolo_psp backend
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
        """Run full inference on one frame, return DetectionResult."""
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame for cage '{cage_id}'.")

        # Step 1 — detector (always YOLOX or SSD)
        yolo_result = self._detector.detect(frame=frame, cage_id=cage_id)

        # Step 2 — level estimation
        if self._backend == "yolo_psp" and self._estimator is not None:
            water = self._water_level(frame, yolo_result)
            food  = self._food_level(frame,  yolo_result)
        else:
            # yolo / ssd / pspnet not loaded → use detector's own level readings
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
        """Store a clean background frame for motion gating."""
        self._bg_subtractor.set_reference(self._preprocessor.resize(frame))

    def has_motion(self, frame: np.ndarray) -> bool:
        """True if frame changed enough from reference to warrant inference."""
        if not self._bg_subtractor.has_reference():
            return True
        return self._bg_subtractor.has_motion(self._preprocessor.resize(frame))

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return annotated copy of the frame (all detections drawn)."""
        result = self._detector.detect(frame=frame, cage_id="__debug__")
        return self._annotator.draw(frame, result)

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # PSPNet level helpers  (only called when backend == "yolo_psp")
    # ------------------------------------------------------------------

    def _load_pspnet(self):
        """Attempt to load PSPNet; return None and log a warning on failure."""
        from pipeline.measurers.pspnet_measurer import LevelEstimator
        cfg = CONFIG["pspnet"]
        water_w = cfg["water_weights"]
        food_w  = cfg["food_weights"]

        if not water_w and not food_w:
            logger.warning(
                "PSP_WATER_WEIGHTS / PSP_FOOD_WEIGHTS not set — "
                "PSPNet disabled; will fall back to YOLOX discrete buckets."
            )
            return None

        try:
            estimator = LevelEstimator(
                water_weights=water_w,
                food_weights=food_w,
                backbone=cfg["backbone"],
                device=CONFIG["device"],
            )
            logger.info(
                "PSPNet loaded — water=%s  food=%s  backbone=%s",
                water_w, food_w, cfg["backbone"],
            )
            return estimator
        except Exception as exc:
            logger.error("PSPNet failed to load: %s", exc)
            return None

    def _water_level(
        self,
        frame: np.ndarray,
        yolo_result: DetectionResult,
    ) -> LevelReading:
        """
        Run PSPNet on the full frame to estimate water fill level.
        Falls back to YOLOX reading on any failure or sentinel return.
        """
        from pipeline.measurers.pspnet_measurer import NO_CONTAINER_SENTINEL
        try:
            pct, status, _ = self._estimator.estimate_water(frame)

            if pct == NO_CONTAINER_SENTINEL:
                logger.warning("Water PSPNet: no container detected — YOLOX fallback")
                return yolo_result.water

            if pct >= 97.0 and yolo_result.water.status != "OK":
                logger.warning(
                    "Water PSPNet saturated (%.1f%%) but YOLOX says %s — YOLOX fallback",
                    pct, yolo_result.water.status,
                )
                return yolo_result.water

            logger.debug("Water PSPNet: %.2f%% [%s]", pct, status)
            return LevelReading(pct=round(pct, 2), status=status)

        except Exception as exc:
            logger.error("PSPNet water failed: %s", exc)
            if CONFIG["pspnet"]["fallback_to_yolox"]:
                return yolo_result.water
            return LevelReading(pct=0.0, status="CRITICAL")

    def _food_level(
        self,
        frame: np.ndarray,
        yolo_result: DetectionResult,
    ) -> LevelReading:
        """
        Crop to the food hopper (YOLOX bbox, then fixed-ROI fallback) and run
        PSPNet to estimate food fill level.
        """
        from pipeline.measurers.pspnet_measurer import NO_CONTAINER_SENTINEL
        try:
            crop = self._get_food_crop(frame, yolo_result)
            if crop is None:
                return yolo_result.food

            pct, status, _ = self._estimator.estimate_food(crop)

            if pct == NO_CONTAINER_SENTINEL:
                logger.warning("Food PSPNet: no container detected — YOLOX fallback")
                return yolo_result.food

            # Sanity gate: PSPNet >40% but YOLOX says CRITICAL → bedding confusion
            if pct > 40.0 and yolo_result.food.status == "CRITICAL":
                logger.warning(
                    "Food PSPNet %.1f%% [%s] conflicts with YOLOX CRITICAL — YOLOX fallback",
                    pct, status,
                )
                return yolo_result.food

            if pct >= 97.0 and yolo_result.food.status != "OK":
                logger.warning(
                    "Food PSPNet saturated (%.1f%%) but YOLOX says %s — YOLOX fallback",
                    pct, yolo_result.food.status,
                )
                return yolo_result.food

            logger.debug("Food PSPNet: %.2f%% [%s]", pct, status)
            return LevelReading(pct=round(pct, 2), status=status)

        except Exception as exc:
            logger.error("PSPNet food failed: %s", exc)
            if CONFIG["pspnet"]["fallback_to_yolox"]:
                return yolo_result.food
            return LevelReading(pct=0.0, status="CRITICAL")

    def _get_food_crop(
        self,
        frame: np.ndarray,
        yolo_result: DetectionResult,
    ) -> Optional[np.ndarray]:
        """
        Return a crop of the food hopper region.
        Priority: YOLOX bbox → fixed-ROI fallback → None (give up).
        """
        h, w = frame.shape[:2]

        # Try YOLOX bbox first
        bbox = yolo_result.food_bbox
        if bbox is not None:
            x1 = max(0, int(bbox.x1)); y1 = max(0, int(bbox.y1))
            x2 = min(w, int(bbox.x2)); y2 = min(h, int(bbox.y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and (x2 - x1) >= _MIN_CROP_PX and (y2 - y1) >= _MIN_CROP_PX:
                return crop
            logger.debug("Food bbox crop degenerate — trying fixed-ROI fallback")

        # Fixed-ROI fallback
        x1f, y1f, x2f, y2f = _FOOD_ROI_FALLBACK
        rx1, ry1 = int(x1f * w), int(y1f * h)
        rx2, ry2 = int(x2f * w), int(y2f * h)
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size > 0 and (rx2 - rx1) >= _MIN_CROP_PX and (ry2 - ry1) >= _MIN_CROP_PX:
            logger.info(
                "Food bbox missing — fixed-ROI fallback (%d,%d,%d,%d)", rx1, ry1, rx2, ry2
            )
            return crop

        logger.warning("Food crop: both YOLOX bbox and fixed ROI are degenerate — skipping PSPNet")
        return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _is_critical(result: DetectionResult) -> bool:
    return result.water.status == "CRITICAL" or result.food.status == "CRITICAL"


def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path