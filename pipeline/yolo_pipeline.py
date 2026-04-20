# pipeline/yolo_pipeline.py
"""
Full single-cage inference pipeline:
    raw frame
        ├─► FramePreprocessor  (resize to 640×640)
        ├─► YOLODetector        (detects mouse + water_container + food_area)
        │       └─► water_bbox, food_bbox  ← NEW: dynamic ROI from YOLO
        ├─► WaterLevelEstimator (HSV on water_container bbox, or fallback zone)
        ├─► FoodLevelEstimator  (HSV on food_area bbox, or fallback zone)
        └─► DetectionResult
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Optional

from core.schemas import DetectionResult, BoundingBox
from core.exceptions import VivariumCVError
from core.config import INPUT_SIZE

from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.background_subtractor import BackgroundSubtractor
from level_estimation.water_level import WaterLevelEstimator
from level_estimation.food_level import FoodLevelEstimator
from detectors.yolo.yolo_detector import YOLODetector


class YOLOPipeline:
    """
    Orchestrates:
      1. Preprocessing (letterbox to 640×640)
      2. YOLO inference → mouse count + container bboxes
      3. Level estimation on YOLO-detected container regions
         (falls back to hardcoded ROI zones if YOLO misses a container)
    """

    def __init__(self, cage_type: str = "default"):
        weights = os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")
        device  = os.getenv("YOLO_DEVICE",  "cpu")

        self.preprocessor  = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor = BackgroundSubtractor()
        self.water_est     = WaterLevelEstimator()
        self.food_est      = FoodLevelEstimator()
        self.detector      = YOLODetector(weights_path=weights, device=device)

        self.detector.warmup()

    # ── Public ────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir:  str  = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame passed for cage '{cage_id}'.")

        # ── Step 1: resize to 640×640 ─────────────────────────────
        frame_640 = self.preprocessor.resize(frame)

        # ── Step 2: YOLO inference ────────────────────────────────
        # Returns dummy result first (no level readings yet), plus bboxes
        # We'll re-run with real level readings in step 4.
        _, water_bbox, food_bbox = self.detector.detect(
            frame=frame_640,
            cage_id=cage_id,
            # Placeholders — we re-call with real readings below
            water_reading=None,
            food_reading=None,
        )

        # ── Step 3: crop ROI for level estimation ─────────────────
        # Use YOLO-detected bbox if available, otherwise fall back to config zone
        water_roi = self._crop_roi(frame_640, water_bbox, fallback_zone="jug")
        food_roi  = self._crop_roi(frame_640, food_bbox,  fallback_zone="hopper")

        water_reading = self.water_est.read(water_roi)
        food_reading  = self.food_est.read(food_roi)

        # ── Step 4: YOLO again with real level readings ────────────
        # (Single-pass alternative: run YOLO once, then estimate levels.
        #  We accept two forwards here because YOLO is fast and the
        #  level estimators are OpenCV-only — no GPU cost.)
        result, water_bbox, food_bbox = self.detector.detect(
            frame=frame_640,
            cage_id=cage_id,
            water_reading=water_reading,
            food_reading=food_reading,
        )

        # ── Step 5: optional frame saving ─────────────────────────
        if save_flagged and self._should_flag(result):
            image_path = self._save_frame(frame_640, cage_id, output_dir)
            result = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        frame_640 = self.preprocessor.resize(frame)
        self.bg_subtractor.set_reference(frame_640)

    def has_motion(self, frame: np.ndarray) -> bool:
        if not self.bg_subtractor.has_reference():
            return True
        frame_640 = self.preprocessor.resize(frame)
        return self.bg_subtractor.has_motion(frame_640)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns annotated 640×640 frame with:
        - YOLO detections (all 3 classes) drawn
        - Water / food level mask overlays
        - Fallback ROI zones shown if YOLO misses a container
        """
        frame_640 = self.preprocessor.resize(frame)

        # Run detection
        _, water_bbox, food_bbox = self.detector.detect(
            frame=frame_640, cage_id="debug"
        )

        viz = frame_640.copy()

        # Draw YOLO bboxes
        _draw_bbox(viz, water_bbox, label="water_container", color=(255, 180, 0))
        _draw_bbox(viz, food_bbox,  label="food_area",       color=(0, 200, 80))

        # Overlay level masks
        water_roi = self._crop_roi(frame_640, water_bbox, fallback_zone="jug")
        food_roi  = self._crop_roi(frame_640, food_bbox,  fallback_zone="hopper")

        water_debug = self.water_est.debug_frame(water_roi)
        food_debug  = self.food_est.debug_frame(food_roi)

        # Paste overlays back
        wy1, wy2, wx1, wx2 = _bbox_to_slice(frame_640, water_bbox, self.preprocessor, "jug")
        fy1, fy2, fx1, fx2 = _bbox_to_slice(frame_640, food_bbox,  self.preprocessor, "hopper")

        viz[wy1:wy2, wx1:wx2] = water_debug
        viz[fy1:fy2, fx1:fx2] = food_debug

        return viz

    # ── Private helpers ───────────────────────────────────────────

    def _crop_roi(
        self,
        frame_640: np.ndarray,
        bbox: Optional[BoundingBox],
        fallback_zone: str,
    ) -> np.ndarray:
        """
        Crop the frame to the container region.
        If YOLO detected the container, use that bbox (with a small padding).
        Otherwise fall back to the hardcoded config ROI zone.
        """
        if bbox is not None:
            h, w = frame_640.shape[:2]
            pad = 8   # px padding around detected container
            x1 = max(0, int(bbox.x1) - pad)
            y1 = max(0, int(bbox.y1) - pad)
            x2 = min(w, int(bbox.x2) + pad)
            y2 = min(h, int(bbox.y2) + pad)
            return frame_640[y1:y2, x1:x2].copy()
        else:
            # Fallback to hardcoded zone
            return self.preprocessor.apply_roi(frame_640, fallback_zone)

    @staticmethod
    def _should_flag(result: DetectionResult) -> bool:
        return (
            result.water.status == "CRITICAL"
            or result.food.status  == "CRITICAL"
        )

    @staticmethod
    def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
        from datetime import datetime, timezone
        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path


# ── Debug drawing helpers ─────────────────────────────────────────────────────

def _draw_bbox(
    img: np.ndarray,
    bbox: Optional[BoundingBox],
    label: str,
    color: tuple,
) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label} {bbox.conf:.2f}", (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _bbox_to_slice(
    frame: np.ndarray,
    bbox: Optional[BoundingBox],
    preprocessor,
    fallback_zone: str,
) -> tuple[int, int, int, int]:
    """Returns (y1, y2, x1, x2) slice indices for pasting debug overlays."""
    if bbox is not None:
        h, w = frame.shape[:2]
        pad = 8
        return (
            max(0, int(bbox.y1) - pad),
            min(h, int(bbox.y2) + pad),
            max(0, int(bbox.x1) - pad),
            min(w, int(bbox.x2) + pad),
        )
    else:
        from core.config import ROI_ZONES
        zones = ROI_ZONES.get(preprocessor.roi_manager.cage_type, ROI_ZONES["default"])
        x, y, w, h = zones[fallback_zone]
        return y, y + h, x, x + w
