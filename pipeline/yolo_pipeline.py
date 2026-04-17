# pipeline/yolo_pipeline.py
"""
Full single-cage inference pipeline wiring:
    raw frame
        ├─► FramePreprocessor (resize to 640x640)
        ├─► WaterLevelEstimator (HSV on jug ROI)
        ├─► FoodLevelEstimator  (HSV on hopper ROI)
        └─► YOLODetector        (mouse count)
            └─► DetectionResult
"""
from __future__ import annotations

import os
import cv2
import numpy as np

from core.schemas import DetectionResult
from core.exceptions import VivariumCVError
from core.config import INPUT_SIZE

from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.background_subtractor import BackgroundSubtractor
from level_estimation.water_level import WaterLevelEstimator
from level_estimation.food_level import FoodLevelEstimator
from detectors.yolo.yolo_detector import YOLODetector


class YOLOPipeline:
    """
    Orchestrates preprocessing → level estimation → YOLO detection
    for a single frame from a single cage.

    Usage:
        pipeline = YOLOPipeline()
        result   = pipeline.run(frame, cage_id="cage_01")
    """

    def __init__(self, cage_type: str = "default"):
        weights = os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")
        device  = os.getenv("YOLO_DEVICE",  "cpu")

        self.preprocessor  = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor = BackgroundSubtractor()
        self.water_est     = WaterLevelEstimator()
        self.food_est      = FoodLevelEstimator()
        self.detector      = YOLODetector(weights_path=weights, device=device)

        # Warm up YOLO so first real frame doesn't pay CUDA init cost
        self.detector.warmup()

    # ── Public ────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir:  str  = "flagged_frames",
    ) -> DetectionResult:
        """
        Full pipeline for one frame.

        Args:
            frame:        Raw BGR frame from camera (any resolution).
            cage_id:      Cage identifier written into the result.
            save_flagged: If True, saves frame to disk when water/food is CRITICAL.
            output_dir:   Directory for flagged frame images.

        Returns:
            DetectionResult
        """
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame passed for cage '{cage_id}'.")

        # ── Step 1: resize to 640×640 (letterbox) ─────────────────
        frame_640 = self.preprocessor.resize(frame)

        # ── Step 2: level estimation (operates on uint8 ROI crops) ─
        water_roi  = self.preprocessor.apply_roi(frame_640, zone="jug")
        food_roi   = self.preprocessor.apply_roi(frame_640, zone="hopper")

        water_reading = self.water_est.read(water_roi)
        food_reading  = self.food_est.read(food_roi)

        # ── Step 3: YOLO mouse detection ───────────────────────────
        # Pass the uint8 640×640 frame — Ultralytics preprocesses internally
        result = self.detector.detect(
            frame=frame_640,
            cage_id=cage_id,
            water_reading=water_reading,
            food_reading=food_reading,
        )

        # ── Step 4: optional frame saving when levels are critical ──
        if save_flagged and self._should_flag(result):
            image_path = self._save_frame(frame_640, cage_id, output_dir)
            result = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """
        Store a clean background frame for motion detection.
        Call once per cage on startup after the cage is clean.
        """
        frame_640 = self.preprocessor.resize(frame)
        self.bg_subtractor.set_reference(frame_640)

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Quick motion check — skip full inference if cage is static.
        Returns True always if no reference frame has been set.
        """
        if not self.bg_subtractor.has_reference():
            return True
        frame_640 = self.preprocessor.resize(frame)
        return self.bg_subtractor.has_motion(frame_640)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns 640×640 frame annotated with:
          - ROI zone rectangles
          - Water mask overlay (blue)
          - Food mask overlay (green)
        Useful during camera calibration.
        """
        frame_640 = self.preprocessor.resize(frame)
        viz       = self.preprocessor.draw_debug(frame)

        water_roi = self.preprocessor.apply_roi(frame_640, zone="jug")
        food_roi  = self.preprocessor.apply_roi(frame_640, zone="hopper")

        # Paste debug overlays back into the full frame
        from core.config import ROI_ZONES
        zones = ROI_ZONES["default"]   # use default for debug

        jx, jy, jw, jh = zones["jug"]
        hx, hy, hw, hh = zones["hopper"]

        viz[jy:jy+jh, jx:jx+jw] = self.water_est.debug_frame(water_roi)
        viz[hy:hy+hh, hx:hx+hw] = self.food_est.debug_frame(food_roi)

        return viz

    # ── Private ───────────────────────────────────────────────────

    @staticmethod
    def _should_flag(result: DetectionResult) -> bool:
        return (
            result.water.status == "CRITICAL"
            or result.food.status  == "CRITICAL"
        )

    @staticmethod
    def _save_frame(
        frame: np.ndarray,
        cage_id: str,
        output_dir: str,
    ) -> str:
        """Save frame as JPEG and return the file path."""
        import os
        from datetime import datetime, timezone

        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path