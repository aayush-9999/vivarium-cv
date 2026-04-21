# pipeline/yolo_pipeline.py
"""
Full single-cage inference pipeline — 9-class version.

    raw frame
        ├─► FramePreprocessor  (letterbox to 640×640)
        └─► YOLODetector        (9 classes)
                ├─► class 0        → mouse_count
                ├─► classes 1-4    → water LevelReading (from class ID)
                └─► classes 5-8    → food  LevelReading (from class ID)
                        └─► DetectionResult

No HSV estimators. No second YOLO pass. One forward → full result.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from typing import Optional

from core.schemas import DetectionResult, BoundingBox
from core.exceptions import VivariumCVError
from core.config import YOLO_CLASS_MAP

from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.background_subtractor import BackgroundSubtractor
from detectors.yolo.yolo_detector import YOLODetector


class YOLOPipeline:

    def __init__(self, cage_type: str = "default"):
        weights = os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")
        device  = os.getenv("YOLO_DEVICE",  "cpu")

        self.preprocessor  = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor = BackgroundSubtractor()
        self.detector      = YOLODetector(weights_path=weights, device=device)

        self.detector.warmup()

    # ── Public ────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir: str = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame passed for cage '{cage_id}'.")

        # Resize to 640×640
        frame_640 = self.preprocessor.resize(frame)

        # Single YOLO pass → mouse count + water level + food level
        result = self.detector.detect(frame=frame_640, cage_id=cage_id)

        # Save frame if flagged
        if save_flagged and self._should_flag(result):
            image_path = self._save_frame(frame_640, cage_id, output_dir)
            result = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self.bg_subtractor.set_reference(self.preprocessor.resize(frame))

    def has_motion(self, frame: np.ndarray) -> bool:
        if not self.bg_subtractor.has_reference():
            return True
        return self.bg_subtractor.has_motion(self.preprocessor.resize(frame))

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Returns 640×640 frame with all YOLO detections drawn."""
        frame_640 = self.preprocessor.resize(frame)
        result    = self.detector.detect(frame=frame_640, cage_id="debug")
        return _draw_result(frame_640, result)

    # ── Private ───────────────────────────────────────────────────

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


# ── Debug helpers ──────────────────────────────────────────────────────────────

# Colour per class
_CLASS_COLORS = {
    0: (255,  80,  80),   # blue   — mouse
    1: (0,    0,  200),   # dark red — water_critical
    2: (0,   140, 255),   # orange   — water_low
    3: (0,   200, 255),   # yellow   — water_ok
    4: (0,   255, 180),   # cyan     — water_full
    5: (0,    0,  200),   # dark red — food_critical
    6: (0,   100, 255),   # orange   — food_low
    7: (80,  200,  80),   # green    — food_ok
    8: (150, 255, 150),   # light green — food_full
}


def _draw_result(img: np.ndarray, result: DetectionResult) -> np.ndarray:
    viz = img.copy()

    for bbox, label_prefix in [
        (result.water_bbox, f"water {result.water.status} {result.water.pct:.0f}%"),
        (result.food_bbox,  f"food  {result.food.status}  {result.food.pct:.0f}%"),
    ]:
        if bbox is not None:
            x1,y1,x2,y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            color = (255, 180, 0) if "water" in label_prefix else (0, 200, 80)
            cv2.rectangle(viz, (x1,y1), (x2,y2), color, 2)
            cv2.putText(viz, f"{label_prefix} ({bbox.conf:.2f})",
                        (x1+4, y1+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Mouse count overlay
    cv2.putText(viz, f"mice: {result.mouse_count}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return viz