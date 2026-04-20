# detectors/yolo/yolo_detector.py
"""
YOLOv8-nano detector — 3-class model.
    class 0: mouse
    class 1: water_container
    class 2: food_area
"""
from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np

from core.base_detector import BaseDetector
from core.schemas import DetectionResult, LevelReading, BoundingBox
from core.config import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
from core.exceptions import DetectorInitError, InferenceError
from detectors.yolo.postprocessor import (
    parse_yolo_results,
    extract_container_bboxes,
)


class YOLODetector(BaseDetector):
    """
    Wraps YOLOv8-nano for 3-class vivarium inference.

    Inference flow:
        raw BGR frame (640×640, uint8)
            → model.predict()
            → extract_container_bboxes()   ← new: pulls water/food bbox
            → parse_yolo_results()
            → DetectionResult
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        super().__init__(weights_path=weights_path, device=device)

    # ── BaseDetector implementation ───────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            self.model.to(self.device)
        except FileNotFoundError as e:
            raise DetectorInitError(
                f"YOLO weights not found at '{self.weights_path}'. "
                "Set YOLO_WEIGHTS in your .env file."
            ) from e
        except Exception as e:
            raise DetectorInitError(f"Failed to load YOLO model: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        cage_id: str,
        water_reading: Optional[LevelReading] = None,
        food_reading:  Optional[LevelReading] = None,
    ) -> Tuple[DetectionResult, Optional[BoundingBox], Optional[BoundingBox]]:
        """
        Run YOLOv8 inference on a 640×640 BGR frame.

        Returns:
            (DetectionResult, water_bbox, food_bbox)
            water_bbox / food_bbox are None when YOLO doesn't detect that class.
            The pipeline uses these bboxes as dynamic ROIs for level estimation.
        """
        if not self.is_ready():
            raise InferenceError("Model is not loaded.")

        if water_reading is None:
            water_reading = LevelReading(pct=0.0, status="CRITICAL")
        if food_reading is None:
            food_reading  = LevelReading(pct=0.0, status="CRITICAL")

        t_start = time.perf_counter_ns()

        try:
            results = self.model.predict(
                source=frame,
                conf=YOLO_CONF_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                imgsz=640,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(f"YOLO inference failed for cage '{cage_id}': {e}") from e

        # Pull container bboxes BEFORE building the result
        water_bbox, food_bbox = extract_container_bboxes(results)

        result = self._postprocess(
            raw_output=results,
            cage_id=cage_id,
            water_reading=water_reading,
            food_reading=food_reading,
            inference_start_ns=t_start,
            water_bbox=water_bbox,
            food_bbox=food_bbox,
        )

        return result, water_bbox, food_bbox

    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        water_reading: Optional[LevelReading] = None,
        food_reading:  Optional[LevelReading] = None,
        inference_start_ns: int = 0,
        water_bbox: Optional[BoundingBox] = None,
        food_bbox:  Optional[BoundingBox] = None,
    ) -> DetectionResult:
        return parse_yolo_results(
            results=raw_output,
            cage_id=cage_id,
            water_reading=water_reading or LevelReading(pct=0.0, status="CRITICAL"),
            food_reading=food_reading   or LevelReading(pct=0.0, status="CRITICAL"),
            inference_start_ns=inference_start_ns,
            water_bbox=water_bbox,
            food_bbox=food_bbox,
        )

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")