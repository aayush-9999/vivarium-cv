# detectors/yolo/yolo_detector.py
"""
YOLOv8-nano detector.
Inherits BaseDetector and implements _load_model, detect, _postprocess.
"""
from __future__ import annotations

import time
import numpy as np

from core.base_detector import BaseDetector
from core.schemas import DetectionResult, LevelReading
from core.config import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
from core.exceptions import DetectorInitError, InferenceError
from detectors.yolo.postprocessor import parse_yolo_results


class YOLODetector(BaseDetector):
    """
    Wraps Ultralytics YOLOv8-nano for mouse detection.

    Inference flow:
        raw BGR frame (640x640, uint8)
            → model.predict()           # Ultralytics handles pre/post internally
            → parse_yolo_results()      # our postprocessor
            → DetectionResult
    
    Level readings (water / food) are injected at call time from the pipeline
    because they come from a separate OpenCV HSV path, not from YOLO.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
    ):
        # BaseDetector.__init__ calls _load_model automatically
        super().__init__(weights_path=weights_path, device=device)

    # ── BaseDetector implementation ───────────────────────────────

    def _load_model(self) -> None:
        """Load YOLOv8-nano weights via Ultralytics."""
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
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
    ) -> DetectionResult:
        """
        Run YOLOv8-nano inference on a preprocessed 640x640 BGR frame.

        Args:
            frame:         uint8 BGR ndarray, shape (640, 640, 3).
                           DO NOT pass the blob (1,3,H,W) here — Ultralytics
                           handles its own preprocessing internally.
            cage_id:       Cage identifier for the result.
            water_reading: Pre-computed water LevelReading (from HSV path).
            food_reading:  Pre-computed food LevelReading (from HSV path).

        Returns:
            DetectionResult with mouse_count, water, food, timing.
        """
        if not self.is_ready():
            raise InferenceError("Model is not loaded. Call _load_model() first.")

        # Provide zero-filled level readings if not supplied
        # (e.g. during warmup or unit tests)
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
                verbose=False,   # suppress Ultralytics stdout per-frame
            )
        except Exception as e:
            raise InferenceError(f"YOLO inference failed for cage '{cage_id}': {e}") from e

        return self._postprocess(
            raw_output=results,
            cage_id=cage_id,
            water_reading=water_reading,
            food_reading=food_reading,
            inference_start_ns=t_start,
        )

    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
        inference_start_ns: int = 0,
    ) -> DetectionResult:
        """Delegates to the standalone postprocessor module."""
        return parse_yolo_results(
            results=raw_output,
            cage_id=cage_id,
            water_reading=water_reading or LevelReading(pct=0.0, status="CRITICAL"),
            food_reading=food_reading   or LevelReading(pct=0.0, status="CRITICAL"),
            inference_start_ns=inference_start_ns,
        )

    # ── Warmup override ───────────────────────────────────────────

    def warmup(self) -> None:
        """
        Send a dummy 640x640 frame through YOLO once so CUDA kernels
        are warm before the first real cage frame arrives.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")