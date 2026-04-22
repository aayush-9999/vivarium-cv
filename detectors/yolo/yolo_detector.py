# detectors/yolo/yolo_detector.py
from __future__ import annotations

import time
import numpy as np

from core.base_detector import BaseDetector
from core.schemas import DetectionResult
from core.config import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
from core.exceptions import DetectorInitError, InferenceError
from detectors.yolo.postprocessor import parse_yolo_results


class YOLODetector(BaseDetector):

    def __init__(self, weights_path: str, device: str = "cpu"):
        super().__init__(weights_path=weights_path, device=device)

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

    def detect(self, frame: np.ndarray, cage_id: str) -> DetectionResult:
        if not self.is_ready():
            raise InferenceError("Model is not loaded.")

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
            raise InferenceError(
                f"YOLO inference failed for cage '{cage_id}': {e}"
            ) from e

        return self._postprocess(results, cage_id, t_start)

    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        inference_start_ns: int = 0,
    ) -> DetectionResult:
        return parse_yolo_results(
            results=raw_output,
            cage_id=cage_id,
            inference_start_ns=inference_start_ns,
        )

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")