# core/base_detector.py
import numpy as np
from abc import ABC, abstractmethod
from .schemas import DetectionResult

class BaseDetector(ABC):

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights onto device. Called once at init."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, cage_id: str) -> DetectionResult:
        """
        Run full detection on a preprocessed frame.
        Returns a fully populated DetectionResult.
        """
        pass

    @abstractmethod
    def _postprocess(self, raw_output, cage_id: str) -> DetectionResult:
        """
        Convert raw model output → DetectionResult.
        Handles confidence filtering, NMS, class mapping.
        """
        pass

    def warmup(self) -> None:
        """
        Run one dummy inference on init so first real frame
        doesn't pay the CUDA kernel launch cost.
        Subclasses must override this with the correct input shape
        for their model (e.g. HWC uint8 for YOLO, NCHW float for ONNX).
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")

    def is_ready(self) -> bool:
        return self.model is not None