# core/base_preprocessor.py
import numpy as np
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):

    @abstractmethod
    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input size."""
        pass

    @abstractmethod
    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize pixel values to 0-1 float32."""
        pass

    @abstractmethod
    def apply_roi(self, frame: np.ndarray, zone: str) -> np.ndarray:
        """Crop frame to a named ROI zone (jug / hopper / floor)."""
        pass

    @abstractmethod
    def to_blob(self, frame: np.ndarray) -> np.ndarray:
        """Convert processed frame to model-ready blob (CHW, batch dim)."""
        pass

    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline in one call.
        Resize → Normalize → Blob.
        ROI is applied separately per zone when needed.
        """
        frame = self.resize(frame)
        frame = self.normalize(frame)
        return self.to_blob(frame)