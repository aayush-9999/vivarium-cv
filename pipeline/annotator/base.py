# pipeline/annotator/base.py
from abc import ABC, abstractmethod
import numpy as np
from core.schemas import DetectionResult

class BaseAnnotator(ABC):
    @abstractmethod
    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        pass