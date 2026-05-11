# pipeline/measurers/base.py
from abc import ABC, abstractmethod
import numpy as np
from core.schemas import LevelReading

class BaseMeasurer(ABC):
    @abstractmethod
    def estimate_water(self, frame: np.ndarray) -> tuple[float, str, np.ndarray]:
        """Returns (pct, status, mask)"""
        pass

    @abstractmethod
    def estimate_food(self, crop: np.ndarray) -> tuple[float, str, np.ndarray]:
        """Returns (pct, status, mask)"""
        pass