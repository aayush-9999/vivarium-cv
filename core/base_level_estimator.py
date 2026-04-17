# core/base_level_estimator.py
import numpy as np
from abc import ABC, abstractmethod
from .schemas import LevelReading

class BaseLevelEstimator(ABC):

    def __init__(self, roi_zone: str):
        """
        roi_zone: one of 'jug' | 'hopper'
        Passed in so the estimator knows which config thresholds to use.
        """
        self.roi_zone = roi_zone

    @abstractmethod
    def extract_mask(self, roi_frame: np.ndarray) -> np.ndarray:
        """
        Given a cropped ROI frame, return a binary mask
        where 255 = detected substance (water / food), 0 = empty.
        """
        pass

    @abstractmethod
    def estimate_pct(self, mask: np.ndarray) -> float:
        """
        Compute fill percentage from binary mask.
        filled_pixels / total_pixels * 100
        """
        pass

    def get_status(self, pct: float) -> str:
        from .config import LEVEL_THRESHOLDS
        if pct < LEVEL_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif pct < LEVEL_THRESHOLDS["LOW"]:
            return "LOW"
        return "OK"

    def read(self, roi_frame: np.ndarray) -> LevelReading:
        """
        Full pipeline in one call:
        ROI frame → mask → pct → status → LevelReading
        """
        mask = self.extract_mask(roi_frame)
        pct  = self.estimate_pct(mask)
        return LevelReading(pct=pct, status=self.get_status(pct))