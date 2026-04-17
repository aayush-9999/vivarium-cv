# level_estimation/water_level.py
import cv2
import numpy as np
from core.base_level_estimator import BaseLevelEstimator
from core.schemas import LevelReading
from core.exceptions import LevelEstimationError
from level_estimation.level_calculator import (
    calc_level,
    combine_contour_masks,
    largest_contour,
    filter_contours_by_area,
)

# HSV range for water detection
# Water in a transparent jug appears as a subtle blue-grey tint
# Adjust lower/upper after testing under your lab lighting
WATER_HSV_LOWER = np.array([90,  20,  60],  dtype=np.uint8)
WATER_HSV_UPPER = np.array([130, 160, 255], dtype=np.uint8)

# Fallback: detect water by darkness contrast inside jug area
# When water has no tint, it appears darker than the air above it
WATER_DARK_LOWER = np.array([0,   0,   30],  dtype=np.uint8)
WATER_DARK_UPPER = np.array([180, 50,  130], dtype=np.uint8)


class WaterLevelEstimator(BaseLevelEstimator):
    """
    Estimates water fill level in the transparent jug ROI.

    Strategy:
    1. Try HSV color range (blue tint of water)
    2. If HSV mask is too sparse → fallback to darkness contrast method
    3. Extract largest contour (there is only one jug)
    4. Fill percentage = contour area / total ROI area
    """

    def __init__(self):
        super().__init__(roi_zone="jug")
        self._last_method: str = "hsv"   # track which path was used, for logging

    # ── BaseLevelEstimator implementation ────────────────────────

    def extract_mask(self, roi_frame: np.ndarray) -> np.ndarray:
        """
        roi_frame: uint8 BGR crop of the jug zone (from FramePreprocessor)
        Returns:   binary mask (uint8, 0 or 255)
        """
        if roi_frame is None or roi_frame.size == 0:
            raise LevelEstimationError("Empty ROI frame passed to WaterLevelEstimator.")

        hsv  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = self._hsv_mask(hsv)

        # Fallback if primary mask is too sparse (< 5% of ROI filled)
        coverage = np.count_nonzero(mask) / mask.size
        if coverage < 0.05:
            mask = self._darkness_mask(hsv)
            self._last_method = "darkness_fallback"
        else:
            self._last_method = "hsv"

        mask = self._clean(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = filter_contours_by_area(contours, min_area=300)
        contours = largest_contour(contours)

        return combine_contour_masks(contours, mask.shape)

    def estimate_pct(self, mask: np.ndarray) -> float:
        reading = calc_level(mask, roi_height=mask.shape[0])
        return reading.pct

    # ── Private ──────────────────────────────────────────────────

    @staticmethod
    def _hsv_mask(hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(hsv, WATER_HSV_LOWER, WATER_HSV_UPPER)

    @staticmethod
    def _darkness_mask(hsv: np.ndarray) -> np.ndarray:
        """
        Detects the water body by its low saturation + low brightness
        compared to the air gap above it inside the jug.
        """
        return cv2.inRange(hsv, WATER_DARK_LOWER, WATER_DARK_UPPER)

    @staticmethod
    def _clean(mask: np.ndarray) -> np.ndarray:
        """
        Larger kernel than background subtractor (7x7) because
        the water body is a solid blob — we want to fill gaps
        from reflections / condensation on the glass.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    # ── Debug ─────────────────────────────────────────────────────

    def debug_frame(self, roi_frame: np.ndarray) -> np.ndarray:
        """
        Returns the ROI with the detected water mask overlaid in blue.
        Call during development to verify HSV range is correct.
        """
        mask  = self.extract_mask(roi_frame)
        viz   = roi_frame.copy()
        viz[mask == 255] = (255, 100, 0)   # blue overlay
        return viz

    @property
    def last_method(self) -> str:
        """Returns 'hsv' or 'darkness_fallback' — useful for logging."""
        return self._last_method