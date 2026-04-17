# level_estimation/food_level.py
import cv2
import numpy as np
from core.base_level_estimator import BaseLevelEstimator
from core.schemas import LevelReading
from core.exceptions import LevelEstimationError
from level_estimation.level_calculator import (
    calc_level,
    combine_contour_masks,
    filter_contours_by_area,
)

# HSV range for food pellet detection
# Standard lab rodent chow is brown/tan — adjust per your pellet brand
FOOD_HSV_LOWER = np.array([8,   40,  40],  dtype=np.uint8)
FOOD_HSV_UPPER = np.array([30, 220, 220], dtype=np.uint8)

# Soiled/darker bedding can bleed into food zone
# This tighter range reduces bedding false positives
FOOD_HSV_LOWER_STRICT = np.array([10,  60,  60],  dtype=np.uint8)
FOOD_HSV_UPPER_STRICT = np.array([25, 200, 200], dtype=np.uint8)

# Minimum contour area — filters out scattered crumbs
# that don't represent meaningful food level
FOOD_MIN_CONTOUR_AREA = 400


class FoodLevelEstimator(BaseLevelEstimator):
    """
    Estimates food fill level in the hopper ROI.

    Strategy:
    1. Apply HSV range for brown/tan food pellets
    2. If mask coverage is suspiciously high (bedding bleed-in),
       retry with stricter HSV range
    3. Filter small crumb contours by area
    4. Fill percentage = total contour area / total ROI area

    Note: Unlike water (single blob), food can be fragmented pellets
    so we keep ALL valid contours, not just the largest.
    """

    def __init__(self):
        super().__init__(roi_zone="hopper")
        self._last_method: str = "hsv"

    # ── BaseLevelEstimator implementation ────────────────────────

    def extract_mask(self, roi_frame: np.ndarray) -> np.ndarray:
        """
        roi_frame: uint8 BGR crop of the hopper zone
        Returns:   binary mask (uint8, 0 or 255)
        """
        if roi_frame is None or roi_frame.size == 0:
            raise LevelEstimationError("Empty ROI frame passed to FoodLevelEstimator.")

        hsv  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = self._hsv_mask(hsv, strict=False)

        # If coverage > 85% the mask likely caught bedding too
        # retry with stricter range
        coverage = np.count_nonzero(mask) / mask.size
        if coverage > 0.85:
            mask = self._hsv_mask(hsv, strict=True)
            self._last_method = "hsv_strict"
        else:
            self._last_method = "hsv"

        mask     = self._clean(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = filter_contours_by_area(contours, min_area=FOOD_MIN_CONTOUR_AREA)

        # Keep all valid contours — food pile can be fragmented
        return combine_contour_masks(contours, mask.shape)

    def estimate_pct(self, mask: np.ndarray) -> float:
        reading = calc_level(mask, roi_height=mask.shape[0])
        return reading.pct

    # ── Private ──────────────────────────────────────────────────

    @staticmethod
    def _hsv_mask(hsv: np.ndarray, strict: bool = False) -> np.ndarray:
        lower = FOOD_HSV_LOWER_STRICT if strict else FOOD_HSV_LOWER
        upper = FOOD_HSV_UPPER_STRICT if strict else FOOD_HSV_UPPER
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def _clean(mask: np.ndarray) -> np.ndarray:
        """
        Smaller kernel than water (5x5) — food pile has natural gaps
        between pellets, we don't want to over-fill them.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    # ── Debug ─────────────────────────────────────────────────────

    def debug_frame(self, roi_frame: np.ndarray) -> np.ndarray:
        """
        Returns the ROI with detected food mask overlaid in green.
        """
        mask = self.extract_mask(roi_frame)
        viz  = roi_frame.copy()
        viz[mask == 255] = (0, 200, 80)   # green overlay
        return viz

    @property
    def last_method(self) -> str:
        return self._last_method