# level_estimation/level_calculator.py
import numpy as np
import cv2
from core.config import LEVEL_THRESHOLDS
from core.schemas import LevelReading
from core.exceptions import LevelEstimationError

def calc_level(mask: np.ndarray, roi_height: int) -> LevelReading:
    """
    Compute fill percentage from a binary mask and return a LevelReading.

    Args:
        mask:       Binary mask (uint8, 0 or 255) — output of HSV range + contour
        roi_height: Pixel height of the ROI zone (used as denominator)

    Returns:
        LevelReading with pct and status
    """
    if mask is None or mask.size == 0:
        raise LevelEstimationError("Empty or null mask passed to calc_level.")

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    filled_pixels = int(np.count_nonzero(mask))
    total_pixels  = int(roi_height * mask.shape[1])

    if total_pixels == 0:
        raise LevelEstimationError(
            f"total_pixels is 0 — roi_height={roi_height}, "
            f"mask.shape={mask.shape}. Check ROI config."
        )

    pct = round((filled_pixels / total_pixels) * 100.0, 1)
    pct = float(np.clip(pct, 0.0, 100.0))   # guard against rounding edge cases

    return LevelReading(pct=pct, status=_get_status(pct))


def _get_status(pct: float) -> str:
    if pct < LEVEL_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif pct < LEVEL_THRESHOLDS["LOW"]:
        return "LOW"
    return "OK"


def combine_contour_masks(
    contours: list,
    shape: tuple[int, int]
) -> np.ndarray:
    """
    Draw filled contours onto a blank mask.
    Used by both water and food estimators after contour filtering.

    Args:
        contours: list of cv2 contours
        shape:    (height, width) of output mask

    Returns:
        Binary mask with filled contours
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def largest_contour(contours: list) -> list:
    """
    Filter to only the single largest contour by area.
    Used when we know there is exactly one jug / one hopper in the ROI.
    """
    if not contours:
        return []
    return [max(contours, key=cv2.contourArea)]


def filter_contours_by_area(
    contours: list,
    min_area: int = 200
) -> list:
    """
    Remove noise contours below min_area pixels.
    Default 200px works well for 640x640 — adjust if ROI is very small.
    """
    return [c for c in contours if cv2.contourArea(c) > min_area]