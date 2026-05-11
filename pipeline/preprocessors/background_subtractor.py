# pipeline/preprocessors/background_subtractor.py
"""
Reference-frame background subtractor.

Lightweight alternative to MOG2 — works well for static cage cameras
where the background never changes between frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config_loader import CONFIG


class BackgroundSubtractor:
    """
    Stores one clean reference frame and flags when a new frame differs
    enough from it to warrant a full inference run.
    """

    def __init__(
        self,
        motion_threshold: Optional[float] = None,
    ) -> None:
        # Pull threshold from CONFIG so it's consistent with the rest of the app.
        self._motion_threshold: float = (
            motion_threshold
            if motion_threshold is not None
            else CONFIG["camera"]["motion_threshold"]
        )
        self._reference: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Reference frame management
    # ------------------------------------------------------------------

    def set_reference(self, frame: np.ndarray) -> None:
        """
        Store a clean empty-cage frame as the background reference.
        Call once during camera setup or after the cage is cleaned.
        """
        self._reference = self._to_gray(frame).astype(np.float32)

    def has_reference(self) -> bool:
        return self._reference is not None

    def clear_reference(self) -> None:
        """Discard the stored reference (e.g. after cage cleaning)."""
        self._reference = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtract reference from current frame.
        Returns binary mask: 255 = foreground, 0 = background.

        Raises
        ------
        RuntimeError : if set_reference() has not been called yet.
        """
        if not self.has_reference():
            raise RuntimeError(
                "No reference frame stored. Call set_reference() first."
            )
        gray  = self._to_gray(frame).astype(np.float32)
        diff  = cv2.absdiff(gray, self._reference)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask    = mask.astype(np.uint8)
        return self._clean_mask(mask)

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Return True if enough pixels changed to warrant inference.
        Used by the motion gate in the camera loop.
        """
        mask            = self.subtract(frame)
        changed_pixels  = np.count_nonzero(mask)
        return (changed_pixels / mask.size) > self._motion_threshold

    def apply_to_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply foreground mask to original colour frame (for visualisation)."""
        if mask is None:
            mask = self.subtract(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological open then close: remove noise, fill holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask


# Fix missing import that only shows up at runtime
from typing import Optional  # noqa: E402  (placed after class to avoid circular)