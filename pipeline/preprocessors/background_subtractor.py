# preprocessing/background_subtractor.py
import numpy as np
import cv2
from core.config import MOTION_PIXEL_THRESHOLD

class BackgroundSubtractor:
    """
    Reference-frame based background subtraction.
    Lightweight alternative to MOG2 — works perfectly for
    static cage cameras where background never changes.
    """

    def __init__(self, motion_threshold: float = MOTION_PIXEL_THRESHOLD):
        self.motion_threshold = motion_threshold  # fraction of total pixels
        self._reference: np.ndarray | None = None

    # ── Reference frame management ───────────────────────────────

    def set_reference(self, frame: np.ndarray) -> None:
        """
        Store a clean empty-cage frame as the background reference.
        Call once during camera setup or after cage is cleaned.
        Frame should be grayscale or will be converted internally.
        """
        self._reference = self._to_gray(frame).astype(np.float32)

    def has_reference(self) -> bool:
        return self._reference is not None

    # ── Core operations ──────────────────────────────────────────

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtract reference from current frame.
        Returns binary mask: 255 = foreground, 0 = background.
        """
        if not self.has_reference():
            raise RuntimeError(
                "No reference frame set. Call set_reference() first."
            )
        gray  = self._to_gray(frame).astype(np.float32)
        diff  = cv2.absdiff(gray, self._reference)

        # Threshold + morphological cleanup to remove noise
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask    = mask.astype(np.uint8)
        mask    = self._clean_mask(mask)
        return mask

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Returns True if enough pixels changed to warrant early inference.
        Used by the motion gate in the inference scheduler.
        """
        mask           = self.subtract(frame)
        changed_pixels = np.count_nonzero(mask)
        total_pixels   = mask.size
        return (changed_pixels / total_pixels) > self.motion_threshold

    def apply_to_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Apply foreground mask to original color frame.
        Useful for visualizing what the subtractor isolated.
        """
        if mask is None:
            mask = self.subtract(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # ── Private ──────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """
        Morphological open (removes noise) then close (fills holes).
        Kernel size 5 works well for mouse-scale objects in 640x640.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask