# preprocessing/frame_preprocessor.py
import cv2
import numpy as np
from core.base_preprocessor import BasePreprocessor
from core.config import INPUT_SIZE
from core.exceptions import VivariumCVError
from preprocessing.roi_manager import ROIManager

class FramePreprocessor(BasePreprocessor):
    """
    Concrete implementation of BasePreprocessor.
    Handles all frame preparation before YOLO inference
    and before HSV level estimation.
    """

    def __init__(self, cage_type: str = "default"):
        self.input_size  = INPUT_SIZE          # (640, 640)
        self.roi_manager = ROIManager(cage_type)
        self._orig_size: tuple[int, int] | None = None  # (h, w) of raw frame

    # ── BasePreprocessor implementation ──────────────────────────

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize to INPUT_SIZE using letterbox strategy.
        Letterbox preserves aspect ratio by padding with grey (114,114,114)
        instead of stretching — YOLO was trained expecting this.
        """
        self._orig_size = frame.shape[:2]   # store for coord rescaling later
        h, w            = frame.shape[:2]
        target_w, target_h = self.input_size

        # Scale factor — limit by whichever dimension hits the boundary first
        scale  = min(target_w / w, target_h / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas and paste resized frame centered
        canvas  = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        # Store padding info for bbox rescaling in postprocessor
        self._pad   = (pad_top, pad_left)
        self._scale = scale

        return canvas

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR uint8 [0,255] → float32 [0.0, 1.0]
        Does NOT convert BGR→RGB here — YOLO via Ultralytics
        handles channel order internally.
        """
        return frame.astype(np.float32) / 255.0

    def apply_roi(self, frame: np.ndarray, zone: str) -> np.ndarray:
        """
        Crop to a named ROI zone via ROIManager.
        Applied on the resized (640x640) frame, before normalize,
        so the ROI crop is in uint8 for HSV processing.
        """
        return self.roi_manager.crop(frame, zone)

    def to_blob(self, frame: np.ndarray) -> np.ndarray:
        """
        HWC float32 → CHW float32 with batch dimension.
        Output shape: (1, 3, 640, 640) — ready for YOLO forward pass.
        """
        chw = np.transpose(frame, (2, 0, 1))    # HWC → CHW
        return np.expand_dims(chw, axis=0)       # CHW → NCHW

    # ── Extra public helpers ──────────────────────────────────────

    def prepare_for_yolo(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline for YOLO detection path.
        Raw frame → letterbox resize → normalize → blob (1,3,640,640)
        """
        try:
            frame = self.resize(frame)
            frame = self.normalize(frame)
            return self.to_blob(frame)
        except Exception as e:
            raise VivariumCVError(f"YOLO preprocessing failed: {e}") from e

    def prepare_for_level(self, frame: np.ndarray, zone: str) -> np.ndarray:
        """
        Full pipeline for level estimation path.
        Raw frame → letterbox resize → ROI crop (uint8, BGR)
        Stops BEFORE normalize because HSV operations need uint8.
        """
        try:
            frame = self.resize(frame)           # 640x640 uint8
            return self.apply_roi(frame, zone)   # cropped uint8 BGR
        except Exception as e:
            raise VivariumCVError(f"Level preprocessing failed: {e}") from e

    def rescale_bbox(
        self,
        bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """
        Convert bbox coords from 640x640 letterboxed space
        back to original frame pixel space.
        Call this in the postprocessor after YOLO inference.
        bbox format: (x1, y1, x2, y2)
        """
        if self._orig_size is None or self._pad is None:
            raise VivariumCVError(
                "rescale_bbox called before resize(). "
                "Run prepare_for_yolo() first."
            )
        pad_top, pad_left = self._pad
        x1, y1, x2, y2   = bbox

        # Remove padding offset
        x1 = (x1 - pad_left) / self._scale
        y1 = (y1 - pad_top)  / self._scale
        x2 = (x2 - pad_left) / self._scale
        y2 = (y2 - pad_top)  / self._scale

        # Clip to original frame bounds
        orig_h, orig_w = self._orig_size
        x1 = float(np.clip(x1, 0, orig_w))
        y1 = float(np.clip(y1, 0, orig_h))
        x2 = float(np.clip(x2, 0, orig_w))
        y2 = float(np.clip(y2, 0, orig_h))

        return x1, y1, x2, y2

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns 640x640 frame with ROI zones drawn.
        Quick visual check during development.
        """
        resized = self.resize(frame)
        return self.roi_manager.draw_zones(resized)

    @property
    def orig_size(self) -> tuple[int, int] | None:
        """Original frame (h, w) from last resize() call."""
        return self._orig_size