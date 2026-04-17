# preprocessing/roi_manager.py
import numpy as np
import cv2
from core.config import ROI_ZONES
from core.exceptions import ROIError

class ROIManager:
    """
    Manages ROI zone extraction for a given cage type.
    All coordinates are in 640x640 pixel space.
    """

    def __init__(self, cage_type: str = "default"):
        if cage_type not in ROI_ZONES:
            raise ROIError(
                f"Cage type '{cage_type}' not found in config. "
                f"Available: {list(ROI_ZONES.keys())}"
            )
        self.cage_type = cage_type
        self._zones = ROI_ZONES[cage_type]

    # ── Public ───────────────────────────────────────────────────

    def crop(self, frame: np.ndarray, zone: str) -> np.ndarray:
        """
        Crop frame to a named zone.
        Returns the cropped region as a new array (copy, not a view).
        """
        x, y, w, h = self._get_zone(zone)
        self._validate_bounds(frame, x, y, w, h, zone)
        return frame[y:y+h, x:x+w].copy()

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all ROI zones onto a copy of the frame.
        Useful for debugging and camera calibration.
        """
        viz = frame.copy()
        colors = {
            "jug":    (255, 180,  0),   # blue
            "hopper": (0,   200, 80),   # green
            "floor":  (0,   120, 255),  # red-orange
        }
        for zone_name, (x, y, w, h) in self._zones.items():
            color = colors.get(zone_name, (200, 200, 200))
            cv2.rectangle(viz, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                viz, zone_name.upper(),
                (x + 4, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA
            )
        return viz

    def get_zone_coords(self, zone: str) -> tuple[int, int, int, int]:
        """Return raw (x, y, w, h) for a zone without cropping."""
        return self._get_zone(zone)

    def available_zones(self) -> list[str]:
        return list(self._zones.keys())

    # ── Private ──────────────────────────────────────────────────

    def _get_zone(self, zone: str) -> tuple[int, int, int, int]:
        if zone not in self._zones:
            raise ROIError(
                f"Zone '{zone}' not found for cage type '{self.cage_type}'. "
                f"Available zones: {self.available_zones()}"
            )
        return self._zones[zone]

    def _validate_bounds(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        zone: str
    ) -> None:
        fh, fw = frame.shape[:2]
        if x < 0 or y < 0 or (x + w) > fw or (y + h) > fh:
            raise ROIError(
                f"Zone '{zone}' coords ({x},{y},{w},{h}) exceed "
                f"frame dimensions ({fw}x{fh}). "
                f"Recalibrate ROI_ZONES in config.py."
            )