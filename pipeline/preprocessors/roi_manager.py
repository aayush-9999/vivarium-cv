# pipeline/preprocessors/roi_manager.py
"""
ROI zone manager.

All coordinates are in 640×640 (letterboxed) pixel space.
Zone definitions live in core/config.py under ROI_ZONES and are accessed
here via CONFIG so the same config_loader path is used everywhere.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config_loader import CONFIG
from core.exceptions import ROIError


class ROIManager:
    """Crop and visualise named ROI zones for a given cage type."""

    # Colours for draw_zones() — BGR
    _ZONE_COLORS = {
        "jug":    (255, 180,   0),   # amber
        "hopper": (  0, 200,  80),   # green
        "floor":  (  0, 120, 255),   # orange-red
    }

    def __init__(self, cage_type: str = "default") -> None:
        roi_zones = CONFIG["roi_zones"]
        if cage_type not in roi_zones:
            raise ROIError(
                f"Cage type '{cage_type}' not found in ROI_ZONES config. "
                f"Available: {list(roi_zones.keys())}"
            )
        self._cage_type = cage_type
        self._zones     = roi_zones[cage_type]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def crop(self, frame: np.ndarray, zone: str) -> np.ndarray:
        """
        Return a copy of the frame cropped to the named zone.

        Raises ROIError if the zone name is unknown or exceeds frame bounds.
        """
        x, y, w, h = self._get_zone(zone)
        self._validate_bounds(frame, x, y, w, h, zone)
        return frame[y : y + h, x : x + w].copy()

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of the frame with all zone rectangles drawn."""
        viz = frame.copy()
        for zone_name, (x, y, w, h) in self._zones.items():
            color = self._ZONE_COLORS.get(zone_name, (200, 200, 200))
            cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                viz, zone_name.upper(),
                (x + 4, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        return viz

    def get_zone_coords(self, zone: str) -> tuple[int, int, int, int]:
        """Return raw (x, y, w, h) for a zone without cropping."""
        return self._get_zone(zone)

    def available_zones(self) -> list[str]:
        return list(self._zones.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_zone(self, zone: str) -> tuple[int, int, int, int]:
        if zone not in self._zones:
            raise ROIError(
                f"Zone '{zone}' not found for cage type '{self._cage_type}'. "
                f"Available: {self.available_zones()}"
            )
        return self._zones[zone]

    def _validate_bounds(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        zone: str,
    ) -> None:
        fh, fw = frame.shape[:2]
        if x < 0 or y < 0 or (x + w) > fw or (y + h) > fh:
            raise ROIError(
                f"Zone '{zone}' coords ({x},{y},{w},{h}) exceed "
                f"frame size ({fw}×{fh}). "
                "Recalibrate ROI_ZONES in core/config.py."
            )