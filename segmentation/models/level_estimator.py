# segmentation/models/level_estimator.py
"""
LevelEstimator — wraps PSPNet and converts segmentation masks to fill percentages.

This is the main inference interface.  Given a cropped image of a container,
it returns a LevelReading (pct + status) compatible with the existing pipeline.

Water class map:
    0 = background
    1 = bottle wall / container boundary
    2 = water fill  (MEASURE THIS)
    3 = empty air   (MEASURE THIS)

Food class map:
    0 = background
    1 = hopper frame / wire mesh
    2 = food pellets (MEASURE THIS)
    3 = empty space  (MEASURE THIS)

Fill percentage = fill_pixels / (fill_pixels + empty_pixels)
               = ignores background and container walls entirely
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from segmentation.models.pspnet import PSPNet, build_water_model, build_food_model

logger = logging.getLogger("vivarium.segmentation")

# ── Class indices (must match annotation scheme) ──────────────────────────────
WATER_FILL_CLASS  = 2
WATER_EMPTY_CLASS = 3

FOOD_FILL_CLASS   = 2
FOOD_EMPTY_CLASS  = 3

# ── Status thresholds (same as existing pipeline) ─────────────────────────────
STATUS_THRESHOLDS = [
    (0.0,  15.0, "CRITICAL"),
    (15.0, 35.0, "LOW"),
    (35.0, 100.1, "OK"),
]


def pct_to_status(pct: float) -> str:
    for lo, hi, status in STATUS_THRESHOLDS:
        if lo <= pct < hi:
            return status
    return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalization (backbone was pretrained on ImageNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop(
    crop_bgr: np.ndarray,
    target_size: tuple[int, int] = (256, 128),   # (H, W) — tall for bottles
) -> torch.Tensor:
    """
    Prepare a BGR crop for PSPNet inference.

    Steps:
        BGR → RGB
        Resize to target_size
        Normalize with ImageNet mean/std
        HWC → CHW → add batch dim
        Output: (1, 3, H, W) float32 tensor
    """
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    normed  = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    chw     = np.transpose(normed, (2, 0, 1))          # HWC → CHW
    tensor  = torch.from_numpy(chw).unsqueeze(0)       # → (1, 3, H, W)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Fill percentage from mask
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_fill_pct(
    mask: np.ndarray,
    fill_class: int,
    empty_class: int,
    use_height_method: bool = True,
) -> float:
    """
    Convert a segmentation mask to a fill percentage.

    Two methods:
    ─────────────────────────────────────────────────────
    1. HEIGHT method (default, more accurate for cylindrical containers):
       Find the lowest row that contains fill pixels (meniscus line).
       Fill % = (container_bottom - meniscus_row) / container_height

       This is more robust because even if the model misses some fill pixels
       at the sides, the meniscus height is still correctly detected.

    2. PIXEL COUNT method (fallback, better for irregular food hoppers):
       Fill % = fill_pixels / (fill_pixels + empty_pixels)

       More appropriate for food because there's no clean meniscus line.
    ─────────────────────────────────────────────────────

    Args:
        mask            : (H, W) integer array of class IDs
        fill_class      : class ID for the filled region
        empty_class     : class ID for the empty region
        use_height_method: True for water (meniscus), False for food (pixel count)

    Returns:
        fill percentage 0.0–100.0
    """
    fill_mask  = (mask == fill_class)
    empty_mask = (mask == empty_class)

    fill_pixels  = int(fill_mask.sum())
    empty_pixels = int(empty_mask.sum())
    total        = fill_pixels + empty_pixels

    if total == 0:
        # Model found neither fill nor empty — container not detected
        logger.warning("No fill or empty pixels found in mask — defaulting to 0%%")
        return 0.0

    if use_height_method and fill_pixels > 0:
        # Find rows that contain fill pixels
        fill_rows = np.where(fill_mask.any(axis=1))[0]
        empty_rows = np.where(empty_mask.any(axis=1))[0]

        if len(fill_rows) == 0 or len(empty_rows) == 0:
            # Fall back to pixel count
            return float(fill_pixels / total * 100.0)

        # Container spans from topmost empty row to bottommost fill row
        container_top    = int(empty_rows.min())
        container_bottom = int(fill_rows.max())
        meniscus_row     = int(fill_rows.min())   # topmost fill row = meniscus

        container_height = container_bottom - container_top
        if container_height <= 0:
            return float(fill_pixels / total * 100.0)

        fill_height = container_bottom - meniscus_row
        pct = float(fill_height / container_height * 100.0)
        return float(np.clip(pct, 0.0, 100.0))

    # Pixel count method
    return float(fill_pixels / total * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# LevelEstimator
# ─────────────────────────────────────────────────────────────────────────────

class LevelEstimator:
    """
    Wraps two PSPNet models (one for water, one for food) and provides
    a simple interface to get fill percentages from container crops.

    Usage:
        estimator = LevelEstimator(
            water_weights="models/psp/water_best.pth",
            food_weights="models/psp/food_best.pth",
        )
        water_pct, water_status = estimator.estimate_water(water_crop_bgr)
        food_pct, food_status   = estimator.estimate_food(food_crop_bgr)
    """

    def __init__(
        self,
        water_weights:  Optional[str] = None,
        food_weights:   Optional[str] = None,
        backbone:       str  = "resnet50",
        device:         str  = "cpu",
        water_size:     tuple[int, int] = (256, 128),   # H, W
        food_size:      tuple[int, int] = (224, 224),   # H, W — hopper is squarish
    ):
        self.device     = device
        self.water_size = water_size
        self.food_size  = food_size

        logger.info("Loading PSPNet water model (backbone=%s)…", backbone)
        self.water_model = build_water_model(
            backbone=backbone,
            pretrained=(water_weights is None),   # pretrained only if no weights
            weights_path=water_weights,
            device=device,
        )

        logger.info("Loading PSPNet food model (backbone=%s)…", backbone)
        self.food_model = build_food_model(
            backbone=backbone,
            pretrained=(food_weights is None),
            weights_path=food_weights,
            device=device,
        )

        self.water_model.eval()
        self.food_model.eval()
        logger.info("PSPNet models ready.")

    # ── Public API ─────────────────────────────────────────────────────────

    def estimate_water(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[float, str, np.ndarray]:
        """
        Estimate water fill from a cropped water bottle image.

        Args:
            crop_bgr : BGR crop of the water bottle (any size)

        Returns:
            (pct, status, mask)
            pct    : 0.0–100.0
            status : "OK" | "LOW" | "CRITICAL"
            mask   : (H, W) segmentation mask for visualization
        """
        mask = self._segment(self.water_model, crop_bgr, self.water_size)
        pct  = mask_to_fill_pct(
            mask,
            fill_class=WATER_FILL_CLASS,
            empty_class=WATER_EMPTY_CLASS,
            use_height_method=True,   # height method for water (meniscus line)
        )
        return pct, pct_to_status(pct), mask

    def estimate_food(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[float, str, np.ndarray]:
        """
        Estimate food fill from a cropped food hopper image.

        Args:
            crop_bgr : BGR crop of the food hopper (any size)

        Returns:
            (pct, status, mask)
            pct    : 0.0–100.0
            status : "OK" | "LOW" | "CRITICAL"
            mask   : (H, W) segmentation mask for visualization
        """
        mask = self._segment(self.food_model, crop_bgr, self.food_size)
        pct  = mask_to_fill_pct(
            mask,
            fill_class=FOOD_FILL_CLASS,
            empty_class=FOOD_EMPTY_CLASS,
            use_height_method=False,   # pixel count for food (no meniscus)
        )
        return pct, pct_to_status(pct), mask

    def is_ready(self) -> bool:
        return self.water_model is not None and self.food_model is not None

    # ── Internal ───────────────────────────────────────────────────────────

    def _segment(
        self,
        model:       PSPNet,
        crop_bgr:    np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Run PSPNet on a crop.
        Returns (H, W) integer mask at original crop size.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            logger.warning("Empty crop passed to segmentation model")
            return np.zeros((1, 1), dtype=np.uint8)

        orig_h, orig_w = crop_bgr.shape[:2]
        tensor = preprocess_crop(crop_bgr, target_size=target_size)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = model(tensor)                              # (1, C, H, W)
            pred   = torch.argmax(logits, dim=1).squeeze(0)    # (H, W)
            mask   = pred.cpu().numpy().astype(np.uint8)

        # Resize mask back to original crop dimensions
        mask = cv2.resize(
            mask, (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,   # nearest neighbor for class maps
        )
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────────────────────────────────────

# Colors for each class (BGR format for OpenCV)
WATER_PALETTE = {
    0: (50,  50,  50),    # background      — dark grey
    1: (200, 200, 200),   # bottle wall     — light grey
    2: (200, 100,  0),    # water fill      — blue
    3: (50,   50, 50),    # empty air       — dark (transparent-ish)
}

FOOD_PALETTE = {
    0: (50,  50,  50),    # background      — dark grey
    1: (100, 100, 180),   # hopper frame    — muted purple
    2: (30,  140, 200),   # food pellets    — orange-brown
    3: (50,   50,  50),   # empty space     — dark
}


def overlay_mask(
    crop_bgr:  np.ndarray,
    mask:      np.ndarray,
    palette:   dict,
    alpha:     float = 0.5,
) -> np.ndarray:
    """
    Overlay a segmentation mask on a BGR crop image.
    Returns the blended visualization.
    """
    color_mask = np.zeros_like(crop_bgr)
    for cls_id, color in palette.items():
        color_mask[mask == cls_id] = color

    return cv2.addWeighted(crop_bgr, 1 - alpha, color_mask, alpha, 0)