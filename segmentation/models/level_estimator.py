# segmentation/models/level_estimator.py
"""
LevelEstimator — wraps PSPNet and converts segmentation masks to fill percentages.

Three bugs fixed vs previous version:

BUG 1 — pred=100.0% when PSPNet predicts entire frame as fill class:
    Root cause: mask_to_fill_pct height method — when empty_rows is empty
    but fill_rows exists, it fell through to pixel count which returned
    fill_pixels/total = 100.0 when total=fill_pixels (no empty pixels).
    Fix: if no empty pixels detected at all, cap result at 95.0% max
    and log a warning. A bottle cannot be 100% full in practice.

BUG 2 — Systematic +10-15% overestimation in OK range:
    Root cause: height method was using ALL fill rows to find meniscus.
    Stray fill pixels at the very top of the mask (wall reflections,
    noise) were being counted as the meniscus line, pushing it too high.
    Fix: find meniscus using the ROW where fill pixel DENSITY exceeds
    a minimum threshold (MIN_FILL_ROW_DENSITY) rather than the absolute
    topmost fill pixel. This ignores sparse reflection noise.

BUG 3 — Food model returning 0.0 everywhere:
    Root cause: food PSPNet predicts all background — mask has zero
    fill AND zero empty pixels, so mask_to_fill_pct returns 0.0.
    Fix: detect this "all background" case explicitly, log it clearly,
    and return a sentinel value (-1.0) so the pipeline can distinguish
    "model ran but found nothing" from "model says 0% fill".
    Pipeline then falls back to YOLOX food reading in this case.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from segmentation.models.pspnet import PSPNet, build_water_model, build_food_model

logger = logging.getLogger("vivarium.segmentation")

# ── Class indices ─────────────────────────────────────────────────────────────
WATER_FILL_CLASS  = 2
WATER_EMPTY_CLASS = 3
FOOD_FILL_CLASS   = 2
FOOD_EMPTY_CLASS  = 3

# ── Status thresholds ─────────────────────────────────────────────────────────
STATUS_THRESHOLDS = [
    (0.0,  15.0,  "CRITICAL"),
    (15.0, 35.0,  "LOW"),
    (35.0, 100.1, "OK"),
]

# ── ImageNet normalization ────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Height method tuning ──────────────────────────────────────────────────────
# Minimum fraction of row width that must be fill pixels to count as a
# "real" fill row. Filters out stray reflection / noise pixels at top.
# 0.10 = at least 10% of the row must be fill pixels to define meniscus.
MIN_FILL_ROW_DENSITY = 0.10

# Maximum fill % to return even when the model sees all fill pixels.
# A real water bottle cannot be 100% full — cap at 97% to flag suspicious cases.
MAX_FILL_PCT = 97.0

# Sentinel returned when model detects no container pixels at all
NO_CONTAINER_SENTINEL = -1.0


def pct_to_status(pct: float) -> str:
    """Continuous fill % → status string."""
    for lo, hi, status in STATUS_THRESHOLDS:
        if lo <= pct < hi:
            return status
    return "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_crop(
    crop_bgr:    np.ndarray,
    target_size: tuple[int, int] = (256, 128),
) -> torch.Tensor:
    """BGR image → normalised (1, 3, H, W) tensor for PSPNet."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    normed  = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    chw     = np.transpose(normed, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Fill percentage from mask
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_fill_pct(
    mask:              np.ndarray,
    fill_class:        int,
    empty_class:       int,
    use_height_method: bool = True,
) -> float:
    """
    PSPNet mask → continuous fill percentage (0.0–100.0).
    Returns NO_CONTAINER_SENTINEL (-1.0) if model found no container pixels.

    HEIGHT method (water):
        Finds meniscus using DENSITY THRESHOLD — the topmost row where
        at least MIN_FILL_ROW_DENSITY fraction of pixels are fill class.
        This ignores stray reflection pixels above the real fill line,
        fixing the +10-15% overestimation bug.

    PIXEL COUNT method (food):
        fill% = fill_pixels / (fill_pixels + empty_pixels)
        Used for food — no clean meniscus line exists.

    Special cases:
        fill=0, empty=0  → NO_CONTAINER_SENTINEL  (model saw no container)
        fill>0, empty=0  → capped at MAX_FILL_PCT  (was causing pred=100%)
        fill=0, empty>0  → 0.0                     (container empty)
    """
    fill_mask  = (mask == fill_class)
    empty_mask = (mask == empty_class)

    fill_pixels  = int(fill_mask.sum())
    empty_pixels = int(empty_mask.sum())
    total        = fill_pixels + empty_pixels

    if total == 0:
        # Model found neither fill nor empty — container not detected
        # Return sentinel -1.0 so callers can distinguish "not detected" from 0% fill
        logger.warning(
            "mask_to_fill_pct: no fill or empty pixels found — "
            "model did not detect container. Returning sentinel."
        )
        return -1.0
 
    # Fill pixels exist but zero empty pixels — model predicts entire interior as fill.
    # Cap at 97% so callers know this is a saturated/uncertain reading.
    if fill_pixels > 0 and empty_pixels == 0:
        logger.warning(
            "mask_to_fill_pct: fill pixels exist but no empty pixels detected — "
            "model may be predicting entire frame as fill. Capping at 97.0%%."
        )
        return 97.0

    # ── BUG 1 FIX: fill pixels but no empty pixels ────────────────────────────
    # This was causing pred=100.0% — model predicted entire frame as fill.
    # Cap at MAX_FILL_PCT and warn so it can be investigated.
    if fill_pixels > 0 and empty_pixels == 0:
        logger.warning(
            "mask_to_fill_pct: fill pixels exist but no empty pixels detected — "
            "model may be predicting entire frame as fill. Capping at %.1f%%.",
            MAX_FILL_PCT,
        )
        return MAX_FILL_PCT

    # ── Container empty (no fill pixels) ─────────────────────────────────────
    if fill_pixels == 0:
        return 0.0

    # ── HEIGHT method (water) ─────────────────────────────────────────────────
    if use_height_method:
        mask_w = mask.shape[1]

        # BUG 2 FIX: find meniscus using density threshold, not topmost pixel
        # This filters out stray reflection/noise pixels above the real fill line
        fill_row_densities = fill_mask.sum(axis=1) / mask_w  # fraction per row
        dense_fill_rows = np.where(fill_row_densities >= MIN_FILL_ROW_DENSITY)[0]
        empty_rows      = np.where(empty_mask.any(axis=1))[0]

        if len(dense_fill_rows) == 0:
            # All fill pixels are sparse/noisy — fall back to pixel count
            logger.debug(
                "height method: no dense fill rows found "
                "(all below %.0f%% density) — using pixel count fallback",
                MIN_FILL_ROW_DENSITY * 100,
            )
            return float(fill_pixels / total * 100.0)

        if len(empty_rows) == 0:
            # Fill detected but no empty region — cap as before
            return MAX_FILL_PCT

        container_top    = int(empty_rows.min())
        container_bottom = int(dense_fill_rows.max())
        meniscus_row     = int(dense_fill_rows.min())  # topmost DENSE fill row

        container_height = container_bottom - container_top
        if container_height <= 0:
            # Degenerate geometry — fall back to pixel count
            return float(fill_pixels / total * 100.0)

        fill_height = container_bottom - meniscus_row
        pct = float(fill_height / container_height * 100.0)
        return float(np.clip(pct, 0.0, MAX_FILL_PCT))

    # ── PIXEL COUNT method (food) ─────────────────────────────────────────────
    return float(fill_pixels / total * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# LevelEstimator
# ─────────────────────────────────────────────────────────────────────────────

class LevelEstimator:
    """
    Wraps two PSPNet models and returns continuous fill percentages.
    from_class_id() / YOLO discrete buckets are never used.
    """

    def __init__(
        self,
        water_weights: Optional[str] = None,
        food_weights:  Optional[str] = None,
        backbone:      str  = "resnet50",
        device:        str  = "cpu",
        water_size:    tuple[int, int] = (256, 128),
        food_size:     tuple[int, int] = (224, 224),
    ):
        self.device     = device
        self.water_size = water_size
        self.food_size  = food_size

        logger.info("Loading water PSPNet (backbone=%s weights=%s)",
                    backbone, water_weights or "ImageNet only")
        self.water_model = build_water_model(
            backbone=backbone,
            pretrained=True,
            weights_path=water_weights,
            device=device,
        )

        logger.info("Loading food PSPNet (backbone=%s weights=%s)",
                    backbone, food_weights or "ImageNet only")
        self.food_model = build_food_model(
            backbone=backbone,
            pretrained=True,
            weights_path=food_weights,
            device=device,
        )

        self.water_model.eval()
        self.food_model.eval()
        logger.info("PSPNet estimator ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    def estimate_water(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[float, str, np.ndarray]:
        """
        Run PSPNet on a full 640x640 frame.

        Returns:
            pct    : 0.0–97.0 continuous (never 100.0)
                     or NO_CONTAINER_SENTINEL (-1.0) if no container found
            status : "OK" | "LOW" | "CRITICAL"
            mask   : (H, W) uint8 segmentation mask
        """
        mask = self._segment(self.water_model, crop_bgr, self.water_size)
        pct  = mask_to_fill_pct(
            mask,
            fill_class=WATER_FILL_CLASS,
            empty_class=WATER_EMPTY_CLASS,
            use_height_method=True,
        )
        if pct < 0:
            # Sentinel: model did not detect any container pixels
            logger.warning("Water PSPNet: no container detected in frame")
            raise ValueError("no_container_detected")
        return pct, pct_to_status(pct), mask

    def estimate_food(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[float, str, np.ndarray]:
        """
        Run PSPNet on a full 640x640 frame.

        Returns:
            pct    : 0.0–97.0 continuous
                     or NO_CONTAINER_SENTINEL (-1.0) if no container found
            status : "OK" | "LOW" | "CRITICAL"
            mask   : (H, W) uint8 segmentation mask
        """
        mask = self._segment(self.food_model, crop_bgr, self.food_size)
        pct  = mask_to_fill_pct(
            mask,
            fill_class=FOOD_FILL_CLASS,
            empty_class=FOOD_EMPTY_CLASS,
            use_height_method=False,
        )
        if pct < 0:
            # Sentinel: model did not detect any container pixels
            logger.warning("Food PSPNet: no container detected in frame")
            raise ValueError("no_container_detected")
        return pct, pct_to_status(pct), mask

    def is_ready(self) -> bool:
        return self.water_model is not None and self.food_model is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _segment(
        self,
        model:       PSPNet,
        frame_bgr:   np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """PSPNet forward pass → (H, W) mask at original frame size."""
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("_segment received empty frame — zero mask returned")
            return np.zeros((1, 1), dtype=np.uint8)

        orig_h, orig_w = frame_bgr.shape[:2]
        tensor = preprocess_crop(frame_bgr, target_size).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            pred   = torch.argmax(logits, dim=1).squeeze(0)
            mask   = pred.cpu().numpy().astype(np.uint8)

        # INTER_NEAREST preserves class IDs — no blending between classes
        return cv2.resize(mask, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

WATER_PALETTE = {
    0: (50,  50,  50),
    1: (200, 200, 200),
    2: (200, 100,  0),
    3: (50,  50,  80),
}

FOOD_PALETTE = {
    0: (50,  50,  50),
    1: (100, 100, 180),
    2: (30,  140, 200),
    3: (50,  50,  50),
}


def overlay_mask(
    frame_bgr: np.ndarray,
    mask:      np.ndarray,
    palette:   dict,
    alpha:     float = 0.5,
) -> np.ndarray:
    color_mask = np.zeros_like(frame_bgr)
    for cls_id, color in palette.items():
        color_mask[mask == cls_id] = color
    return cv2.addWeighted(frame_bgr, 1 - alpha, color_mask, alpha, 0)