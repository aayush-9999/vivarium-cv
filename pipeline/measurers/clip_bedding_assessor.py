# pipeline/measurers/clip_bedding_assessor.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.config_loader import CONFIG
from core.schemas import BeddingReading, BoundingBox

logger = logging.getLogger("vivarium.clip_bedding")

CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"
CLIP_LOCAL_PATH = os.getenv("CLIP_LOCAL_PATH", "")

CLEAN_PROMPTS = [
    "clean white wood shaving bedding in a mouse cage",
    "fresh dry rodent bedding with no soiling",
    "light coloured clean cage substrate",
]
DIRTY_PROMPTS = [
    "soiled wet bedding in a mouse cage with dark stains",
    "dirty discoloured cage bedding that needs changing",
    "heavily soiled rodent substrate with urine and droppings",
]

CLIP_DIRTY_THRESHOLD: float = 0.55
_AREA_THRESHOLD: float = CONFIG["bedding"]["area_threshold"]
FLOOR_ROI_FALLBACK = (0.20, 0.55, 0.80, 0.95)
_MIN_CROP_PX = 32


def _to_tensor(output) -> torch.Tensor:
    """
    Safely extract a plain tensor from whatever transformers returns.
    Handles all known return types across transformers versions:
      - plain torch.Tensor                  (some versions)
      - BaseModelOutputWithPooling          (.pooler_output)
      - CLIPTextModelOutput / CLIPVisionModelOutput  (.pooler_output)
    Falls back to the first tensor-valued attribute if nothing else matches.
    """
    if isinstance(output, torch.Tensor):
        return output

    # Try the most common attribute names in order of likelihood
    for attr in ("pooler_output", "last_hidden_state", "text_embeds", "image_embeds"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            # pooler_output is (B, D); last_hidden_state is (B, seq, D) — take CLS token
            if val.dim() == 3:
                val = val[:, 0, :]
            return val

    # Last resort: find any tensor attribute
    for val in vars(output).values():
        if isinstance(val, torch.Tensor) and val.dim() in (1, 2):
            return val

    raise AttributeError(
        f"Cannot extract tensor from {type(output)}. "
        f"Available attrs: {list(vars(output).keys())}"
    )


class ClipBeddingAssessor:

    def __init__(
        self,
        device:          Optional[str] = None,
        dirty_threshold: float = CLIP_DIRTY_THRESHOLD,
    ) -> None:
        self._device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dirty_threshold = dirty_threshold
        self._model           = None
        self._processor       = None
        self._text_features   = None
        self._load_model()

    def assess(
        self,
        frame:            np.ndarray,
        bedding_bbox:     Optional[BoundingBox] = None,
        bedding_area_pct: float = 0.0,
    ) -> BeddingReading:
        if self._model is None:
            logger.warning("CLIP model not loaded — returning not_detected()")
            return BeddingReading.not_detected()

        crop = self._get_crop(frame, bedding_bbox)
        if crop is None:
            logger.warning("Bedding crop degenerate — returning not_detected()")
            return BeddingReading.not_detected()

        try:
            dirty_prob = self._clip_dirty_probability(crop)
        except Exception as exc:
            logger.error("CLIP inference failed: %s — returning not_detected()", exc)
            return BeddingReading.not_detected()

        condition = self._decide_condition(dirty_prob, bedding_area_pct)
        logger.debug("Bedding: dirty_prob=%.3f  area_pct=%.1f%%  → %s",
                     dirty_prob, bedding_area_pct, condition)

        return BeddingReading(area_pct=round(bedding_area_pct, 2), condition=condition)

    def is_ready(self) -> bool:
        return self._model is not None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        from transformers import CLIPProcessor, CLIPModel

        local = Path(CLIP_LOCAL_PATH) if CLIP_LOCAL_PATH else None
        if local and local.exists():
            source = str(local)
            logger.info("Loading CLIP from local path: %s", source)
        else:
            source = CLIP_MODEL_ID
            logger.warning(
                "CLIP_LOCAL_PATH not set or not found — downloading from HuggingFace. "
                "Run this once to save locally and avoid re-downloading:\n"
                "  python -c \""
                "from transformers import CLIPModel, CLIPProcessor; "
                "CLIPModel.from_pretrained('openai/clip-vit-base-patch32').save_pretrained('models/clip/clip-vit-base-patch32'); "
                "CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32').save_pretrained('models/clip/clip-vit-base-patch32')"
                "\""
            )

        try:
            self._model     = CLIPModel.from_pretrained(source).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(source)
            logger.info("CLIP model ready (device=%s).", self._device)
        except Exception as exc:
            logger.error("Failed to load CLIP model: %s", exc)
            self._model = None

    # ── Crop extraction ───────────────────────────────────────────────────────

    def _get_crop(self, frame: np.ndarray, bedding_bbox: Optional[BoundingBox]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]

        if bedding_bbox is not None:
            x1 = max(0, int(bedding_bbox.x1))
            y1 = max(0, int(bedding_bbox.y1))
            x2 = min(w, int(bedding_bbox.x2))
            y2 = min(h, int(bedding_bbox.y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and (x2 - x1) >= _MIN_CROP_PX and (y2 - y1) >= _MIN_CROP_PX:
                return crop

        x1f, y1f, x2f, y2f = FLOOR_ROI_FALLBACK
        crop = frame[int(y1f * h):int(y2f * h), int(x1f * w):int(x2f * w)]
        if crop.size > 0 and crop.shape[1] >= _MIN_CROP_PX and crop.shape[0] >= _MIN_CROP_PX:
            return crop

        return None

    # ── CLIP inference ────────────────────────────────────────────────────────

    def _get_text_features(self) -> torch.Tensor:
        if self._text_features is not None:
            return self._text_features

        inputs = self._processor(
            text=CLEAN_PROMPTS + DIRTY_PROMPTS,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            raw   = self._model.get_text_features(**inputs)
            feats = _to_tensor(raw)                  # ← safe extraction
            feats = F.normalize(feats, dim=-1)

        self._text_features = feats
        logger.debug("CLIP text features cached (%d prompts).", len(CLEAN_PROMPTS + DIRTY_PROMPTS))
        return self._text_features

    def _clip_dirty_probability(self, crop_bgr: np.ndarray) -> float:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs  = self._processor(images=pil_img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            raw       = self._model.get_image_features(**inputs)
            img_feats = _to_tensor(raw)              # ← safe extraction
            img_feats = F.normalize(img_feats, dim=-1)

        text_feats = self._get_text_features()

        sims      = (img_feats @ text_feats.T).squeeze(0)
        sim_clean = sims[:len(CLEAN_PROMPTS)].mean().item()
        sim_dirty = sims[len(CLEAN_PROMPTS):].mean().item()

        dirty_prob = torch.softmax(torch.tensor([sim_clean, sim_dirty]), dim=0)[1].item()
        logger.debug("CLIP sims: clean=%.4f  dirty=%.4f  P(dirty)=%.4f",
                     sim_clean, sim_dirty, dirty_prob)
        return dirty_prob

    # ── Decision ──────────────────────────────────────────────────────────────

    def _decide_condition(self, dirty_prob: float, area_pct: float) -> str:
        clip_dirty  = dirty_prob >= self._dirty_threshold
        area_large  = area_pct   >= _AREA_THRESHOLD

        if area_pct == 0.0:
            condition = "BAD" if clip_dirty else "GOOD"
        else:
            condition = "BAD" if (clip_dirty and area_large) else "GOOD"

        logger.debug("Bedding decision: clip_dirty=%s  area_large=%s → %s",
                     clip_dirty, area_large, condition)
        return condition