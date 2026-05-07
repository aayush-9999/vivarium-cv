# pipeline/pipeline_factory.py  (UPDATED — replaces existing file)
"""
Factory for runtime backend selection + orchestrator access.

Now supports three pipeline modes:
    BACKEND=yolo      → pure YOLOX (original, 4-bucket level classification)
    BACKEND=yolo_psp  → YOLOX detection + PSPNet level estimation (NEW — continuous %)
    BACKEND=ssd       → SSD MobileNet (existing)

Set in .env:
    BACKEND=yolo_psp
    PSP_WATER_WEIGHTS=models/psp/water_best.pth
    PSP_FOOD_WEIGHTS=models/psp/food_best.pth
    PSP_BACKBONE=resnet50
"""

from __future__ import annotations

import os
from typing import Optional

from pipeline.orchestrator import VivariumOrchestrator, OrchestratorConfig


# ── Inference pipeline ────────────────────────────────────────────────────────

def get_pipeline(cage_type: str = "default"):
    """
    Return the inference pipeline for the active backend.

    Backends:
        yolo     : pure YOLOX, 4-bucket level classification (original)
        yolo_psp : YOLOX bbox detection + PSPNet continuous level estimation (NEW)
        ssd      : SSD MobileNet (existing)
    """
    backend = os.getenv("BACKEND", "yolo").lower()

    if backend == "yolo":
        from pipeline.yolo_pipeline import YOLOPipeline
        return YOLOPipeline(cage_type=cage_type)

    elif backend == "yolo_psp":
        from pipeline.yolo_psp_pipeline import YOLOPSPPipeline
        return YOLOPSPPipeline(
            cage_type=cage_type,
            water_psp_weights=os.getenv("PSP_WATER_WEIGHTS"),
            food_psp_weights=os.getenv("PSP_FOOD_WEIGHTS"),
            psp_backbone=os.getenv("PSP_BACKBONE", "resnet50"),
            fallback_to_yolox=True,
        )

    elif backend == "ssd":
        from pipeline.ssd_pipeline import SSDPipeline
        return SSDPipeline(cage_type=cage_type)

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Set BACKEND=yolo, BACKEND=yolo_psp, or BACKEND=ssd in your .env file."
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> VivariumOrchestrator:
    """
    Return a VivariumOrchestrator — single entry-point for all workflows.
    PSPNet training is also accessible through the orchestrator.
    """
    return VivariumOrchestrator(config=config)