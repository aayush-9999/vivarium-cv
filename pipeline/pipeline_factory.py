# pipeline/pipeline_factory.py
"""
Factory for runtime backend selection + orchestrator access.

Usage:
    # Inference backend (existing behaviour)
    from pipeline.pipeline_factory import get_pipeline
    pipeline = get_pipeline()

    # Full orchestrator (data prep, training, validation, inference)
    from pipeline.pipeline_factory import get_orchestrator
    orch = get_orchestrator()
"""

from __future__ import annotations

import os
from typing import Optional

from pipeline.orchestrator import VivariumOrchestrator, OrchestratorConfig


# ── Inference pipeline (unchanged API) ───────────────────────────────────────

def get_pipeline(cage_type: str = "default"):
    """
    Return the inference pipeline for the active backend (BACKEND env var).
    Backend: 'yolo' (default) | 'ssd'
    """
    backend = os.getenv("BACKEND", "yolo").lower()

    if backend == "yolo":
        from pipeline.yolo_pipeline import YOLOPipeline
        return YOLOPipeline(cage_type=cage_type)

    elif backend == "ssd":
        from pipeline.ssd_pipeline import SSDPipeline
        return SSDPipeline(cage_type=cage_type)

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Set BACKEND=yolo or BACKEND=ssd in your .env file."
        )


# ── Orchestrator (data prep + training + inference) ───────────────────────────

def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> VivariumOrchestrator:
    """
    Return a VivariumOrchestrator — the single entry-point for all workflows.

    Args:
        config: Optional OrchestratorConfig to override defaults.

    Example:
        from pipeline.pipeline_factory import get_orchestrator

        orch = get_orchestrator()

        # Data pipeline
        orch.run_data_pipeline(n_augments=50)

        # Training
        orch.train(epochs=100)

        # Single frame inference
        result = orch.infer(frame=bgr_frame, cage_id="cage_01")
    """
    return VivariumOrchestrator(config=config)