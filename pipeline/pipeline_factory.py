# pipeline/pipeline_factory.py
"""
Factory helpers — the only two functions the rest of the codebase needs.

    get_pipeline()      → InferencePipeline   (inference / API use)
    get_orchestrator()  → VivariumOrchestrator (training / data-prep use)

Backend is selected from CONFIG["backend"] (set via BACKEND= in .env):
    "yolo"     → YOLOX only, discrete 4-bucket levels
    "yolo_psp" → YOLOX + PSPNet continuous levels
    "ssd"      → SSD MobileNet (legacy)
"""

from __future__ import annotations

from typing import Optional

from pipeline.pipeline import InferencePipeline
from pipeline.orchestrator import VivariumOrchestrator, OrchestratorConfig


def get_pipeline(
    cage_type: str = "default",
    backend: Optional[str] = None,
) -> InferencePipeline:
    """
    Return a ready-to-use InferencePipeline.

    Parameters
    ----------
    cage_type : str
        Cage configuration key (maps to ROI zones in config.py).
    backend : str | None
        Override the backend from .env.  "yolo" | "yolo_psp" | "ssd".
        If None, CONFIG["backend"] is used.
    """
    return InferencePipeline(cage_type=cage_type, backend=backend)


def get_orchestrator(
    config: Optional[OrchestratorConfig] = None,
) -> VivariumOrchestrator:
    """
    Return a VivariumOrchestrator for data-prep / training workflows.

    Parameters
    ----------
    config : OrchestratorConfig | None
        Optional config override.  Uses sensible env-based defaults if None.
    """
    return VivariumOrchestrator(config=config)