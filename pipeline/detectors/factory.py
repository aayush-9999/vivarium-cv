# pipeline/detectors/factory.py
"""
Detector factory.

Returns the right detector instance for the active (or requested) backend.
All callers should go through get_detector() rather than importing a
concrete detector class directly.
"""

from __future__ import annotations

from core.config_loader import CONFIG


def get_detector(
    cage_type: str = "default",
    backend: str | None = None,
):
    """
    Return a concrete BaseDetector for the requested backend.

    Parameters
    ----------
    cage_type : str
        Passed to detectors that are cage-type-aware (e.g. SSD).
    backend : str | None
        Override CONFIG["backend"].  "yolo" | "yolo_psp" | "ssd".
        If None, CONFIG["backend"] is used.

    Notes
    -----
    "yolo_psp" uses the same YOLODetector as "yolo" — PSPNet lives in
    the pipeline layer (pipeline/pipeline.py), not in the detector.
    """
    active = (backend or CONFIG["backend"]).lower()

    if active in ("yolo", "yolo_psp"):
        from pipeline.detectors.yolo.yolo_detector import YOLODetector
        return YOLODetector(
            weights_path = CONFIG["yolox"]["weights"],
            device       = CONFIG["device"],
        )

    if active == "ssd":
        from pipeline.detectors.ssd.ssd_detector import SSDDetector   # type: ignore[import]
        return SSDDetector(weights_path=CONFIG["ssd"]["weights"])

    raise ValueError(
        f"Unknown backend '{active}'. "
        "Set BACKEND=yolo, BACKEND=yolo_psp, or BACKEND=ssd in your .env file."
    )