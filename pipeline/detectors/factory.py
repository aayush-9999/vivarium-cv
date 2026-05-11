# pipeline/detectors/factory.py
import os
from core.config_loader import CONFIG

def get_detector(cage_type: str = "default"):
    backend = CONFIG.get("backend", os.getenv("BACKEND", "yolo")).lower()
    if backend in ("yolo", "yolo_psp"):
        from pipeline.detectors.yolo.yolo_detector import YOLODetector
        return YOLODetector(
            weights_path=CONFIG["yolox"]["weights"],
            device=CONFIG["device"],
        )
    elif backend == "ssd":
        from pipeline.detectors.ssd.ssd_detector import SSDDetector
        return SSDDetector(weights_path=CONFIG["ssd"]["weights"])
    raise ValueError(f"Unknown backend: {backend}")