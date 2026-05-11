# pipeline/annotator/opencv_annotator.py
import cv2
import numpy as np
from core.schemas import DetectionResult
from pipeline.annotator.base import BaseAnnotator

STATUS_COLORS = {
    "OK":       (80,  200,  80),
    "LOW":      (0,   180, 255),
    "CRITICAL": (0,     0, 255),
}

class OpenCVAnnotator(BaseAnnotator):
    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        viz = frame.copy()
        for bbox, label in [
            (result.water_bbox, f"water {result.water.pct:.1f}% [{result.water.status}]"),
            (result.food_bbox,  f"food  {result.food.pct:.1f}%  [{result.food.status}]"),
        ]:
            if bbox is not None:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                color = (255, 180, 0) if "water" in label else (0, 200, 80)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz, f"{label} ({bbox.conf:.2f})",
                            (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        summary = [
            ("OK",                f"Mice : {result.mouse_count}"),
            (result.water.status, f"Water: {result.water.pct:.1f}%  [{result.water.status}]"),
            (result.food.status,  f"Food : {result.food.pct:.1f}%  [{result.food.status}]"),
        ]
        for i, (status, line) in enumerate(summary):
            color = STATUS_COLORS.get(status, (160, 160, 160))
            y = 20 + i * 22
            cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, cv2.LINE_AA)
        return viz