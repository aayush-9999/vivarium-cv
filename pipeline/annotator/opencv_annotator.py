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

BEDDING_COLORS = {
    # YOLOX 4-class path (from_class_id)
    "PERFECT":      ( 80, 200,  80),   # green
    "OK":           (  0, 180, 255),   # amber-ish
    "BAD":          (  0, 100, 255),   # orange-red
    "WORST":        (  0,   0, 255),   # red
    # ClipBeddingAssessor binary path
    "GOOD":         ( 80, 200,  80),   # same as PERFECT
    # BeddingReading.not_detected() sentinel
    "NOT_DETECTED": (160, 160, 160),   # grey
}

# Mouse bbox colour — distinct cyan so it doesn't clash with water/food
MOUSE_COLOR = (255, 220, 0)    # bright cyan-yellow

class OpenCVAnnotator(BaseAnnotator):
    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        viz = frame.copy()
        h, w = frame.shape[:2]

        # ── Scale font/thickness relative to frame size ───────────────────
        scale_factor  = min(w, h) / 640.0          # 1.0 at 640px, 0.4 at 255px
        font_scale    = max(0.3, 0.45 * scale_factor)
        thickness     = max(1, int(2 * scale_factor))
        text_offset   = max(8, int(16 * scale_factor))
        summary_scale = max(0.3, 0.6 * scale_factor)
        summary_step  = max(12, int(22 * scale_factor))
        # ─────────────────────────────────────────────────────────────────

        # ── Per-mouse bounding boxes ──────────────────────────────────────
        mouse_bboxes = getattr(result, "mouse_bboxes", None) or []
        for i, bbox in enumerate(mouse_bboxes, start=1):
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cv2.rectangle(viz, (x1, y1), (x2, y2), MOUSE_COLOR, thickness)
            cv2.putText(
                viz,
                f"mouse {i} ({bbox.conf:.2f})",
                (x1 + 4, y1 + text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, MOUSE_COLOR, 1, cv2.LINE_AA,
            )

        # ── Water / food bounding boxes ───────────────────────────────────
        container_items = [
            (result.water_bbox,
             f"water {result.water.pct:.1f}% [{result.water.status}]"),
            (result.food_bbox,
             f"food  {result.food.pct:.1f}%  [{result.food.status}]"),
        ]
        for bbox, label in container_items:
            if bbox is not None:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                color = (255, 180, 0) if "water" in label else (0, 200, 80)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(viz, f"{label} ({bbox.conf:.2f})",
                            (x1 + 4, y1 + text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # ── Bedding bbox ──────────────────────────────────────────────────
        if result.bedding_bbox is not None:
            bb = result.bedding_bbox
            x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
            b_color = BEDDING_COLORS.get(result.bedding.condition, (200, 200, 200))
            cv2.rectangle(viz, (x1, y1), (x2, y2), b_color, thickness)
            b_label = (
                f"bedding {result.bedding.area_pct:.0f}% [{result.bedding.condition}]"
                f" ({bb.conf:.2f})"
            )
            cv2.putText(viz, b_label,
                        (x1 + 4, y1 + text_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, b_color, 1, cv2.LINE_AA)

        # ── Summary overlay ───────────────────────────────────────────────
        summary = [
            ("OK",                f"Mice    : {result.mouse_count}"),
            (result.water.status, f"Water   : {result.water.pct:.1f}% [{result.water.status}]"),
            (result.food.status,  f"Food    : {result.food.pct:.1f}%  [{result.food.status}]"),
            (None,                f"Bedding : {result.bedding.area_pct:.0f}% [{result.bedding.condition}]"),
        ]

        for i, (status, line) in enumerate(summary):
            if status is None:
                color = BEDDING_COLORS.get(result.bedding.condition, (160, 160, 160))
            else:
                color = STATUS_COLORS.get(status, (160, 160, 160))
            y = 12 + i * summary_step
            cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        summary_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(viz, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        summary_scale, color, 1, cv2.LINE_AA)
        return viz