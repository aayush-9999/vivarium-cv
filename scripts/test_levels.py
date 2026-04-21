"""
scripts/test_levels.py
======================
Runs the FULL pipeline on val images:
    YOLO detection → crop ROI → HSV level estimation → water % + food %

This is the same pipeline that runs in production (yolo_pipeline.py),
so what you see here is exactly what the API would return.

Output:
    runs/level_test/
        annotated/   ← images with boxes + level % overlaid
        results.txt  ← per-image: mouse, water%, water_status, food%, food_status

Usage:
    python scripts/test_levels.py

    # Custom paths / thresholds
    python scripts/test_levels.py --weights runs/detect/vivarium_v1/weights/best.pt
    python scripts/test_levels.py --val-dir dataset/split/val/images
    python scripts/test_levels.py --conf 0.45 --iou 0.30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make sure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/detect/vivarium_v1/weights/best.pt"
DEFAULT_VAL_DIR = "dataset/split/val/images"
DEFAULT_OUT_DIR = "runs/level_test"
DEFAULT_CONF    = 0.45
DEFAULT_IOU     = 0.30
IMGSZ           = 640

CLASS_COLORS = {
    0: (255,  80,  80),   # blue  — mouse
    1: (255, 180,   0),   # cyan  — water_container
    2: ( 80, 200,  80),   # green — food_area
}
STATUS_COLORS = {
    "OK":       (80,  200,  80),   # green
    "LOW":      (0,   180, 255),   # orange
    "CRITICAL": (0,    0,  255),   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_bbox(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_level_bar(img, x, y, w, h, pct, status, label):
    """
    Draw a vertical fill bar next to the container bbox showing level %.
    """
    bar_x  = x + w + 6
    bar_w  = 14
    bar_h  = h
    bar_y  = y

    # Background (empty)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # Fill (from bottom)
    fill_h   = int(bar_h * (pct / 100.0))
    fill_y   = bar_y + bar_h - fill_h
    color    = STATUS_COLORS.get(status, (200, 200, 200))
    cv2.rectangle(img, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), color, -1)

    # Border
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    # Percentage text
    pct_label = f"{pct:.0f}%"
    cv2.putText(img, pct_label,
                (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Status text below bar
    cv2.putText(img, status,
                (bar_x - 2, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def run_level_estimation(roi_frame, estimator):
    """
    Safely run level estimation — returns LevelReading or fallback CRITICAL.
    """
    try:
        if roi_frame is None or roi_frame.size == 0:
            from core.schemas import LevelReading
            return LevelReading(pct=0.0, status="CRITICAL")
        return estimator.read(roi_frame)
    except Exception as e:
        from core.schemas import LevelReading
        print(f"    [WARN] Level estimation failed: {e}")
        return LevelReading(pct=0.0, status="CRITICAL")


def crop_roi(frame, bbox, preprocessor, fallback_zone):
    """Crop to YOLO bbox if available, else fall back to config ROI zone."""
    if bbox is not None:
        h, w = frame.shape[:2]
        pad = 8
        x1 = max(0, int(bbox.x1) - pad)
        y1 = max(0, int(bbox.y1) - pad)
        x2 = min(w, int(bbox.x2) + pad)
        y2 = min(h, int(bbox.y2) + pad)
        return frame[y1:y2, x1:x2].copy()
    else:
        return preprocessor.apply_roi(frame, fallback_zone)


def main(weights, val_dir, out_dir, conf, iou):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        return

    from preprocessing.frame_preprocessor import FramePreprocessor
    from level_estimation.water_level import WaterLevelEstimator
    from level_estimation.food_level  import FoodLevelEstimator
    from detectors.yolo.postprocessor import extract_container_bboxes

    weights_path = Path(weights)
    val_path     = Path(val_dir)
    out_path     = Path(out_dir)
    ann_path     = out_path / "annotated"
    ann_path.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return
    if not val_path.exists():
        print(f"[ERROR] Val dir not found: {val_path}")
        return

    img_paths = sorted(
        p for p in val_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        print(f"[ERROR] No images in {val_path}")
        return

    print(f"Loading model  : {weights_path}")
    model        = YOLO(str(weights_path))
    preprocessor = FramePreprocessor()
    water_est    = WaterLevelEstimator()
    food_est     = FoodLevelEstimator()

    total       = len(img_paths)
    results_log = []

    print(f"Running full pipeline on {total} images  (conf={conf}, iou={iou})\n")
    print(f"  {'Image':<55} {'Mouse':>6}  {'Water':>12}  {'Food':>12}")
    print(f"  {'─'*55} {'─'*6}  {'─'*12}  {'─'*12}")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP: {img_path.name}")
            continue

        # ── Step 1: resize to 640×640 ─────────────────────────────
        frame_640 = preprocessor.resize(img)

        # ── Step 2: YOLO inference ────────────────────────────────
        results = model.predict(
            source=frame_640,
            conf=conf,
            iou=iou,
            imgsz=IMGSZ,
            verbose=False,
        )

        r = results[0]
        mouse_count = 0
        viz = frame_640.copy()

        # Draw all boxes
        if r.boxes is not None and len(r.boxes) > 0:
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs_  = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, c, cls in zip(boxes, confs_, classes):
                cls = int(cls)
                color = CLASS_COLORS.get(cls, (200, 200, 200))
                labels_map = {0: "mouse", 1: "water", 2: "food"}
                draw_bbox(viz,
                          int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                          f"{labels_map.get(cls, str(cls))} {c:.2f}", color)
                if cls == 0:
                    mouse_count += 1

        # ── Step 3: extract container bboxes ─────────────────────
        water_bbox, food_bbox = extract_container_bboxes(results)

        # ── Step 4: crop ROIs ─────────────────────────────────────
        water_roi = crop_roi(frame_640, water_bbox, preprocessor, "jug")
        food_roi  = crop_roi(frame_640, food_bbox,  preprocessor, "hopper")

        # ── Step 5: level estimation ──────────────────────────────
        water_reading = run_level_estimation(water_roi, water_est)
        food_reading  = run_level_estimation(food_roi,  food_est)

        # ── Step 6: draw level bars ───────────────────────────────
        if water_bbox is not None:
            wx1, wy1, wx2, wy2 = int(water_bbox.x1), int(water_bbox.y1), int(water_bbox.x2), int(water_bbox.y2)
            draw_level_bar(viz, wx1, wy1, wx2 - wx1, wy2 - wy1,
                           water_reading.pct, water_reading.status, "water")
        else:
            # Fallback zone — draw bar at config position
            from core.config import ROI_ZONES
            jx, jy, jw, jh = ROI_ZONES["default"]["jug"]
            draw_level_bar(viz, jx, jy, jw, jh,
                           water_reading.pct, water_reading.status, "water")

        if food_bbox is not None:
            fx1, fy1, fx2, fy2 = int(food_bbox.x1), int(food_bbox.y1), int(food_bbox.x2), int(food_bbox.y2)
            draw_level_bar(viz, fx1, fy1, fx2 - fx1, fy2 - fy1,
                           food_reading.pct, food_reading.status, "food")
        else:
            from core.config import ROI_ZONES
            hx, hy, hw, hh = ROI_ZONES["default"]["hopper"]
            draw_level_bar(viz, hx, hy, hw, hh,
                           food_reading.pct, food_reading.status, "food")

        # ── Step 7: stamp summary text on image ───────────────────
        summary_lines = [
            f"Mouse : {mouse_count}",
            f"Water : {water_reading.pct:.1f}%  [{water_reading.status}]",
            f"Food  : {food_reading.pct:.1f}%  [{food_reading.status}]",
        ]
        for i, line in enumerate(summary_lines):
            status_key = "OK"
            if "CRITICAL" in line: status_key = "CRITICAL"
            elif "LOW"      in line: status_key = "LOW"
            color = STATUS_COLORS.get(status_key, (220, 220, 220))
            cv2.putText(viz, line, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(viz, line, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,   1, cv2.LINE_AA)

        # ── Save ──────────────────────────────────────────────────
        cv2.imwrite(str(ann_path / img_path.name), viz,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        w_str = f"{water_reading.pct:5.1f}% {water_reading.status:8s}"
        f_str = f"{food_reading.pct:5.1f}% {food_reading.status:8s}"
        print(f"  [{idx:3d}/{total}] {img_path.name:<55} {mouse_count:>6}  {w_str}  {f_str}")

        results_log.append(
            f"{img_path.name}\t"
            f"mouse={mouse_count}\t"
            f"water_pct={water_reading.pct:.1f}\twater_status={water_reading.status}\t"
            f"food_pct={food_reading.pct:.1f}\tfood_status={food_reading.status}\t"
            f"water_method={water_est.last_method}\t"
            f"food_method={food_est.last_method}"
        )

    # ── Write results ─────────────────────────────────────────────
    results_txt = out_path / "results.txt"
    header = (
        f"conf={conf}  iou={iou}  weights={weights_path}\n"
        f"filename\tmouse\twater_pct\twater_status\tfood_pct\tfood_status"
        f"\twater_method\tfood_method\n"
    )
    results_txt.write_text(header + "\n".join(results_log), encoding="utf-8")

    print(f"\n{'─'*65}")
    print(f"Done")
    print(f"  Images processed : {total}")
    print(f"  Annotated images : {ann_path}")
    print(f"  Results log      : {results_txt}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Full pipeline test: YOLO + water/food level estimation on val images."
    )
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    ap.add_argument("--out",     default=DEFAULT_OUT_DIR)
    ap.add_argument("--conf",    type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou",     type=float, default=DEFAULT_IOU)
    args = ap.parse_args()
    main(args.weights, args.val_dir, args.out, args.conf, args.iou)