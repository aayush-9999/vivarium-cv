"""
scripts/test_val.py
===================
Runs inference on all images in the val split and saves
annotated output images to a separate folder.

Outputs:
    runs/val_test/
        ├── annotated/   ← images with YOLO boxes drawn
        └── results.txt  ← per-image summary (mouse count, water, food detections)

Usage:
    python scripts/test_val.py

    # Custom paths
    python scripts/test_val.py --weights runs/detect/vivarium_v1/weights/best.pt
    python scripts/test_val.py --val-dir dataset/split/val/images
    python scripts/test_val.py --out runs/val_test

    # Tune thresholds
    python scripts/test_val.py --conf 0.45 --iou 0.30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/detect/vivarium_v1/weights/best.pt"
DEFAULT_VAL_DIR = "dataset/split/val/images"
DEFAULT_OUT_DIR = "runs/val_test"
DEFAULT_CONF    = 0.45
DEFAULT_IOU     = 0.30
IMGSZ           = 640

CLASS_NAMES  = {0: "mouse", 1: "water_container", 2: "food_area"}
CLASS_COLORS = {
    0: (255,  80,  80),   # blue  — mouse
    1: (255, 180,   0),   # cyan  — water_container
    2: ( 80, 200,  80),   # green — food_area
}


# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(img: np.ndarray, boxes, confs, classes) -> np.ndarray:
    viz = img.copy()
    for box, conf, cls in zip(boxes, confs, classes):
        cls   = int(cls)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(viz, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(viz, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return viz


def summarise(classes) -> dict:
    counts = {0: 0, 1: 0, 2: 0}
    for c in classes:
        counts[int(c)] = counts.get(int(c), 0) + 1
    return counts


def main(weights: str, val_dir: str, out_dir: str, conf: float, iou: float) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        return

    weights_path = Path(weights)
    val_path     = Path(val_dir)
    out_path     = Path(out_dir)
    ann_path     = out_path / "annotated"
    ann_path.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return
    if not val_path.exists():
        print(f"[ERROR] Val directory not found: {val_path}")
        return

    img_paths = sorted(
        p for p in val_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        print(f"[ERROR] No images found in {val_path}")
        return

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    total      = len(img_paths)
    results_log = []

    print(f"Running inference on {total} val images  (conf={conf}, iou={iou})\n")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx:4d}/{total}] SKIP (unreadable): {img_path.name}")
            continue

        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=IMGSZ,
            verbose=False,
        )

        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs_  = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            counts  = summarise(classes)
            annotated = draw_boxes(img, boxes, confs_, classes)
        else:
            boxes   = []
            counts  = {0: 0, 1: 0, 2: 0}
            annotated = img.copy()
            # Stamp "NO DETECTIONS" on image
            cv2.putText(annotated, "NO DETECTIONS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Save annotated image
        out_img_path = ann_path / img_path.name
        cv2.imwrite(str(out_img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        status = (
            f"mouse={counts[0]}  water={counts[1]}  food={counts[2]}"
            if any(counts.values()) else "⚠  NO DETECTIONS"
        )
        print(f"  [{idx:4d}/{total}] {img_path.name:<60} {status}")

        results_log.append(
            f"{img_path.name}\tmouse={counts[0]}\twater={counts[1]}\tfood={counts[2]}"
        )

    # Write results summary
    results_txt = out_path / "results.txt"
    header = f"conf={conf}  iou={iou}  weights={weights_path}\n"
    header += "filename\tmouse\twater\tfood\n"
    header += "─" * 60 + "\n"
    results_txt.write_text(header + "\n".join(results_log), encoding="utf-8")

    # Count stats
    no_det = sum(1 for l in results_log if "mouse=0\twater=0\tfood=0" in l)

    print(f"""
{'─'*65}
Done
  Images processed : {total}
  No detections    : {no_det}
  Annotated images : {ann_path}
  Results log      : {results_txt}
{'─'*65}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test YOLO on val split and save annotated outputs."
    )
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help=f"Path to best.pt (default: {DEFAULT_WEIGHTS})")
    ap.add_argument("--val-dir", default=DEFAULT_VAL_DIR,
                    help=f"Val images folder (default: {DEFAULT_VAL_DIR})")
    ap.add_argument("--out",     default=DEFAULT_OUT_DIR,
                    help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--conf",    type=float, default=DEFAULT_CONF,
                    help=f"Confidence threshold (default: {DEFAULT_CONF})")
    ap.add_argument("--iou",     type=float, default=DEFAULT_IOU,
                    help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    args = ap.parse_args()
    main(args.weights, args.val_dir, args.out, args.conf, args.iou)