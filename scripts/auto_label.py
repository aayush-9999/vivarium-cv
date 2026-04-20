"""
scripts/auto_label.py
=====================
Bootstrap YOLO annotations for vivarium images using a pretrained YOLOv8n
(COCO weights). Since 'mouse' is not a COCO class, we detect animals via
a broad set of animal-adjacent classes and let you manually verify/fix them.

Strategy:
    - Primary:  COCO class 15 = 'cat', 16 = 'dog', 20 = 'sheep'  (closest shapes)
    - Fallback: any detection with conf > FALLBACK_CONF in animal classes
    - All detections are written as class 0 (mouse) in output .txt files
    - A review_needed.txt is written listing images with 0 detections
      so you know which ones need manual labelling.

Usage:
    # Label augmented images (default)
    python scripts/auto_label.py

    # Label a specific folder
    python scripts/auto_label.py --src dataset/augmented/images --dst dataset/augmented/labels

    # Lower confidence threshold if too few detections
    python scripts/auto_label.py --conf 0.15

Output layout:
    dataset/augmented/labels/
        image_aug0001_wok_fok.txt    ← YOLO format: class cx cy w h (normalised)
        ...
    dataset/augmented/review_needed.txt  ← images with 0 mouse detections

YOLO label format (one line per detection):
    <class_id> <cx> <cy> <width> <height>
    All values normalised 0–1 relative to image dimensions.
    class_id is always 0 (mouse) regardless of what COCO detected.

After running:
    1. Open flagged images in LabelImg or Roboflow to fix bad boxes.
    2. Then run:  python scripts/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── COCO classes used as mouse proxies ───────────────────────────────────────
# Full COCO class list: https://cocodataset.org/#explore
# We use animal classes that share rough body shape / size with a lab mouse.
ANIMAL_CLASSES = {
    15: "cat",
    16: "dog",
    17: "horse",    # unlikely but catches large white blobs
    18: "sheep",
    19: "cow",
    20: "elephant",
    # Smaller objects that sometimes fire on rodents:
    74: "clock",    # round white objects — often false-fires on mice in top-down
}

PRIMARY_CLASSES  = {15, 16, 18}   # cat / dog / sheep — highest priority
FALLBACK_CLASSES = set(ANIMAL_CLASSES.keys()) - PRIMARY_CLASSES

PRIMARY_CONF  = 0.20   # lower than normal — rodents are small, conf is naturally low
FALLBACK_CONF = 0.30   # slightly stricter for less-similar classes

# ── Suppress non-animal COCO detections entirely ──────────────────────────────
IGNORE_CLASSES = set(range(80)) - set(ANIMAL_CLASSES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy bbox → normalised YOLO cx cy w h."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def filter_detections(boxes, confs, classes) -> list[tuple[float, float, float, float]]:
    """
    Return xyxy boxes kept after class + confidence filtering.
    All surviving boxes are treated as class 0 (mouse).
    """
    kept = []
    for box, conf, cls in zip(boxes, confs, classes):
        cls = int(cls)
        if cls in PRIMARY_CLASSES and conf >= PRIMARY_CONF:
            kept.append(tuple(float(v) for v in box))
        elif cls in FALLBACK_CLASSES and conf >= FALLBACK_CONF:
            kept.append(tuple(float(v) for v in box))
    return kept


def draw_debug(img: np.ndarray, boxes_xyxy: list, img_path: Path) -> None:
    """Save a debug JPEG with detected boxes drawn — goes to labels/../debug/."""
    debug_dir = img_path.parent.parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    viz = img.copy()
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 80), 2)
        cv2.putText(viz, "mouse?", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)

    out = debug_dir / img_path.name
    cv2.imwrite(str(out), viz, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(src: Path, dst: Path, conf: float, debug: bool) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not img_paths:
        print(f"[ERROR] No images found in {src}")
        sys.exit(1)

    print(f"Loading pretrained YOLOv8n (COCO) …")
    # Downloads ~6 MB on first run, cached in ~/.cache/ultralytics
    model = YOLO("yolov8n.pt")

    total_imgs      = len(img_paths)
    total_dets      = 0
    review_needed   = []

    print(f"Labelling {total_imgs} images from '{src}' …\n")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}, skipping.")
            continue

        img_h, img_w = img.shape[:2]

        results = model.predict(
            source=img_path,
            conf=PRIMARY_CONF,      # low threshold — we filter manually below
            iou=0.45,
            imgsz=640,
            verbose=False,
        )

        r = results[0]
        kept_boxes: list[tuple[float, float, float, float]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs   = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            kept_boxes = filter_detections(boxes, confs, classes)

        # Write YOLO label file
        label_path = dst / (img_path.stem + ".txt")
        lines = []
        for (x1, y1, x2, y2) in kept_boxes:
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"0 {cx} {cy} {bw} {bh}")

        label_path.write_text("\n".join(lines))

        if not kept_boxes:
            review_needed.append(img_path.name)

        if debug and kept_boxes:
            draw_debug(img, kept_boxes, img_path)

        total_dets += len(kept_boxes)

        status = f"{len(kept_boxes)} det(s)" if kept_boxes else "⚠ NO DETECTIONS"
        print(f"  [{idx:4d}/{total_imgs}] {img_path.name:<55} {status}")

    # Write review list
    review_path = dst.parent / "review_needed.txt"
    review_path.write_text("\n".join(review_needed))

    # ── Summary ───────────────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-labelling complete
  Images processed : {total_imgs}
  Total detections : {total_dets}
  Avg per image    : {total_dets / max(total_imgs, 1):.1f}
  Need review      : {len(review_needed)}  → {review_path}
  Labels written   : {dst}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:
  1. Open images in 'review_needed.txt' with LabelImg and add/fix boxes.
       pip install labelImg && labelImg {src} {dst / 'classes.txt'}
  2. Then run:  python scripts/train.py
""")

    # Write classes.txt for LabelImg compatibility
    classes_txt = dst / "classes.txt"
    classes_txt.write_text("mouse\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-label vivarium images with pretrained YOLOv8n")
    ap.add_argument("--src",   type=Path, default=Path("dataset/augmented/images"),
                    help="Folder of images to label")
    ap.add_argument("--dst",   type=Path, default=Path("dataset/augmented/labels"),
                    help="Output folder for .txt label files")
    ap.add_argument("--conf",  type=float, default=PRIMARY_CONF,
                    help=f"Minimum confidence for primary animal classes (default {PRIMARY_CONF})")
    ap.add_argument("--debug", action="store_true",
                    help="Save debug images with drawn boxes to dataset/augmented/debug/")
    args = ap.parse_args()

    main(args.src, args.dst, args.conf, args.debug)