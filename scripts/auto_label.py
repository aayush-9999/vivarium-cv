"""
scripts/auto_label.py
=====================
Bootstrap YOLO annotations for vivarium images using pretrained YOLOv8n (COCO).

3-class output:
    class 0 = mouse            (COCO animal proxies)
    class 1 = water_container  (COCO bottle=39, cup=41, bowl=45)
    class 2 = food_area        (COCO bowl=45, dining table=60, dish-adjacent shapes)

THE FIX vs old version:
    Previously everything was written as class 0. Now each detected object gets
    its correct class ID (0, 1, or 2) based on what COCO class fired.

Usage:
    python scripts/auto_label.py
    python scripts/auto_label.py --src dataset/augmented/images --dst dataset/augmented/labels
    python scripts/auto_label.py --conf 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── COCO class → Vivarium class mapping ──────────────────────────────────────
#
# Vivarium class 0: mouse
MOUSE_COCO_CLASSES = {
    15: "cat",      # closest body shape proxy for a mouse in COCO
    16: "dog",
    18: "sheep",    # sometimes fires on huddled white mice
}
MOUSE_CONF = 0.20   # low — rodents are small, COCO was never trained on mice

# Vivarium class 1: water_container (jug / bottle / sipper tube)
WATER_COCO_CLASSES = {
    39: "bottle",   # closest match — lab water bottles/jugs
    41: "cup",
    75: "vase",     # sometimes fires on transparent jugs
}
WATER_CONF = 0.25

# Vivarium class 2: food_area (hopper / dish / feed pile)
FOOD_COCO_CLASSES = {
    45: "bowl",          # closest match — food dish / hopper opening
    60: "dining table",  # fires on flat food areas in overhead shots
    46: "wine glass",    # occasionally fires on conical hoppers
}
FOOD_CONF = 0.25

# Build unified lookup: coco_cls → (vivarium_cls, min_conf)
COCO_TO_VIVARIUM: dict[int, tuple[int, float]] = {}
for coco_cls in MOUSE_COCO_CLASSES:
    COCO_TO_VIVARIUM[coco_cls] = (0, MOUSE_CONF)
for coco_cls in WATER_COCO_CLASSES:
    COCO_TO_VIVARIUM[coco_cls] = (1, WATER_CONF)
for coco_cls in FOOD_COCO_CLASSES:
    COCO_TO_VIVARIUM[coco_cls] = (2, FOOD_CONF)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def filter_detections(
    boxes: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
) -> list[tuple[int, float, float, float, float, float]]:
    """
    Returns list of (vivarium_class_id, x1, y1, x2, y2) for each kept detection.
    Each detection gets the CORRECT class id — not hardcoded to 0.
    """
    kept = []
    for box, conf, cls in zip(boxes, confs, classes):
        coco_cls = int(cls)
        if coco_cls not in COCO_TO_VIVARIUM:
            continue

        viv_cls, min_conf = COCO_TO_VIVARIUM[coco_cls]
        if conf >= min_conf:
            kept.append((viv_cls, float(box[0]), float(box[1]), float(box[2]), float(box[3])))

    return kept


def draw_debug(
    img: np.ndarray,
    detections: list[tuple[int, float, float, float, float]],
    img_path: Path,
) -> None:
    debug_dir = img_path.parent.parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    CLASS_COLORS = {
        0: (0,   255,  80),    # green  — mouse
        1: (255, 180,   0),    # blue   — water_container
        2: (0,   140, 255),    # orange — food_area
    }
    CLASS_LABELS = {0: "mouse", 1: "water", 2: "food"}

    viz = img.copy()
    for (viv_cls, x1, y1, x2, y2) in detections:
        color = CLASS_COLORS.get(viv_cls, (200, 200, 200))
        label = CLASS_LABELS.get(viv_cls, str(viv_cls))
        cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(viz, label, (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(debug_dir / img_path.name), viz, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(src: Path, dst: Path, conf_override: float | None, debug: bool) -> None:
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

    print("Loading pretrained YOLOv8n (COCO) …")
    model = YOLO("yolov8n.pt")

    total_imgs    = len(img_paths)
    total_dets    = 0
    review_needed = []

    print(f"Labelling {total_imgs} images …\n")
    print(f"  Class 0 = mouse  (COCO proxies: cat/dog/sheep, conf≥{MOUSE_CONF})")
    print(f"  Class 1 = water  (COCO proxies: bottle/cup/vase, conf≥{WATER_CONF})")
    print(f"  Class 2 = food   (COCO proxies: bowl/table, conf≥{FOOD_CONF})\n")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}, skipping.")
            continue

        img_h, img_w = img.shape[:2]

        # Use the lowest configured threshold for the model call —
        # we filter per-class below
        model_conf = conf_override if conf_override else min(MOUSE_CONF, WATER_CONF, FOOD_CONF)

        results = model.predict(
            source=img_path,
            conf=model_conf,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )

        r = results[0]
        kept: list[tuple[int, float, float, float, float]] = []

        if r.boxes is not None and len(r.boxes) > 0:
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs   = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            kept    = filter_detections(boxes, confs, classes)

        # Write YOLO label file — with CORRECT class IDs
        label_path = dst / (img_path.stem + ".txt")
        lines = []
        for (viv_cls, x1, y1, x2, y2) in kept:
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{viv_cls} {cx} {cy} {bw} {bh}")

        label_path.write_text("\n".join(lines))

        if not kept:
            review_needed.append(img_path.name)

        if debug and kept:
            draw_debug(img, kept, img_path)

        total_dets += len(kept)

        counts = {0: 0, 1: 0, 2: 0}
        for (cls, *_) in kept:
            counts[cls] = counts.get(cls, 0) + 1

        status = (
            f"mouse={counts[0]} water={counts[1]} food={counts[2]}"
            if kept else "⚠ NO DETECTIONS"
        )
        print(f"  [{idx:4d}/{total_imgs}] {img_path.name:<55} {status}")

    review_path = dst.parent / "review_needed.txt"
    review_path.write_text("\n".join(review_needed))

    # Write classes.txt for LabelImg
    classes_txt = dst / "classes.txt"
    classes_txt.write_text("mouse\nwater_container\nfood_area\n")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-labelling complete
  Images processed : {total_imgs}
  Total detections : {total_dets}
  Avg per image    : {total_dets / max(total_imgs, 1):.1f}
  Need review      : {len(review_needed)}  → {review_path}
  Labels written   : {dst}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOTE: COCO was not trained on mice or lab equipment.
      Water/food container detection quality depends on visual similarity
      to bottles/bowls in COCO. Review debug/ images and fix with LabelImg:

        pip install labelImg
        labelImg {src} {dst / 'classes.txt'}

Then run:  python scripts/split_dataset.py && python scripts/train.py
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   type=Path, default=Path("dataset/augmented/images"))
    ap.add_argument("--dst",   type=Path, default=Path("dataset/augmented/labels"))
    ap.add_argument("--conf",  type=float, default=None,
                    help="Override minimum confidence for all classes")
    ap.add_argument("--debug", action="store_true",
                    help="Save debug images with coloured boxes per class")
    args = ap.parse_args()

    main(args.src, args.dst, args.conf, args.debug)