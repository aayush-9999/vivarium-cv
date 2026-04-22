"""
scripts/test_single.py
======================
Run the full pipeline on a SINGLE image and save annotated output.

Usage:
    python scripts/test_single.py --image path/to/image.jpg
    python scripts/test_single.py --image path/to/image.jpg --weights runs/detect/vivarium_v1/weights/best.pt
    python scripts/test_single.py --image path/to/image.jpg --conf 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_WEIGHTS = "runs/detect/vivarium_v1/weights/best.pt"
DEFAULT_OUT     = "runs/single_test"
DEFAULT_CONF    = 0.35
DEFAULT_IOU     = 0.45

# 9-class map
CLASS_NAMES = {
    0: "mouse",
    1: "water_critical", 2: "water_low", 3: "water_ok", 4: "water_full",
    5: "food_critical",  6: "food_low",  7: "food_ok",  8: "food_full",
}

# class_id → (representative_pct, status)
CLASS_TO_LEVEL = {
    1: (5.0,  "CRITICAL"), 2: (20.0, "LOW"), 3: (60.0, "OK"), 4: (95.0, "OK"),
    5: (5.0,  "CRITICAL"), 6: (20.0, "LOW"), 7: (60.0, "OK"), 8: (95.0, "OK"),
}

WATER_IDS = {1, 2, 3, 4}
FOOD_IDS  = {5, 6, 7, 8}

CLASS_COLORS = {
    0: (255,  80,  80),   # mouse  — blue
    1: (  0,   0, 255),   # water_critical — red
    2: (  0, 180, 255),   # water_low      — orange
    3: ( 80, 200,  80),   # water_ok       — green
    4: ( 80, 200,  80),   # water_full     — green
    5: (  0,   0, 255),   # food_critical  — red
    6: (  0, 180, 255),   # food_low       — orange
    7: ( 80, 200,  80),   # food_ok        — green
    8: ( 80, 200,  80),   # food_full      — green
}

STATUS_COLORS = {
    "OK":       ( 80, 200,  80),
    "LOW":      (  0, 180, 255),
    "CRITICAL": (  0,   0, 255),
    "UNKNOWN":  (160, 160, 160),
}


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_bbox(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_level_bar(img, x, y, w, h, pct, status):
    bar_x = min(x + w + 6, img.shape[1] - 22)
    bar_w = 14
    color = STATUS_COLORS.get(status, (160, 160, 160))

    # background
    cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + h), (40, 40, 40), -1)
    # fill from bottom
    fill_h = int(h * (pct / 100.0))
    cv2.rectangle(img, (bar_x, y + h - fill_h), (bar_x + bar_w, y + h), color, -1)
    # border
    cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + h), (200, 200, 200), 1)
    # % above
    cv2.putText(img, f"{pct:.0f}%", (bar_x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    # status below
    cv2.putText(img, status, (bar_x - 2, y + h + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(image_path, weights, out_dir, conf, iou):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics"); return

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}"); return

    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}"); return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nImage   : {img_path}")
    print(f"Weights : {weights_path}\n")

    # ── Load + resize ─────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        print("[ERROR] Cannot read image."); return

    h0, w0 = img.shape[:2]
    scale  = 640 / max(h0, w0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_t  = (640 - nh) // 2
    pad_l  = (640 - nw) // 2
    canvas[pad_t:pad_t+nh, pad_l:pad_l+nw] = cv2.resize(img, (nw, nh))
    frame_640 = canvas
    viz = frame_640.copy()

    # ── YOLO inference ────────────────────────────────────────────
    model   = YOLO(str(weights_path))
    results = model.predict(source=frame_640, conf=conf, iou=iou,
                            imgsz=640, verbose=False)
    r = results[0]

    mouse_count = 0
    water_pct, water_status, water_box = 0.0, "UNKNOWN", None
    food_pct,  food_status,  food_box  = 0.0, "UNKNOWN", None
    water_best_conf = -1.0
    food_best_conf  = -1.0

    if r.boxes is not None and len(r.boxes) > 0:
        boxes   = r.boxes.xyxy.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, cf, cls in zip(boxes, confs, classes):
            cls = int(cls)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            label = f"{CLASS_NAMES.get(cls, str(cls))} {cf:.2f}"
            draw_bbox(viz, x1, y1, x2, y2, label, color)

            if cls == 0:
                mouse_count += 1

            elif cls in WATER_IDS and cf > water_best_conf:
                water_best_conf = cf
                water_pct, water_status = CLASS_TO_LEVEL[cls]
                water_box = (x1, y1, x2, y2)

            elif cls in FOOD_IDS and cf > food_best_conf:
                food_best_conf = cf
                food_pct, food_status = CLASS_TO_LEVEL[cls]
                food_box = (x1, y1, x2, y2)

    # ── Draw level bars ───────────────────────────────────────────
    if water_box:
        x1, y1, x2, y2 = water_box
        draw_level_bar(viz, x1, y1, x2-x1, y2-y1, water_pct, water_status)
    else:
        # fallback position — right side
        draw_level_bar(viz, 480, 80, 140, 300, water_pct, water_status)
        cv2.putText(viz, "[no detection]", (490, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    if food_box:
        x1, y1, x2, y2 = food_box
        draw_level_bar(viz, x1, y1, x2-x1, y2-y1, food_pct, food_status)
    else:
        # fallback position — left side
        draw_level_bar(viz, 20, 80, 160, 200, food_pct, food_status)
        cv2.putText(viz, "[no detection]", (22, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # ── Summary overlay ───────────────────────────────────────────
    summary = [
        f"Mouse : {mouse_count}",
        f"Water : {water_pct:.0f}%  [{water_status}]",
        f"Food  : {food_pct:.0f}%  [{food_status}]",
    ]
    for i, line in enumerate(summary):
        s = "CRITICAL" if "CRITICAL" in line else "LOW" if "LOW" in line else "OK"
        c = STATUS_COLORS[s]
        cv2.putText(viz, line, (8, 20 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(viz, line, (8, 20 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c,     1, cv2.LINE_AA)

    # ── Save ──────────────────────────────────────────────────────
    out_file = out_path / f"{img_path.stem}_level_test.jpg"
    cv2.imwrite(str(out_file), viz, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"{'─'*50}")
    for line in summary:
        print(f"  {line}")
    print(f"{'─'*50}")
    print(f"\n  Output → {out_file}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",   required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out",     default=DEFAULT_OUT)
    ap.add_argument("--conf",    type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou",     type=float, default=DEFAULT_IOU)
    args = ap.parse_args()
    main(args.image, args.weights, args.out, args.conf, args.iou)



# python scripts/label_tools.py verify --img-dir dataset/augmented/images
# python scripts/label_tools.py dedup --dry-run
# python scripts/label_tools.py clean --dry-run
# python scripts/label_tools.py fix-classes --dry-run