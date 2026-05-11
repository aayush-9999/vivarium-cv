"""
labelme_to_masks.py
=======================
Converts LabelMe JSON annotations to single-channel PNG segmentation masks.

Water → saves full 640x640 frame (bottle is large)
Food  → saves tight crop around hopper_frame polygon (hopper is small)

Usage:
    python labelme_to_masks_v2.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ORIG_DIR   = Path("dataset/original")
IMG_SIZE   = 640

# ── Label → class ID ─────────────────────────────────────────────────────────
WATER_LABEL_TO_ID = {
    "background":  0,
    "bottle_wall": 1,
    "water_fill":  2,
    "empty_air":   3,
}
WATER_DRAW_ORDER = ["bottle_wall", "empty_air", "water_fill"]

FOOD_LABEL_TO_ID = {
    "background":   0,
    "hopper_frame": 1,
    "food_pellets": 2,
    "empty_space":  3,
}
FOOD_DRAW_ORDER = ["hopper_frame", "empty_space", "food_pellets"]

# ── Output dirs ───────────────────────────────────────────────────────────────
WATER_OUT = Path("dataset/segmentation/water")
FOOD_OUT  = Path("dataset/segmentation/food")

for d in [
    WATER_OUT / "images", WATER_OUT / "masks",
    FOOD_OUT  / "images", FOOD_OUT  / "masks",
]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_t = (size - nh) // 2
    pad_l = (size - nw) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, pad_t, pad_l


def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def scale_points(points, scale, pad_t, pad_l):
    return [[p[0] * scale + pad_l, p[1] * scale + pad_t] for p in points]


def draw_mask(mask, shapes_by_label, draw_order, label_to_id, scale, pad_t, pad_l):
    for label in draw_order:
        class_id = label_to_id[label]
        for pts in shapes_by_label.get(label, []):
            if len(pts) < 3:
                continue
            scaled = scale_points(pts, scale, pad_t, pad_l)
            poly = np.array([[int(p[0]), int(p[1])] for p in scaled], dtype=np.int32)
            cv2.fillPoly(mask, [poly], color=class_id)
    return mask


def get_polygon_bbox(points, scale, pad_t, pad_l, img_size=640, margin=0.05):
    """Get bbox of polygon in letterboxed space with optional margin."""
    scaled = scale_points(points, scale, pad_t, pad_l)
    xs = [p[0] for p in scaled]
    ys = [p[1] for p in scaled]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Add margin
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * margin)
    y1 = max(0, y1 - bh * margin)
    x2 = min(img_size, x2 + bw * margin)
    y2 = min(img_size, y2 + bh * margin)

    return int(x1), int(y1), int(x2), int(y2)


# ── Process each JSON ─────────────────────────────────────────────────────────

def process(json_path: Path):
    # Find image
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        print(f"  [SKIP] No image for {json_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot read {img_path.name}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_lb, scale, pad_t, pad_l = letterbox(img, IMG_SIZE)

    # Group shapes by label
    shapes_by_label = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        shapes_by_label.setdefault(label, []).append(shape.get("points", []))

    stem = json_path.stem
    water_saved = False
    food_saved  = False

    # ── WATER — full 640x640 frame ────────────────────────────────────────
    if "bottle_wall" in shapes_by_label:
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        mask = draw_mask(mask, shapes_by_label, WATER_DRAW_ORDER,
                         WATER_LABEL_TO_ID, scale, pad_t, pad_l)

        cv2.imwrite(str(WATER_OUT / "images" / f"{stem}.jpg"), img_lb,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(WATER_OUT / "masks"  / f"{stem}.png"), mask)

        fill_px  = int((mask == 2).sum())
        air_px   = int((mask == 3).sum())
        total    = fill_px + air_px
        fill_pct = fill_px / total * 100 if total > 0 else 0
        print(f"  WATER ✓ {stem:<22} fill={fill_pct:.0f}%  "
              f"wall={int((mask==1).sum())}px  "
              f"fill={fill_px}px  air={air_px}px")
        water_saved = True

    # ── FOOD — tight crop around hopper_frame polygon ─────────────────────
    if "hopper_frame" in shapes_by_label:
        # Use largest hopper_frame polygon for bbox
        wall_pts = max(shapes_by_label["hopper_frame"], key=polygon_area)
        x1, y1, x2, y2 = get_polygon_bbox(wall_pts, scale, pad_t, pad_l,
                                           img_size=IMG_SIZE, margin=0.08)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 20 or crop_h < 20:
            print(f"  FOOD  [SKIP] {stem} — crop too small ({crop_w}x{crop_h})")
        else:
            # Full mask first, then crop
            full_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            full_mask = draw_mask(full_mask, shapes_by_label, FOOD_DRAW_ORDER,
                                  FOOD_LABEL_TO_ID, scale, pad_t, pad_l)

            crop_img  = img_lb[y1:y2, x1:x2].copy()
            crop_mask = full_mask[y1:y2, x1:x2].copy()

            # Resize crop to standard size for PSPNet (224x224 for food)
            TARGET = (224, 224)
            crop_img_r  = cv2.resize(crop_img,  TARGET, interpolation=cv2.INTER_LINEAR)
            crop_mask_r = cv2.resize(crop_mask, TARGET, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(str(FOOD_OUT / "images" / f"{stem}.jpg"), crop_img_r,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(FOOD_OUT / "masks"  / f"{stem}.png"), crop_mask_r)

            fill_px  = int((crop_mask_r == 2).sum())
            air_px   = int((crop_mask_r == 3).sum())
            total    = fill_px + air_px
            fill_pct = fill_px / total * 100 if total > 0 else 0
            print(f"  FOOD  ✓ {stem:<22} fill={fill_pct:.0f}%  "
                  f"crop=({crop_w}x{crop_h})→{TARGET}  "
                  f"fill={fill_px}px  air={air_px}px")
            food_saved = True

    if not water_saved and not food_saved:
        print(f"  [SKIP] {stem} — no water or food annotations found")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    json_files = sorted(ORIG_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files in {ORIG_DIR}")
        return

    print(f"Found {len(json_files)} JSON files\n")
    print(f"Water → full 640x640 frames  → {WATER_OUT}")
    print(f"Food  → tight hopper crops   → {FOOD_OUT}\n")

    for jp in json_files:
        process(jp)

    # Summary
    water_imgs = list((WATER_OUT / "images").glob("*.jpg"))
    food_imgs  = list((FOOD_OUT  / "images").glob("*.jpg"))

    print(f"\n{'─'*60}")
    print(f"Water images : {len(water_imgs)}  masks: {len(list((WATER_OUT/'masks').glob('*.png')))}")
    print(f"Food  images : {len(food_imgs)}   masks: {len(list((FOOD_OUT/'masks').glob('*.png')))}")
    print(f"\nNext — retrain food PSPNet on crops:")
    print(f"  python -m segmentation.trainers.psp_trainer \\")
    print(f"      --container food \\")
    print(f"      --data-root dataset/segmentation \\")
    print(f"      --output-dir runs/pspnet/food_v2 \\")
    print(f"      --backbone resnet50 \\")
    print(f"      --epochs 80 \\")
    print(f"      --batch-size 4 \\")
    print(f"      --device cuda")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()