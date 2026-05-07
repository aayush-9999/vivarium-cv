# scripts/generate_pseudo_masks.py
"""
Generate pseudo segmentation masks from existing YOLO bounding box labels.

This bootstraps PSPNet training WITHOUT requiring manual pixel annotation.
The masks are approximate but good enough to start training — you then
refine them by annotating real cage images with CVAT.

How it works:
─────────────
Your YOLO labels already tell us:
  - WHERE the water container is  (bbox)
  - HOW FULL it is  (class ID 1-4 → fill percentage)

We use this to generate approximate masks:

    Water container bbox:
    ┌─────────────┐
    │░░░ empty ░░░│  ← class 3 (above fill line)
    │░░░ air   ░░░│
    ├─────────────┤  ← fill line at known height
    │███ water ███│  ← class 2 (below fill line)
    │███ fill  ███│
    └─────────────┘
    Outer ring = class 1 (bottle wall)
    Outside bbox = class 0 (background)

For food:
    ┌─────────────┐
    │░░░░░░░░░░░░░│  ← class 3 (empty space) at top
    ├─────────────┤  ← fill line
    │█████████████│  ← class 2 (food pellets) at bottom
    └─────────────┘
    Outer ring = class 1 (hopper frame)

Fill percentages from class IDs:
    Water: 1=7.5% 2=25.0% 3=57.5% 4=90.0%
    Food:  5=7.5% 6=25.0% 7=57.5% 8=90.0%

Usage:
    # Generate masks for augmented dataset
    python scripts/generate_pseudo_masks.py

    # Custom paths
    python scripts/generate_pseudo_masks.py \
        --img-dir   dataset/augmented/images \
        --label-dir dataset/augmented/labels \
        --out-dir   dataset/segmentation

    # Dry run — generate debug images to verify masks look correct
    python scripts/generate_pseudo_masks.py --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Class ID → fill percentage (midpoint of each bucket) ─────────────────────
WATER_CLASS_TO_PCT = {1: 0.075, 2: 0.250, 3: 0.575, 4: 0.900}
FOOD_CLASS_TO_PCT  = {5: 0.075, 6: 0.250, 7: 0.575, 8: 0.900}

WATER_CLASS_IDS = {1, 2, 3, 4}
FOOD_CLASS_IDS  = {5, 6, 7, 8}

# ── Mask class values ─────────────────────────────────────────────────────────
BG         = 0   # background (outside container)
WALL       = 1   # container wall / frame
FILL       = 2   # water / food fill region
EMPTY      = 3   # empty air / empty hopper

WALL_THICKNESS = 4   # pixels for wall border


def read_labels(path: Path) -> list[tuple]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                out.append((int(parts[0]),
                            float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return out


def yolo_to_pixels(cx_n, cy_n, bw_n, bh_n, img_w, img_h):
    """Convert normalised YOLO bbox to pixel coords."""
    bw = bw_n * img_w
    bh = bh_n * img_h
    x1 = max(0, int(cx_n * img_w - bw / 2))
    y1 = max(0, int(cy_n * img_h - bh / 2))
    x2 = min(img_w, int(cx_n * img_w + bw / 2))
    y2 = min(img_h, int(cy_n * img_h + bh / 2))
    return x1, y1, x2, y2


def generate_container_mask(
    mask:     np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    fill_pct: float,
) -> None:
    """
    Draw a container pseudo-mask into `mask` in-place.

    Args:
        mask      : (H, W) uint8 array, class IDs
        x1,y1,x2,y2: container bbox in pixels
        fill_pct  : 0.0 – 1.0 fill level
    """
    wt = WALL_THICKNESS

    # Container interior bounds (inside walls)
    ix1 = x1 + wt
    iy1 = y1 + wt
    ix2 = x2 - wt
    iy2 = y2 - wt

    container_h = iy2 - iy1
    if container_h <= 0:
        return

    # Fill region starts at this y (from bottom up)
    fill_h    = int(container_h * fill_pct)
    fill_y1   = iy2 - fill_h   # top of fill region

    # Paint in order: fill → empty → walls
    # (so walls overwrite fill/empty at the border)

    # Empty region (above fill line)
    if fill_y1 > iy1:
        mask[iy1:fill_y1, ix1:ix2] = EMPTY

    # Fill region (below fill line)
    if fill_y1 < iy2:
        mask[fill_y1:iy2, ix1:ix2] = FILL

    # Walls (outer ring of bbox)
    mask[y1:y2, x1:x1 + wt] = WALL    # left
    mask[y1:y2, x2 - wt:x2] = WALL    # right
    mask[y1:y1 + wt, x1:x2] = WALL    # top
    mask[y2 - wt:y2, x1:x2] = WALL    # bottom


def process_image(
    img_path:   Path,
    label_path: Path,
    water_out:  Path,
    food_out:   Path,
    debug:      bool = False,
) -> dict:
    """
    Generate water and food pseudo-masks for one image.
    Saves crops + masks into separate water/ and food/ subdirs.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {"status": "unreadable"}

    img_h, img_w = img.shape[:2]
    labels = read_labels(label_path)

    stats = {"water": 0, "food": 0}

    for cls_id, cx_n, cy_n, bw_n, bh_n in labels:
        x1, y1, x2, y2 = yolo_to_pixels(cx_n, cy_n, bw_n, bh_n, img_w, img_h)
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w < 20 or bbox_h < 20:
            continue   # too small

        if cls_id in WATER_CLASS_IDS:
            fill_pct = WATER_CLASS_TO_PCT[cls_id]
            container_type = "water"
            out_root = water_out
        elif cls_id in FOOD_CLASS_IDS:
            fill_pct = FOOD_CLASS_TO_PCT[cls_id]
            container_type = "food"
            out_root = food_out
        else:
            continue

        # Create mask (same size as full image, then crop)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        generate_container_mask(full_mask, x1, y1, x2, y2, fill_pct)

        # Crop both image and mask to bbox
        crop_img  = img[y1:y2, x1:x2].copy()
        crop_mask = full_mask[y1:y2, x1:x2].copy()

        stem = f"{img_path.stem}_{container_type}_x{x1}_y{y1}"

        # Save image crop
        img_save  = out_root / "images" / f"{stem}.jpg"
        mask_save = out_root / "masks"  / f"{stem}.png"

        cv2.imwrite(str(img_save),  crop_img,  [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_save), crop_mask)  # single-channel PNG

        # Debug visualization
        if debug:
            debug_save = out_root / "debug" / f"{stem}_debug.jpg"
            viz = _visualize_mask(crop_img, crop_mask)
            cv2.imwrite(str(debug_save), viz)

        stats[container_type] += 1

    return stats


def _visualize_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay coloured mask on image for visual verification."""
    palette = {
        BG:    (30,  30,  30),
        WALL:  (180, 180, 180),
        FILL:  (200, 100,  20),   # blue-ish for water
        EMPTY: (40,  40,  80),
    }
    color_mask = np.zeros_like(img_bgr)
    for cls_id, color in palette.items():
        color_mask[mask == cls_id] = color
    return cv2.addWeighted(img_bgr, 0.6, color_mask, 0.4, 0)


def main(
    img_dir:   Path,
    label_dir: Path,
    out_dir:   Path,
    debug:     bool = False,
) -> None:
    # Create output dirs
    water_out = out_dir / "water"
    food_out  = out_dir / "food"

    for d in [
        water_out / "images", water_out / "masks",
        food_out  / "images", food_out  / "masks",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if debug:
        (water_out / "debug").mkdir(exist_ok=True)
        (food_out  / "debug").mkdir(exist_ok=True)

    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not img_paths:
        print(f"[ERROR] No images in {img_dir}")
        return

    total_water = 0
    total_food  = 0

    print(f"\nGenerating pseudo-masks for {len(img_paths)} images …\n")
    print(f"  Source images : {img_dir}")
    print(f"  Source labels : {label_dir}")
    print(f"  Output        : {out_dir}")
    print(f"  Debug images  : {'yes' if debug else 'no'}\n")

    for idx, img_path in enumerate(img_paths, 1):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        stats = process_image(img_path, label_path, water_out, food_out, debug)
        total_water += stats.get("water", 0)
        total_food  += stats.get("food",  0)

        if idx % 100 == 0 or idx == len(img_paths):
            print(f"  [{idx:5d}/{len(img_paths)}]  water_crops={total_water}  food_crops={total_food}")

    print(f"""
{'─'*60}
Pseudo-mask generation complete

  Water crops : {total_water}
    images    : {water_out}/images/
    masks     : {water_out}/masks/
    {'debug     : ' + str(water_out) + '/debug/' if debug else ''}

  Food crops  : {total_food}
    images    : {food_out}/images/
    masks     : {food_out}/masks/

{'─'*60}

NEXT STEPS — ANNOTATION:
  These pseudo-masks are approximate rectangles.
  For best PSPNet accuracy, refine them with real annotations.

  Recommended tool: CVAT (free, web-based)
  https://cvat.ai

  Import format  : "Segmentation mask" (PNG, single channel)
  Export format  : "Segmentation mask" (same)

  Annotation priority:
    1. Annotate ~50 real cage images per container type first
    2. Train PSPNet on pseudo-masks + 50 real images
    3. Run inference, check error, annotate more if needed

  Label classes (in order, 0-indexed):
    Water model: background | bottle_wall | water_fill | empty_air
    Food model : background | hopper_frame | food_pellets | empty_space

  Then run:
    python segmentation/trainers/psp_trainer.py --container water
    python segmentation/trainers/psp_trainer.py --container food

  Set in .env:
    BACKEND=yolo_psp
    PSP_WATER_WEIGHTS=runs/pspnet/water/best.pth
    PSP_FOOD_WEIGHTS=runs/pspnet/food/best.pth
{'─'*60}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate pseudo segmentation masks from YOLO labels"
    )
    ap.add_argument("--img-dir",   type=Path, default=Path("dataset/augmented/images"))
    ap.add_argument("--label-dir", type=Path, default=Path("dataset/augmented/labels"))
    ap.add_argument("--out-dir",   type=Path, default=Path("dataset/segmentation"))
    ap.add_argument("--debug",     action="store_true",
                    help="Save coloured debug images to verify mask quality")
    args = ap.parse_args()
    main(args.img_dir, args.label_dir, args.out_dir, args.debug)