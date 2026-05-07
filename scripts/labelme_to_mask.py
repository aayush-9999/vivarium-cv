"""
labelme_to_masks.py
====================
Converts LabelMe JSON annotations to single-channel PNG segmentation masks
for PSPNet training.

Label → class ID mapping:
    background  = 0  (anything not annotated)
    bottle_wall = 1
    water_fill  = 2
    empty_air   = 3

Usage:
    python labelme_to_masks.py

Output structure:
    dataset/segmentation/water/
        images/   ← copies of original images
        masks/    ← single-channel PNG masks (pixel value = class ID)
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ORIG_DIR   = Path("dataset/original")           # where your .jpg + .json files are
OUT_ROOT   = Path("dataset/segmentation/water") # output root
IMG_SIZE   = 640                                # resize everything to 640x640

LABEL_TO_ID = {
    "background":  0,
    "bottle_wall": 1,
    "water_fill":  2,
    "empty_air":   3,
}

# Draw order — larger areas first so smaller regions paint on top
DRAW_ORDER = ["bottle_wall", "empty_air", "water_fill"]

# ── Output dirs ───────────────────────────────────────────────────────────────
(OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "masks").mkdir(parents=True, exist_ok=True)


def letterbox(img: np.ndarray, size: int = 640):
    """Resize image to size×size with letterboxing (grey padding)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_t = (size - nh) // 2
    pad_l = (size - nw) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, pad_t, pad_l


def process(json_path: Path):
    img_path = json_path.with_suffix(".jpg")
    if not img_path.exists():
        img_path = json_path.with_suffix(".png")
    if not img_path.exists():
        print(f"  [SKIP] No image for {json_path.name}")
        return

    # Load image + letterbox
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot read {img_path.name}")
        return

    img_lb, scale, pad_t, pad_l = letterbox(img, IMG_SIZE)

    # Load JSON
    data = json.loads(json_path.read_text(encoding="utf-8"))
    orig_w = data.get("imageWidth",  img.shape[1])
    orig_h = data.get("imageHeight", img.shape[0])

    # Build mask (starts all zeros = background)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Group shapes by label
    shapes_by_label = {label: [] for label in DRAW_ORDER}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label in shapes_by_label:
            shapes_by_label[label].append(shape)

    # Draw polygons in order (bottle_wall first, water_fill last = on top)
    for label in DRAW_ORDER:
        class_id = LABEL_TO_ID[label]
        for shape in shapes_by_label[label]:
            pts = shape.get("points", [])
            if len(pts) < 3:
                continue

            # Scale points from original image space → letterboxed 640x640 space
            scaled = []
            for x, y in pts:
                # First scale by letterbox scale factor
                sx = x * scale + pad_l
                sy = y * scale + pad_t
                scaled.append([int(sx), int(sy)])

            poly = np.array(scaled, dtype=np.int32)
            cv2.fillPoly(mask, [poly], color=class_id)

    # Save mask as single-channel PNG
    stem = json_path.stem
    mask_path = OUT_ROOT / "masks" / f"{stem}.png"
    cv2.imwrite(str(mask_path), mask)

    # Save letterboxed image copy
    img_out = OUT_ROOT / "images" / f"{stem}.jpg"
    cv2.imwrite(str(img_out), img_lb, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Stats
    total_px = IMG_SIZE * IMG_SIZE
    counts = {
        "background":  int((mask == 0).sum()),
        "bottle_wall": int((mask == 1).sum()),
        "water_fill":  int((mask == 2).sum()),
        "empty_air":   int((mask == 3).sum()),
    }
    fill_pct = counts["water_fill"] / max(counts["water_fill"] + counts["empty_air"], 1) * 100

    print(f"  ✓ {stem:<20}  "
          f"wall={counts['bottle_wall']:5d}px  "
          f"fill={counts['water_fill']:5d}px  "
          f"air={counts['empty_air']:5d}px  "
          f"→ {fill_pct:.0f}% full")

    return fill_pct


def main():
    json_files = sorted(ORIG_DIR.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON files found in {ORIG_DIR}")
        print("  Make sure your LabelMe JSONs are saved in dataset/original/")
        return

    print(f"Found {len(json_files)} JSON files in {ORIG_DIR}\n")
    print(f"Output → {OUT_ROOT}\n")

    fill_pcts = []
    skipped = 0

    for jp in json_files:
        result = process(jp)
        if result is not None:
            fill_pcts.append(result)
        else:
            skipped += 1

    print(f"\n{'─'*60}")
    print(f"Done")
    print(f"  Processed : {len(fill_pcts)} images")
    print(f"  Skipped   : {skipped}")
    if fill_pcts:
        print(f"  Avg fill  : {sum(fill_pcts)/len(fill_pcts):.1f}%")
        print(f"  Min fill  : {min(fill_pcts):.1f}%")
        print(f"  Max fill  : {max(fill_pcts):.1f}%")
    print(f"\n  Images → {OUT_ROOT}/images/")
    print(f"  Masks  → {OUT_ROOT}/masks/")
    print(f"\nNext step — retrain PSPNet:")
    print(f"  python -m segmentation.trainers.psp_trainer \\")
    print(f"      --container water \\")
    print(f"      --data-root dataset/segmentation \\")
    print(f"      --output-dir runs/pspnet/water_real \\")
    print(f"      --backbone resnet50 \\")
    print(f"      --epochs 80 \\")
    print(f"      --batch-size 4 \\")
    print(f"      --device cuda")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()