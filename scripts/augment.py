"""
scripts/augment.py
==================
Augments vivarium cage images and AUTO-WRITES YOLO label files.

Key change vs previous version:
    Since we know exactly where the synthetic water/food overlays are drawn
    (they use ROI_ZONES), we generate class 1 (water_container) and
    class 2 (food_area) labels FOR FREE during augmentation.
    You only need to manually label class 0 (mouse) afterwards.

Output layout:
    dataset/augmented/
        images/   ← augmented JPEGs
        labels/   ← YOLO .txt files (class cx cy w h, normalised)
                     class 1 + 2 are auto-generated
                     class 0 (mouse) lines are EMPTY — fill in with LabelImg
        meta/     ← per-image JSON with augmentation params

Usage:
    python scripts/augment.py --src dataset/original --dst dataset/augmented --n 50

After running:
    1. Use LabelImg to add mouse (class 0) boxes to the generated .txt files.
       The water/food boxes are already there — you only draw mice.
         pip install labelImg
         labelImg dataset/augmented/images dataset/augmented/labels/classes.txt
    2. Run: python scripts/gdino_label.py --only-review   (to fill in any missed mice)
    3. Run: python scripts/split_dataset.py && python scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ── ROI zones (must match core/config.py) ────────────────────────────────────
ROI_ZONES = {
    "default": {
        "jug":    (480, 80,  140, 300),   # (x, y, w, h) in 640×640 space
        "hopper": (20,  80,  160, 200),
    },
    "type_b": {
        "jug":    (460, 60,  160, 320),
        "hopper": (10,  60,  180, 220),
    },
}

# Level scenarios: (label, fill_fraction_of_roi_height)
WATER_LEVELS = [
    ("critical", 0.05),
    ("low",      0.20),
    ("ok",       0.60),
    ("full",     0.90),
]
FOOD_LEVELS = [
    ("critical", 0.05),
    ("low",      0.20),
    ("ok",       0.55),
    ("full",     0.85),
]

WATER_OVERLAY_COLOR = (180, 120,  60)
FOOD_OVERLAY_COLOR  = ( 60, 150, 190)

IMG_SIZE = 640


# ─────────────────────────────────────────────────────────────────────────────
# Label generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def roi_to_yolo(
    x: int, y: int, w: int, h: int,
    fill_frac: float,
    img_size: int = IMG_SIZE,
) -> str:
    """
    Convert an ROI zone + fill fraction to a YOLO bounding box string.

    For water/food we label the ENTIRE zone rectangle (not just the filled part).
    The model learns to detect the container regardless of fill level —
    level % is then computed by the HSV estimator, not by bbox height.

    Returns: "cx cy bw bh" (normalised, no class prefix)
    """
    cx = (x + w / 2) / img_size
    cy = (y + h / 2) / img_size
    bw = w / img_size
    bh = h / img_size
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def generate_container_labels(cage_type: str, img_size: int = IMG_SIZE) -> list[str]:
    """
    Generate YOLO label lines for class 1 (water_container) and class 2 (food_area).
    Returns list of label strings — append mouse boxes (class 0) manually later.
    """
    zones = ROI_ZONES.get(cage_type, ROI_ZONES["default"])
    lines = []

    jx, jy, jw, jh = zones["jug"]
    hx, hy, hw, hh = zones["hopper"]

    # class 1 = water_container (full jug zone)
    lines.append(f"1 {roi_to_yolo(jx, jy, jw, jh, 1.0, img_size)}")

    # class 2 = food_area (full hopper zone)
    lines.append(f"2 {roi_to_yolo(hx, hy, hw, hh, 1.0, img_size)}")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def adjust_brightness_contrast(img, alpha, beta):
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

def shift_hsv(img, dh, ds, dv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + dh, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + ds, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gaussian_blur(img, ksize):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def random_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def horizontal_flip(img):
    return cv2.flip(img, 1)

def _draw_level_overlay(img, roi, fill_frac, color, alpha=0.45):
    out = img.copy()
    x, y, w, h = roi
    fill_h  = max(1, int(h * fill_frac))
    y_start = y + h - fill_h
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y_start), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def apply_synthetic_levels(img, cage_type, water_frac, food_frac):
    zones = ROI_ZONES.get(cage_type, ROI_ZONES["default"])
    img = _draw_level_overlay(img, zones["jug"],    water_frac, WATER_OVERLAY_COLOR)
    img = _draw_level_overlay(img, zones["hopper"], food_frac,  FOOD_OVERLAY_COLOR)
    return img

def _letterbox(img, size=640):
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

def _sample_params(rng):
    w_label, w_frac = rng.choice(WATER_LEVELS)
    f_label, f_frac = rng.choice(FOOD_LEVELS)
    cage_type       = rng.choice(list(ROI_ZONES.keys()))
    return {
        "cage_type":     cage_type,
        "alpha":         rng.uniform(0.6, 1.5),
        "beta":          rng.uniform(-60, 60),
        "dh":            rng.randint(-18, 18),
        "ds":            rng.randint(-40, 40),
        "dv":            rng.randint(-40, 40),
        "blur_k":        rng.choice([0, 3, 5, 7]),
        "noise_sigma":   rng.uniform(0, 18),
        "flip":          rng.random() < 0.5,
        "angle":         rng.uniform(-15, 15),
        "water_label":   w_label,
        "water_frac":    w_frac,
        "food_label":    f_label,
        "food_frac":     f_frac,
        "overlay_alpha": rng.uniform(0.3, 0.6),
    }

def augment_one(img, p):
    out = _letterbox(img, IMG_SIZE)
    out = adjust_brightness_contrast(out, p["alpha"], p["beta"])
    out = shift_hsv(out, p["dh"], p["ds"], p["dv"])
    if p["blur_k"] > 0:
        out = gaussian_blur(out, p["blur_k"])
    if p["noise_sigma"] > 0:
        out = add_gaussian_noise(out, p["noise_sigma"])
    if p["flip"]:
        out = horizontal_flip(out)
    if abs(p["angle"]) > 0.5:
        out = random_rotate(out, p["angle"])
    out = apply_synthetic_levels(
        out, cage_type=p["cage_type"],
        water_frac=p["water_frac"], food_frac=p["food_frac"],
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(src: Path, dst: Path, n: int, seed: int) -> None:
    img_dir   = dst / "images"
    label_dir = dst / "labels"
    meta_dir  = dst / "meta"

    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not source_images:
        print(f"[ERROR] No images found in {src}")
        sys.exit(1)

    print(f"Found {len(source_images)} source image(s) in '{src}'")
    print(f"Generating {n} augmented variants per image → {len(source_images) * n} total")
    print(f"Auto-writing class 1 (water) + class 2 (food) labels.")
    print(f"You will need to ADD class 0 (mouse) boxes in LabelImg afterwards.\n")

    # Write classes.txt for LabelImg
    (label_dir / "classes.txt").write_text("mouse\nwater_container\nfood_area\n")

    rng   = random.Random(seed)
    total = 0

    for src_path in source_images:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [WARN] Could not read {src_path.name}, skipping.")
            continue

        stem = src_path.stem

        # Save the original resized version too — with a label file
        orig_img  = _letterbox(img, IMG_SIZE)
        orig_name = f"{stem}_orig.jpg"
        cv2.imwrite(str(img_dir / orig_name), orig_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # For the original, write container labels with "default" cage type
        orig_label_path = label_dir / f"{stem}_orig.txt"
        orig_label_path.write_text("\n".join(generate_container_labels("default")))

        for i in range(n):
            params  = _sample_params(rng)
            aug_img = augment_one(img, params)

            out_name = (
                f"{stem}_aug{i:04d}"
                f"_w{params['water_label']}"
                f"_f{params['food_label']}.jpg"
            )
            cv2.imwrite(str(img_dir / out_name), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # ── Auto-generate label file ──────────────────────────
            # class 1 + class 2 are known from ROI_ZONES + cage_type
            # class 0 (mouse) = empty for now, added later via LabelImg or GDINO
            label_lines = generate_container_labels(params["cage_type"])
            # Note: if source image had mouse annotations, you could
            # transform them here (handle flip/rotation). For now, left empty.

            label_path = label_dir / out_name.replace(".jpg", ".txt")
            label_path.write_text("\n".join(label_lines))

            # Meta
            meta = {"source": src_path.name, "output": out_name, "aug_index": i, **params}
            (meta_dir / out_name.replace(".jpg", ".json")).write_text(json.dumps(meta, indent=2))

            total += 1

        print(f"  ✓ {stem}: {n} variants saved  (water+food labels auto-written)")

    print(f"\nDone — {total} augmented images")
    print(f"Labels (class 1+2 only) → {label_dir}")
    print(f"\nNext steps:")
    print(f"  OPTION A (fastest): Run GDINO to fill in mouse boxes automatically:")
    print(f"    python scripts/gdino_label.py")
    print(f"  OPTION B (highest quality): Add mouse boxes manually in LabelImg:")
    print(f"    labelImg {img_dir} {label_dir / 'classes.txt'}")
    print(f"  Then: python scripts/split_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  type=Path, default=Path("dataset/original"))
    ap.add_argument("--dst",  type=Path, default=Path("dataset/augmented"))
    ap.add_argument("--n",    type=int,  default=50)
    ap.add_argument("--seed", type=int,  default=42)
    args = ap.parse_args()
    main(args.src, args.dst, args.n, args.seed)