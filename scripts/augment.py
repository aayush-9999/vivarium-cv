"""
scripts/augment.py
==================
Augments vivarium cage images for YOLO training data preparation.

Usage:
    python scripts/augment.py \
        --src  dataset/original \
        --dst  dataset/augmented \
        --n    50

Output layout:
    dataset/augmented/
        images/   ← augmented JPEGs
        meta/     ← per-image JSON with augmentation params applied

Augmentations applied (full suite):
    • Brightness / contrast jitter
    • HSV hue + saturation shift  (simulates different lab lighting)
    • Gaussian blur               (simulates focus variation)
    • Horizontal flip
    • Random rotation ±15°
    • Gaussian noise
    • Synthetic water level fill  (overlays translucent blue in jug ROI)
    • Synthetic food level fill   (overlays translucent tan in hopper ROI)

Water / food levels cycle through: CRITICAL (5%), LOW (20%), OK (60%), FULL (90%)
so the visual dataset covers all alert states.
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
# (x, y, w, h) in 640×640 pixel space
ROI_ZONES = {
    "default": {
        "jug":    (480, 80,  140, 300),
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

# Colours for synthetic overlays (BGR)
WATER_OVERLAY_COLOR = (180, 120,  60)   # blueish
FOOD_OVERLAY_COLOR  = ( 60, 150, 190)   # tan/brown


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def adjust_brightness_contrast(
    img: np.ndarray,
    alpha: float,   # contrast  [0.6 – 1.6]
    beta:  float,   # brightness [-60 – +60]
) -> np.ndarray:
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def shift_hsv(
    img: np.ndarray,
    dh: int,   # hue shift      [-18, +18]
    ds: int,   # sat shift      [-40, +40]
    dv: int,   # value shift    [-40, +40]
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + dh, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + ds, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic level overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_level_overlay(
    img: np.ndarray,
    roi: tuple[int, int, int, int],   # (x, y, w, h)
    fill_frac: float,                  # 0.0 – 1.0 fraction of ROI height filled
    color: tuple[int, int, int],       # BGR
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Paint a semi-transparent filled rectangle in the bottom `fill_frac`
    of the given ROI, simulating fluid / food pile level.
    """
    out = img.copy()
    x, y, w, h = roi

    fill_h   = max(1, int(h * fill_frac))
    y_start  = y + h - fill_h          # fill from the bottom up
    overlay  = out.copy()

    cv2.rectangle(overlay, (x, y_start), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def apply_synthetic_levels(
    img: np.ndarray,
    cage_type: str,
    water_label: str,
    water_frac:  float,
    food_label:  str,
    food_frac:   float,
) -> np.ndarray:
    zones = ROI_ZONES.get(cage_type, ROI_ZONES["default"])
    img = _draw_level_overlay(img, zones["jug"],    water_frac, WATER_OVERLAY_COLOR)
    img = _draw_level_overlay(img, zones["hopper"], food_frac,  FOOD_OVERLAY_COLOR)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Random parameter samplers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(rng: random.Random) -> dict:
    w_label, w_frac = rng.choice(WATER_LEVELS)
    f_label, f_frac = rng.choice(FOOD_LEVELS)
    cage_type       = rng.choice(list(ROI_ZONES.keys()))

    return {
        "cage_type":       cage_type,
        "alpha":           rng.uniform(0.6, 1.5),
        "beta":            rng.uniform(-60, 60),
        "dh":              rng.randint(-18, 18),
        "ds":              rng.randint(-40, 40),
        "dv":              rng.randint(-40, 40),
        "blur_k":          rng.choice([0, 3, 5, 7]),
        "noise_sigma":     rng.uniform(0, 18),
        "flip":            rng.random() < 0.5,
        "angle":           rng.uniform(-15, 15),
        "water_label":     w_label,
        "water_frac":      w_frac,
        "food_label":      f_label,
        "food_frac":       f_frac,
        "overlay_alpha":   rng.uniform(0.3, 0.6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment_one(img: np.ndarray, p: dict) -> np.ndarray:
    """Apply one full augmentation pass using sampled params dict."""
    out = img.copy()

    # 1. Resize to 640×640 (letterbox if needed)
    out = _letterbox(out, 640)

    # 2. Photometric
    out = adjust_brightness_contrast(out, p["alpha"], p["beta"])
    out = shift_hsv(out, p["dh"], p["ds"], p["dv"])

    # 3. Blur
    if p["blur_k"] > 0:
        out = gaussian_blur(out, p["blur_k"])

    # 4. Noise
    if p["noise_sigma"] > 0:
        out = add_gaussian_noise(out, p["noise_sigma"])

    # 5. Geometric
    if p["flip"]:
        out = horizontal_flip(out)
    if abs(p["angle"]) > 0.5:
        out = random_rotate(out, p["angle"])

    # 6. Synthetic level overlays (applied last so they're always visible)
    out = apply_synthetic_levels(
        out,
        cage_type=p["cage_type"],
        water_label=p["water_label"],
        water_frac=p["water_frac"],
        food_label=p["food_label"],
        food_frac=p["food_frac"],
    )

    return out


def _letterbox(img: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    scale   = size / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    top     = (size - new_h) // 2
    left    = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(src: Path, dst: Path, n: int, seed: int) -> None:
    img_dir  = dst / "images"
    meta_dir = dst / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
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

    rng     = random.Random(seed)
    total   = 0

    for src_path in source_images:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [WARN] Could not read {src_path.name}, skipping.")
            continue

        stem = src_path.stem

        # Always save the original resized version too
        orig_out = img_dir / f"{stem}_orig.jpg"
        cv2.imwrite(str(orig_out), _letterbox(img, 640), [cv2.IMWRITE_JPEG_QUALITY, 92])

        for i in range(n):
            params  = _sample_params(rng)
            aug_img = augment_one(img, params)

            out_name = f"{stem}_aug{i:04d}_w{params['water_label']}_f{params['food_label']}.jpg"
            out_path = img_dir / out_name

            cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            meta = {
                "source":      src_path.name,
                "output":      out_name,
                "aug_index":   i,
                **params,
            }
            meta_path = meta_dir / out_name.replace(".jpg", ".json")
            meta_path.write_text(json.dumps(meta, indent=2))

            total += 1

        print(f"  ✓ {stem}: {n} variants saved")

    print(f"\nDone — {total} augmented images → {img_dir}")
    print(f"Metadata  → {meta_dir}")
    print("\nNext step: run  python scripts/auto_label.py  to generate YOLO annotations.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vivarium image augmentation pipeline")
    ap.add_argument("--src",  type=Path, default=Path("dataset/original"),
                    help="Folder containing original cage images")
    ap.add_argument("--dst",  type=Path, default=Path("dataset/augmented"),
                    help="Output folder for augmented images + metadata")
    ap.add_argument("--n",    type=int,  default=50,
                    help="Number of augmented variants per source image")
    ap.add_argument("--seed", type=int,  default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    main(args.src, args.dst, args.n, args.seed)