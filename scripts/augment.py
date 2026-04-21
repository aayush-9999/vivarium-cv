"""
scripts/augment.py
==================
Augments vivarium cage images and correctly transforms source labels.

Key fixes vs previous version:
    1. NO hardcoded ROI zones used for label generation.
       Container labels come from real GDINO-generated source labels,
       not invented pixel coordinates.
    2. Mouse + water + food boxes are all propagated from source labels.
    3. Geometric transforms (flip, rotation) are applied to labels too.
    4. --src-labels lets you point at where the original labels live.
    5. Seed is randomised per-run by default (pass --seed N for reproducibility).
    6. --img-size and --quality are configurable.
    7. Synthetic level overlays are drawn INSIDE the actual detected container
       bboxes — not at hardcoded pixel positions.

Workflow:
    # Step 1: label your originals first
    python scripts/gdino_label_originals.py

    # Step 2: augment — labels are auto-transformed from source
    python scripts/augment.py --src dataset/original --dst dataset/augmented --n 50

    # Step 3: split and train
    python scripts/split_dataset.py
    python scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 640   # default, overridable via --img-size


def glob_escape(s: str) -> str:
    """Escape glob special characters in a filename stem."""
    for ch in ["[", "]", "?", "*"]:
        s = s.replace(ch, f"[{ch}]")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Label I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO label file → list of (class_id, cx, cy, bw, bh).
    Uses open() with str path to handle Windows filenames with spaces/parentheses.
    """
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            content = f.read()
    except (FileNotFoundError, OSError):
        return []
    out = []
    for line in content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                out.append((int(parts[0]), float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return out


def write_labels(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Label transforms
# ─────────────────────────────────────────────────────────────────────────────

def flip_labels(labels: list[tuple]) -> list[tuple]:
    """Horizontal flip: cx → 1 - cx."""
    return [(c, 1.0 - cx, cy, bw, bh) for c, cx, cy, bw, bh in labels]


def rotate_labels(
    labels: list[tuple],
    angle_deg: float,
    img_size: int,
) -> list[tuple]:
    """
    Rotate bounding boxes to match cv2.warpAffine rotation.
    Converts each box to 4 corners, rotates them, re-fits axis-aligned bbox.
    Boxes that collapse outside the frame are dropped.
    """
    if abs(angle_deg) < 0.5:
        return labels

    rad    = math.radians(-angle_deg)
    cos_a  = math.cos(rad)
    sin_a  = math.sin(rad)
    ctr    = img_size / 2.0

    def rot(px, py):
        dx, dy = px - ctr, py - ctr
        return cos_a * dx - sin_a * dy + ctr, sin_a * dx + cos_a * dy + ctr

    out = []
    for (cls, cx_n, cy_n, bw_n, bh_n) in labels:
        cx_px = cx_n * img_size
        cy_px = cy_n * img_size
        bw_px = bw_n * img_size
        bh_px = bh_n * img_size

        x1, y1 = cx_px - bw_px / 2, cy_px - bh_px / 2
        x2, y2 = cx_px + bw_px / 2, cy_px - bh_px / 2
        x3, y3 = cx_px + bw_px / 2, cy_px + bh_px / 2
        x4, y4 = cx_px - bw_px / 2, cy_px + bh_px / 2

        corners = [rot(x, y) for x, y in [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]

        rx1 = max(0, min(xs))
        ry1 = max(0, min(ys))
        rx2 = min(img_size, max(xs))
        ry2 = min(img_size, max(ys))

        rbw = rx2 - rx1
        rbh = ry2 - ry1
        if rbw < 2 or rbh < 2:
            continue

        out.append((
            cls,
            (rx1 + rx2) / 2 / img_size,
            (ry1 + ry2) / 2 / img_size,
            rbw / img_size,
            rbh / img_size,
        ))
    return out


def clip_labels(labels: list[tuple]) -> list[tuple]:
    """Clip all coords to [0,1] and drop sub-pixel boxes."""
    out = []
    for (c, cx, cy, bw, bh) in labels:
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if bw > 0.005 and bh > 0.005:
            out.append((c, cx, cy, bw, bh))
    return out


def letterbox_labels(
    labels: list[tuple],
    scale: float,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
    img_size: int,
) -> list[tuple]:
    """
    Transform labels from original pixel space into letterboxed img_size space.
    Must be called once per source image before any augmentation transforms.
    """
    out = []
    for (cls, cx_n, cy_n, bw_n, bh_n) in labels:
        cx_px = cx_n * orig_w
        cy_px = cy_n * orig_h
        bw_px = bw_n * orig_w
        bh_px = bh_n * orig_h

        new_cx = (cx_px * scale + pad_left) / img_size
        new_cy = (cy_px * scale + pad_top)  / img_size
        new_bw = (bw_px * scale)            / img_size
        new_bh = (bh_px * scale)            / img_size

        out.append((cls, new_cx, new_cy, new_bw, new_bh))

    return clip_labels(out)


# ─────────────────────────────────────────────────────────────────────────────
# Image augmentation helpers
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


def _letterbox(img, size):
    """Returns (letterboxed_img, scale, pad_top, pad_left)."""
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, top, left


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic level overlays — drawn INSIDE real container bboxes
# ─────────────────────────────────────────────────────────────────────────────

WATER_LEVELS = [("critical", 0.05), ("low", 0.20), ("ok", 0.60), ("full", 0.90)]
FOOD_LEVELS  = [("critical", 0.05), ("low", 0.20), ("ok", 0.55), ("full", 0.85)]
WATER_COLOR  = (180, 120,  60)
FOOD_COLOR   = ( 60, 150, 190)


def _bbox_to_px(cx_n, cy_n, bw_n, bh_n, img_w, img_h):
    x1 = int((cx_n - bw_n / 2) * img_w)
    y1 = int((cy_n - bh_n / 2) * img_h)
    x2 = int((cx_n + bw_n / 2) * img_w)
    y2 = int((cy_n + bh_n / 2) * img_h)
    return x1, y1, x2, y2


def apply_synthetic_levels(img, water_bbox, food_bbox, water_frac, food_frac):
    """
    Draw fill overlays strictly inside the detected container bboxes.
    Skips overlay entirely if bbox is None — no fake rectangles drawn elsewhere.
    """
    h, w = img.shape[:2]
    out  = img.copy()

    for bbox, frac, color in [
        (water_bbox, water_frac, WATER_COLOR),
        (food_bbox,  food_frac,  FOOD_COLOR),
    ]:
        if bbox is None:
            continue

        x1, y1, x2, y2 = _bbox_to_px(*bbox, w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        fill_h  = max(1, int((y2 - y1) * frac))
        ys      = y2 - fill_h
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, ys), (x2, y2), color, thickness=-1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    return out

def _letterbox(img, size=640):
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampler
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(rng: random.Random) -> dict:
    w_label, w_frac = rng.choice(WATER_LEVELS)
    f_label, f_frac = rng.choice(FOOD_LEVELS)
    return {
        "alpha":       rng.uniform(0.6, 1.5),
        "beta":        rng.uniform(-60, 60),
        "dh":          rng.randint(-18, 18),
        "ds":          rng.randint(-40, 40),
        "dv":          rng.randint(-40, 40),
        "blur_k":      rng.choice([0, 3, 5, 7]),
        "noise_sigma": rng.uniform(0, 18),
        "flip":        rng.random() < 0.5,
        "angle":       rng.uniform(-15, 15),
        "water_label": w_label,
        "water_frac":  w_frac,
        "food_label":  f_label,
        "food_frac":   f_frac,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    src: Path,
    dst: Path,
    src_labels: Path | None,
    n: int,
    seed: int,
    img_size: int,
    quality: int,
) -> None:
    img_dir   = dst / "images"
    label_dir = dst / "labels"
    meta_dir  = dst / "meta"

    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (label_dir / "classes.txt").write_text("mouse\nwater_container\nfood_area\n")

    source_images = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not source_images:
        print(f"[ERROR] No images found in {src}")
        sys.exit(1)

    # ── Resolve label source directory ───────────────────────────
    # Checks in order: --src-labels arg → labels/ next to src → labels/ in dst
    candidates = [src_labels, src.parent / "labels", dst / "labels"]
    resolved_label_dir = next((p for p in candidates if p and p.exists()), None)

    if resolved_label_dir is None:
        print(
            "[WARN] No source labels found. Augmented images will have empty labels.\n"
            "       Run gdino_label_originals.py first, then re-run augment.py.\n"
            f"       Checked: {[str(p) for p in candidates if p]}"
        )
    else:
        print(f"Source labels  : {resolved_label_dir}")

    print(f"Source images  : {len(source_images)}")
    print(f"Variants each  : {n}  →  {len(source_images) * n} total")
    print(f"Image size     : {img_size}×{img_size}")
    print(f"Seed           : {seed}\n")

    rng   = random.Random(seed)
    total = 0
    no_label_count = 0

    for src_path in source_images:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [WARN] Cannot read {src_path.name}, skipping.")
            continue

        orig_h, orig_w = img.shape[:2]
        stem = src_path.stem

        # ── Load + letterbox-transform source labels ──────────────
        raw_labels: list[tuple] = []
        if resolved_label_dir is not None:
            import os
            # Use os.path.join with str() to avoid Windows Path issues
            # with spaces and parentheses in filenames
            label_path = Path(os.path.join(str(resolved_label_dir), stem + ".txt"))
            raw_labels = read_labels(label_path)
            if not raw_labels:
                # Fallback: try glob match in case of encoding differences
                matches = list(resolved_label_dir.glob(f"{glob_escape(stem)}.txt"))
                if matches:
                    raw_labels = read_labels(matches[0])
            if not raw_labels:
                no_label_count += 1
                print(f"  [WARN] No labels for {stem} — augmented variants will have empty labels")

        lb_img, scale, pad_top, pad_left = _letterbox(img, img_size)

        lb_labels = letterbox_labels(
            raw_labels, scale, pad_top, pad_left, orig_h, orig_w, img_size
        ) if raw_labels else []

        # ── Save original letterboxed image + labels ──────────────
        orig_name = f"{stem}_orig.jpg"
        cv2.imwrite(str(img_dir / orig_name), lb_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        write_labels(label_dir / f"{stem}_orig.txt", lb_labels)

        # Extract container bboxes (cx_n, cy_n, bw_n, bh_n) for overlay positioning
        water_bbox = next(((cx,cy,bw,bh) for (c,cx,cy,bw,bh) in lb_labels if c==1), None)
        food_bbox  = next(((cx,cy,bw,bh) for (c,cx,cy,bw,bh) in lb_labels if c==2), None)

        # ── Generate augmented variants ───────────────────────────
        for i in range(n):
            p = _sample_params(rng)

            # Image transforms
            aug = lb_img.copy()
            aug = adjust_brightness_contrast(aug, p["alpha"], p["beta"])
            aug = shift_hsv(aug, p["dh"], p["ds"], p["dv"])
            if p["blur_k"] > 0:
                aug = gaussian_blur(aug, p["blur_k"])
            if p["noise_sigma"] > 0:
                aug = add_gaussian_noise(aug, p["noise_sigma"])
            if p["flip"]:
                aug = horizontal_flip(aug)
            if abs(p["angle"]) > 0.5:
                aug = random_rotate(aug, p["angle"])

            # Update container bbox positions to match transforms (for overlay)
            ov_water = water_bbox
            ov_food  = food_bbox
            if p["flip"]:
                if ov_water: ov_water = (1.0-ov_water[0], *ov_water[1:])
                if ov_food:  ov_food  = (1.0-ov_food[0],  *ov_food[1:])

            aug = apply_synthetic_levels(
                aug, ov_water, ov_food, p["water_frac"], p["food_frac"]
            )

            # Label transforms (must match image transforms exactly)
            aug_labels = list(lb_labels)
            if p["flip"]:
                aug_labels = flip_labels(aug_labels)
            if abs(p["angle"]) > 0.5:
                aug_labels = rotate_labels(aug_labels, p["angle"], img_size)
            aug_labels = clip_labels(aug_labels)

            # Save
            out_name = f"{stem}_aug{i:04d}_w{p['water_label']}_f{p['food_label']}.jpg"
            cv2.imwrite(str(img_dir / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
            write_labels(label_dir / out_name.replace(".jpg", ".txt"), aug_labels)

            meta = {
                "source":    src_path.name,
                "output":    out_name,
                "aug_index": i,
                "label_counts": {
                    "mouse": sum(1 for (c,*_) in aug_labels if c==0),
                    "water": sum(1 for (c,*_) in aug_labels if c==1),
                    "food":  sum(1 for (c,*_) in aug_labels if c==2),
                },
                **p,
            }
            (meta_dir / out_name.replace(".jpg", ".json")).write_text(
                json.dumps(meta, indent=2)
            )
            total += 1

        m = sum(1 for (c,*_) in lb_labels if c==0)
        w = sum(1 for (c,*_) in lb_labels if c==1)
        f = sum(1 for (c,*_) in lb_labels if c==2)
        print(f"  ✓ {stem:<40} {n} variants  (src: mouse={m} water={w} food={f})")

    print(f"\nDone — {total} augmented images")
    if no_label_count:
        print(f"\n⚠  {no_label_count} source images had no labels.")
        print("   Run gdino_label_originals.py, then re-run augment.py.")
    else:
        print("\nNext: python scripts/split_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",        type=Path, default=Path("dataset/original"))
    ap.add_argument("--dst",        type=Path, default=Path("dataset/augmented"))
    ap.add_argument("--src-labels", type=Path, default=None,
                    help="Folder with source .txt labels (auto-detected if not set)")
    ap.add_argument("--n",          type=int,  default=50)
    ap.add_argument("--seed",       type=int,  default=random.randint(0, 99999),
                    help="Random seed (randomised each run by default)")
    ap.add_argument("--img-size",   type=int,  default=640)
    ap.add_argument("--quality",    type=int,  default=90)
    args = ap.parse_args()

    main(
        src=args.src,
        dst=args.dst,
        src_labels=args.src_labels,
        n=args.n,
        seed=args.seed,
        img_size=args.img_size,
        quality=args.quality,
    )
