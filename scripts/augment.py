"""
scripts/augment.py
==================
Augments vivarium cage images and writes correct 9-class YOLO labels.

9-class scheme:
    0            mouse
    1            water_critical   (fill 0–15%)
    2            water_low        (fill 15–35%)
    3            water_ok         (fill 35–80%)
    4            water_full       (fill 80–100%)
    5            food_critical    (fill 0–15%)
    6            food_low         (fill 15–35%)
    7            food_ok          (fill 35–80%)
    8            food_full        (fill 80–100%)

For each augmented variant:
  - Water bbox class  = determined by water_frac sampled for that variant
  - Food bbox class   = determined by food_frac sampled for that variant
  - Mouse boxes       = propagated from source labels (class 0), with flip/rotate applied

The augmentation script knows the exact fill fraction it drew, so the
class assignment is 100% accurate — no guessing, no HSV.

Workflow:
    python scripts/gdino_label_originals.py --label-out dataset/original_labels
    python scripts/augment.py \
        --src dataset/original \
        --src-labels dataset/original_labels \
        --dst dataset/augmented \
        --n 50
    python scripts/split_dataset.py
    python scripts/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import math
import os
import random
import sys
import cv2
import numpy as np

IMG_SIZE = 640


# ── Class boundary helpers ────────────────────────────────────────────────────

def _frac_to_water_class(frac: float) -> int:
    from core.config import WATER_CLASS_BOUNDARIES
    for lo, hi, cls in WATER_CLASS_BOUNDARIES:
        if lo <= frac < hi:
            return cls
    return 3   # default: ok

def _frac_to_food_class(frac: float) -> int:
    from core.config import FOOD_CLASS_BOUNDARIES
    for lo, hi, cls in FOOD_CLASS_BOUNDARIES:
        if lo <= frac < hi:
            return cls
    return 7   # default: ok


# ── Glob escape ───────────────────────────────────────────────────────────────

def glob_escape(s: str) -> str:
    for ch in ["[", "]", "?", "*"]:
        s = s.replace(ch, f"[{ch}]")
    return s


# ── Label I/O ─────────────────────────────────────────────────────────────────

def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
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


def write_labels(path: Path, labels: list[tuple]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Label transforms ──────────────────────────────────────────────────────────

def flip_labels(labels):
    return [(c, 1.0 - cx, cy, bw, bh) for c, cx, cy, bw, bh in labels]


def rotate_labels(labels, angle_deg, img_size):
    if abs(angle_deg) < 0.5:
        return labels
    rad   = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    ctr   = img_size / 2.0

    def rot(px, py):
        dx, dy = px - ctr, py - ctr
        return cos_a*dx - sin_a*dy + ctr, sin_a*dx + cos_a*dy + ctr

    out = []
    for (cls, cx_n, cy_n, bw_n, bh_n) in labels:
        cx_px, cy_px = cx_n*img_size, cy_n*img_size
        bw_px, bh_px = bw_n*img_size, bh_n*img_size
        x1,y1 = cx_px-bw_px/2, cy_px-bh_px/2
        x2,y2 = cx_px+bw_px/2, cy_py-bh_px/2 if False else cy_px-bh_px/2
        corners = [rot(x,y) for x,y in [
            (cx_px-bw_px/2, cy_px-bh_px/2),
            (cx_px+bw_px/2, cy_px-bh_px/2),
            (cx_px+bw_px/2, cy_px+bh_px/2),
            (cx_px-bw_px/2, cy_px+bh_px/2),
        ]]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        rx1,ry1 = max(0,min(xs)), max(0,min(ys))
        rx2,ry2 = min(img_size,max(xs)), min(img_size,max(ys))
        rbw,rbh = rx2-rx1, ry2-ry1
        if rbw < 2 or rbh < 2:
            continue
        out.append((cls, (rx1+rx2)/2/img_size, (ry1+ry2)/2/img_size,
                    rbw/img_size, rbh/img_size))
    return out


def clip_labels(labels):
    out = []
    for (c, cx, cy, bw, bh) in labels:
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if bw > 0.005 and bh > 0.005:
            out.append((c, cx, cy, bw, bh))
    return out


def letterbox_labels(labels, scale, pad_top, pad_left, orig_h, orig_w, img_size):
    out = []
    for (cls, cx_n, cy_n, bw_n, bh_n) in labels:
        new_cx = (cx_n*orig_w*scale + pad_left) / img_size
        new_cy = (cy_n*orig_h*scale + pad_top)  / img_size
        new_bw = (bw_n*orig_w*scale)             / img_size
        new_bh = (bh_n*orig_h*scale)             / img_size
        out.append((cls, new_cx, new_cy, new_bw, new_bh))
    return clip_labels(out)


# ── Image helpers ─────────────────────────────────────────────────────────────

def adjust_brightness_contrast(img, alpha, beta):
    return np.clip(alpha*img.astype(np.float32)+beta, 0, 255).astype(np.uint8)

def shift_hsv(img, dh, ds, dv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,0] = np.clip(hsv[:,:,0]+dh, 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1]+ds, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2]+dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gaussian_blur(img, ksize):
    ksize = ksize if ksize%2==1 else ksize+1
    return cv2.GaussianBlur(img, (ksize,ksize), 0)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)

def random_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

def horizontal_flip(img):
    return cv2.flip(img, 1)

def _letterbox(img, size):
    h, w   = img.shape[:2]
    scale  = size / max(h,w)
    nw,nh  = int(w*scale), int(h*scale)
    res    = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size,size,3), 114, dtype=np.uint8)
    top    = (size-nh)//2
    left   = (size-nw)//2
    canvas[top:top+nh, left:left+nw] = res
    return canvas, scale, top, left


# ── Synthetic level overlay ───────────────────────────────────────────────────

WATER_LEVELS = [("critical",0.05),("low",0.20),("ok",0.60),("full",0.90)]
FOOD_LEVELS  = [("critical",0.05),("low",0.20),("ok",0.55),("full",0.85)]
WATER_COLOR  = (180,120, 60)
FOOD_COLOR   = ( 60,150,190)


def apply_synthetic_levels(img, water_bbox_norm, food_bbox_norm, water_frac, food_frac):
    """Draw fill overlay INSIDE the actual detected bbox. Skip if bbox unknown."""
    h, w = img.shape[:2]
    out  = img.copy()
    for bbox_n, frac, color in [
        (water_bbox_norm, water_frac, WATER_COLOR),
        (food_bbox_norm,  food_frac,  FOOD_COLOR),
    ]:
        if bbox_n is None:
            continue
        cx_n, cy_n, bw_n, bh_n = bbox_n
        x1 = max(0, int((cx_n-bw_n/2)*w))
        y1 = max(0, int((cy_n-bh_n/2)*h))
        x2 = min(w, int((cx_n+bw_n/2)*w))
        y2 = min(h, int((cy_n+bh_n/2)*h))
        fill_h  = max(1, int((y2-y1)*frac))
        ys      = y2 - fill_h
        overlay = out.copy()
        cv2.rectangle(overlay, (x1,ys), (x2,y2), color, -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    return out


# ── Parameter sampler ─────────────────────────────────────────────────────────

def _sample_params(rng):
    w_label, w_frac = rng.choice(WATER_LEVELS)
    f_label, f_frac = rng.choice(FOOD_LEVELS)
    return {
        "alpha":       rng.uniform(0.6, 1.5),
        "beta":        rng.uniform(-60, 60),
        "dh":          rng.randint(-18, 18),
        "ds":          rng.randint(-40, 40),
        "dv":          rng.randint(-40, 40),
        "blur_k":      rng.choice([0,3,5,7]),
        "noise_sigma": rng.uniform(0, 18),
        "flip":        rng.random() < 0.5,
        "angle":       rng.uniform(-15, 15),
        "water_label": w_label,
        "water_frac":  w_frac,
        "food_label":  f_label,
        "food_frac":   f_frac,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(src, dst, src_labels, n, seed, img_size, quality):
    img_dir   = dst / "images"
    label_dir = dst / "labels"
    meta_dir  = dst / "meta"
    for d in [img_dir, label_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (label_dir / "classes.txt").write_text(
        "mouse\nwater_critical\nwater_low\nwater_ok\nwater_full\n"
        "food_critical\nfood_low\nfood_ok\nfood_full\n"
    )

    source_images = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}
    )
    if not source_images:
        print(f"[ERROR] No images in {src}"); sys.exit(1)

    # Resolve label source
    candidates = [src_labels, src.parent/"labels", dst/"labels"]
    resolved   = next((p for p in candidates if p and p.exists()), None)
    if resolved is None:
        print("[WARN] No source labels found — all labels will be empty.")
    else:
        print(f"Source labels  : {resolved}")

    print(f"Source images  : {len(source_images)}")
    print(f"Variants each  : {n}  →  {len(source_images)*n} total")
    print(f"Seed           : {seed}\n")

    rng   = random.Random(seed)
    total = 0

    for src_path in source_images:
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [WARN] Cannot read {src_path.name}"); continue

        orig_h, orig_w = img.shape[:2]
        stem = src_path.stem

        # Load source labels — keep only mouse (class 0) boxes from GDINO
        raw_labels = []
        if resolved is not None:
            label_path = Path(os.path.join(str(resolved), stem + ".txt"))
            raw_labels = read_labels(label_path)
            if not raw_labels:
                matches = list(resolved.glob(f"{glob_escape(stem)}.txt"))
                if matches:
                    raw_labels = read_labels(matches[0])

        # Only propagate mouse boxes — water/food class will be set by augment
        mouse_labels = [(c,cx,cy,bw,bh) for (c,cx,cy,bw,bh) in raw_labels if c == 0]

        lb_img, scale, pad_top, pad_left = _letterbox(img, img_size)

        lb_mouse = letterbox_labels(mouse_labels, scale, pad_top, pad_left,
                                    orig_h, orig_w, img_size)

        # Extract water/food bbox positions from GDINO labels for overlay positioning
        # GDINO used classes 1/2; we use them only for overlay position, not class
        all_lb = letterbox_labels(raw_labels, scale, pad_top, pad_left,
                                  orig_h, orig_w, img_size)
        water_bbox_n = next(((cx,cy,bw,bh) for (c,cx,cy,bw,bh) in all_lb if c==1), None)
        food_bbox_n  = next(((cx,cy,bw,bh) for (c,cx,cy,bw,bh) in all_lb if c==2), None)

        # Save original
        orig_name = f"{stem}_orig.jpg"
        cv2.imwrite(str(img_dir/orig_name), lb_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # For orig: use median fill level (ok)
        orig_labels = list(lb_mouse)
        if water_bbox_n:
            cx,cy,bw,bh = water_bbox_n
            orig_labels.append((3, cx, cy, bw, bh))  # water_ok
        if food_bbox_n:
            cx,cy,bw,bh = food_bbox_n
            orig_labels.append((7, cx, cy, bw, bh))  # food_ok
        write_labels(label_dir/f"{stem}_orig.txt", clip_labels(orig_labels))

        for i in range(n):
            p = _sample_params(rng)

            # Image transforms
            aug = lb_img.copy()
            aug = adjust_brightness_contrast(aug, p["alpha"], p["beta"])
            aug = shift_hsv(aug, p["dh"], p["ds"], p["dv"])
            if p["blur_k"] > 0:  aug = gaussian_blur(aug, p["blur_k"])
            if p["noise_sigma"] > 0: aug = add_gaussian_noise(aug, p["noise_sigma"])
            if p["flip"]:  aug = horizontal_flip(aug)
            if abs(p["angle"]) > 0.5: aug = random_rotate(aug, p["angle"])

            # Container bbox positions after transforms (for overlay)
            ov_water = water_bbox_n
            ov_food  = food_bbox_n
            if p["flip"]:
                if ov_water: ov_water = (1.0-ov_water[0], *ov_water[1:])
                if ov_food:  ov_food  = (1.0-ov_food[0],  *ov_food[1:])
            aug = apply_synthetic_levels(aug, ov_water, ov_food,
                                         p["water_frac"], p["food_frac"])

            # Label transforms
            aug_mouse = list(lb_mouse)
            if p["flip"]:        aug_mouse = flip_labels(aug_mouse)
            if abs(p["angle"]) > 0.5: aug_mouse = rotate_labels(aug_mouse, p["angle"], img_size)
            aug_mouse = clip_labels(aug_mouse)

            # Determine water/food class from fill fraction
            w_cls = _frac_to_water_class(p["water_frac"])
            f_cls = _frac_to_food_class(p["food_frac"])

            # Build final label set: mouse boxes + 1 water box + 1 food box
            aug_labels = list(aug_mouse)
            if ov_water:
                cx,cy,bw,bh = ov_water
                aug_labels.append((w_cls, cx, cy, bw, bh))
            if ov_food:
                cx,cy,bw,bh = ov_food
                aug_labels.append((f_cls, cx, cy, bw, bh))
            aug_labels = clip_labels(aug_labels)

            out_name = f"{stem}_aug{i:04d}_w{p['water_label']}_f{p['food_label']}.jpg"
            cv2.imwrite(str(img_dir/out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
            write_labels(label_dir/out_name.replace(".jpg",".txt"), aug_labels)

            meta = {
                "source": src_path.name, "output": out_name, "aug_index": i,
                "water_class": w_cls, "food_class": f_cls,
                "label_counts": {
                    "mouse": sum(1 for (c,*_) in aug_labels if c==0),
                    "water": sum(1 for (c,*_) in aug_labels if c in {1,2,3,4}),
                    "food":  sum(1 for (c,*_) in aug_labels if c in {5,6,7,8}),
                },
                **p,
            }
            (meta_dir/out_name.replace(".jpg",".json")).write_text(
                json.dumps(meta, indent=2))
            total += 1

        m = len(lb_mouse)
        print(f"  ✓ {stem:<40} {n} variants  (mouse={m} water={'yes' if water_bbox_n else 'NO'} food={'yes' if food_bbox_n else 'NO'})")

    print(f"\nDone — {total} augmented images → {img_dir}")
    print("Next: python scripts/split_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",        type=Path, default=Path("dataset/original"))
    ap.add_argument("--dst",        type=Path, default=Path("dataset/augmented"))
    ap.add_argument("--src-labels", type=Path, default=None)
    ap.add_argument("--n",          type=int,  default=50)
    ap.add_argument("--seed",       type=int,  default=random.randint(0,99999))
    ap.add_argument("--img-size",   type=int,  default=640)
    ap.add_argument("--quality",    type=int,  default=90)
    args = ap.parse_args()
    main(args.src, args.dst, args.src_labels, args.n,
         args.seed, args.img_size, args.quality)