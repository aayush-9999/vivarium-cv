"""
scripts/gdino_label_originals.py
=================================
THE CORRECT APPROACH:
  Run GDINO only on ORIGINAL (pre-augmentation) images.
  Then copy + propagate those labels to all augmented variants.

Why this fixes the duplicate/false-box problem:
  - Augmented images have synthetic water/food overlays drawn on them.
  - GDINO was detecting those overlay rectangles AS water/food containers.
  - Original images have no overlays → clean detections.
  - Since augmentation doesn't move the jug/hopper (only colour/brightness/flip),
    the same bounding boxes are valid for all variants of the same source.

IMPORTANT — flip handling:
  If an augmented variant was horizontally flipped, container bbox x-coords
  are mirrored. This script reads the augmentation metadata JSON to detect
  flips and mirrors the bbox accordingly.

Usage:
    # Step 1: label originals
    python scripts/gdino_label_originals.py

    # Step 2: if results look good in debug_gdino_orig/, propagate to augmented
    python scripts/gdino_label_originals.py --propagate

    # Tune thresholds if needed
    python scripts/gdino_label_originals.py --mouse-thresh 0.20 --container-thresh 0.25

Output:
    dataset/augmented/labels/          ← updated label files
    dataset/augmented/debug_gdino_orig/ ← debug images for originals only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ORIG_DIR    = Path("dataset/original")           # your raw source images
AUG_DIR     = Path("dataset/augmented")
IMG_DIR     = AUG_DIR / "images"
LABEL_DIR   = AUG_DIR / "labels"
META_DIR    = AUG_DIR / "meta"
DEBUG_DIR   = AUG_DIR / "debug_gdino_orig"

IMG_SIZE    = 640

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# ── Prompts (focused, no overlap between classes) ─────────────────────────────
PROMPTS = {
    0: "small white mouse. small brown mouse. lab rodent. small furry animal sitting.",
    1: "transparent water bottle. plastic water jug. sipper tube on cage wall. drinking bottle.",
    2: "brown food pellets in tray. rodent food dish. food hopper. grain pile in container.",
}

# ── Thresholds ────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    0: 0.22,
    1: 0.30,
    2: 0.28,
}

NMS_IOU     = 0.40   # tighter than before to kill duplicate container boxes

CLASS_COLORS = {
    0: (0,   255,  80),
    1: (255, 180,   0),
    2: (0,   140, 255),
}
CLASS_NAMES = {0: "mouse", 1: "water", 2: "food"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_ID} …")
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("[ERROR] pip install transformers")
        sys.exit(1)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (AutoModelForZeroShotObjectDetection
             .from_pretrained(MODEL_ID).to(DEVICE))
    model.eval()
    print("Model ready.\n")
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_gdino(processor, model, pil_img, class_id, threshold):
    img_w, img_h = pil_img.size
    inputs = processor(images=pil_img, text=PROMPTS[class_id],
                       return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            target_sizes=[(img_h, img_w)],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=threshold, text_threshold=threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()
    keep   = scores >= threshold
    return list(zip(boxes[keep].tolist(), scores[keep].tolist()))


def nms(detections, iou_thresh):
    """detections: list of ([x1,y1,x2,y2], score)"""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if _iou(best[0], d[0]) < iou_thresh]
    return kept


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo_str(x1, y1, x2, y2, img_w, img_h):
    cx = max(0.0, min(1.0, ((x1+x2)/2)/img_w))
    cy = max(0.0, min(1.0, ((y1+y2)/2)/img_h))
    bw = max(0.0, min(1.0, (x2-x1)/img_w))
    bh = max(0.0, min(1.0, (y2-y1)/img_h))
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def mirror_yolo_box(cx, cy, bw, bh):
    """Horizontally flip a normalised YOLO box."""
    return 1.0 - cx, cy, bw, bh


def save_debug(pil_img, all_dets, img_path, debug_dir):
    debug_dir.mkdir(exist_ok=True)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for cls_id, dets in all_dets.items():
        color = CLASS_COLORS[cls_id]
        label = CLASS_NAMES[cls_id]
        for (x1,y1,x2,y2), score in dets:
            cv2.rectangle(img_cv,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            cv2.putText(img_cv,f"{label} {score:.2f}",
                        (int(x1),max(int(y1)-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    cv2.imwrite(str(debug_dir/img_path.name), img_cv)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Label originals
# ─────────────────────────────────────────────────────────────────────────────

def label_originals(processor, model, thresholds):
    """
    Run GDINO on original images (from dataset/original/).
    Also runs on *_orig.jpg in augmented/images/ as fallback.
    Saves labels to labels/{stem}.txt and debug images.
    Returns dict: stem → list of (class_id, cx, cy, bw, bh) normalised boxes
    """
    # Collect original images — prefer dataset/original/, fallback to *_orig.jpg
    orig_paths = []
    if ORIG_DIR.exists():
        orig_paths = sorted(
            p for p in ORIG_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not orig_paths:
        print(f"  dataset/original/ empty or missing — using *_orig.jpg from augmented/images/")
        orig_paths = sorted(IMG_DIR.glob("*_orig.jpg"))

    if not orig_paths:
        print("[ERROR] No original images found. Put source images in dataset/original/")
        sys.exit(1)

    print(f"Labelling {len(orig_paths)} original image(s) …\n")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    stem_to_boxes: dict[str, list] = {}

    for idx, img_path in enumerate(orig_paths, 1):
        try:
            pil_img      = Image.open(img_path).convert("RGB")
            # Letterbox to 640×640 — same as augmentation pipeline
            pil_img      = _letterbox_pil(pil_img, IMG_SIZE)
            img_w, img_h = pil_img.size   # always 640×640 after letterbox

            all_dets: dict[int, list] = {}
            label_lines = []
            boxes_norm  = []

            for cls_id in [0, 1, 2]:
                raw     = run_gdino(processor, model, pil_img, cls_id, thresholds[cls_id])
                cleaned = nms(raw, NMS_IOU)
                all_dets[cls_id] = cleaned

                for (x1,y1,x2,y2), score in cleaned:
                    coords = xyxy_to_yolo_str(x1,y1,x2,y2, img_w, img_h)
                    label_lines.append(f"{cls_id} {coords}")

                    # Store normalised for propagation
                    cx = ((x1+x2)/2)/img_w
                    cy = ((y1+y2)/2)/img_h
                    bw = (x2-x1)/img_w
                    bh = (y2-y1)/img_h
                    boxes_norm.append((cls_id, cx, cy, bw, bh))

            # Write label for this original / *_orig image
            label_path = LABEL_DIR / (img_path.stem + ".txt")
            label_path.write_text("\n".join(label_lines))

            # The "base" stem used to match augmented variants
            # e.g. "image_orig" → base = "image"
            # e.g. "image (3)_orig" → base = "image (3)"
            base_stem = img_path.stem
            if base_stem.endswith("_orig"):
                base_stem = base_stem[:-5]
            stem_to_boxes[base_stem] = boxes_norm

            counts = {c: len(d) for c,d in all_dets.items()}
            print(f"  [{idx:3d}] {img_path.name:<50} "
                  f"mouse={counts[0]} water={counts[1]} food={counts[2]}")

            save_debug(pil_img, all_dets, img_path, DEBUG_DIR)

        except Exception as e:
            print(f"  [{idx:3d}] {img_path.name:<50} ERROR: {e}")

    return stem_to_boxes


def _letterbox_pil(pil_img: Image.Image, size: int) -> Image.Image:
    w, h   = pil_img.size
    scale  = size / max(w, h)
    nw, nh = int(w*scale), int(h*scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas  = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size-nw)//2, (size-nh)//2))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Propagate to augmented variants
# ─────────────────────────────────────────────────────────────────────────────

def propagate_labels(stem_to_boxes: dict):
    """
    Copy base-image labels to all augmented variants.
    Reads meta JSON to detect horizontal flips and mirrors bbox x accordingly.
    """
    aug_images = sorted(IMG_DIR.glob("*.jpg"))
    updated    = 0
    skipped    = 0

    print(f"\nPropagating labels to {len(aug_images)} augmented images …\n")

    for img_path in aug_images:
        stem = img_path.stem

        # Skip originals — already labelled
        if stem.endswith("_orig"):
            continue

        # Find base stem: "image_aug0001_wok_fcritical" → "image"
        base = _extract_base_stem(stem)
        if base not in stem_to_boxes:
            skipped += 1
            continue

        boxes = stem_to_boxes[base]   # list of (cls, cx, cy, bw, bh)

        # Check if this variant was flipped — read from meta JSON
        meta_path = META_DIR / (stem + ".json")
        was_flipped = False
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                was_flipped = bool(meta.get("flip", False))
            except Exception:
                pass

        # Build label lines, mirroring x if flipped
        lines = []
        for (cls_id, cx, cy, bw, bh) in boxes:
            if was_flipped:
                cx, cy, bw, bh = mirror_yolo_box(cx, cy, bw, bh)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = LABEL_DIR / (stem + ".txt")
        label_path.write_text("\n".join(lines))
        updated += 1

    print(f"  Updated : {updated} label files")
    print(f"  Skipped : {skipped} (no matching base image found)")


def _extract_base_stem(aug_stem: str) -> str:
    """
    'image_aug0001_wok_fcritical'    → 'image'
    'image (3)_aug0012_wfull_flow'   → 'image (3)'
    """
    # Split on '_aug' — everything before it is the base name
    if "_aug" in aug_stem:
        return aug_stem[:aug_stem.index("_aug")]
    return aug_stem


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(propagate: bool, thresholds: dict):
    processor, model = load_model()

    print("=" * 60)
    print("Step 1: Label original images")
    print("=" * 60)
    stem_to_boxes = label_originals(processor, model, thresholds)

    if propagate:
        print("\n" + "=" * 60)
        print("Step 2: Propagate labels to augmented variants")
        print("=" * 60)
        propagate_labels(stem_to_boxes)

    print(f"""
{'─'*60}
Done
  Original images labelled : {len(stem_to_boxes)}
  Debug images             : {DEBUG_DIR}
{'─'*60}

Check debug images in {DEBUG_DIR}
If they look good, run with --propagate to copy to all augmented variants:
  python scripts/gdino_label_originals.py --propagate

Then:
  python scripts/split_dataset.py
  python scripts/train.py
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--propagate",        action="store_true",
                    help="After labelling originals, copy labels to all augmented variants")
    ap.add_argument("--mouse-thresh",     type=float, default=DEFAULT_THRESHOLDS[0])
    ap.add_argument("--container-thresh", type=float, default=DEFAULT_THRESHOLDS[1])
    ap.add_argument("--food-thresh",      type=float, default=DEFAULT_THRESHOLDS[2])
    args = ap.parse_args()

    thresholds = {
        0: args.mouse_thresh,
        1: args.container_thresh,
        2: args.food_thresh,
    }
    main(propagate=args.propagate, thresholds=thresholds)