"""
scripts/gdino_label_originals.py
=================================
Run GDINO only on ORIGINAL (pre-augmentation) images.
Then copy + propagate those labels to all augmented variants.

Class rules:
    0 = mouse           → multiple allowed (real lab cages can have many)
    1 = water_container → max 1 (one bottle per cage)
    2 = food_area       → multiple allowed (hopper + pile can be separate detections)

Prompts are color-agnostic — GDINO handles visual matching internally.
Water container is still described by shape/position since transparent
objects need context clues more than color ones.

Usage:
    python scripts/gdino_label_originals.py
    python scripts/gdino_label_originals.py --propagate
    python scripts/gdino_label_originals.py --mouse-thresh 0.25
    python scripts/gdino_label_originals.py --container-thresh 0.30
    python scripts/gdino_label_originals.py --food-thresh 0.28
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
ORIG_DIR  = Path("dataset/original")
AUG_DIR   = Path("dataset/augmented")
IMG_DIR   = AUG_DIR / "images"
LABEL_DIR = AUG_DIR / "labels"
META_DIR  = AUG_DIR / "meta"
DEBUG_DIR = AUG_DIR / "debug_gdino_orig"

IMG_SIZE  = 640

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# ── Prompts ───────────────────────────────────────────────────────────────────
# Color-agnostic — GDINO handles visual matching internally.
# Describe shape, context, and location rather than color.
PROMPTS = {
    0: (
        "laboratory mouse. "
        "small rodent sitting on bedding. "
        "rat inside cage. "
        "mouse on wood shavings."
    ),
    1: (
        "inverted transparent plastic bottle on top of cage. "
        "clear upside down water bottle. "
        "transparent drinking bottle mounted on cage. "
        "see-through plastic water dispenser on cage lid."
    ),
    2: (
        "food pellets in wire hopper. "
        "compressed rodent chow in cage feeder. "
        "food biscuits in wire basket. "
        "pellets in cage food container. "
        "rodent food in feeder."
    ),
}

# ── Confidence thresholds ─────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    0: 0.25,   # mouse      — moderate, multiple mice expected
    1: 0.32,   # water      — higher, only 1 bottle, want confident detection
    2: 0.28,   # food       — moderate, multiple food areas expected
}

# ── NMS IoU per class ─────────────────────────────────────────────────────────
NMS_IOU = {
    0: 0.40,   # mouse  — allow some overlap (mice can huddle together)
    1: 0.30,   # water  — tight, kill duplicate bottle detections
    2: 0.45,   # food   — looser, hopper opening + food pile are different regions
}

# ── Max boxes per class ───────────────────────────────────────────────────────
MAX_BOXES = {
    0: 8,      # up to 8 mice per cage (group housing)
    1: 1,      # exactly 1 water container per cage
    2: 4,      # up to 4 food area detections
}

# ── Size filters (normalised 0-1 in 640x640 space) ───────────────────────────
# Rejects noise (too small) and full-image false detections (too large)
SIZE_FILTERS = {
    # mouse: small animal, shouldn't cover more than 30% of image
    0: dict(min_area=0.002, max_area=0.30, min_w=0.02, max_w=0.55, min_h=0.02, max_h=0.65),
    # water: bottle is medium sized, sits on top corner of cage
    1: dict(min_area=0.010, max_area=0.35, min_w=0.05, max_w=0.55, min_h=0.05, max_h=0.70),
    # food: hopper is small-medium, shouldn't be full image width
    2: dict(min_area=0.005, max_area=0.25, min_w=0.03, max_w=0.55, min_h=0.03, max_h=0.60),
}

CLASS_COLORS = {0: (0, 255, 80), 1: (255, 180, 0), 2: (0, 140, 255)}
CLASS_NAMES  = {0: "mouse", 1: "water", 2: "food"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Device : {DEVICE}")
    print(f"Loading {MODEL_ID} …")
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("[ERROR] pip install transformers")
        sys.exit(1)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = (AutoModelForZeroShotObjectDetection
                 .from_pretrained(MODEL_ID).to(DEVICE))
    model.eval()
    print("Model ready.\n")
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_gdino(processor, model, pil_img, class_id, threshold):
    img_w, img_h = pil_img.size
    inputs = processor(
        images=pil_img,
        text=PROMPTS[class_id],
        return_tensors="pt",
    ).to(DEVICE)

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
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()
    keep   = scores >= threshold
    return list(zip(boxes[keep].tolist(), scores[keep].tolist()))


def nms(detections, iou_thresh):
    """Greedy NMS — keeps highest confidence boxes first."""
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
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Size filtering
# ─────────────────────────────────────────────────────────────────────────────

def passes_size_filter(x1, y1, x2, y2, img_w, img_h, class_id) -> bool:
    """Reject boxes that are clearly noise or full-image false detections."""
    f    = SIZE_FILTERS[class_id]
    bw   = (x2 - x1) / img_w
    bh   = (y2 - y1) / img_h
    area = bw * bh

    if area < f["min_area"] or area > f["max_area"]:
        return False
    if bw < f["min_w"] or bw > f["max_w"]:
        return False
    if bh < f["min_h"] or bh > f["max_h"]:
        return False
    return True


def filter_and_cap(detections, class_id, img_w, img_h):
    """
    Apply size filter then cap to MAX_BOXES[class_id].
    NMS already sorted by confidence so we take the top N after filtering.
    """
    filtered = [
        d for d in detections
        if passes_size_filter(
            d[0][0], d[0][1], d[0][2], d[0][3],
            img_w, img_h, class_id
        )
    ]
    return filtered[:MAX_BOXES[class_id]]


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo_str(x1, y1, x2, y2, img_w, img_h):
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
    bw = max(0.0, min(1.0, (x2 - x1) / img_w))
    bh = max(0.0, min(1.0, (y2 - y1) / img_h))
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def mirror_yolo_box(cx, cy, bw, bh):
    """Horizontally flip a normalised YOLO box."""
    return 1.0 - cx, cy, bw, bh


def save_debug(pil_img, all_dets, rejected, img_path, debug_dir):
    """
    Save debug image:
        Coloured boxes = kept detections
        Grey boxes     = rejected by size filter
    """
    debug_dir.mkdir(exist_ok=True)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Rejected — grey, thin
    for cls_id, dets in rejected.items():
        for (x1, y1, x2, y2), score in dets:
            cv2.rectangle(img_cv,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          (100, 100, 100), 1)
            cv2.putText(img_cv,
                        f"SKIP {CLASS_NAMES[cls_id]} {score:.2f}",
                        (int(x1), max(int(y1) - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Kept — class color, thick
    for cls_id, dets in all_dets.items():
        color = CLASS_COLORS[cls_id]
        for (x1, y1, x2, y2), score in dets:
            cv2.rectangle(img_cv,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            cv2.putText(img_cv,
                        f"{CLASS_NAMES[cls_id]} {score:.2f}",
                        (int(x1), max(int(y1) - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(debug_dir / img_path.name), img_cv)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Label originals
# ─────────────────────────────────────────────────────────────────────────────

def label_originals(processor, model, thresholds):
    orig_paths = []
    if ORIG_DIR.exists():
        orig_paths = sorted(
            p for p in ORIG_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not orig_paths:
        print("  dataset/original/ empty — using *_orig.jpg from augmented/images/")
        orig_paths = sorted(IMG_DIR.glob("*_orig.jpg"))

    if not orig_paths:
        print("[ERROR] No original images found.")
        sys.exit(1)

    print(f"Labelling {len(orig_paths)} original image(s) …\n")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    stem_to_boxes: dict[str, list] = {}

    for idx, img_path in enumerate(orig_paths, 1):
        try:
            pil_img      = Image.open(img_path).convert("RGB")
            pil_img      = _letterbox_pil(pil_img, IMG_SIZE)
            img_w, img_h = pil_img.size   # always 640x640

            all_dets: dict[int, list] = {}
            rejected:  dict[int, list] = {}
            label_lines = []
            boxes_norm  = []

            for cls_id in [0, 1, 2]:
                raw       = run_gdino(processor, model, pil_img, cls_id,
                                      thresholds[cls_id])
                after_nms = nms(raw, NMS_IOU[cls_id])
                kept      = filter_and_cap(after_nms, cls_id, img_w, img_h)
                rej       = [d for d in after_nms if d not in kept]

                all_dets[cls_id] = kept
                rejected[cls_id] = rej

                for (x1, y1, x2, y2), score in kept:
                    coords = xyxy_to_yolo_str(x1, y1, x2, y2, img_w, img_h)
                    label_lines.append(f"{cls_id} {coords}")
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    boxes_norm.append((cls_id, cx, cy, bw, bh))

            label_path = LABEL_DIR / (img_path.stem + ".txt")
            label_path.write_text("\n".join(label_lines))

            base_stem = img_path.stem
            if base_stem.endswith("_orig"):
                base_stem = base_stem[:-5]
            stem_to_boxes[base_stem] = boxes_norm

            counts     = {c: len(d) for c, d in all_dets.items()}
            rej_counts = {c: len(d) for c, d in rejected.items()}
            print(
                f"  [{idx:3d}] {img_path.name:<50} "
                f"mouse={counts[0]}  water={counts[1]}  food={counts[2]}  "
                f"| skipped: m={rej_counts[0]} w={rej_counts[1]} f={rej_counts[2]}"
            )

            save_debug(pil_img, all_dets, rejected, img_path, DEBUG_DIR)

        except Exception as e:
            print(f"  [{idx:3d}] {img_path.name:<50} ERROR: {e}")

    return stem_to_boxes


def _letterbox_pil(pil_img, size):
    w, h    = pil_img.size
    scale   = size / max(w, h)
    nw, nh  = int(w * scale), int(h * scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas  = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Propagate to augmented variants
# ─────────────────────────────────────────────────────────────────────────────

def propagate_labels(stem_to_boxes):
    aug_images = sorted(IMG_DIR.glob("*.jpg"))
    updated = 0
    skipped = 0

    print(f"\nPropagating labels to {len(aug_images)} augmented images …\n")

    for img_path in aug_images:
        stem = img_path.stem
        if stem.endswith("_orig"):
            continue

        base = _extract_base_stem(stem)
        if base not in stem_to_boxes:
            skipped += 1
            continue

        boxes       = stem_to_boxes[base]
        meta_path   = META_DIR / (stem + ".json")
        was_flipped = False

        if meta_path.exists():
            try:
                meta        = json.loads(meta_path.read_text())
                was_flipped = bool(meta.get("flip", False))
            except Exception:
                pass

        lines = []
        for (cls_id, cx, cy, bw, bh) in boxes:
            if was_flipped:
                cx, cy, bw, bh = mirror_yolo_box(cx, cy, bw, bh)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (LABEL_DIR / (stem + ".txt")).write_text("\n".join(lines))
        updated += 1

    print(f"  Updated : {updated} label files")
    print(f"  Skipped : {skipped} (no matching base found)")


def _extract_base_stem(aug_stem):
    if "_aug" in aug_stem:
        return aug_stem[:aug_stem.index("_aug")]
    return aug_stem


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(propagate, thresholds):
    processor, model = load_model()

    print("=" * 60)
    print("Step 1: Label original images")
    print("=" * 60)
    stem_to_boxes = label_originals(processor, model, thresholds)

    if propagate:
        print("\n" + "=" * 60)
        print("Step 2: Propagate to augmented variants")
        print("=" * 60)
        propagate_labels(stem_to_boxes)

    print(f"""
{'─'*60}
Done
  Originals labelled : {len(stem_to_boxes)}
  Debug images       : {DEBUG_DIR}

  Coloured boxes = kept
  Grey boxes     = rejected by size filter

Review debug images first, then run with --propagate.

Threshold defaults:
  mouse : {DEFAULT_THRESHOLDS[0]}  (lower = more mice detected)
  water : {DEFAULT_THRESHOLDS[1]}  (higher = fewer false bottles)
  food  : {DEFAULT_THRESHOLDS[2]}  (lower = more food areas detected)

To tune:
  python scripts/gdino_label_originals.py --mouse-thresh 0.20
{'─'*60}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--propagate",        action="store_true",
                    help="Copy labels from originals to all augmented variants")
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