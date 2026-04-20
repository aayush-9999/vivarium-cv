"""
scripts/gdino_label.py
======================
Labels ALL images (not just review_needed) using Grounding DINO.
Outputs correct 3-class YOLO labels:
    0 = mouse
    1 = water_container
    2 = food_area

Strategy:
  - Run 3 separate GDINO passes per image, one prompt per class.
    This is more reliable than a single multi-class prompt because
    GDINO's attention is cleaner when focused on one object type.
  - Merge all detections, apply per-class NMS to kill duplicates.
  - Write YOLO .txt with CORRECT class IDs (not all 0).
  - Save coloured debug images so you can spot bad boxes fast.

Usage:
    # Label everything from scratch
    python scripts/gdino_label.py

    # Label only the review_needed list (faster if most labels are already OK)
    python scripts/gdino_label.py --only-review

    # Preview without writing anything
    python scripts/gdino_label.py --dry-run

    # Adjust thresholds if too many / too few boxes
    python scripts/gdino_label.py --mouse-thresh 0.20 --container-thresh 0.28

Output:
    dataset/augmented/labels/   ← YOLO .txt files (class cx cy w h)
    dataset/augmented/debug_gdino/  ← debug JPEGs with coloured boxes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
IMG_DIR     = Path("dataset/augmented/images")
LABEL_DIR   = Path("dataset/augmented/labels")
DEBUG_DIR   = Path("dataset/augmented/debug_gdino")
REVIEW_FILE = Path("dataset/augmented/review_needed.txt")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-tiny"   # ~700 MB, cached after first run

# ── Per-class prompts ─────────────────────────────────────────────────────────
# GDINO is very prompt-sensitive. Specific visual descriptions work much better
# than generic class names. Tweak these if your lab setup looks different.
PROMPTS = {
    0: "small white mouse. small brown mouse. lab rodent. small furry animal.",
    1: (
        "transparent water bottle. plastic water jug. glass water container. "
        "sipper tube. drinking bottle mounted on cage. water dispenser."
    ),
    2: (
        "food hopper. brown food pellets. rodent chow. grain pile. "
        "feeding dish. food container. pellet dispenser."
    ),
}

# ── Thresholds (tune per class — mice are harder) ─────────────────────────────
DEFAULT_THRESHOLDS = {
    0: 0.22,   # mouse      — lower because mice are small + GDINO rarely sees them
    1: 0.28,   # water_container
    2: 0.25,   # food_area
}

# ── NMS IoU for deduplication within each class ───────────────────────────────
NMS_IOU = 0.45

# ── Debug colours (BGR) ───────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0,   255,  80),    # green  — mouse
    1: (255, 180,   0),    # blue   — water_container
    2: (0,   140, 255),    # orange — food_area
}
CLASS_NAMES = {0: "mouse", 1: "water", 2: "food"}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Device: {DEVICE}")
    print(f"Loading Grounding DINO ({MODEL_ID}) …")
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("[ERROR] Install transformers:  pip install transformers")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(MODEL_ID)
        .to(DEVICE)
    )
    model.eval()
    print("Model ready.\n")
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_gdino_for_class(
    processor,
    model,
    pil_img: Image.Image,
    class_id: int,
    threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    """
    Run GDINO for a single class prompt.
    Returns list of (x1, y1, x2, y2, score) in pixel coords.
    """
    img_w, img_h = pil_img.size
    prompt = PROMPTS[class_id]

    inputs = processor(
        images=pil_img,
        text=prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process — API-safe across transformers versions
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(img_h, img_w)],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

    boxes  = results["boxes"].cpu()
    scores = results["scores"].cpu()

    keep = scores >= threshold
    boxes  = boxes[keep].tolist()
    scores = scores[keep].tolist()

    return [(x1, y1, x2, y2, s) for (x1, y1, x2, y2), s in zip(boxes, scores)]


def nms(
    detections: list[tuple[float, float, float, float, float]],
    iou_thresh: float,
) -> list[tuple[float, float, float, float, float]]:
    """Simple greedy NMS. Input: list of (x1,y1,x2,y2,score)."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_thresh]

    return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h) -> str:
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
    bw = max(0.0, min(1.0, (x2 - x1) / img_w))
    bh = max(0.0, min(1.0, (y2 - y1) / img_h))
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_debug(
    pil_img: Image.Image,
    all_detections: dict[int, list],   # class_id → [(x1,y1,x2,y2,score)]
    img_path: Path,
) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for cls_id, dets in all_detections.items():
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        for (x1, y1, x2, y2, score) in dets:
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                img_cv,
                f"{label} {score:.2f}",
                (int(x1), max(int(y1) - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    cv2.imwrite(str(DEBUG_DIR / img_path.name), img_cv)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    processor,
    model,
    img_path: Path,
    thresholds: dict[int, float],
    dry_run: bool,
) -> dict:
    """
    Run GDINO on one image, write label file, save debug image.
    Returns stats dict.
    """
    pil_img      = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size

    all_detections: dict[int, list] = {}
    label_lines = []

    for cls_id in [0, 1, 2]:
        raw = run_gdino_for_class(processor, model, pil_img, cls_id, thresholds[cls_id])
        cleaned = nms(raw, NMS_IOU)
        all_detections[cls_id] = cleaned

        for (x1, y1, x2, y2, score) in cleaned:
            coords = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(f"{cls_id} {coords}")

    counts = {cls_id: len(dets) for cls_id, dets in all_detections.items()}

    if not dry_run:
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))
        save_debug(pil_img, all_detections, img_path)

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    only_review: bool,
    dry_run: bool,
    thresholds: dict[int, float],
) -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    if only_review:
        if not REVIEW_FILE.exists():
            print(f"[ERROR] {REVIEW_FILE} not found. Run auto_label.py first or use without --only-review.")
            sys.exit(1)
        stems = [Path(l).stem for l in REVIEW_FILE.read_text().strip().splitlines() if l.strip()]
        img_paths = []
        for stem in stems:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = IMG_DIR / (stem + ext)
                if p.exists():
                    img_paths.append(p)
                    break
        print(f"Mode: review_needed only ({len(img_paths)} images)")
    else:
        img_paths = sorted(
            p for p in IMG_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"Mode: ALL images ({len(img_paths)} images)")

    if not img_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"Thresholds → mouse={thresholds[0]}, water={thresholds[1]}, food={thresholds[2]}")
    print(f"Dry run: {dry_run}\n")

    processor, model = load_model()

    total         = len(img_paths)
    still_empty   = []
    summary       = {0: 0, 1: 0, 2: 0}

    for idx, img_path in enumerate(img_paths, 1):
        try:
            counts = process_image(processor, model, img_path, thresholds, dry_run)
        except Exception as e:
            print(f"  [{idx:4d}/{total}] {img_path.name:<50} ERROR: {e}")
            continue

        for cls_id, n in counts.items():
            summary[cls_id] += n

        total_boxes = sum(counts.values())
        if total_boxes == 0:
            still_empty.append(img_path.name)

        status = (
            f"mouse={counts[0]}  water={counts[1]}  food={counts[2]}"
            if total_boxes > 0
            else "⚠  NO DETECTIONS"
        )
        print(f"  [{idx:4d}/{total}] {img_path.name:<50} {status}")

    # Write still-empty list
    still_path = Path("dataset/augmented/still_needs_review.txt")
    if not dry_run:
        still_path.write_text("\n".join(still_empty))

    print(f"""
{'─'*60}
{'DRY RUN — nothing written' if dry_run else 'Grounding DINO labelling complete'}
  Images processed  : {total}
  Still empty       : {len(still_empty)}
  Total mouse boxes : {summary[0]}
  Total water boxes : {summary[1]}
  Total food  boxes : {summary[2]}
  Debug images      : {DEBUG_DIR}
{'─'*60}

If mouse box count is very low (< images/3):
  Lower --mouse-thresh to 0.15 and re-run.

If water/food boxes are wrong or missing:
  Check debug_gdino/ images.
  If your jug/hopper look unusual, edit PROMPTS in this script
  to describe them more specifically.

After verifying debug images:
  1. Fix remaining bad labels manually:
       pip install labelImg
       labelImg {IMG_DIR} {LABEL_DIR / 'classes.txt'}
  2. Run: python scripts/split_dataset.py
  3. Run: python scripts/train.py
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Label vivarium images with Grounding DINO (3-class)")
    ap.add_argument("--only-review",      action="store_true",
                    help="Only process images in review_needed.txt")
    ap.add_argument("--dry-run",          action="store_true",
                    help="Run inference but don't write label files")
    ap.add_argument("--mouse-thresh",     type=float, default=DEFAULT_THRESHOLDS[0],
                    help=f"Score threshold for mouse class (default {DEFAULT_THRESHOLDS[0]})")
    ap.add_argument("--mouse-container",  type=float, default=DEFAULT_THRESHOLDS[1],
                    help=f"Score threshold for water_container (default {DEFAULT_THRESHOLDS[1]})")
    ap.add_argument("--food-thresh",      type=float, default=DEFAULT_THRESHOLDS[2],
                    help=f"Score threshold for food_area (default {DEFAULT_THRESHOLDS[2]})")
    args = ap.parse_args()

    thresholds = {
        0: args.mouse_thresh,
        1: args.mouse_container,
        2: args.food_thresh,
    }

    main(
        only_review=args.only_review,
        dry_run=args.dry_run,
        thresholds=thresholds,
    )
