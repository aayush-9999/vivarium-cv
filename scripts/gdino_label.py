"""
scripts/gdino_label.py
======================
Re-labels review_needed images using Grounding DINO via HuggingFace Transformers.
Fixed for transformers >= 4.38 where post_process_grounded_object_detection()
no longer accepts box_threshold / text_threshold as kwargs — filtering is now
done manually on the returned scores tensor.

Usage:
    python scripts/gdino_label.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
REVIEW_FILE = Path("dataset/augmented/review_needed.txt")
IMG_DIR     = Path("dataset/augmented/images")
LABEL_DIR   = Path("dataset/augmented/labels")
DEBUG_DIR   = Path("dataset/augmented/debug_gdino")

BOX_THRESH  = 0.25   # filter boxes below this score after post-processing
PROMPT      = "a mouse. a rodent. a small animal."
MODEL_ID    = "IDEA-Research/grounding-dino-tiny"   # ~700 MB, cached after first run

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: Running on CPU — expect ~20-40s per image.\n")

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading Grounding DINO (downloads ~700 MB on first run, cached after)…")
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError:
    print("[ERROR] transformers not installed.\n  pip install transformers")
    sys.exit(1)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("Model ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo(
    x1: float, y1: float,
    x2: float, y2: float,
    img_w: int, img_h: int,
) -> str:
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
    bw = max(0.0, min(1.0, (x2 - x1) / img_w))
    bh = max(0.0, min(1.0, (y2 - y1) / img_h))
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def run_gdino(pil_img: Image.Image) -> tuple[list, list]:
    """
    Run Grounding DINO and return (boxes_xyxy, scores) after threshold filtering.

    Compatible with transformers >= 4.38:
      - Do NOT pass box_threshold / text_threshold to post_process_…()
      - Instead, pass threshold= to the post-process call if supported,
        OR filter the scores tensor ourselves (most reliable across versions).
    """
    img_w, img_h = pil_img.size

    inputs = processor(
        images=pil_img,
        text=PROMPT,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # ── API-safe post-processing ──────────────────────────────────
    # We call without threshold kwargs (works on all transformers versions)
    # and filter by score manually below.
    try:
        # transformers >= 4.38: signature accepts no threshold kwargs
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(img_h, img_w)],
        )[0]
    except TypeError:
        # Very old transformers: try with threshold kwargs as fallback
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESH,
            text_threshold=BOX_THRESH,
            target_sizes=[(img_h, img_w)],
        )[0]

    boxes_tensor  = results["boxes"]   # (N, 4) xyxy in pixel space
    scores_tensor = results["scores"]  # (N,)

    # Manual threshold filter — works regardless of API version
    keep   = scores_tensor >= BOX_THRESH
    boxes  = boxes_tensor[keep].cpu().tolist()
    scores = scores_tensor[keep].cpu().tolist()

    return boxes, scores


def save_debug(pil_img: Image.Image, boxes: list, img_path: Path) -> None:
    import cv2
    import numpy as np
    DEBUG_DIR.mkdir(exist_ok=True)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 80), 2)
        cv2.putText(img_cv, "mouse?", (int(x1), max(int(y1) - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
    cv2.imwrite(str(DEBUG_DIR / img_path.name), img_cv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not REVIEW_FILE.exists():
        print(f"[ERROR] {REVIEW_FILE} not found.")
        print("Run auto_label.py first to generate review_needed.txt")
        sys.exit(1)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_lines    = REVIEW_FILE.read_text().strip().splitlines()
    review_stems = [Path(l).stem for l in raw_lines if l.strip()]

    print(f"Images to re-label : {len(review_stems)}")
    print(f"Prompt             : {PROMPT}")
    print(f"Score threshold    : {BOX_THRESH}\n")

    labelled    = 0
    still_empty = []
    errors      = []

    for idx, stem in enumerate(review_stems, 1):
        # Try jpg then png
        img_path = IMG_DIR / (stem + ".jpg")
        if not img_path.exists():
            img_path = IMG_DIR / (stem + ".png")
        if not img_path.exists():
            errors.append(f"NOT FOUND: {stem}")
            print(f"  [{idx:4d}/{len(review_stems)}] {stem:<55} NOT FOUND")
            continue

        try:
            pil_img       = Image.open(img_path).convert("RGB")
            img_w, img_h  = pil_img.size

            boxes, scores = run_gdino(pil_img)

            label_path = LABEL_DIR / (stem + ".txt")

            if boxes:
                lines = [
                    xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                    for (x1, y1, x2, y2) in boxes
                ]
                label_path.write_text("\n".join(lines))
                save_debug(pil_img, boxes, img_path)
                labelled += 1
                status = f"{len(boxes)} box(es)  scores={[round(s, 2) for s in scores]}"
            else:
                label_path.write_text("")   # empty label = no mouse visible
                still_empty.append(img_path.name)
                status = "no detection → empty label written"

        except Exception as e:
            errors.append(f"ERROR {stem}: {e}")
            status = f"ERROR: {e}"

        print(f"  [{idx:4d}/{len(review_stems)}] {stem[:52]:<52} {status}")

    # ── Write still_needs_review.txt ─────────────────────────────────────────
    still_path = Path("dataset/augmented/still_needs_review.txt")
    still_path.write_text("\n".join(still_empty))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"""
{'─'*60}
Grounding DINO labelling complete
  Reviewed       : {len(review_stems)}
  Newly labelled : {labelled}
  Still empty    : {len(still_empty)}  → {still_path}
  Errors         : {len(errors)}
  Debug images   : {DEBUG_DIR}
{'─'*60}

Next steps:
  1. Check dataset/augmented/debug_gdino/ to verify boxes look correct.
  2. For anything in still_needs_review.txt — either:
       a) Leave empty label (valid if no mouse is visible in frame), OR
       b) Label manually:
            pip install labelImg
            labelImg {IMG_DIR} {LABEL_DIR / 'classes.txt'}
  3. Run:  python scripts/train.py
""")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()