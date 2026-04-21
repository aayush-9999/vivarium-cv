"""
scripts/gdino_label_originals.py
=================================
Run GDINO only on ORIGINAL images. Saves labels to a SEPARATE folder
(dataset/original_labels/) so augment.py can find them cleanly.

Usage:
    # Step 1: label originals
    python scripts/gdino_label_originals.py

    # Step 2: check debug images
    #   dataset/augmented/debug_gdino_orig/

    # Step 3: augment using those labels
    python scripts/augment.py \
        --src dataset/original \
        --src-labels dataset/original_labels \
        --dst dataset/augmented \
        --n 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ORIG_DIR   = Path("dataset/original")
DEBUG_DIR  = Path("dataset/augmented/debug_gdino_orig")
IMG_SIZE   = 640
MODEL_ID   = "IDEA-Research/grounding-dino-tiny"

PROMPTS = {
    0: "small white mouse. small brown mouse. lab rodent. small furry animal sitting.",
    1: (
        "transparent water bottle. plastic water jug. sipper tube on cage wall. "
        "drinking bottle. water dispenser."
    ),
    2: (
        "brown food pellets in tray. rodent food dish. food hopper. "
        "grain pile in container. pellet dispenser."
    ),
}

DEFAULT_THRESHOLDS = {0: 0.22, 1: 0.30, 2: 0.28}
NMS_IOU      = 0.40
CLASS_COLORS = {0: (0, 255, 80), 1: (255, 180, 0), 2: (0, 140, 255)}
CLASS_NAMES  = {0: "mouse", 1: "water", 2: "food"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_ID} ...")
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
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0.0


def xyxy_to_yolo_str(x1, y1, x2, y2, img_w, img_h):
    cx = max(0.0, min(1.0, ((x1+x2)/2)/img_w))
    cy = max(0.0, min(1.0, ((y1+y2)/2)/img_h))
    bw = max(0.0, min(1.0, (x2-x1)/img_w))
    bh = max(0.0, min(1.0, (y2-y1)/img_h))
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_debug(pil_img, all_dets, img_path):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for cls_id, dets in all_dets.items():
        color = CLASS_COLORS[cls_id]
        label = CLASS_NAMES[cls_id]
        for (x1,y1,x2,y2), score in dets:
            cv2.rectangle(img_cv,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            cv2.putText(img_cv, f"{label} {score:.2f}",
                        (int(x1), max(int(y1)-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(DEBUG_DIR / img_path.name), img_cv)


def _letterbox_pil(pil_img, size):
    w, h    = pil_img.size
    scale   = size / max(w, h)
    nw, nh  = int(w*scale), int(h*scale)
    resized = pil_img.resize((nw, nh), Image.BILINEAR)
    canvas  = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size-nw)//2, (size-nh)//2))
    return canvas


def main(label_out: Path, thresholds: dict):
    orig_paths = sorted(
        p for p in ORIG_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if ORIG_DIR.exists() else []

    if not orig_paths:
        print(f"[ERROR] No images found in {ORIG_DIR}")
        sys.exit(1)

    label_out.mkdir(parents=True, exist_ok=True)
    (label_out / "classes.txt").write_text("mouse\nwater_container\nfood_area\n")

    print(f"Labelling {len(orig_paths)} original image(s)")
    print(f"Saving labels → {label_out}")
    print(f"Thresholds    → mouse={thresholds[0]}, water={thresholds[1]}, food={thresholds[2]}\n")

    processor, model = load_model()
    still_empty = []

    for idx, img_path in enumerate(orig_paths, 1):
        try:
            pil_img      = Image.open(img_path).convert("RGB")
            pil_img      = _letterbox_pil(pil_img, IMG_SIZE)
            img_w, img_h = pil_img.size

            all_dets: dict[int, list] = {}
            label_lines = []

            for cls_id in [0, 1, 2]:
                raw     = run_gdino(processor, model, pil_img, cls_id, thresholds[cls_id])
                cleaned = nms(raw, NMS_IOU)
                all_dets[cls_id] = cleaned
                for (x1,y1,x2,y2), _ in cleaned:
                    coords = xyxy_to_yolo_str(x1,y1,x2,y2, img_w, img_h)
                    label_lines.append(f"{cls_id} {coords}")

            # Save using original stem — augment.py matches by stem name
            (label_out / (img_path.stem + ".txt")).write_text("\n".join(label_lines))

            counts = {c: len(d) for c,d in all_dets.items()}
            if sum(counts.values()) == 0:
                still_empty.append(img_path.name)

            print(f"  [{idx:3d}/{len(orig_paths)}] {img_path.name:<50} "
                  f"mouse={counts[0]} water={counts[1]} food={counts[2]}")

            save_debug(pil_img, all_dets, img_path)

        except Exception as e:
            print(f"  [{idx:3d}/{len(orig_paths)}] {img_path.name:<50} ERROR: {e}")

    print(f"""
{'─'*60}
Done
  Originals labelled : {len(orig_paths)}
  Still empty        : {len(still_empty)}
  Labels saved to    : {label_out}
  Debug images       : {DEBUG_DIR}
{'─'*60}

If debug images look good, run:

  python scripts/augment.py \\
      --src dataset/original \\
      --src-labels {label_out} \\
      --dst dataset/augmented \\
      --n 50
""")

    if still_empty:
        print("Images with no detections:")
        for name in still_empty:
            print(f"  {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-out",        type=Path, default=Path("dataset/original_labels"),
                    help="Output folder for original labels (default: dataset/original_labels/)")
    ap.add_argument("--mouse-thresh",     type=float, default=DEFAULT_THRESHOLDS[0])
    ap.add_argument("--container-thresh", type=float, default=DEFAULT_THRESHOLDS[1])
    ap.add_argument("--food-thresh",      type=float, default=DEFAULT_THRESHOLDS[2])
    args = ap.parse_args()

    main(
        label_out=Path(args.label_out),
        thresholds={0: args.mouse_thresh, 1: args.container_thresh, 2: args.food_thresh},
    )