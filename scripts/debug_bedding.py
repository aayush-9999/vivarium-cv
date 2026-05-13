# scripts/debug_bedding.py
"""
Raw YOLOX detection dump — shows every detection the model fires,
including low-confidence ones, so you can see if bedding classes
9-12 are being output at all.

Usage:
    python scripts/debug_bedding.py --image path/to/frame.jpg
    python scripts/debug_bedding.py --image path/to/frame.jpg --conf 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok",  "water_full",
    "food_critical",  "food_low",  "food_ok",   "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

BEDDING_CLASS_IDS = {9, 10, 11, 12}

# BGR colours per group
COLOR_MOUSE   = (0, 220, 255)
COLOR_WATER   = (255, 180, 0)
COLOR_FOOD    = (0, 200, 80)
COLOR_BEDDING = (180, 0, 255)
COLOR_UNKNOWN = (160, 160, 160)

def cls_color(cls: int):
    if cls == 0:              return COLOR_MOUSE
    if 1 <= cls <= 4:         return COLOR_WATER
    if 5 <= cls <= 8:         return COLOR_FOOD
    if 9 <= cls <= 12:        return COLOR_BEDDING
    return COLOR_UNKNOWN


def run(image_path: str, conf_thresh: float, nms_thresh: float) -> None:
    from core.config_loader import (
        CONFIG, YOLOX_EXP_FILE, YOLOX_INPUT_SIZE,
    )
    from yolox.exp import get_exp
    from yolox.data.data_augment import ValTransform
    from yolox.utils import postprocess

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    weights = CONFIG["yolox"]["weights"]
    print(f"\n{'='*60}")
    print(f"  Image   : {image_path}")
    print(f"  Weights : {weights}")
    print(f"  conf_thresh={conf_thresh}  nms_thresh={nms_thresh}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────
    exp = get_exp(str(YOLOX_EXP_FILE), exp_name=None)
    exp.num_classes = 13
    model = exp.get_model()
    model.eval()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))

    # ── Preprocess ────────────────────────────────────────────────
    preproc = ValTransform(legacy=False)
    tensor, ratio = preproc(img, None, YOLOX_INPUT_SIZE)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()

    # ── Inference ─────────────────────────────────────────────────
    with torch.no_grad():
        raw = model(tensor)

    outputs = postprocess(
        raw,
        num_classes = 13,
        conf_thre   = conf_thresh,
        nms_thre    = nms_thresh,
    )

    if outputs[0] is None:
        print(f"  [!] NO detections at conf_thresh={conf_thresh}")
        print("      Try lowering --conf to 0.01 to see if the model fires at all.")
        return

    dets    = outputs[0].cpu().numpy()
    if isinstance(ratio, (tuple, list, np.ndarray)):
        ratio = float(np.array(ratio).flatten()[0])
    ratio = float(ratio) if ratio else 1.0

    boxes   = dets[:, :4] / ratio
    scores  = dets[:, 4] * dets[:, 5]
    classes = dets[:, 6].astype(int)

    # ── Print ALL detections ──────────────────────────────────────
    print(f"  Total detections: {len(dets)}\n")
    print(f"  {'cls':>3}  {'name':<18}  {'score':>6}  {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}")
    print(f"  {'─'*65}")

    bedding_found = False
    for box, score, cls in sorted(zip(boxes, scores, classes), key=lambda x: -x[1]):
        name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls_{cls}"
        flag  = "  ← BEDDING" if cls in BEDDING_CLASS_IDS else ""
        if cls in BEDDING_CLASS_IDS:
            bedding_found = True
        print(f"  {cls:>3}  {name:<18}  {score:>6.3f}  "
              f"{box[0]:>6.1f} {box[1]:>6.1f} {box[2]:>6.1f} {box[3]:>6.1f}{flag}")

    print()
    if not bedding_found:
        print("  [!] ZERO bedding detections (classes 9-12) at this threshold.")
        print("      Possible causes:")
        print("      1. Model not trained on bedding classes yet")
        print("      2. conf_thresh too high — try --conf 0.01")
        print("      3. num_classes mismatch — model head has wrong output size")
        print("      4. Bedding visible in frame but model hasn't learned it")
    else:
        print("  [OK] Bedding detections found — check condition mapping above.")

    # ── Annotated image with ALL detections ───────────────────────
    viz = img.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = cls_color(cls)
        name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls_{cls}"
        label = f"{name} {score:.2f}"
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        cv2.putText(viz, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(viz, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── Class distribution summary ────────────────────────────────
    from collections import Counter
    dist = Counter(int(c) for c in classes)
    print(f"\n  Class distribution:")
    for cls_id, count in sorted(dist.items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        bar  = "█" * count
        group = "BEDDING" if cls_id in BEDDING_CLASS_IDS else (
                "WATER"   if 1 <= cls_id <= 4 else (
                "FOOD"    if 5 <= cls_id <= 8 else "MOUSE"))
        print(f"    [{group:>7}]  cls {cls_id:>2}  {name:<18}  {bar} ({count})")

    out_dir  = Path("runs/debug_bedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = Path(image_path).stem
    out_path = out_dir / f"{stem}_raw_detections.jpg"
    cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Annotated output → {out_path}")

    # ── Also dump near-zero conf to catch weak bedding signals ────
    print(f"\n{'─'*60}")
    print("  Re-running at conf=0.01 to catch any weak bedding signals...")
    outputs_low = postprocess(raw, num_classes=13, conf_thre=0.01, nms_thre=nms_thresh)
    if outputs_low[0] is not None:
        dets_low    = outputs_low[0].cpu().numpy()
        classes_low = dets_low[:, 6].astype(int)
        scores_low  = dets_low[:, 4] * dets_low[:, 5]
        bedding_low = [
            (int(c), float(s))
            for c, s in zip(classes_low, scores_low)
            if int(c) in BEDDING_CLASS_IDS
        ]
        if bedding_low:
            print(f"  Weak bedding signals found at conf=0.01:")
            for cls_id, sc in sorted(bedding_low, key=lambda x: -x[1]):
                name = CLASS_NAMES[cls_id]
                print(f"    cls {cls_id} ({name})  score={sc:.4f}")
            print(f"\n  → Your CONF_THRESHOLD is too high for bedding.")
            print(f"    Lower YOLO_CONF_THRESHOLD in .env, or train more bedding samples.")
        else:
            print("  No bedding signals even at conf=0.01.")
            print("  → Model has not learned bedding classes at all.")
            print("    Check your training data has class IDs 9-12 labelled.")
    else:
        print("  No detections even at conf=0.01 — model output is empty entirely.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Debug bedding detection — dump raw YOLOX output")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold (default 0.25)")
    ap.add_argument("--nms",   type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    args = ap.parse_args()
    run(args.image, args.conf, args.nms)