# scripts/test_psp_inference.py
"""
Test PSPNet level estimation on a single image.

Usage:
    # Full pipeline test (YOLOX + PSPNet)
    python scripts/test_psp_inference.py --image cage_frame.jpg

    # Test PSPNet alone on a pre-cropped container image
    python scripts/test_psp_inference.py \
        --image water_bottle_crop.jpg \
        --container water \
        --crop-only

    # With trained weights
    python scripts/test_psp_inference.py \
        --image cage_frame.jpg \
        --water-weights runs/pspnet/water/best.pth \
        --food-weights  runs/pspnet/food/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_full_pipeline(
    image_path:     str,
    water_weights:  str = None,
    food_weights:   str = None,
    out_dir:        str = "runs/psp_test",
) -> None:
    """Test the full YOLOX + PSPNet hybrid pipeline on a cage frame."""
    import os
    if water_weights:
        os.environ["PSP_WATER_WEIGHTS"] = water_weights
    if food_weights:
        os.environ["PSP_FOOD_WEIGHTS"] = food_weights
    os.environ["BACKEND"] = "yolo_psp"

    from pipeline.yolo_psp_pipeline import YOLOPSPPipeline, _draw_result

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    print(f"\nImage      : {image_path}")
    print(f"Water PSP  : {water_weights or 'not loaded (using YOLOX fallback)'}")
    print(f"Food PSP   : {food_weights  or 'not loaded (using YOLOX fallback)'}\n")

    pipeline = YOLOPSPPipeline(
        water_psp_weights=water_weights,
        food_psp_weights=food_weights,
    )

    result = pipeline.run(frame=img, cage_id="test")

    print(f"{'─'*50}")
    print(f"  Mouse count : {result.mouse_count}")
    print(f"  Water       : {result.water.pct:.1f}%  [{result.water.status}]")
    print(f"  Food        : {result.food.pct:.1f}%  [{result.food.status}]")
    print(f"  Inference   : {result.inference_ms} ms")
    print(f"{'─'*50}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(image_path).stem}_psp_result.jpg"
    viz = _draw_result(img, result)
    cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Output → {out_path}\n")


def test_crop_only(
    image_path: str,
    container:  str,
    weights:    str = None,
    out_dir:    str = "runs/psp_test",
) -> None:
    """Test PSPNet alone on a pre-cropped container image."""
    from segmentation.models.level_estimator import (
        LevelEstimator, overlay_mask, WATER_PALETTE, FOOD_PALETTE
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    water_w = weights if container == "water" else None
    food_w  = weights if container == "food"  else None

    estimator = LevelEstimator(
        water_weights=water_w,
        food_weights=food_w,
    )

    if container == "water":
        pct, status, mask = estimator.estimate_water(img)
        palette = WATER_PALETTE
    else:
        pct, status, mask = estimator.estimate_food(img)
        palette = FOOD_PALETTE

    print(f"\n{'─'*50}")
    print(f"  Container : {container}")
    print(f"  Fill      : {pct:.1f}%")
    print(f"  Status    : {status}")
    print(f"{'─'*50}\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save side-by-side: original | mask overlay
    overlay = overlay_mask(img, mask, palette, alpha=0.5)
    combined = np.hstack([img, overlay])

    # Draw fill % on image
    label = f"{pct:.1f}% [{status}]"
    color = (80, 200, 80) if status == "OK" else (0, 180, 255) if status == "LOW" else (0, 0, 255)
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
    cv2.putText(combined, "Original", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(combined, "PSPNet mask", (img.shape[1] + 10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    out_path = Path(out_dir) / f"{Path(image_path).stem}_{container}_psp.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Output → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",         required=True, help="Input image path")
    ap.add_argument("--container",     choices=["water", "food"], default=None,
                    help="Container type for --crop-only mode")
    ap.add_argument("--crop-only",     action="store_true",
                    help="Test PSPNet alone on a pre-cropped container image")
    ap.add_argument("--water-weights", default=None)
    ap.add_argument("--food-weights",  default=None)
    ap.add_argument("--weights",       default=None,
                    help="Single weights path for --crop-only mode")
    ap.add_argument("--out",           default="runs/psp_test")
    args = ap.parse_args()

    if args.crop_only:
        if not args.container:
            ap.error("--container is required with --crop-only")
        test_crop_only(
            image_path=args.image,
            container=args.container,
            weights=args.weights,
            out_dir=args.out,
        )
    else:
        test_full_pipeline(
            image_path=args.image,
            water_weights=args.water_weights,
            food_weights=args.food_weights,
            out_dir=args.out,
        )