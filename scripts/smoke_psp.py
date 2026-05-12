# scripts/smoke_psp.py
"""
Smoke-test inference on a single image using the unified InferencePipeline.

Supports three modes:

  1. YOLOX only (default, no PSPNet weights needed)
     python scripts/smoke_psp.py --image cage_frame.jpg

  2. YOLOX + PSPNet hybrid
     python scripts/smoke_psp.py --image cage_frame.jpg --backend yolo_psp \\
         --water-weights runs/pspnet/water/best.pth \\
         --food-weights  runs/pspnet/food/best.pth

  3. PSPNet crop-only (test measurer in isolation)
     python scripts/smoke_psp.py --image water_crop.jpg \\
         --crop-only --container water --weights runs/pspnet/water/best.pth

13-class scheme
───────────────
    0  mouse
    1  water_critical   4  water_full
    2  water_low        5  food_critical
    3  water_ok         6  food_low
                        7  food_ok
                        8  food_full
    9  bedding_worst   12  bedding_perfect
    10 bedding_bad
    11 bedding_ok
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 13-class names for display
CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

STATUS_COLORS = {
    "OK":       (80,  200,  80),
    "LOW":      (0,   180, 255),
    "CRITICAL": (0,     0, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 + 2 — full pipeline (yolo or yolo_psp)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline(
    image_path:    str,
    backend:       str = "yolo",
    yolo_weights:  str | None = None,
    water_weights: str | None = None,
    food_weights:  str | None = None,
    out_dir:       str = "runs/smoke_test",
) -> None:
    """Run the full InferencePipeline on one image and save annotated output."""

    # Override env vars before importing pipeline so CONFIG picks them up
    os.environ["BACKEND"] = backend
    if yolo_weights:
        os.environ["YOLO_WEIGHTS"] = yolo_weights
    if water_weights:
        os.environ["PSP_WATER_WEIGHTS"] = water_weights
    if food_weights:
        os.environ["PSP_FOOD_WEIGHTS"] = food_weights

    from pipeline.pipeline import InferencePipeline
    from pipeline.annotator.factory import get_annotator

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    print(f"\n{'='*55}")
    print(f"  Smoke test — backend={backend}")
    print(f"  Image        : {image_path}")
    print(f"  YOLO weights : {yolo_weights or os.getenv('YOLO_WEIGHTS', 'from .env')}")
    if backend == "yolo_psp":
        print(f"  Water PSP    : {water_weights or os.getenv('PSP_WATER_WEIGHTS', 'NOT SET')}")
        print(f"  Food PSP     : {food_weights  or os.getenv('PSP_FOOD_WEIGHTS',  'NOT SET')}")
    print(f"{'='*55}\n")

    pipeline  = InferencePipeline(backend=backend)
    result    = pipeline.run(frame=img, cage_id="smoke_test")
    annotator = get_annotator()
    viz       = annotator.draw(img, result)

    # Print results
    print(f"{'─'*55}")
    print(f"  Mouse count  : {result.mouse_count}")
    print(f"  Water        : {result.water.pct:.1f}%  [{result.water.status}]")
    print(f"  Food         : {result.food.pct:.1f}%   [{result.food.status}]")
    print(f"  Inference    : {result.inference_ms} ms")
    if result.water_bbox:
        b = result.water_bbox
        print(f"  Water bbox   : ({b.x1:.0f},{b.y1:.0f}) → ({b.x2:.0f},{b.y2:.0f})  conf={b.conf:.2f}")
    if result.food_bbox:
        b = result.food_bbox
        print(f"  Food bbox    : ({b.x1:.0f},{b.y1:.0f}) → ({b.x2:.0f},{b.y2:.0f})  conf={b.conf:.2f}")
    print(f"{'─'*55}\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(image_path).stem}_{backend}_result.jpg"
    cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Annotated output → {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — PSPNet crop-only
# ─────────────────────────────────────────────────────────────────────────────

def test_crop_only(
    image_path: str,
    container:  str,
    weights:    str | None = None,
    out_dir:    str = "runs/smoke_test",
) -> None:
    """Test the PSPNet measurer in isolation on a pre-cropped container image."""
    from pipeline.measurers.pspnet_measurer import (
        LevelEstimator, overlay_mask, WATER_PALETTE, FOOD_PALETTE,
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    estimator = LevelEstimator(
        water_weights = weights if container == "water" else None,
        food_weights  = weights if container == "food"  else None,
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

    overlay  = overlay_mask(img, mask, palette, alpha=0.5)
    combined = np.hstack([img, overlay])

    label = f"{pct:.1f}% [{status}]"
    color = STATUS_COLORS.get(status, (160, 160, 160))
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
    cv2.putText(combined, "Original",    (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(combined, "PSPNet mask", (img.shape[1] + 10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(image_path).stem}_{container}_psp.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Output → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Smoke-test inference (13-class YOLOX + optional PSPNet).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLOX only (uses YOLO_WEIGHTS from .env)
  python scripts/smoke_psp.py --image dataset/source/images/image_1.jpg

  # YOLOX with explicit weights
  python scripts/smoke_psp.py --image dataset/source/images/image_1.jpg \\
      --yolo-weights YOLOX_outputs/vivarium_yolox_tiny/best_ckpt.pth

  # YOLOX + PSPNet
  python scripts/smoke_psp.py --image cage_frame.jpg --backend yolo_psp \\
      --water-weights runs/pspnet/water/best.pth \\
      --food-weights  runs/pspnet/food/best.pth

  # PSPNet measurer only (crop test)
  python scripts/smoke_psp.py --image water_crop.jpg \\
      --crop-only --container water --weights runs/pspnet/water/best.pth
        """,
    )

    ap.add_argument("--image",         required=True,  help="Input image path")
    ap.add_argument("--backend",       default="yolo", choices=["yolo", "yolo_psp", "ssd"],
                    help="Inference backend (default: yolo)")
    ap.add_argument("--yolo-weights",  default=None,
                    help="YOLOX checkpoint path (overrides YOLO_WEIGHTS in .env)")
    ap.add_argument("--water-weights", default=None,
                    help="PSPNet water checkpoint (yolo_psp backend only)")
    ap.add_argument("--food-weights",  default=None,
                    help="PSPNet food checkpoint (yolo_psp backend only)")
    ap.add_argument("--crop-only",     action="store_true",
                    help="Test PSPNet measurer alone on a pre-cropped image")
    ap.add_argument("--container",     choices=["water", "food"], default=None,
                    help="Container type for --crop-only mode")
    ap.add_argument("--weights",       default=None,
                    help="PSPNet weights for --crop-only mode")
    ap.add_argument("--out",           default="runs/smoke_test",
                    help="Output directory for annotated images")

    args = ap.parse_args()

    if args.crop_only:
        if not args.container:
            ap.error("--container is required with --crop-only")
        test_crop_only(
            image_path = args.image,
            container  = args.container,
            weights    = args.weights,
            out_dir    = args.out,
        )
    else:
        test_full_pipeline(
            image_path    = args.image,
            backend       = args.backend,
            yolo_weights  = args.yolo_weights,
            water_weights = args.water_weights,
            food_weights  = args.food_weights,
            out_dir       = args.out,
        )