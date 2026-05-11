# scripts/test_psp_vs_annotations.py
"""
Evaluate PSP pipeline water AND food level predictions against LabelMe JSON ground truth.

PSPNet receives the full 640x640 letterboxed frame — same as during training.
YOLOX is used only for mouse count and bbox overlay drawing.

Annotation label scheme expected:
    Water:  bottle_wall  + water_fill   + empty_air
    Food:   hopper_frame + food_pellets + empty_space

A JSON that has bottle_wall   → evaluated as water.
A JSON that has hopper_frame  → evaluated as food.
A JSON can have both          → evaluated for both.

Usage:
    # Reads weights from .env automatically
    python scripts/test_psp_vs_annotations.py

    # Override weights explicitly
    python scripts/test_psp_vs_annotations.py \
        --water-weights runs/pspnet/water/best.pth \
        --food-weights  runs/pspnet/food/best.pth

    # Evaluate only one container type
    python scripts/test_psp_vs_annotations.py --only water
    python scripts/test_psp_vs_annotations.py --only food

    # Custom dataset dir
    python scripts/test_psp_vs_annotations.py --orig-dir dataset/original_v2
"""

from __future__ import annotations

# ── MUST be first — before any pipeline/model imports ────────────────────────
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os

import cv2
import numpy as np

IMG_SIZE = 640
IMG_EXTS = {".jpg", ".jpeg", ".png"}

STATUS_THRESHOLDS = [
    (0.0,  15.0,  "CRITICAL"),
    (15.0, 35.0,  "LOW"),
    (35.0, 100.1, "OK"),
]

STATUS_COLORS = {
    "OK":       (80,  200,  80),
    "LOW":      (0,   180, 255),
    "CRITICAL": (0,     0, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct_to_status(pct: float) -> str:
    for lo, hi, s in STATUS_THRESHOLDS:
        if lo <= pct < hi:
            return s
    return "CRITICAL"


def polygon_area(points: list) -> float:
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bbox_pixels(points: list) -> tuple:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def scale_points(points: list, scale: float, pad_l: int, pad_t: int) -> list:
    return [[p[0] * scale + pad_l, p[1] * scale + pad_t] for p in points]


def letterbox_image(img: np.ndarray, size: int = 640):
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_t   = (size - nh) // 2
    pad_l   = (size - nw) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, pad_l, pad_t


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_water_gt(shapes_by_label: dict, json_name: str) -> dict | None:
    """
    Extract water GT from parsed LabelMe shapes.
    Requires: bottle_wall + water_fill + empty_air
    """
    if "bottle_wall" not in shapes_by_label:
        return None

    fill_area  = sum(polygon_area(p) for p in shapes_by_label.get("water_fill", []))
    empty_area = sum(polygon_area(p) for p in shapes_by_label.get("empty_air",  []))
    total      = fill_area + empty_area

    if total == 0:
        print(f"  [SKIP water] {json_name}: zero water_fill+empty_air area")
        return None

    fill_pct = fill_area / total * 100.0
    return {
        "fill_pct":     fill_pct,
        "status":       pct_to_status(fill_pct),
        "anchor_pts":   max(shapes_by_label["bottle_wall"], key=polygon_area),
        "anchor_label": "GT water",
    }


def extract_food_gt(shapes_by_label: dict, json_name: str) -> dict | None:
    """
    Extract food GT from parsed LabelMe shapes.
    Requires: hopper_frame + food_pellets + empty_space
    """
    if "hopper_frame" not in shapes_by_label:
        return None

    fill_area  = sum(polygon_area(p) for p in shapes_by_label.get("food_pellets", []))
    empty_area = sum(polygon_area(p) for p in shapes_by_label.get("empty_space",  []))
    total      = fill_area + empty_area

    if total == 0:
        print(f"  [SKIP food] {json_name}: zero food_pellets+empty_space area")
        return None

    fill_pct = fill_area / total * 100.0
    return {
        "fill_pct":     fill_pct,
        "status":       pct_to_status(fill_pct),
        "anchor_pts":   max(shapes_by_label["hopper_frame"], key=polygon_area),
        "anchor_label": "GT food",
    }


def parse_json(json_path: Path) -> tuple[dict | None, dict | None]:
    """
    Parse a LabelMe JSON and return (water_gt, food_gt).
    Either can be None if that container type isn't annotated.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Cannot parse {json_path.name}: {e}")
        return None, None

    shapes_by_label: dict[str, list] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        shapes_by_label.setdefault(label, []).append(shape.get("points", []))

    water_gt = extract_water_gt(shapes_by_label, json_path.name)
    food_gt  = extract_food_gt(shapes_by_label,  json_path.name)
    return water_gt, food_gt


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_gt_polygon(
    viz:    np.ndarray,
    gt:     dict,
    scale:  float,
    pad_l:  int,
    pad_t:  int,
) -> None:
    pts_lb = scale_points(gt["anchor_pts"], scale, pad_l, pad_t)
    poly   = np.array([[int(p[0]), int(p[1])] for p in pts_lb], dtype=np.int32)
    color  = STATUS_COLORS.get(gt["status"], (200, 200, 200))
    cv2.polylines(viz, [poly], isClosed=True, color=color, thickness=2)
    x1, y1, _, _ = polygon_bbox_pixels(pts_lb)
    label = f"{gt['anchor_label']} {gt['fill_pct']:.0f}% [{gt['status']}]"
    cv2.putText(viz, label, (max(0, x1), max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(viz, label, (max(0, x1), max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    pairs:         list[tuple[Path, Path]],
    water_weights: str | None,
    food_weights:  str | None,
    out_dir:       Path,
    only:          str | None,
) -> tuple[list[dict], list[dict]]:
    """
    Run full YOLOPSPPipeline on each image.
    Returns (water_results, food_results).
    """
    if water_weights:
        os.environ["PSP_WATER_WEIGHTS"] = water_weights
    if food_weights:
        os.environ["PSP_FOOD_WEIGHTS"] = food_weights
    os.environ["BACKEND"] = "yolo_psp"

    from pipeline.yolo_psp_pipeline import YOLOPSPPipeline, _draw_result

    pipeline = YOLOPSPPipeline(
        water_psp_weights=os.getenv("PSP_WATER_WEIGHTS"),
        food_psp_weights=os.getenv("PSP_FOOD_WEIGHTS"),
        psp_backbone=os.getenv("PSP_BACKBONE", "resnet50"),
    )

    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    water_results: list[dict] = []
    food_results:  list[dict] = []

    for img_path, json_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue

        img_lb, scale, pad_l, pad_t = letterbox_image(img, IMG_SIZE)
        water_gt, food_gt = parse_json(json_path)

        has_water = water_gt is not None and only in (None, "water")
        has_food  = food_gt  is not None and only in (None, "food")
        if not has_water and not has_food:
            continue

        try:
            result = pipeline.run(frame=img_lb, cage_id=img_path.stem)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        viz = _draw_result(img_lb, result)
        print(f"  {img_path.name}")

        # ── Water ──────────────────────────────────────────────────────────
        if has_water:
            pred_pct    = result.water.pct
            pred_status = result.water.status
            gt_pct      = water_gt["fill_pct"]
            gt_status   = water_gt["status"]
            error       = abs(pred_pct - gt_pct)
            match       = pred_status == gt_status

            water_results.append({
                "image":        img_path.name,
                "gt_pct":       round(gt_pct,   2),
                "pred_pct":     round(pred_pct, 2),
                "error":        round(error,    2),
                "gt_status":    gt_status,
                "pred_status":  pred_status,
                "status_match": match,
                "bbox_found":   result.water_bbox is not None,
                "inference_ms": result.inference_ms,
            })
            _draw_gt_polygon(viz, water_gt, scale, pad_l, pad_t)
            symbol = "✓" if match else "✗"
            print(
                f"    water {symbol}  gt={gt_pct:.1f}% [{gt_status:<8}]  "
                f"pred={pred_pct:.1f}% [{pred_status:<8}]  err={error:.1f}%  "
                f"bbox={'yes' if result.water_bbox else 'NO'}"
            )

        # ── Food ───────────────────────────────────────────────────────────
        if has_food:
            pred_pct    = result.food.pct
            pred_status = result.food.status
            gt_pct      = food_gt["fill_pct"]
            gt_status   = food_gt["status"]
            error       = abs(pred_pct - gt_pct)
            match       = pred_status == gt_status

            food_results.append({
                "image":        img_path.name,
                "gt_pct":       round(gt_pct,   2),
                "pred_pct":     round(pred_pct, 2),
                "error":        round(error,    2),
                "gt_status":    gt_status,
                "pred_status":  pred_status,
                "status_match": match,
                "bbox_found":   result.food_bbox is not None,
                "inference_ms": result.inference_ms,
            })
            _draw_gt_polygon(viz, food_gt, scale, pad_l, pad_t)
            symbol = "✓" if match else "✗"
            print(
                f"    food  {symbol}  gt={gt_pct:.1f}% [{gt_status:<8}]  "
                f"pred={pred_pct:.1f}% [{pred_status:<8}]  err={error:.1f}%  "
                f"bbox={'yes' if result.food_bbox else 'NO'}"
            )

        cv2.imwrite(str(viz_dir / img_path.name), viz, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return water_results, food_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], container: str, out_dir: Path) -> None:
    if not results:
        print(f"\n  [INFO] No {container} results to summarise.")
        return

    errors      = [r["error"] for r in results]
    status_hits = [r["status_match"] for r in results]
    mae         = float(np.mean(errors))
    rmse        = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    max_err     = float(np.max(errors))
    status_acc  = float(np.mean(status_hits)) * 100.0
    bbox_found  = sum(1 for r in results if r["bbox_found"])

    by_status: dict[str, list[float]] = {}
    for r in results:
        by_status.setdefault(r["gt_status"], []).append(r["error"])

    tag = container.upper()
    print(f"\n{'═'*65}")
    print(f"  PSP {tag} LEVEL — EVALUATION SUMMARY")
    print(f"{'═'*65}")
    print(f"  Images evaluated  : {len(results)}")
    print(f"  MAE               : {mae:.2f}%")
    print(f"  RMSE              : {rmse:.2f}%")
    print(f"  Max error         : {max_err:.2f}%")
    print(f"  Status accuracy   : {status_acc:.1f}%  ({sum(status_hits)}/{len(status_hits)})")
    print(f"  YOLOX bbox found  : {bbox_found}/{len(results)}")
    print(f"\n  Per-status MAE:")
    for status in ["CRITICAL", "LOW", "OK"]:
        errs = by_status.get(status, [])
        if errs:
            print(f"    {status:<8}: MAE={np.mean(errs):.2f}%  n={len(errs)}")
        else:
            print(f"    {status:<8}: n=0")

    worst = sorted(results, key=lambda r: r["error"], reverse=True)[:5]
    print(f"\n  Worst predictions:")
    for r in worst:
        print(f"    {r['image']:<40} gt={r['gt_pct']:.1f}%  "
              f"pred={r['pred_pct']:.1f}%  err={r['error']:.1f}%")
    print(f"{'═'*65}")

    csv_path = out_dir / f"results_{container}.csv"
    lines = ["image,gt_pct,pred_pct,error,gt_status,pred_status,status_match,bbox_found"]
    for r in results:
        lines.append(
            f"{r['image']},{r['gt_pct']},{r['pred_pct']},{r['error']},"
            f"{r['gt_status']},{r['pred_status']},{r['status_match']},{r['bbox_found']}"
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Results CSV → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    orig_dir:      Path,
    water_weights: str | None,
    food_weights:  str | None,
    out_dir:       Path,
    only:          str | None,
) -> None:
    print(f"\n  PSP_WATER_WEIGHTS = {os.getenv('PSP_WATER_WEIGHTS')}")
    print(f"  PSP_FOOD_WEIGHTS  = {os.getenv('PSP_FOOD_WEIGHTS')}")
    print(f"  BACKEND           = {os.getenv('BACKEND')}\n")

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(orig_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((img_path, json_path))

    if not pairs:
        print(f"[ERROR] No image+JSON pairs in {orig_dir}")
        sys.exit(1)

    # Quick scan to report annotation coverage
    n_water = n_food = 0
    for _, jp in pairs:
        try:
            data   = json.loads(jp.read_text(encoding="utf-8"))
            labels = {s.get("label", "") for s in data.get("shapes", [])}
            if "bottle_wall"  in labels: n_water += 1
            if "hopper_frame" in labels: n_food  += 1
        except Exception:
            pass

    print(f"\n{'─'*65}")
    print(f"PSP Level Evaluation vs LabelMe Annotations")
    print(f"{'─'*65}")
    print(f"  Source dir      : {orig_dir}")
    print(f"  Total pairs     : {len(pairs)}")
    print(f"  Water annotated : {n_water}  (have bottle_wall)")
    print(f"  Food annotated  : {n_food}   (have hopper_frame)")
    print(f"  Evaluating      : {only or 'both'}")
    print(f"  Water weights   : {water_weights or os.getenv('PSP_WATER_WEIGHTS') or 'NOT SET'}")
    print(f"  Food weights    : {food_weights  or os.getenv('PSP_FOOD_WEIGHTS')  or 'NOT SET'}")
    print(f"  Output          : {out_dir}")
    print(f"{'─'*65}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    water_results, food_results = run_eval(
        pairs=pairs,
        water_weights=water_weights,
        food_weights=food_weights,
        out_dir=out_dir,
        only=only,
    )

    if only in (None, "water"):
        print_summary(water_results, "water", out_dir)
    if only in (None, "food"):
        print_summary(food_results, "food", out_dir)

    print(f"\n  Visualizations → {out_dir / 'visualizations'}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_psp_vs_annotations.py
  python scripts/test_psp_vs_annotations.py --only water
  python scripts/test_psp_vs_annotations.py --only food
  python scripts/test_psp_vs_annotations.py \\
      --water-weights runs/pspnet/water/best.pth \\
      --food-weights  runs/pspnet/food/best.pth
        """,
    )
    ap.add_argument("--orig-dir",      type=Path, default=Path("dataset/original"))
    ap.add_argument("--water-weights", type=str,  default=None,
                    help="Override PSP_WATER_WEIGHTS from .env")
    ap.add_argument("--food-weights",  type=str,  default=None,
                    help="Override PSP_FOOD_WEIGHTS from .env")
    ap.add_argument("--out",           type=Path, default=Path("runs/psp_eval"))
    ap.add_argument("--only",          type=str,  default=None,
                    choices=["water", "food"],
                    help="Evaluate only one container type (default: both)")
    args = ap.parse_args()

    main(
        orig_dir=args.orig_dir,
        water_weights=args.water_weights,
        food_weights=args.food_weights,
        out_dir=args.out,
        only=args.only,
    )