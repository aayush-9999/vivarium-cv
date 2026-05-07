"""
migration/convert_to_coco.py
============================
One-time conversion utility: YOLO .txt label format → COCO JSON format.

Required because YOLOx natively expects COCO JSON annotations, while the
existing vivarium pipeline uses YOLO .txt format (class cx cy bw bh, normalised).

This script is intentionally placed in migration/ rather than scripts/ so it
is NOT part of the normal pipeline. Run it once before switching to YOLOx,
then keep the originals in dataset/split/ untouched as a fallback.

Output structure
─────────────────
    dataset/
      coco/
        train.json       ← COCO annotations for train split
        val.json         ← COCO annotations for val split
        images/          ← symlinks (or copies) of original images
          train/
          val/

COCO JSON format (what YOLOx expects)
───────────────────────────────────────
{
  "images":      [{"id": int, "file_name": str, "width": int, "height": int}],
  "annotations": [{"id": int, "image_id": int, "category_id": int,
                   "bbox": [x, y, w, h],   ← top-left x,y in PIXELS (not normalised)
                   "area": float, "iscrowd": 0}],
  "categories":  [{"id": int, "name": str, "supercategory": str}]
}

Note: COCO bbox format is [x_topleft, y_topleft, width, height] in absolute pixels,
whereas YOLO format is [cx, cy, bw, bh] normalised 0-1. The conversion below handles
this precisely.

IMPORTANT: category_id in COCO is 1-indexed by convention.
           YOLO class IDs are 0-indexed.
           This script converts: coco_category_id = yolo_class_id + 1

Usage
──────
    # Default: converts dataset/split/ → dataset/coco/
    python migration/convert_to_coco.py

    # Custom paths
    python migration/convert_to_coco.py \
        --split-dir  dataset/split \
        --output-dir dataset/coco \
        --copy-images          # copies images instead of creating symlinks

    # Dry run (prints stats, writes nothing)
    python migration/convert_to_coco.py --dry-run

Rollback
─────────
    Original YOLO labels in dataset/split/ are never touched.
    To revert to YOLOv8: set BACKEND=yolo in .env, done.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2

# ── 9-class scheme (must match core/config.py) ───────────────────────────────
YOLO_CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
]

# COCO categories: id is 1-indexed (YOLO class_id + 1)
COCO_CATEGORIES = [
    {"id": i, "name": name, "supercategory": "vivarium"}
    for i, name in enumerate(YOLO_CLASS_NAMES)
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_image_size(img_path: Path) -> tuple[int, int]:
    """Returns (width, height). Uses OpenCV for consistency with the pipeline."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return w, h


def _yolo_to_coco_bbox(
    cx_n: float, cy_n: float, bw_n: float, bh_n: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """
    Convert YOLO normalised [cx, cy, bw, bh] → COCO absolute [x, y, w, h].
    x, y are the TOP-LEFT corner in pixels.
    """
    bw_px = bw_n * img_w
    bh_px = bh_n * img_h
    x_tl  = (cx_n * img_w) - (bw_px / 2)
    y_tl  = (cy_n * img_h) - (bh_px / 2)
    # Clip to image bounds (guards against tiny float rounding errors)
    x_tl  = max(0.0, x_tl)
    y_tl  = max(0.0, y_tl)
    bw_px = min(bw_px, img_w - x_tl)
    bh_px = min(bh_px, img_h - y_tl)
    return round(x_tl, 2), round(y_tl, 2), round(bw_px, 2), round(bh_px, 2)


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Returns list of (class_id, cx, cy, bw, bh) from a YOLO .txt label file."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            out.append((
                int(parts[0]),
                float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            ))
        except ValueError:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-split conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_split(
    split_name:   str,
    img_dir:      Path,
    label_dir:    Path,
    out_json:     Path,
    out_img_dir:  Path,
    copy_images:  bool = False,
    dry_run:      bool = False,
) -> dict:
    """
    Convert one split (train or val) to a COCO JSON file.

    Returns stats dict: {images, annotations, skipped_no_label, skipped_unreadable}
    """
    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )

    if not img_paths:
        print(f"  [WARN] No images found in {img_dir}")
        return {}

    images_list:      list[dict] = []
    annotations_list: list[dict] = []

    img_id  = 0
    ann_id  = 0
    skipped_no_label    = 0
    skipped_unreadable  = 0

    print(f"\n  Processing {split_name} split: {len(img_paths)} images")
    print(f"  Labels   : {label_dir}")
    print(f"  Output   : {out_json}")

    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        labels     = _read_yolo_labels(label_path)

        if not labels:
            skipped_no_label += 1
            # Still include image in COCO (empty annotation list is valid)
            # but warn so the user knows

        # Read image dimensions
        try:
            img_w, img_h = _read_image_size(img_path)
        except ValueError:
            skipped_unreadable += 1
            print(f"    [SKIP] Unreadable: {img_path.name}")
            continue

        img_id += 1
        images_list.append({
            "id":        img_id,
            "file_name": img_path.name,
            "width":     img_w,
            "height":    img_h,
        })

        # Convert each YOLO box to COCO annotation
        for cls_id, cx_n, cy_n, bw_n, bh_n in labels:
            x, y, w, h = _yolo_to_coco_bbox(cx_n, cy_n, bw_n, bh_n, img_w, img_h)
            if w <= 0 or h <= 0:
                continue   # degenerate box — skip
            ann_id += 1
            annotations_list.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": cls_id,   # COCO is 1-indexed
                "bbox":        [x, y, w, h],
                "area":        round(w * h, 2),
                "iscrowd":     0,
            })

        # Handle image copying / symlinking
        if not dry_run:
            out_img_dir.mkdir(parents=True, exist_ok=True)
            dest = out_img_dir / img_path.name
            if not dest.exists():
                if copy_images:
                    shutil.copy2(img_path, dest)
                else:
                    # Symlink saves disk space — YOLOx can read through symlinks
                    try:
                        dest.symlink_to(img_path.resolve())
                    except OSError:
                        # Windows may not support symlinks without elevated permissions
                        shutil.copy2(img_path, dest)

    # Assemble COCO JSON
    coco_dict = {
        "images":      images_list,
        "annotations": annotations_list,
        "categories":  COCO_CATEGORIES,
    }

    stats = {
        "split":               split_name,
        "images":              len(images_list),
        "annotations":         len(annotations_list),
        "skipped_no_label":    skipped_no_label,
        "skipped_unreadable":  skipped_unreadable,
    }

    if not dry_run:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(coco_dict, indent=2), encoding="utf-8")
        print(f"  Written  : {out_json}  "
              f"({len(images_list)} images, {len(annotations_list)} annotations)")
    else:
        print(f"  [DRY RUN] Would write {out_json}  "
              f"({len(images_list)} images, {len(annotations_list)} annotations)")

    if skipped_no_label > 0:
        print(f"  [INFO] {skipped_no_label} images had no label file "
              f"(included in JSON with zero annotations)")
    if skipped_unreadable > 0:
        print(f"  [WARN] {skipped_unreadable} images were unreadable and skipped entirely")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_coco_json(json_path: Path) -> list[str]:
    """
    Basic sanity checks on the produced COCO JSON.
    Returns list of issue strings (empty = clean).
    """
    issues: list[str] = []
    data = json.loads(json_path.read_text(encoding="utf-8"))

    img_ids  = {img["id"] for img in data.get("images", [])}
    cat_ids  = {cat["id"] for cat in data.get("categories", [])}
    ann_list = data.get("annotations", [])

    for ann in ann_list:
        if ann["image_id"] not in img_ids:
            issues.append(f"Ann {ann['id']}: image_id {ann['image_id']} not in images")
        if ann["category_id"] not in cat_ids:
            issues.append(f"Ann {ann['id']}: category_id {ann['category_id']} not in categories")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            issues.append(f"Ann {ann['id']}: degenerate bbox {ann['bbox']}")
        if x < 0 or y < 0:
            issues.append(f"Ann {ann['id']}: negative bbox origin {ann['bbox']}")

    # Check category IDs are exactly 1..9
    expected_ids = set(range(0, 9))
    if cat_ids != expected_ids:
        issues.append(f"Category IDs mismatch: got {sorted(cat_ids)}, expected {sorted(expected_ids)}")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    split_dir:   Path,
    output_dir:  Path,
    copy_images: bool = False,
    dry_run:     bool = False,
    verify:      bool = True,
) -> None:
    print("=" * 60)
    print("YOLO → COCO JSON conversion (for YOLOx)")
    print("=" * 60)
    print(f"Source splits : {split_dir}")
    print(f"Output        : {output_dir}")
    print(f"Copy images   : {copy_images} ({'copies' if copy_images else 'symlinks'})")
    print(f"Dry run       : {dry_run}")
    print()

    all_stats = []

    for split_name in ["train", "val"]:
        img_dir   = split_dir / split_name / "images"
        label_dir = split_dir / split_name / "labels"

        if not img_dir.exists():
            print(f"  [SKIP] {split_name}: images dir not found at {img_dir}")
            continue

        if not label_dir.exists():
            print(f"  [WARN] {split_name}: labels dir not found at {label_dir}")

        out_json    = output_dir / f"{split_name}.json"
        out_img_dir = output_dir / "images" / split_name

        stats = convert_split(
            split_name   = split_name,
            img_dir      = img_dir,
            label_dir    = label_dir,
            out_json     = out_json,
            out_img_dir  = out_img_dir,
            copy_images  = copy_images,
            dry_run      = dry_run,
        )
        if stats:
            all_stats.append(stats)

        # Verify the written JSON
        if verify and not dry_run and out_json.exists():
            issues = verify_coco_json(out_json)
            if issues:
                print(f"\n  [VERIFY FAILED] {len(issues)} issue(s) in {out_json.name}:")
                for iss in issues:
                    print(f"    {iss}")
            else:
                print(f"  [VERIFY OK] {out_json.name} passed all checks")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Conversion summary")
    print(f"{'─' * 60}")
    for s in all_stats:
        print(
            f"  {s['split']:<6}  images={s['images']}  "
            f"annotations={s['annotations']}  "
            f"no_label={s['skipped_no_label']}  "
            f"unreadable={s['skipped_unreadable']}"
        )

    if not dry_run:
        print(f"\nNext steps:")
        print(f"  1. Update exps/vivarium_yolox_tiny.py:")
        print(f"       self.data_dir  = '{output_dir}'")
        print(f"       self.train_ann = 'train.json'")
        print(f"       self.val_ann   = 'val.json'")
        print(f"  2. Run: python -m yolox.tools.train -f exps/vivarium_yolox_tiny.py -d 1 -b 16")
        print(f"\nOriginal YOLO labels remain untouched in {split_dir}")
        print(f"To revert to YOLOv8: set BACKEND=yolo in .env")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert YOLO .txt labels to COCO JSON for YOLOx.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migration/convert_to_coco.py
  python migration/convert_to_coco.py --dry-run
  python migration/convert_to_coco.py --split-dir dataset/split --output-dir dataset/coco
  python migration/convert_to_coco.py --copy-images   # full copy instead of symlinks
        """,
    )
    ap.add_argument(
        "--split-dir",   type=Path, default=Path("dataset/split"),
        help="Root of the existing YOLO train/val split (default: dataset/split)",
    )
    ap.add_argument(
        "--output-dir",  type=Path, default=Path("dataset/coco"),
        help="Where to write COCO JSON files and image links (default: dataset/coco)",
    )
    ap.add_argument(
        "--copy-images", action="store_true",
        help="Copy images instead of symlinking (use on Windows or if you need portability)",
    )
    ap.add_argument(
        "--dry-run",     action="store_true",
        help="Print stats without writing any files",
    )
    ap.add_argument(
        "--no-verify",   action="store_true",
        help="Skip post-write JSON verification",
    )
    args = ap.parse_args()

    main(
        split_dir   = args.split_dir,
        output_dir  = args.output_dir,
        copy_images = args.copy_images,
        dry_run     = args.dry_run,
        verify      = not args.no_verify,
    )