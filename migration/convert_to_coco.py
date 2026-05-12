"""
migration/convert_to_coco.py
============================
Converts YOLO .txt labels → COCO JSON format for YOLOx training.

Supports two input layouts:

  Layout A — flat (dataset/source/)   ← DEFAULT
  ─────────────────────────────────
    dataset/source/
        images/   ← all images here
        labels/   ← matching .txt files here
    Automatically splits into train/val by --train-ratio (default 0.85).

  Layout B — pre-split (dataset/split/)
  ──────────────────────────────────────
    dataset/split/
        train/images/  train/labels/
        val/images/    val/labels/
    Pass --pre-split to use this layout.

Output
──────
    dataset/coco/
        train.json
        val.json
        images/
            train2017/   ← images (symlinks or copies)
            val2017/

13-class scheme
───────────────
    0  mouse
    1  water_critical
    2  water_low
    3  water_ok
    4  water_full
    5  food_critical
    6  food_low
    7  food_ok
    8  food_full
    9  bedding_worst
    10 bedding_bad
    11 bedding_ok
    12 bedding_perfect

Usage
──────
    # Layout A (flat source) — most common
    python migration/convert_to_coco.py

    # Custom source path
    python migration/convert_to_coco.py \
        --source-dir dataset/source \
        --output-dir dataset/coco

    # Layout B (already split)
    python migration/convert_to_coco.py \
        --pre-split \
        --split-dir dataset/split \
        --output-dir dataset/coco

    # Copy images instead of symlinks (required on Windows without admin)
    python migration/convert_to_coco.py --copy-images

    # Dry run — print stats, write nothing
    python migration/convert_to_coco.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


# ── 13-class scheme (must match core/config.py) ──────────────────────────────
YOLO_CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

NUM_CLASSES = len(YOLO_CLASS_NAMES)   # 13

COCO_CATEGORIES = [
    {"id": i, "name": name, "supercategory": "vivarium"}
    for i, name in enumerate(YOLO_CLASS_NAMES)
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_image_size(img_path: Path) -> tuple[int, int]:
    """Returns (width, height)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return w, h


def _yolo_to_coco_bbox(
    cx_n: float, cy_n: float, bw_n: float, bh_n: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """YOLO normalised [cx,cy,bw,bh] → COCO absolute [x_tl, y_tl, w, h]."""
    bw_px = bw_n * img_w
    bh_px = bh_n * img_h
    x_tl  = max(0.0, (cx_n * img_w) - (bw_px / 2))
    y_tl  = max(0.0, (cy_n * img_h) - (bh_px / 2))
    bw_px = min(bw_px, img_w - x_tl)
    bh_px = min(bh_px, img_h - y_tl)
    return round(x_tl, 2), round(y_tl, 2), round(bw_px, 2), round(bh_px, 2)


def _read_yolo_labels(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO .txt label file into list of (cls, cx, cy, bw, bh)."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            if cls < 0 or cls >= NUM_CLASSES:
                print(f"    [WARN] Invalid class id {cls} in {label_path.name} — skipped")
                continue
            out.append((cls, float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return out


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Symlink src → dst, fall back to copy on failure (e.g. Windows)."""
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_split(
    split_name:  str,
    img_paths:   list[Path],
    label_dir:   Path,
    out_json:    Path,
    out_img_dir: Path,
    copy_images: bool = False,
    dry_run:     bool = False,
) -> dict:
    """
    Convert a list of image paths + their YOLO labels into one COCO JSON file.
    Returns stats dict.
    """
    images_list:      list[dict] = []
    annotations_list: list[dict] = []
    img_id             = 0
    ann_id             = 0
    skipped_no_label   = 0
    skipped_unreadable = 0

    print(f"\n  [{split_name}]  {len(img_paths)} images  →  {out_json}")

    for img_path in sorted(img_paths):
        label_path = label_dir / (img_path.stem + ".txt")
        labels     = _read_yolo_labels(label_path)

        if not labels:
            skipped_no_label += 1

        try:
            img_w, img_h = _read_image_size(img_path)
        except ValueError:
            skipped_unreadable += 1
            print(f"    [SKIP unreadable] {img_path.name}")
            continue

        img_id += 1
        images_list.append({
            "id":        img_id,
            "file_name": img_path.name,
            "width":     img_w,
            "height":    img_h,
        })

        for cls_id, cx_n, cy_n, bw_n, bh_n in labels:
            x, y, w, h = _yolo_to_coco_bbox(cx_n, cy_n, bw_n, bh_n, img_w, img_h)
            if w <= 0 or h <= 0:
                continue
            ann_id += 1
            annotations_list.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": cls_id,
                "bbox":        [x, y, w, h],
                "area":        round(w * h, 2),
                "iscrowd":     0,
            })

        if not dry_run:
            out_img_dir.mkdir(parents=True, exist_ok=True)
            _link_or_copy(img_path, out_img_dir / img_path.name, copy_images)

    coco_dict = {
        "images":      images_list,
        "annotations": annotations_list,
        "categories":  COCO_CATEGORIES,
    }

    if not dry_run:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(coco_dict, indent=2), encoding="utf-8")

    stats = {
        "split":              split_name,
        "images":             len(images_list),
        "annotations":        len(annotations_list),
        "skipped_no_label":   skipped_no_label,
        "skipped_unreadable": skipped_unreadable,
    }

    tag = "[DRY RUN] " if dry_run else ""
    print(f"    {tag}images={len(images_list)}  annotations={len(annotations_list)}"
          f"  no_label={skipped_no_label}  unreadable={skipped_unreadable}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_coco_json(json_path: Path) -> list[str]:
    """Basic sanity checks on a produced COCO JSON. Returns list of issues."""
    issues: list[str] = []
    data    = json.loads(json_path.read_text(encoding="utf-8"))
    img_ids = {img["id"] for img in data.get("images", [])}
    cat_ids = {cat["id"] for cat in data.get("categories", [])}

    for ann in data.get("annotations", []):
        if ann["image_id"] not in img_ids:
            issues.append(f"Ann {ann['id']}: unknown image_id {ann['image_id']}")
        if ann["category_id"] not in cat_ids:
            issues.append(f"Ann {ann['id']}: unknown category_id {ann['category_id']}")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            issues.append(f"Ann {ann['id']}: degenerate bbox {ann['bbox']}")
        if x < 0 or y < 0:
            issues.append(f"Ann {ann['id']}: negative bbox origin {ann['bbox']}")

    expected_ids = set(range(NUM_CLASSES))
    if cat_ids != expected_ids:
        issues.append(
            f"Category IDs mismatch: got {sorted(cat_ids)}, "
            f"expected {sorted(expected_ids)}"
        )
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_flat(
    source_dir: Path,
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], Path]:
    """
    Collect all images from source_dir/images/, shuffle, split into train/val.
    Returns (train_img_paths, val_img_paths, label_dir).
    """
    img_dir   = source_dir / "images"
    label_dir = source_dir / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {label_dir}")

    all_images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not all_images:
        raise ValueError(f"No images found in {img_dir}")

    random.seed(seed)
    random.shuffle(all_images)
    split_idx   = int(len(all_images) * train_ratio)
    train_imgs  = all_images[:split_idx]
    val_imgs    = all_images[split_idx:]

    # Ensure at least 1 in val
    if not val_imgs and train_imgs:
        val_imgs   = train_imgs[-1:]
        train_imgs = train_imgs[:-1]

    return train_imgs, val_imgs, label_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    source_dir:  Path,
    split_dir:   Path,
    output_dir:  Path,
    pre_split:   bool  = False,
    train_ratio: float = 0.85,
    seed:        int   = 42,
    copy_images: bool  = False,
    dry_run:     bool  = False,
    verify:      bool  = True,
) -> None:
    print("=" * 60)
    print(f"YOLO .txt → COCO JSON  ({NUM_CLASSES} classes)")
    print("=" * 60)
    print(f"  Mode        : {'pre-split' if pre_split else 'flat source'}")
    print(f"  Output      : {output_dir}")
    print(f"  Copy images : {copy_images}")
    print(f"  Dry run     : {dry_run}")
    print()

    all_stats = []

    if pre_split:
        # Layout B — train/ and val/ already exist
        for split_name in ["train", "val"]:
            img_dir   = split_dir / split_name / "images"
            label_dir = split_dir / split_name / "labels"

            if not img_dir.exists():
                print(f"  [SKIP] {split_name}: {img_dir} not found")
                continue

            img_paths = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            )
            stats = convert_split(
                split_name  = split_name,
                img_paths   = img_paths,
                label_dir   = label_dir,
                out_json    = output_dir / f"{split_name}.json",
                out_img_dir = output_dir / "images" / f"{split_name}2017",
                copy_images = copy_images,
                dry_run     = dry_run,
            )
            all_stats.append(stats)

    else:
        # Layout A — flat source, split here
        print(f"  Source      : {source_dir}")
        print(f"  Train ratio : {train_ratio}  seed={seed}")

        train_imgs, val_imgs, label_dir = _collect_flat(
            source_dir, train_ratio, seed
        )
        print(f"  Split       : {len(train_imgs)} train / {len(val_imgs)} val")

        for split_name, img_paths in [("train", train_imgs), ("val", val_imgs)]:
            stats = convert_split(
                split_name  = split_name,
                img_paths   = img_paths,
                label_dir   = label_dir,
                out_json    = output_dir / f"{split_name}.json",
                out_img_dir = output_dir / "images" / f"{split_name}2017",
                copy_images = copy_images,
                dry_run     = dry_run,
            )
            all_stats.append(stats)

    # ── Verify ────────────────────────────────────────────────────────────────
    if verify and not dry_run:
        print()
        for s in all_stats:
            json_path = output_dir / f"{s['split']}.json"
            if json_path.exists():
                issues = verify_coco_json(json_path)
                if issues:
                    print(f"  [VERIFY FAILED] {json_path.name} — {len(issues)} issue(s):")
                    for iss in issues:
                        print(f"    {iss}")
                else:
                    print(f"  [VERIFY OK] {json_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Summary")
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
        print(f"  1. Verify exps/vivarium_yolox_tiny.py has:")
        print(f"       self.num_classes = 13")
        print(f"       self.data_dir    = r'{output_dir.resolve()}'")
        print(f"       self.train_ann   = 'train.json'")
        print(f"       self.val_ann     = 'val.json'")
        print(f"  2. Train:")
        print(f"       python scripts/train.py -f exps/vivarium_yolox_tiny.py -d 1 -b 16")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert YOLO .txt labels to COCO JSON for YOLOx (13-class).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--source-dir", type=Path, default=Path("dataset/source"),
        help="Flat source dir containing images/ and labels/ (default: dataset/source)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("dataset/coco"),
        help="Output dir for COCO JSONs and image links (default: dataset/coco)",
    )
    ap.add_argument(
        "--pre-split", action="store_true",
        help="Use a pre-split dataset/split/ layout instead of flat source",
    )
    ap.add_argument(
        "--split-dir", type=Path, default=Path("dataset/split"),
        help="Pre-split root dir (only used with --pre-split, default: dataset/split)",
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.85,
        help="Train/val split ratio for flat source mode (default: 0.85)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split (default: 42)",
    )
    ap.add_argument(
        "--copy-images", action="store_true",
        help="Copy images instead of symlinking (use on Windows without admin rights)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing any files",
    )
    ap.add_argument(
        "--no-verify", action="store_true",
        help="Skip post-write JSON verification",
    )

    args = ap.parse_args()

    main(
        source_dir  = args.source_dir,
        split_dir   = args.split_dir,
        output_dir  = args.output_dir,
        pre_split   = args.pre_split,
        train_ratio = args.train_ratio,
        seed        = args.seed,
        copy_images = args.copy_images,
        dry_run     = args.dry_run,
        verify      = not args.no_verify,
    )