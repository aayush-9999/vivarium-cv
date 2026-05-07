"""
json_to_yolo_labels.py
========================
Generates correct YOLO bounding box labels from your existing LabelMe
JSON annotations (the ones you drew for PSPNet).

It takes the bounding box of the 'bottle_wall' polygon as the water
detection bbox, and uses the fill ratio (water_fill / (water_fill + empty_air))
to determine the correct water class (1-4).

Usage:
    python json_to_yolo_labels.py --dry-run   # preview only
    python json_to_yolo_labels.py             # apply and overwrite labels

Output:
    Overwrites dataset/original_labels_9class/*.txt water class lines
    with correct tight bboxes from your LabelMe annotations.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ORIG_DIR   = Path("dataset/original")
LABEL_DIR  = Path("dataset/original_labels_9class")
BACKUP_DIR = Path("dataset/original_labels_9class_backup_v2")
IMG_SIZE   = 640

WATER_CLASSES = {1, 2, 3, 4}

# Fill % → YOLO class
def fill_to_class(fill_pct: float) -> int:
    if fill_pct < 15:  return 1  # water_critical
    if fill_pct < 35:  return 2  # water_low
    if fill_pct < 80:  return 3  # water_ok
    return 4                      # water_full


def polygon_bbox(points, img_w, img_h):
    """Returns YOLO format cx, cy, w, h (normalized) from polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def polygon_area(points):
    """Pixel area of a polygon using shoelace formula."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def process_json(json_path: Path, dry_run: bool):
    # Find matching label file
    label_path = LABEL_DIR / (json_path.stem + ".txt")
    if not label_path.exists():
        print(f"  [SKIP] No label file: {label_path.name}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = data.get("imageWidth",  IMG_SIZE)
    img_h = data.get("imageHeight", IMG_SIZE)

    # Collect shapes by label
    shapes = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        shapes.setdefault(label, []).append(shape.get("points", []))

    # Need bottle_wall for bbox
    if "bottle_wall" not in shapes:
        print(f"  [SKIP] No bottle_wall in {json_path.name}")
        return

    # Use largest bottle_wall polygon for bbox
    wall_pts = max(shapes["bottle_wall"], key=lambda p: polygon_area(p))
    cx, cy, w, h = polygon_bbox(wall_pts, img_w, img_h)

    # Calculate fill %
    fill_area = sum(polygon_area(p) for p in shapes.get("water_fill", []))
    air_area  = sum(polygon_area(p) for p in shapes.get("empty_air",  []))
    total     = fill_area + air_area
    fill_pct  = (fill_area / total * 100) if total > 0 else 0.0
    water_cls = fill_to_class(fill_pct)

    # Read existing label, replace water lines
    orig_lines = label_path.read_text().strip().splitlines()
    new_lines  = []
    replaced   = False

    for line in orig_lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in WATER_CLASSES:
            if not replaced:
                new_lines.append(f"{water_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                replaced = True
            # Skip duplicate water lines
        else:
            new_lines.append(line)

    if not replaced:
        # No existing water line — add one
        new_lines.append(f"{water_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    old_water = [l for l in orig_lines if l.strip() and int(l.split()[0]) in WATER_CLASSES]
    print(f"  {'[DRY]' if dry_run else '[FIX]'} {json_path.stem:<20} "
          f"fill={fill_pct:.0f}%  class={water_cls}  "
          f"bbox=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})")
    if old_water:
        old = old_water[0].split()
        print(f"         OLD: w={float(old[3]):.3f} h={float(old[4]):.3f}  "
              f"→  NEW: w={w:.3f} h={h:.3f}")

    if not dry_run:
        label_path.write_text("\n".join(new_lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    json_files = sorted(ORIG_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files in {ORIG_DIR}")
        return

    print(f"Found {len(json_files)} LabelMe JSONs\n")

    if not args.dry_run:
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(LABEL_DIR, BACKUP_DIR)
        print(f"Backup → {BACKUP_DIR}\n")

    for jp in json_files:
        process_json(jp, args.dry_run)

    print(f"\n{'─'*60}")
    if args.dry_run:
        print("Dry run complete — no files changed.")
        print("Run without --dry-run to apply changes.")
    else:
        print("Labels updated from LabelMe annotations.")
        print("\nNext — re-augment and retrain YOLOX:")
        print("  python scripts/augment.py \\")
        print("      --src dataset/original \\")
        print("      --src-labels dataset/original_labels_9class \\")
        print("      --dst dataset/augmented_v2 --n 50")


if __name__ == "__main__":
    main()