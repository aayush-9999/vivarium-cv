"""
scripts/dedup_labels.py
=======================
Deduplicates YOLO label files according to vivarium class rules:

    Class 0 (mouse)           → keep ALL, but NMS dedup overlapping boxes
                                 (multiple mice per cage is normal)
    Class 1 (water_container) → keep ONLY the largest box (1 jug per cage)
    Class 2 (food_area)       → keep ALL, but NMS dedup overlapping boxes

Usage:
    # Preview what would change
    python scripts/dedup_labels.py --dry-run

    # Apply
    python scripts/dedup_labels.py

    # Tune NMS IoU thresholds
    python scripts/dedup_labels.py --mouse-iou 0.45 --food-iou 0.50
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LABEL_DIR = Path("dataset/augmented/labels")

MOUSE_NMS_IOU = 0.45
FOOD_NMS_IOU  = 0.50


def read_labels(path: Path) -> list[tuple]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                out.append((int(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return out


def write_labels(path: Path, labels: list[tuple]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
             for c, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines))


def box_area(bw: float, bh: float) -> float:
    return bw * bh


def iou_cxcywh(a: tuple, b: tuple) -> float:
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def keep_largest_one(boxes: list[tuple]) -> list[tuple]:
    """Keep only the single largest box. Used for water container only."""
    if not boxes:
        return []
    return [max(boxes, key=lambda b: box_area(b[3], b[4]))]


def nms_dedup(boxes: list[tuple], iou_thresh: float) -> list[tuple]:
    """
    Greedy NMS — removes near-duplicate overlapping boxes while
    keeping genuinely separate detections (different mice, different
    food areas). Sorts by area descending so largest box wins ties.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: box_area(b[3], b[4]), reverse=True)
    kept = []
    for candidate in sorted_boxes:
        if all(iou_cxcywh(candidate, k) < iou_thresh for k in kept):
            kept.append(candidate)
    return kept


def dedup_label_file(
    labels: list[tuple],
    mouse_iou: float,
    food_iou: float,
) -> tuple[list[tuple], dict]:
    mice  = [b for b in labels if b[0] == 0]
    water = [b for b in labels if b[0] == 1]
    food  = [b for b in labels if b[0] == 2]

    mice_clean  = nms_dedup(mice,  mouse_iou)
    water_clean = keep_largest_one(water)
    food_clean  = nms_dedup(food,  food_iou)

    cleaned = mice_clean + water_clean + food_clean

    stats = {
        "mouse_before": len(mice),  "mouse_after": len(mice_clean),
        "water_before": len(water), "water_after": len(water_clean),
        "food_before":  len(food),  "food_after":  len(food_clean),
    }
    return cleaned, stats


def main(labels_dir: Path, mouse_iou: float, food_iou: float, dry_run: bool) -> None:
    label_files = sorted(
        f for f in labels_dir.glob("*.txt")
        if f.name != "classes.txt"
    )

    if not label_files:
        print(f"[ERROR] No .txt label files found in: {labels_dir}")
        return

    total_files   = len(label_files)
    changed_files = 0
    total = {"mouse_removed": 0, "water_removed": 0, "food_removed": 0}

    print(f"{'DRY RUN — ' if dry_run else ''}Scanning {total_files} label files …")
    print(f"Rules:")
    print(f"  mouse → NMS dedup (iou={mouse_iou})  — multiple mice preserved")
    print(f"  water → keep largest 1               — 1 jug per cage")
    print(f"  food  → NMS dedup (iou={food_iou})   — multiple food areas preserved\n")

    for lf in label_files:
        labels = read_labels(lf)
        if not labels:
            continue

        cleaned, stats = dedup_label_file(labels, mouse_iou, food_iou)

        mouse_removed = stats["mouse_before"] - stats["mouse_after"]
        water_removed = stats["water_before"] - stats["water_after"]
        food_removed  = stats["food_before"]  - stats["food_after"]
        any_change    = (mouse_removed + water_removed + food_removed) > 0

        if any_change:
            changed_files += 1
            total["mouse_removed"] += mouse_removed
            total["water_removed"] += water_removed
            total["food_removed"]  += food_removed

            parts = []
            if mouse_removed:
                parts.append(f"mouse {stats['mouse_before']}→{stats['mouse_after']}")
            if water_removed:
                parts.append(f"water {stats['water_before']}→{stats['water_after']}")
            if food_removed:
                parts.append(f"food  {stats['food_before']}→{stats['food_after']}")

            action = "would fix" if dry_run else "fixed"
            print(f"  {action}: {lf.name:<60}  [{', '.join(parts)}]")

        if not dry_run and any_change:
            write_labels(lf, cleaned)

    print(f"""
{'─'*65}
{'DRY RUN — no files written' if dry_run else 'Dedup complete'}
  Files scanned        : {total_files}
  Files changed        : {changed_files}
  Mouse labels removed : {total['mouse_removed']}  (overlapping duplicates only)
  Water labels removed : {total['water_removed']}  (kept 1 per cage)
  Food  labels removed : {total['food_removed']}   (overlapping duplicates only)
{'─'*65}

Class rules:
  Class 0 (mouse)           → NMS iou={mouse_iou}  (multiple mice preserved)
  Class 1 (water_container) → keep largest 1       (1 per cage)
  Class 2 (food_area)       → NMS iou={food_iou}   (multiple areas preserved)
""")

    if dry_run:
        print("Re-run without --dry-run to apply changes.")
    else:
        print("Done. Run verify_labels.py to confirm integrity.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Dedup vivarium YOLO labels: NMS mice, 1 water, NMS food."
    )
    ap.add_argument("--labels",    type=Path,  default=DEFAULT_LABEL_DIR)
    ap.add_argument("--mouse-iou", type=float, default=MOUSE_NMS_IOU,
                    help=f"IoU for mouse NMS (default {MOUSE_NMS_IOU})")
    ap.add_argument("--food-iou",  type=float, default=FOOD_NMS_IOU,
                    help=f"IoU for food NMS (default {FOOD_NMS_IOU})")
    ap.add_argument("--dry-run",   action="store_true")
    args = ap.parse_args()
    main(args.labels, args.mouse_iou, args.food_iou, args.dry_run)