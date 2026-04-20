"""
scripts/merge_labels.py
=======================
Merges two label folders without losing good annotations.

Use case:
    You have existing labels (some good, some empty/wrong).
    GDINO has produced a new set of labels.
    You want to MERGE them: keep existing non-empty labels,
    fill in empty ones from GDINO output.

Merge rules (per image):
    - If existing label has boxes → KEEP existing (don't overwrite human work)
    - If existing label is empty or missing → USE GDINO label
    - If both have boxes → MERGE (union), then apply NMS per class to deduplicate

Usage:
    python scripts/merge_labels.py \
        --existing  dataset/augmented/labels \
        --new       dataset/augmented/labels_gdino \
        --out       dataset/augmented/labels_merged

    # Then review and rename:
    # cp -r dataset/augmented/labels dataset/augmented/labels_backup
    # cp -r dataset/augmented/labels_merged/* dataset/augmented/labels/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


NMS_IOU = 0.45   # IoU threshold for deduplicating merged boxes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file → list of (class_id, cx, cy, w, h)."""
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            out.append((cls, cx, cy, bw, bh))
    return out


def write_labels(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines))


def iou_cxcywh(a, b) -> float:
    """IoU between two (cx, cy, w, h) boxes."""
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def nms_labels(
    labels: list[tuple[int, float, float, float, float]],
    iou_thresh: float,
) -> list[tuple[int, float, float, float, float]]:
    """Greedy NMS per class on YOLO-format labels (no confidence, so just dedup overlapping boxes)."""
    if not labels:
        return []

    by_class: dict[int, list] = {}
    for item in labels:
        by_class.setdefault(item[0], []).append(item)

    kept = []
    for cls_id, boxes in by_class.items():
        remaining = list(boxes)
        while remaining:
            chosen = remaining.pop(0)
            kept.append(chosen)
            remaining = [b for b in remaining if iou_cxcywh(chosen, b) < iou_thresh]

    return kept


def merge_pair(
    existing: list,
    new: list,
) -> tuple[list, str]:
    """
    Merge existing + new label lists.
    Returns (merged_labels, strategy_used).
    """
    if existing and not new:
        return existing, "kept_existing"
    if not existing and new:
        return new, "used_new"
    if not existing and not new:
        return [], "both_empty"

    # Both have boxes — merge and dedup
    merged = nms_labels(existing + new, NMS_IOU)
    return merged, "merged"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(existing_dir: Path, new_dir: Path, out_dir: Path, dry_run: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy classes.txt
    for src_dir in [existing_dir, new_dir]:
        classes_txt = src_dir / "classes.txt"
        if classes_txt.exists():
            shutil.copy(classes_txt, out_dir / "classes.txt")
            break

    # Collect all stems from both directories
    all_stems = set()
    for p in existing_dir.glob("*.txt"):
        if p.name != "classes.txt":
            all_stems.add(p.stem)
    for p in new_dir.glob("*.txt"):
        if p.name != "classes.txt":
            all_stems.add(p.stem)

    stats = {"kept_existing": 0, "used_new": 0, "merged": 0, "both_empty": 0}

    for stem in sorted(all_stems):
        existing_labels = read_labels(existing_dir / f"{stem}.txt")
        new_labels      = read_labels(new_dir      / f"{stem}.txt")

        merged, strategy = merge_pair(existing_labels, new_labels)
        stats[strategy] += 1

        if not dry_run:
            write_labels(out_dir / f"{stem}.txt", merged)

        e_count = len(existing_labels)
        n_count = len(new_labels)
        m_count = len(merged)
        print(f"  {stem[:50]:<50}  [{strategy}]  existing={e_count} new={n_count} → {m_count}")

    print(f"""
{'─'*60}
{'DRY RUN — nothing written' if dry_run else f'Merge complete → {out_dir}'}
  kept_existing : {stats['kept_existing']}  (existing had boxes, new didn't add anything)
  used_new      : {stats['used_new']}       (existing was empty, filled from GDINO)
  merged        : {stats['merged']}         (both had boxes — unioned + NMS)
  both_empty    : {stats['both_empty']}     (still no labels — needs manual work)
{'─'*60}

Next:
  1. Review merged labels:
       labelImg dataset/augmented/images {out_dir / 'classes.txt'}
  2. Copy to main labels dir when happy:
       cp -r {existing_dir} {existing_dir}_backup
       cp {out_dir}/*.txt {existing_dir}/
  3. Run: python scripts/split_dataset.py && python scripts/train.py
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", type=Path, default=Path("dataset/augmented/labels"),
                    help="Your current label folder")
    ap.add_argument("--new",      type=Path, default=Path("dataset/augmented/labels_gdino"),
                    help="GDINO output folder (set --dst in gdino_label.py to this path)")
    ap.add_argument("--out",      type=Path, default=Path("dataset/augmented/labels_merged"),
                    help="Output folder for merged labels")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()
    main(args.existing, args.new, args.out, args.dry_run)