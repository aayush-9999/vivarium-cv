"""
scripts/fix_labels.py
=====================
ONE-TIME FIX for existing label files where everything was written as class 0.

This script re-reads every .txt label in your labels folder and re-maps class IDs
based on bounding box position within the 640×640 frame:

  - If the box centre is in the RIGHT ~25% of the frame → class 1 (water_container)
    (because the jug ROI is at x=480–620 in the default config)
  - If the box centre is in the LEFT ~25% of the frame  → class 2 (food_area)
    (because the hopper ROI is at x=20–180 in the default config)
  - Everything else                                     → class 0 (mouse)

⚠️  This is a HEURISTIC fix. After running it, visually verify the labels
    with LabelImg before training. This is much faster than relabelling from scratch.

Usage:
    python scripts/fix_labels.py
    python scripts/fix_labels.py --labels dataset/augmented/labels --dry-run

Args:
    --labels   Path to label folder (default: dataset/augmented/labels)
    --dry-run  Print what would change without writing anything
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ── Zone definitions (must match core/config.py ROI_ZONES default) ────────────
# These are x-ranges (normalised 0-1) that define each zone in 640×640 space
WATER_X_MIN = 480 / 640   # ≈ 0.75
WATER_X_MAX = 620 / 640   # ≈ 0.97

FOOD_X_MIN  =  20 / 640   # ≈ 0.03
FOOD_X_MAX  = 180 / 640   # ≈ 0.28


def infer_class(cx: float, cy: float) -> int:
    """
    Infer vivarium class from normalised box centre.
    Priority: water > food > mouse (containers are at known positions).
    """
    if WATER_X_MIN <= cx <= WATER_X_MAX:
        return 1   # water_container
    if FOOD_X_MIN <= cx <= FOOD_X_MAX:
        return 2   # food_area
    return 0       # mouse


def fix_label_file(
    label_path: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Fix one label file. Returns (total_lines, changed_lines).
    """
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return 0, 0

    new_lines = []
    changed = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            new_lines.append(line)   # malformed — leave as-is
            continue

        orig_cls = int(parts[0])
        cx, cy   = float(parts[1]), float(parts[2])

        new_cls = infer_class(cx, cy)
        if new_cls != orig_cls:
            changed += 1

        new_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    if not dry_run and changed > 0:
        label_path.write_text("\n".join(new_lines))

    return len(lines), changed


def main(labels_dir: Path, dry_run: bool) -> None:
    label_files = sorted(labels_dir.glob("*.txt"))

    if not label_files:
        print(f"[ERROR] No .txt files found in {labels_dir}")
        return

    total_files   = 0
    total_changed = 0

    print(f"{'DRY RUN — ' if dry_run else ''}Processing {len(label_files)} label files …\n")

    for lf in label_files:
        n_lines, n_changed = fix_label_file(lf, dry_run)
        if n_changed > 0:
            action = "would fix" if dry_run else "fixed"
            print(f"  {action} {n_changed}/{n_lines} lines  →  {lf.name}")
            total_changed += n_changed
        total_files += 1

    print(f"""
{'─'*50}
{'DRY RUN summary' if dry_run else 'Fix complete'}
  Files scanned  : {total_files}
  Lines changed  : {total_changed}
{'  (no files written — remove --dry-run to apply)' if dry_run else f'  Labels updated : {labels_dir}'}
{'─'*50}

Class re-mapping used:
  cx in [{WATER_X_MIN:.2f}, {WATER_X_MAX:.2f}] → class 1 (water_container)
  cx in [{FOOD_X_MIN:.2f}, {FOOD_X_MAX:.2f}]  → class 2 (food_area)
  everything else                       → class 0 (mouse)

⚠️  Verify with LabelImg before training:
    pip install labelImg
    labelImg dataset/augmented/images {labels_dir / 'classes.txt'}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",  type=Path, default=Path("dataset/augmented/labels"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview changes without writing files")
    args = ap.parse_args()
    main(args.labels, args.dry_run)