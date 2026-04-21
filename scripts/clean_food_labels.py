"""
scripts/clean_food_labels.py
=============================
Removes food_area (class 2) boxes that are too large —
these are usually the full-cage bedding detections that cause
the model to draw a giant food box over the entire image.

Rule:
    Remove class 2 boxes where box AREA (bw * bh) > MAX_FOOD_AREA
    or box WIDTH  (bw)  > MAX_FOOD_W
    or box HEIGHT (bh)  > MAX_FOOD_H

Defaults are tuned for a standard vivarium hopper:
    The hopper is roughly 160x220px in 640x640 space
    → normalised: w ≈ 0.25, h ≈ 0.34, area ≈ 0.085

Anything significantly larger is likely bedding / full-cage noise.

Usage:
    # Preview only
    python scripts/clean_food_labels.py --dry-run

    # Apply
    python scripts/clean_food_labels.py

    # Tune thresholds if needed
    python scripts/clean_food_labels.py --max-area 0.10 --max-w 0.35 --max-h 0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LABEL_DIR = Path("dataset/augmented/labels")

# ── Thresholds (normalised 0-1) ───────────────────────────────────────────────
# A legitimate hopper box is ~0.25w × 0.34h = area ~0.085
# Give some headroom → cap at 0.12 area, 0.40w, 0.50h
MAX_FOOD_AREA = 0.12   # bw * bh
MAX_FOOD_W    = 0.40   # bw alone
MAX_FOOD_H    = 0.50   # bh alone


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
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines))


def is_oversized_food(cls: int, bw: float, bh: float,
                      max_area: float, max_w: float, max_h: float) -> bool:
    if cls != 2:
        return False
    area = bw * bh
    return area > max_area or bw > max_w or bh > max_h


def clean_file(labels: list[tuple], max_area: float, max_w: float, max_h: float
               ) -> tuple[list[tuple], int]:
    kept    = []
    removed = 0
    for box in labels:
        cls, cx, cy, bw, bh = box
        if is_oversized_food(cls, bw, bh, max_area, max_w, max_h):
            removed += 1
        else:
            kept.append(box)
    return kept, removed


def main(labels_dir: Path, max_area: float, max_w: float, max_h: float, dry_run: bool) -> None:
    label_files = sorted(f for f in labels_dir.glob("*.txt") if f.name != "classes.txt")

    if not label_files:
        print(f"[ERROR] No label files found in: {labels_dir}")
        return

    total_files   = len(label_files)
    changed_files = 0
    total_removed = 0

    print(f"{'DRY RUN — ' if dry_run else ''}Scanning {total_files} label files …")
    print(f"Removing class 2 (food_area) boxes where:")
    print(f"  area > {max_area}  OR  width > {max_w}  OR  height > {max_h}\n")

    for lf in label_files:
        labels = read_labels(lf)
        if not labels:
            continue

        cleaned, removed = clean_file(labels, max_area, max_w, max_h)

        if removed > 0:
            changed_files += 1
            total_removed += removed

            # Show what was removed for inspection
            oversized = [b for b in labels if is_oversized_food(b[0], b[3], b[4], max_area, max_w, max_h)]
            details = "  ".join(
                f"[bw={b[3]:.2f} bh={b[4]:.2f} area={b[3]*b[4]:.3f}]"
                for b in oversized
            )
            action = "would remove" if dry_run else "removed"
            print(f"  {action} {removed}x food  {lf.name:<55}  {details}")

            if not dry_run:
                write_labels(lf, cleaned)

    print(f"""
{'─'*65}
{'DRY RUN — no files written' if dry_run else 'Clean complete'}
  Files scanned        : {total_files}
  Files changed        : {changed_files}
  Food boxes removed   : {total_removed}
{'─'*65}

Thresholds used:
  max area : {max_area}  (bw × bh)
  max width: {max_w}
  max height:{max_h}

If too many/few boxes were caught, tune with:
  --max-area  --max-w  --max-h
""")

    if dry_run:
        print("Re-run without --dry-run to apply.")
    else:
        print("✅ Done. Re-run dedup_labels.py then split + train.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",   type=Path,  default=DEFAULT_LABEL_DIR)
    ap.add_argument("--max-area", type=float, default=MAX_FOOD_AREA)
    ap.add_argument("--max-w",    type=float, default=MAX_FOOD_W)
    ap.add_argument("--max-h",    type=float, default=MAX_FOOD_H)
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()
    main(args.labels, args.max_area, args.max_w, args.max_h, args.dry_run)