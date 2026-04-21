# scripts/label_tools.py
"""
All label hygiene operations in one place.

Functions:
    verify(...)       — check label/image pairing and coordinate validity
    clean_food(...)   — remove oversized food_area boxes (bedding false positives)
    dedup(...)        — NMS dedup mice/food, keep 1 water box per cage
    fix_classes(...)  — heuristic remap of wrong class IDs by box position

Each function works standalone or via the orchestrator:

    from scripts.label_tools import verify, clean_food, dedup, fix_classes

    issues = verify()
    clean_food(dry_run=True)
    dedup()
    fix_classes(dry_run=True)

Or via orchestrator:
    orch.verify_labels()
    orch.clean_food_labels()
    orch.dedup_labels()
    orch.fix_labels()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_IMG_DIR   = Path("dataset/augmented/images")
DEFAULT_LABEL_DIR = Path("dataset/augmented/labels")

# clean_food defaults — hopper is ~0.25w × 0.34h = area ~0.085
MAX_FOOD_AREA = 0.12
MAX_FOOD_W    = 0.40
MAX_FOOD_H    = 0.50

# dedup defaults
MOUSE_NMS_IOU = 0.45
FOOD_NMS_IOU  = 0.50

# fix_classes zone boundaries (normalised, from default ROI_ZONES in config.py)
WATER_X_MIN = 480 / 640   # ≈ 0.75
WATER_X_MAX = 620 / 640   # ≈ 0.97
FOOD_X_MIN  =  20 / 640   # ≈ 0.03
FOOD_X_MAX  = 180 / 640   # ≈ 0.28


# ─────────────────────────────────────────────────────────────────────────────
# Shared I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> list[tuple]:
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


def _write(path: Path, labels: list[tuple]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
             for c, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines))


def _label_files(label_dir: Path) -> list[Path]:
    return sorted(f for f in label_dir.glob("*.txt") if f.name != "classes.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 1. VERIFY
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    img_dir:   Path = DEFAULT_IMG_DIR,
    label_dir: Path = DEFAULT_LABEL_DIR,
    verbose:   bool = True,
) -> list[str]:
    """
    Check label/image pairing and coordinate validity.

    Returns:
        List of issue strings. Empty list = all clean.
    """
    issues = []

    for label_file in _label_files(label_dir):
        img_file = img_dir / (label_file.stem + ".jpg")

        if not img_file.exists():
            issues.append(f"MISSING IMAGE : {label_file.name}")
            continue

        lines = label_file.read_text().strip().splitlines()

        if not lines:
            issues.append(f"EMPTY LABEL   : {label_file.name}")
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"BAD FORMAT    : {label_file.name} → '{line}'")
                continue
            _, cx, cy, w, h = parts
            if not (0 <= float(cx) <= 1 and 0 <= float(cy) <= 1):
                issues.append(f"OUT OF BOUNDS : {label_file.name}")

    if verbose:
        if issues:
            print(f"{len(issues)} issues found:")
            for i in issues:
                print(f"  {i}")
        else:
            total = len(_label_files(label_dir))
            print(f"✅ All {total} label files are clean.")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN FOOD LABELS
# ─────────────────────────────────────────────────────────────────────────────

def clean_food(
    label_dir: Path  = DEFAULT_LABEL_DIR,
    max_area:  float = MAX_FOOD_AREA,
    max_w:     float = MAX_FOOD_W,
    max_h:     float = MAX_FOOD_H,
    dry_run:   bool  = False,
) -> dict:
    """
    Remove food_area (class 2) boxes that are too large —
    these are usually full-cage bedding detections, not the hopper.

    Args:
        label_dir: Folder containing .txt label files.
        max_area:  Max normalised bw*bh (default 0.12).
        max_w:     Max normalised width  (default 0.40).
        max_h:     Max normalised height (default 0.50).
        dry_run:   Preview without writing.

    Returns:
        {"files_changed": N, "boxes_removed": M}
    """
    files_changed = 0
    boxes_removed = 0

    print(f"{'DRY RUN — ' if dry_run else ''}Cleaning oversized food labels …")
    print(f"  Removing class 2 where area>{max_area} OR w>{max_w} OR h>{max_h}\n")

    for lf in _label_files(label_dir):
        labels  = _read(lf)
        kept    = []
        removed = 0

        for (c, cx, cy, bw, bh) in labels:
            if c == 2 and (bw * bh > max_area or bw > max_w or bh > max_h):
                removed += 1
            else:
                kept.append((c, cx, cy, bw, bh))

        if removed:
            files_changed += 1
            boxes_removed += removed
            action = "would remove" if dry_run else "removed"
            print(f"  {action} {removed}× food  {lf.name}")
            if not dry_run:
                _write(lf, kept)

    print(f"\n{'─'*55}")
    print(f"{'DRY RUN — no files written' if dry_run else 'Clean complete'}")
    print(f"  Files changed  : {files_changed}")
    print(f"  Boxes removed  : {boxes_removed}")

    return {"files_changed": files_changed, "boxes_removed": boxes_removed}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEDUP LABELS
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
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


def _nms(boxes: list[tuple], iou_thresh: float) -> list[tuple]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[3] * b[4], reverse=True)
    kept = []
    for candidate in sorted_boxes:
        if all(_iou(candidate, k) < iou_thresh for k in kept):
            kept.append(candidate)
    return kept


def dedup(
    label_dir:  Path  = DEFAULT_LABEL_DIR,
    mouse_iou:  float = MOUSE_NMS_IOU,
    food_iou:   float = FOOD_NMS_IOU,
    dry_run:    bool  = False,
) -> dict:
    """
    Deduplicate labels per class rules:
        mouse (0) → NMS dedup  (multiple mice per cage is normal)
        water (1) → keep only the largest box (1 jug per cage)
        food  (2) → NMS dedup  (multiple food areas allowed)

    Args:
        label_dir: Folder containing .txt label files.
        mouse_iou: IoU threshold for mouse NMS.
        food_iou:  IoU threshold for food NMS.
        dry_run:   Preview without writing.

    Returns:
        {"files_changed": N, "total_removed": M}
    """
    files_changed = 0
    total_removed = 0

    print(f"{'DRY RUN — ' if dry_run else ''}Deduplicating labels …")
    print(f"  mouse → NMS iou={mouse_iou}  |  water → keep largest 1  |  food → NMS iou={food_iou}\n")

    for lf in _label_files(label_dir):
        labels = _read(lf)
        if not labels:
            continue

        mice  = [b for b in labels if b[0] == 0]
        water = [b for b in labels if b[0] == 1]
        food  = [b for b in labels if b[0] == 2]

        mice_clean  = _nms(mice, mouse_iou)
        water_clean = [max(water, key=lambda b: b[3] * b[4])] if water else []
        food_clean  = _nms(food, food_iou)

        removed = (len(mice) - len(mice_clean) +
                   len(water) - len(water_clean) +
                   len(food) - len(food_clean))

        if removed:
            files_changed += 1
            total_removed += removed
            action = "would fix" if dry_run else "fixed"
            parts = []
            if len(mice)  != len(mice_clean):  parts.append(f"mouse {len(mice)}→{len(mice_clean)}")
            if len(water) != len(water_clean):  parts.append(f"water {len(water)}→{len(water_clean)}")
            if len(food)  != len(food_clean):   parts.append(f"food {len(food)}→{len(food_clean)}")
            print(f"  {action}: {lf.name:<60} [{', '.join(parts)}]")

            if not dry_run:
                _write(lf, mice_clean + water_clean + food_clean)

    print(f"\n{'─'*55}")
    print(f"{'DRY RUN — no files written' if dry_run else 'Dedup complete'}")
    print(f"  Files changed  : {files_changed}")
    print(f"  Boxes removed  : {total_removed}")

    return {"files_changed": files_changed, "total_removed": total_removed}


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIX CLASS IDs
# ─────────────────────────────────────────────────────────────────────────────

def _infer_class(cx: float) -> int:
    """Infer vivarium class from normalised box centre x."""
    if WATER_X_MIN <= cx <= WATER_X_MAX:
        return 1   # water_container
    if FOOD_X_MIN <= cx <= FOOD_X_MAX:
        return 2   # food_area
    return 0       # mouse


def fix_classes(
    label_dir: Path = DEFAULT_LABEL_DIR,
    dry_run:   bool = False,
) -> dict:
    """
    One-time heuristic fix for datasets where every box was written as class 0.
    Re-maps class IDs based on bounding box x-position:
        cx in [0.75, 0.97] → class 1 (water_container)
        cx in [0.03, 0.28] → class 2 (food_area)
        everything else    → class 0 (mouse)

    ⚠ Heuristic only — verify with LabelImg after running.

    Args:
        label_dir: Folder containing .txt label files.
        dry_run:   Preview without writing.

    Returns:
        {"files_changed": N, "lines_changed": M}
    """
    files_changed = 0
    lines_changed = 0

    print(f"{'DRY RUN — ' if dry_run else ''}Fixing class IDs by position …\n")

    for lf in _label_files(label_dir):
        labels = _read(lf)
        if not labels:
            continue

        new_labels = []
        changed = 0

        for (orig_cls, cx, cy, bw, bh) in labels:
            new_cls = _infer_class(cx)
            if new_cls != orig_cls:
                changed += 1
            new_labels.append((new_cls, cx, cy, bw, bh))

        if changed:
            files_changed += 1
            lines_changed += changed
            action = "would fix" if dry_run else "fixed"
            print(f"  {action} {changed} lines  →  {lf.name}")
            if not dry_run:
                _write(lf, new_labels)

    print(f"\n{'─'*55}")
    print(f"{'DRY RUN — no files written' if dry_run else 'Fix complete'}")
    print(f"  Files changed  : {files_changed}")
    print(f"  Lines changed  : {lines_changed}")
    print(f"\n⚠ Verify with LabelImg before training.")

    return {"files_changed": files_changed, "lines_changed": lines_changed}


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run any operation directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Vivarium label hygiene tools")
    ap.add_argument(
        "op",
        choices=["verify", "clean", "dedup", "fix"],
        help="Operation to run",
    )
    ap.add_argument("--labels",    type=Path,  default=DEFAULT_LABEL_DIR)
    ap.add_argument("--images",    type=Path,  default=DEFAULT_IMG_DIR)
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--max-area",  type=float, default=MAX_FOOD_AREA)
    ap.add_argument("--max-w",     type=float, default=MAX_FOOD_W)
    ap.add_argument("--max-h",     type=float, default=MAX_FOOD_H)
    ap.add_argument("--mouse-iou", type=float, default=MOUSE_NMS_IOU)
    ap.add_argument("--food-iou",  type=float, default=FOOD_NMS_IOU)
    args = ap.parse_args()

    if args.op == "verify":
        verify(img_dir=args.images, label_dir=args.labels)

    elif args.op == "clean":
        clean_food(
            label_dir=args.labels,
            max_area=args.max_area,
            max_w=args.max_w,
            max_h=args.max_h,
            dry_run=args.dry_run,
        )

    elif args.op == "dedup":
        dedup(
            label_dir=args.labels,
            mouse_iou=args.mouse_iou,
            food_iou=args.food_iou,
            dry_run=args.dry_run,
        )

    elif args.op == "fix":
        fix_classes(label_dir=args.labels, dry_run=args.dry_run)