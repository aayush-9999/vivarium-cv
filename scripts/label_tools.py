"""
scripts/label_tools.py
======================
Shared label utilities for the 9-class vivarium pipeline.

9-class scheme:
    0            mouse
    1            water_critical   (fill 0–15%)
    2            water_low        (fill 15–35%)
    3            water_ok         (fill 35–80%)
    4            water_full       (fill 80–100%)
    5            food_critical    (fill 0–15%)
    6            food_low         (fill 15–35%)
    7            food_ok          (fill 35–80%)
    8            food_full        (fill 80–100%)

Key rules enforced here:
    - Water (classes 1–4): EXACTLY 1 box per image, must sit in top-right quadrant
      (cx > TOP_RIGHT_CX_MIN, cy < TOP_RIGHT_CY_MAX).  If multiple water boxes exist,
      keep the one closest to the top-right corner; if none pass the spatial filter,
      keep the largest box regardless (fallback, logged as a warning).
    - Food  (classes 5–8): NMS dedup, keep all non-overlapping
    - Mouse (class 0):     NMS dedup, keep all non-overlapping

Functions exported / used by orchestrator:
    verify(img_dir, label_dir)          → list[str]   issues
    clean_food(label_dir, ...)          → dict        stats
    dedup(label_dir, ...)               → dict        stats
    fix_classes(label_dir, dry_run)     → dict        stats
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vivarium.label_tools")

# ── Class groups ──────────────────────────────────────────────────────────────
CLASS_MOUSE      = 0
WATER_CLASS_IDS  = {1, 2, 3, 4}
FOOD_CLASS_IDS   = {5, 6, 7, 8}
ALL_VALID_IDS    = {0, 1, 2, 3, 4, 5, 6, 7, 8}

CLASS_NAMES = {
    0: "mouse",
    1: "water_critical", 2: "water_low", 3: "water_ok", 4: "water_full",
    5: "food_critical",  6: "food_low",  7: "food_ok",  8: "food_full",
}

# ── Top-right jug spatial constraint ─────────────────────────────────────────
# The water bottle sits in the top-right corner of the cage image (640×640).
# Normalised thresholds — tune to your cage setup if needed.
TOP_RIGHT_CX_MIN = 0.45   # cx must be right of centre
TOP_RIGHT_CY_MAX = 0.55   # cy must be in upper half

# ── NMS thresholds ────────────────────────────────────────────────────────────
MOUSE_NMS_IOU = 0.45
FOOD_NMS_IOU  = 0.50

# ── Food size filter thresholds (normalised 0–1) ──────────────────────────────
# A legitimate hopper box is roughly 0.25w × 0.34h  →  area ≈ 0.085
MAX_FOOD_AREA = 0.12
MAX_FOOD_W    = 0.40
MAX_FOOD_H    = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

Label = tuple[int, float, float, float, float]   # (cls, cx, cy, bw, bh)


def read_labels(path: Path) -> list[Label]:
    if not path.exists():
        return []
    out: list[Label] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                out.append((
                    int(parts[0]),
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                ))
            except ValueError:
                pass
    return out


def write_labels(path: Path, labels: list[Label]) -> None:
    lines = [
        f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for cls, cx, cy, bw, bh in labels
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: Label, b: Label) -> float:
    """IoU between two YOLO-format (cx,cy,bw,bh) boxes — class ID ignored."""
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0.0 else 0.0


def _area(b: Label) -> float:
    return b[3] * b[4]


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────

def _nms(boxes: list[Label], iou_thresh: float) -> list[Label]:
    """
    Greedy NMS sorted by area descending (largest box wins ties).
    Preserves genuinely separate detections (e.g. multiple mice).
    """
    if not boxes:
        return []
    remaining = sorted(boxes, key=_area, reverse=True)
    kept: list[Label] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [b for b in remaining if _iou(best, b) < iou_thresh]
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Water jug: exactly-one + top-right enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _top_right_score(b: Label) -> float:
    """
    Score that rewards boxes close to the top-right corner.
    Higher = better candidate for the single jug detection.
    Uses cx (want large) and cy (want small).
    """
    _, cx, cy, _, _ = b
    return cx - cy   # maximise cx, minimise cy


# CHANGE: Added optional label_name parameter so callers can get meaningful
# log messages that include the filename when a water box is outside the
# expected top-right quadrant.  This resolves the TODO comment that was
# left in the verify() function below.
def _passes_top_right(b: Label, label_name: str = "") -> bool:
    """
    Returns True if the water box sits in the expected top-right quadrant
    (cx >= TOP_RIGHT_CX_MIN and cy <= TOP_RIGHT_CY_MAX).

    If label_name is provided and the box fails, a logger.warning is emitted
    so the file can be found and reviewed without needing to run verify().
    """
    _, cx, cy, _, _ = b
    passes = cx >= TOP_RIGHT_CX_MIN and cy <= TOP_RIGHT_CY_MAX
    if not passes and label_name:
        logger.warning(
            "%s: water box at cx=%.3f cy=%.3f is outside top-right quadrant "
            "(expected cx>=%.2f, cy<=%.2f) — review manually.",
            label_name, cx, cy, TOP_RIGHT_CX_MIN, TOP_RIGHT_CY_MAX,
        )
    return passes


def _keep_one_water(water_boxes: list[Label], label_name: str = "") -> Label | None:
    """
    From all water-class detections, enforce exactly one jug in the top-right.

    Selection priority:
      1. Among boxes that pass the spatial filter, pick the one with the
         highest top-right score (cx - cy).
      2. If NO box passes the spatial filter (unusual cage framing, etc.),
         fall back to the largest box and log a warning so it can be reviewed.

    Returns None if the input list is empty.
    """
    if not water_boxes:
        return None

    # First apply NMS within the water group to collapse true duplicates
    after_nms = _nms(water_boxes, iou_thresh=0.30)

    # CHANGE: pass label_name into _passes_top_right so warnings are actionable
    in_quadrant = [b for b in after_nms if _passes_top_right(b, label_name=label_name)]

    if in_quadrant:
        return max(in_quadrant, key=_top_right_score)

    # Fallback — keep largest, emit warning
    fallback = max(after_nms, key=_area)
    if label_name:
        logger.warning(
            "%s: no water box in top-right quadrant "
            "(cx>%.2f, cy<%.2f) — kept largest box as fallback "
            "(cx=%.3f cy=%.3f). Review manually.",
            label_name,
            TOP_RIGHT_CX_MIN, TOP_RIGHT_CY_MAX,
            fallback[1], fallback[2],
        )
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Food: size filter
# ─────────────────────────────────────────────────────────────────────────────

def _is_oversized_food(b: Label) -> bool:
    cls, _, _, bw, bh = b
    if cls not in FOOD_CLASS_IDS:
        return False
    return (bw * bh) > MAX_FOOD_AREA or bw > MAX_FOOD_W or bh > MAX_FOOD_H


# ─────────────────────────────────────────────────────────────────────────────
# Per-file dedup
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_labels(
    labels: list[Label],
    mouse_iou: float,
    food_iou: float,
    label_name: str = "",
) -> tuple[list[Label], dict]:
    """
    Apply all dedup rules to one label file's worth of boxes.
    Returns (cleaned_labels, stats_dict).
    """
    mice  = [b for b in labels if b[0] == CLASS_MOUSE]
    water = [b for b in labels if b[0] in WATER_CLASS_IDS]
    food  = [b for b in labels if b[0] in FOOD_CLASS_IDS]

    mice_clean  = _nms(mice, mouse_iou)
    water_clean = []
    best_water  = _keep_one_water(water, label_name=label_name)
    if best_water is not None:
        water_clean = [best_water]
    food_clean  = _nms(food, food_iou)

    cleaned = mice_clean + water_clean + food_clean

    stats = {
        "mouse_before": len(mice),   "mouse_after": len(mice_clean),
        "water_before": len(water),  "water_after": len(water_clean),
        "food_before":  len(food),   "food_after":  len(food_clean),
    }
    return cleaned, stats


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    img_dir: Path,
    label_dir: Path,
) -> list[str]:
    """
    Verify label/image pairing and coordinate validity for a 9-class dataset.

    Checks:
      - Every image has a corresponding label file
      - Every label file has a corresponding image
      - All class IDs are in 0–8
      - All coordinates are in [0, 1]
      - Exactly one water box (classes 1–4) per labelled image
      - Water box sits in top-right quadrant (with warning, not error)

    Returns a list of human-readable issue strings (empty = all clean).
    """
    issues: list[str] = []

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_stems = {
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in img_extensions
    }
    label_stems = {
        p.stem for p in label_dir.glob("*.txt")
        if p.name != "classes.txt"
    }

    # Missing labels
    for stem in sorted(img_stems - label_stems):
        issues.append(f"MISSING_LABEL: {stem}")

    # Orphan labels (no image)
    for stem in sorted(label_stems - img_stems):
        issues.append(f"ORPHAN_LABEL: {stem}")

    # Per-file content checks
    for stem in sorted(label_stems & img_stems):
        lbl_path = label_dir / f"{stem}.txt"
        labels   = read_labels(lbl_path)

        if not labels:
            # Empty label file is valid (image with no detectable objects)
            continue

        water_boxes = []
        for i, (cls, cx, cy, bw, bh) in enumerate(labels):
            # Valid class
            if cls not in ALL_VALID_IDS:
                issues.append(
                    f"INVALID_CLASS: {stem}.txt line {i+1} cls={cls}"
                )
            # Valid coordinates
            for name, val in [("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)]:
                if not (0.0 <= val <= 1.0):
                    issues.append(
                        f"OUT_OF_RANGE: {stem}.txt line {i+1} {name}={val:.4f}"
                    )
            # Zero-size box
            if bw <= 0.0 or bh <= 0.0:
                issues.append(
                    f"ZERO_SIZE: {stem}.txt line {i+1} bw={bw:.4f} bh={bh:.4f}"
                )

            if cls in WATER_CLASS_IDS:
                water_boxes.append((cls, cx, cy, bw, bh))

        # Water box count
        if len(water_boxes) == 0:
            issues.append(f"NO_WATER_BOX: {stem}.txt — no water class (1–4) detected")
        elif len(water_boxes) > 1:
            issues.append(
                f"MULTIPLE_WATER: {stem}.txt — {len(water_boxes)} water boxes "
                f"(should be exactly 1 after dedup)"
            )
        elif len(water_boxes) == 1:
            _, cx, cy, _, _ = water_boxes[0]
            # CHANGE: pass label_name=stem so the warning log includes the filename
            if not _passes_top_right(water_boxes[0], label_name=stem):
                issues.append(
                    f"WATER_NOT_TOP_RIGHT: {stem}.txt — "
                    f"water box cx={cx:.3f} cy={cy:.3f} is outside expected "
                    f"top-right region (cx>={TOP_RIGHT_CX_MIN}, cy<={TOP_RIGHT_CY_MAX})"
                )

    return issues


def clean_food(
    label_dir: Path,
    max_area: float = MAX_FOOD_AREA,
    max_w: float    = MAX_FOOD_W,
    max_h: float    = MAX_FOOD_H,
    dry_run: bool   = False,
) -> dict:
    """
    Remove food boxes (classes 5–8) that are too large — these are usually
    bedding false positives that cover most of the cage floor.

    Returns stats dict: {files_scanned, files_changed, boxes_removed}.
    """
    label_files = sorted(
        f for f in label_dir.glob("*.txt")
        if f.name != "classes.txt"
    )

    files_scanned  = len(label_files)
    files_changed  = 0
    boxes_removed  = 0

    print(
        f"{'[DRY RUN] ' if dry_run else ''}clean_food: "
        f"scanning {files_scanned} files  "
        f"(max_area={max_area}, max_w={max_w}, max_h={max_h})"
    )

    for lf in label_files:
        labels = read_labels(lf)
        if not labels:
            continue

        kept    = []
        removed = 0
        for b in labels:
            cls, _, _, bw, bh = b
            if cls in FOOD_CLASS_IDS and (
                (bw * bh) > max_area or bw > max_w or bh > max_h
            ):
                removed += 1
            else:
                kept.append(b)

        if removed > 0:
            files_changed += 1
            boxes_removed += removed
            oversized_info = "  ".join(
                f"cls={b[0]} bw={b[3]:.2f} bh={b[4]:.2f} area={b[3]*b[4]:.3f}"
                for b in labels if b[0] in FOOD_CLASS_IDS and (
                    (b[3] * b[4]) > max_area or b[3] > max_w or b[4] > max_h
                )
            )
            action = "would remove" if dry_run else "removed"
            print(
                f"  {action} {removed}x food  {lf.name:<55}  {oversized_info}"
            )
            if not dry_run:
                write_labels(lf, kept)

    stats = {
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "boxes_removed": boxes_removed,
    }
    print(
        f"  → {'[DRY RUN] ' if dry_run else ''}done: "
        f"{files_changed} files changed, {boxes_removed} food boxes removed"
    )
    return stats


def dedup(
    label_dir: Path,
    mouse_iou: float = MOUSE_NMS_IOU,
    food_iou:  float = FOOD_NMS_IOU,
    dry_run:   bool  = False,
) -> dict:
    """
    Deduplicate all label files according to 9-class rules:

      - Mouse (0):      NMS — keep all non-overlapping
      - Water (1–4):    keep exactly ONE box in the top-right quadrant
      - Food  (5–8):    NMS — keep all non-overlapping

    Returns stats dict.
    """
    label_files = sorted(
        f for f in label_dir.glob("*.txt")
        if f.name != "classes.txt"
    )

    files_scanned = len(label_files)
    files_changed = 0
    totals = {
        "mouse_removed": 0,
        "water_removed": 0,
        "food_removed":  0,
    }

    print(
        f"{'[DRY RUN] ' if dry_run else ''}dedup: "
        f"scanning {files_scanned} files  "
        f"(mouse_iou={mouse_iou}, food_iou={food_iou})"
    )
    print(
        f"  Rules: mouse=NMS  |  water=1 box top-right  |  food=NMS"
    )

    for lf in label_files:
        labels = read_labels(lf)
        if not labels:
            continue

        cleaned, stats = _dedup_labels(
            labels,
            mouse_iou=mouse_iou,
            food_iou=food_iou,
            label_name=lf.name,
        )

        m_rem = stats["mouse_before"] - stats["mouse_after"]
        w_rem = stats["water_before"] - stats["water_after"]
        f_rem = stats["food_before"]  - stats["food_after"]
        any_change = (m_rem + w_rem + f_rem) > 0

        if any_change:
            files_changed             += 1
            totals["mouse_removed"]   += m_rem
            totals["water_removed"]   += w_rem
            totals["food_removed"]    += f_rem

            parts = []
            if m_rem: parts.append(f"mouse {stats['mouse_before']}→{stats['mouse_after']}")
            if w_rem: parts.append(f"water {stats['water_before']}→{stats['water_after']}")
            if f_rem: parts.append(f"food  {stats['food_before']}→{stats['food_after']}")
            action = "would fix" if dry_run else "fixed"
            print(f"  {action}: {lf.name:<60}  [{', '.join(parts)}]")

            if not dry_run:
                write_labels(lf, cleaned)

    stats_out = {
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        **totals,
    }
    print(
        f"  → {'[DRY RUN] ' if dry_run else ''}done: "
        f"{files_changed} files changed  "
        f"(mouse -{totals['mouse_removed']}, "
        f"water -{totals['water_removed']}, "
        f"food -{totals['food_removed']})"
    )
    return stats_out


def fix_classes(
    label_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    One-time migration helper: converts old 3-class labels to 9-class defaults.

      Old class 0 (mouse)            → 0  (unchanged)
      Old class 1 (water_container)  → 3  (water_ok)
      Old class 2 (food_area)        → 7  (food_ok)

    Any class ID already in 0–8 and > 2 is left untouched (already migrated).
    Any class ID outside 0–8 is left untouched and reported.

    After remapping, dedup rules are applied so the water box ends up
    correctly placed in the top-right quadrant.

    Returns stats dict.
    """
    OLD_TO_NEW = {1: 3, 2: 7}   # 3-class → 9-class defaults

    label_files = sorted(
        f for f in label_dir.glob("*.txt")
        if f.name != "classes.txt"
    )

    files_scanned  = len(label_files)
    files_changed  = 0
    boxes_remapped = 0
    unknown_classes: dict[int, int] = {}

    print(
        f"{'[DRY RUN] ' if dry_run else ''}fix_classes: "
        f"scanning {files_scanned} files"
    )
    print(
        "  Mapping: cls 1 → 3 (water_ok)   cls 2 → 7 (food_ok)"
    )

    for lf in label_files:
        labels = read_labels(lf)
        if not labels:
            continue

        remapped: list[Label] = []
        n_remapped = 0

        for b in labels:
            cls, cx, cy, bw, bh = b
            if cls in OLD_TO_NEW:
                new_cls = OLD_TO_NEW[cls]
                remapped.append((new_cls, cx, cy, bw, bh))
                n_remapped += 1
            else:
                if cls not in ALL_VALID_IDS:
                    unknown_classes[cls] = unknown_classes.get(cls, 0) + 1
                remapped.append(b)

        if n_remapped > 0:
            files_changed  += 1
            boxes_remapped += n_remapped

            # Also apply dedup so water lands in top-right
            remapped, _ = _dedup_labels(
                remapped,
                mouse_iou=MOUSE_NMS_IOU,
                food_iou=FOOD_NMS_IOU,
                label_name=lf.name,
            )

            print(f"  {'[skip] ' if dry_run else ''}remapped {n_remapped} boxes  {lf.name}")
            if not dry_run:
                write_labels(lf, remapped)

    if unknown_classes:
        for cls_id, count in sorted(unknown_classes.items()):
            print(
                f"  [WARN] Unknown class ID {cls_id} found {count}× — not remapped, review manually"
            )

    stats = {
        "files_scanned":  files_scanned,
        "files_changed":  files_changed,
        "boxes_remapped": boxes_remapped,
        "unknown_classes": unknown_classes,
    }
    print(
        f"  → {'[DRY RUN] ' if dry_run else ''}done: "
        f"{files_changed} files changed, {boxes_remapped} boxes remapped"
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI (for direct script use / debugging)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="9-class vivarium label utilities"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # verify
    p_ver = sub.add_parser("verify", help="Verify label integrity")
    p_ver.add_argument("--img-dir", type=Path, default=Path("dataset/original"))
    p_ver.add_argument("--label-dir", type=Path, default=Path("dataset/original_labels_9class"))

    # clean
    p_cln = sub.add_parser("clean", help="Remove oversized food boxes")
    p_cln.add_argument("--label-dir", type=Path, default=Path("dataset/original_labels_9class"))
    p_cln.add_argument("--max-area",  type=float, default=MAX_FOOD_AREA)
    p_cln.add_argument("--max-w",     type=float, default=MAX_FOOD_W)
    p_cln.add_argument("--max-h",     type=float, default=MAX_FOOD_H)
    p_cln.add_argument("--dry-run",   action="store_true")

    # dedup
    p_ded = sub.add_parser("dedup", help="Deduplicate labels (9-class rules)")
    p_ded.add_argument("--label-dir",  type=Path,  default=Path("dataset/original_labels_9class"))
    p_ded.add_argument("--mouse-iou",  type=float, default=MOUSE_NMS_IOU)
    p_ded.add_argument("--food-iou",   type=float, default=FOOD_NMS_IOU)
    p_ded.add_argument("--dry-run",    action="store_true")

    # fix-classes
    p_fix = sub.add_parser("fix-classes", help="Migrate old 3-class labels to 9-class defaults")
    p_fix.add_argument("--label-dir", type=Path, default=Path("dataset/original_labels_9class"))
    p_fix.add_argument("--dry-run",   action="store_true")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s  %(message)s",
    )

    if args.cmd == "verify":
        issues = verify(args.img_dir, args.label_dir)
        if issues:
            print(f"\n{len(issues)} issue(s) found:\n")
            for iss in issues:
                print(f"  {iss}")
        else:
            print("\n✅ All labels are clean.\n")

    elif args.cmd == "clean":
        clean_food(
            args.label_dir,
            max_area=args.max_area,
            max_w=args.max_w,
            max_h=args.max_h,
            dry_run=args.dry_run,
        )

    elif args.cmd == "dedup":
        dedup(
            args.label_dir,
            mouse_iou=args.mouse_iou,
            food_iou=args.food_iou,
            dry_run=args.dry_run,
        )

    elif args.cmd == "fix-classes":
        fix_classes(args.label_dir, dry_run=args.dry_run)