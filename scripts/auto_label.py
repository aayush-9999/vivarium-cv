"""
scripts/auto_label.py
=====================
Auto-label new images using the trained YOLOv8 model.

This is the core of the active learning loop:

    Cycle 1:  train on synthetic data
    Cycle 2+: auto-label real images → human reviews uncertain ones → retrain
              → model gets better → next cycle needs fewer human corrections

HOW IT WORKS
─────────────
Every prediction box gets a confidence score from YOLO (0.0–1.0).
We split predictions into three buckets based on that score:

    conf >= AUTO_ACCEPT  (default 0.80)  → written as-is, no human needed
    conf >= REVIEW       (default 0.45)  → written but flagged for human review
    conf <  REVIEW                       → box dropped, image flagged for manual label

Output folder structure
────────────────────────
    dst/
      labels/
        auto/        ← high-conf, accepted automatically, ready for training
        review/      ← medium-conf, open in LabelImg and correct these
        rejected/    ← low-conf images, label these manually from scratch
      debug/         ← annotated JPEGs showing all predictions + bucket colour
      review_report.txt  ← human-readable list of every file needing attention

Typical result on a reasonably trained model:
    ~70-80% auto-accepted  (zero human time)
    ~15-25% needs review   (3-5x faster than labelling from scratch)
    ~5-10%  rejected       (label manually)

USAGE
──────
    # Via orchestrator (recommended)
    from pipeline.pipeline_factory import get_orchestrator
    orch = get_orchestrator()
    orch.auto_label(
        src=Path("dataset/real/images"),
        dst=Path("dataset/real"),
        debug=True,
    )

    # Direct CLI
    python scripts/auto_label.py --src dataset/real/images --dst dataset/real
    python scripts/auto_label.py --src dataset/real/images --dst dataset/real --debug
    python scripts/auto_label.py --src dataset/real/images --dst dataset/real \\
        --auto-accept 0.85 --review 0.50

AFTER RUNNING
──────────────
    1. Open dst/review_report.txt — lists every file needing human attention
    2. Open dst/labels/review/ images in LabelImg:
           labelImg dataset/real/images dataset/real/labels/review/
       Correct boxes, save.
    3. Copy corrected review/ labels to a merged labels folder
    4. Merge with auto/ labels → run split_dataset → retrain

THE LOOP
─────────
    New images → auto_label → correct review/ → retrain → better model
        ↑                                                       │
        └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Tune these based on your model's calibration.
# After first real-image cycle, raise AUTO_ACCEPT if model is performing well.
AUTO_ACCEPT_THRESH = 0.80   # >= this → write label, no review needed
REVIEW_THRESH      = 0.45   # >= this → write label, flag for human review
                             # <  this → drop box, flag image for manual label

# ── Class info (mirrors core/config.py) ──────────────────────────────────────
CLASS_NAMES = {
    0: "mouse",
    1: "water_critical", 2: "water_low", 3: "water_ok", 4: "water_full",
    5: "food_critical",  6: "food_low",  7: "food_ok",  8: "food_full",
}

WATER_IDS = {1, 2, 3, 4}
FOOD_IDS  = {5, 6, 7, 8}

# Debug annotation colours per bucket
COLOR_AUTO   = (80,  200,  80)    # green   — accepted
COLOR_REVIEW = (0,   180, 255)    # orange  — needs review
COLOR_DROP   = (0,     0, 200)    # red     — dropped / below review thresh

# Per-class colours for the label text
CLASS_COLORS = {
    0: (255,  80,  80),
    1: (0,     0, 200), 2: (0,  140, 255), 3: (80, 200, 80), 4: (80, 200, 80),
    5: (0,     0, 200), 6: (0,  140, 255), 7: (80, 200, 80), 8: (80, 200, 80),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, size: int = 640):
    h, w   = img.shape[:2]
    scale  = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_t   = (size - nh) // 2
    pad_l   = (size - nw) // 2
    canvas[pad_t:pad_t+nh, pad_l:pad_l+nw] = resized
    return canvas, scale, pad_t, pad_l


def _bucket(conf: float) -> str:
    if conf >= AUTO_ACCEPT_THRESH:
        return "auto"
    if conf >= REVIEW_THRESH:
        return "review"
    return "drop"


def _draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _write_labels(path: Path, boxes: list[tuple]) -> None:
    """Write YOLO format label file. boxes = [(cls, cx, cy, bw, bh), ...]"""
    lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
             for cls, cx, cy, bw, bh in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_path:    Path,
    model,
    auto_dir:    Path,
    review_dir:  Path,
    reject_dir:  Path,
    debug_dir:   Path | None,
    auto_thresh: float,
    review_thresh: float,
) -> dict:
    """
    Run YOLO on one image, bucket each prediction, write label files.

    Returns a stats dict for the summary report.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {"status": "unreadable", "stem": img_path.stem}

    frame_640, scale, pad_t, pad_l = _letterbox(img, 640)

    results = model.predict(
        source=frame_640,
        imgsz=640,
        verbose=False,
        conf=review_thresh,   # let YOLO return everything above review threshold
                               # we bucket manually below
    )

    r = results[0]

    auto_boxes:   list[tuple] = []
    review_boxes: list[tuple] = []
    drop_count = 0
    image_needs_review  = False
    image_needs_manual  = False

    box_details = []   # for the review report

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy    = r.boxes.xyxy.cpu().numpy()
        confs   = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(xyxy, confs, classes):
            conf = float(conf)
            cls  = int(cls)
            x1, y1, x2, y2 = box

            # Convert to YOLO normalised format
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 640
            bw = (x2 - x1) / 640
            bh = (y2 - y1) / 640
            cx = float(np.clip(cx, 0, 1))
            cy = float(np.clip(cy, 0, 1))
            bw = float(np.clip(bw, 0, 1))
            bh = float(np.clip(bh, 0, 1))

            bucket = _bucket(conf)

            if bucket == "auto":
                auto_boxes.append((cls, cx, cy, bw, bh))
            elif bucket == "review":
                review_boxes.append((cls, cx, cy, bw, bh))
                image_needs_review = True
            else:
                drop_count += 1
                image_needs_manual = True

            box_details.append({
                "cls":    cls,
                "name":   CLASS_NAMES.get(cls, str(cls)),
                "conf":   round(conf, 3),
                "bucket": bucket,
                "xyxy":   [round(float(v), 1) for v in [x1, y1, x2, y2]],
            })

    # ── Determine where this image's labels go ────────────────────────────────
    # Rules:
    #   - If ALL boxes are auto → write to auto/, done
    #   - If any box is review  → write combined (auto+review) to review/
    #                             human sees all boxes, corrects the uncertain ones
    #   - If any box was dropped (low conf) → copy image stem to rejected/
    #                             human needs to check if something was missed
    #   - If no boxes at all    → write to rejected/ (might be empty cage or miss)

    all_boxes   = auto_boxes + review_boxes
    label_stem  = img_path.stem + ".txt"

    if not all_boxes and drop_count == 0:
        # YOLO found nothing at all — could be empty cage or model failure
        # Write empty label to rejected so human can verify
        _write_labels(reject_dir / label_stem, [])
        dest = "rejected"

    elif image_needs_manual and not all_boxes:
        # Only low-conf boxes, nothing usable — full manual label needed
        _write_labels(reject_dir / label_stem, [])
        dest = "rejected"

    elif image_needs_review or image_needs_manual:
        # Mix of good + uncertain boxes — write all to review/
        # Human verifies the uncertain ones but gets the good ones pre-filled
        _write_labels(review_dir / label_stem, all_boxes)
        if image_needs_manual:
            # Also note in rejected that this image had dropped boxes
            # so reviewer knows to check for missed detections
            (reject_dir / label_stem).write_text(
                f"# dropped {drop_count} low-conf boxes — check for misses\n"
                f"# review labels already written to review/{label_stem}\n",
                encoding="utf-8",
            )
        dest = "review"

    else:
        # All boxes are high-confidence → auto-accept
        _write_labels(auto_dir / label_stem, auto_boxes)
        dest = "auto"

    # ── Debug annotation ──────────────────────────────────────────────────────
    if debug_dir is not None:
        viz = frame_640.copy()
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy    = r.boxes.xyxy.cpu().numpy()
            confs_  = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(xyxy, confs_, classes):
                conf   = float(conf)
                bucket = _bucket(conf)
                color  = COLOR_AUTO if bucket == "auto" else \
                         COLOR_REVIEW if bucket == "review" else COLOR_DROP
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                label  = f"{CLASS_NAMES.get(int(cls), str(cls))} {conf:.2f} [{bucket}]"
                _draw_box(viz, x1, y1, x2, y2, label, color)

        # Stamp destination bucket on image
        stamp_color = COLOR_AUTO if dest == "auto" else \
                      COLOR_REVIEW if dest == "review" else COLOR_DROP
        cv2.putText(viz, f"→ {dest.upper()}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(viz, f"→ {dest.upper()}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stamp_color, 2, cv2.LINE_AA)

        cv2.imwrite(
            str(debug_dir / img_path.name),
            viz,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

    return {
        "status":       "ok",
        "stem":         img_path.stem,
        "dest":         dest,
        "auto_boxes":   len(auto_boxes),
        "review_boxes": len(review_boxes),
        "drop_count":   drop_count,
        "boxes":        box_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    src:           Path,
    dst:           Path,
    weights:       str  = None,
    auto_thresh:   float = AUTO_ACCEPT_THRESH,
    review_thresh: float = REVIEW_THRESH,
    conf_override: float = None,   # orchestrator compat — sets review_thresh
    debug:         bool  = False,
) -> dict:
    """
    Auto-label all images in src/ using the trained YOLO model.

    Args:
        src           : folder of images to label
        dst           : output root  (labels/ and debug/ created here)
        weights       : path to best.pt  (defaults to YOLO_WEIGHTS from config)
        auto_thresh   : confidence >= this → auto-accept  (default 0.80)
        review_thresh : confidence >= this → review       (default 0.45)
        conf_override : shorthand to set review_thresh (orchestrator compat)
        debug         : save annotated debug images to dst/debug/

    Returns:
        stats dict with counts per bucket and list of files needing review
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        sys.exit(1)

    from core.config import YOLO_WEIGHTS
    if conf_override is not None:
        review_thresh = conf_override

    weights_path = Path(weights) if weights else YOLO_WEIGHTS
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("  Set YOLO_WEIGHTS in your .env or pass --weights explicitly.")
        sys.exit(1)

    img_paths = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        print(f"[ERROR] No images found in {src}")
        sys.exit(1)

    # ── Create output dirs ────────────────────────────────────────────────────
    auto_dir   = dst / "labels" / "auto"
    review_dir = dst / "labels" / "review"
    reject_dir = dst / "labels" / "rejected"
    debug_dir  = (dst / "debug") if debug else None

    for d in [auto_dir, review_dir, reject_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Write classes.txt to all label dirs
    classes_txt = "\n".join(
        ["mouse",
         "water_critical", "water_low", "water_ok", "water_full",
         "food_critical",  "food_low",  "food_ok",  "food_full"]
    )
    for d in [auto_dir, review_dir, reject_dir]:
        (d / "classes.txt").write_text(classes_txt, encoding="utf-8")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nWeights      : {weights_path}")
    print(f"Source       : {src}  ({len(img_paths)} images)")
    print(f"Output       : {dst}")
    print(f"Auto-accept  : conf >= {auto_thresh}")
    print(f"Review       : conf >= {review_thresh}")
    print(f"Drop         : conf <  {review_thresh}")
    print(f"Debug images : {'yes → ' + str(debug_dir) if debug else 'no'}\n")

    model = YOLO(str(weights_path))

    # ── Process each image ────────────────────────────────────────────────────
    results_log = []
    counts = {"auto": 0, "review": 0, "rejected": 0, "unreadable": 0}

    for idx, img_path in enumerate(img_paths, 1):
        stats = process_image(
            img_path      = img_path,
            model         = model,
            auto_dir      = auto_dir,
            review_dir    = review_dir,
            reject_dir    = reject_dir,
            debug_dir     = debug_dir,
            auto_thresh   = auto_thresh,
            review_thresh = review_thresh,
        )

        dest   = stats.get("dest", stats.get("status", "unreadable"))
        counts[dest if dest in counts else "unreadable"] += 1
        results_log.append(stats)

        # Console line
        if dest == "auto":
            marker = "✓"
        elif dest == "review":
            marker = "⚠"
        else:
            marker = "✗"

        print(
            f"  [{idx:4d}/{len(img_paths)}] {marker} {img_path.name:<50} "
            f"→ {dest:<8}  "
            f"(auto={stats.get('auto_boxes',0)} "
            f"review={stats.get('review_boxes',0)} "
            f"drop={stats.get('drop_count',0)})"
        )

    # ── Write review report ───────────────────────────────────────────────────
    report_path = dst / "review_report.txt"
    _write_review_report(report_path, results_log, auto_thresh, review_thresh, weights_path)

    # ── Write machine-readable stats JSON ─────────────────────────────────────
    stats_path = dst / "auto_label_stats.json"
    stats_out = {
        "run_at":        datetime.now(tz=timezone.utc).isoformat(),
        "weights":       str(weights_path),
        "source":        str(src),
        "total_images":  len(img_paths),
        "auto_thresh":   auto_thresh,
        "review_thresh": review_thresh,
        "counts":        counts,
        "auto_rate_pct": round(counts["auto"] / max(len(img_paths), 1) * 100, 1),
        "images":        results_log,
    }
    stats_path.write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    # ── Final summary ─────────────────────────────────────────────────────────
    total = len(img_paths)
    print(f"""
{'─'*65}
Auto-labelling complete
  Total images    : {total}
  ✓ Auto-accepted : {counts['auto']:4d}  ({counts['auto']/total*100:.1f}%)  → {auto_dir}
  ⚠ Needs review  : {counts['review']:4d}  ({counts['review']/total*100:.1f}%)  → {review_dir}
  ✗ Rejected      : {counts['rejected']:4d}  ({counts['rejected']/total*100:.1f}%)  → {reject_dir}
  Unreadable      : {counts['unreadable']:4d}

  Review report   : {report_path}
  Stats JSON      : {stats_path}
{'─'*65}

Next steps:
  1. Open review_report.txt to see what needs human attention
  2. Correct labels in {review_dir}
       labelImg {src} {review_dir}
  3. Manually label images in {reject_dir}
  4. Merge auto/ + review/ + rejected/ into one labels folder
  5. Run split_dataset → retrain → repeat
""")

    return stats_out


def _write_review_report(
    path:          Path,
    results:       list[dict],
    auto_thresh:   float,
    review_thresh: float,
    weights_path:  Path,
) -> None:
    """
    Human-readable report listing every image needing attention,
    with specific boxes flagged and why.
    """
    review_items   = [r for r in results if r.get("dest") == "review"]
    rejected_items = [r for r in results if r.get("dest") == "rejected"]
    unreadable     = [r for r in results if r.get("status") == "unreadable"]

    lines = [
        "=" * 65,
        "VIVARIUM AUTO-LABEL REVIEW REPORT",
        f"Generated : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Weights   : {weights_path}",
        f"Thresholds: auto-accept >= {auto_thresh}  |  review >= {review_thresh}",
        "=" * 65,
        "",
        f"SUMMARY",
        f"  Auto-accepted : {sum(1 for r in results if r.get('dest')=='auto')}",
        f"  Needs review  : {len(review_items)}",
        f"  Rejected      : {len(rejected_items)}",
        f"  Unreadable    : {len(unreadable)}",
        "",
    ]

    if review_items:
        lines += [
            "─" * 65,
            f"FILES NEEDING REVIEW  ({len(review_items)} files)",
            "  Open these in LabelImg. Predicted boxes already written.",
            "  Check + correct uncertain boxes, add any missed detections.",
            "─" * 65,
            "",
        ]
        for item in review_items:
            lines.append(f"  {item['stem']}")
            for box in item.get("boxes", []):
                if box["bucket"] == "review":
                    lines.append(
                        f"    ⚠  {box['name']:<20} conf={box['conf']:.3f}  "
                        f"box={box['xyxy']}"
                    )
                elif box["bucket"] == "drop":
                    lines.append(
                        f"    ✗  {box['name']:<20} conf={box['conf']:.3f}  "
                        f"DROPPED — check if this detection was real"
                    )
            if item.get("drop_count", 0) > 0:
                lines.append(
                    f"    ↳ {item['drop_count']} box(es) dropped (conf < {review_thresh})"
                    f" — verify nothing was missed"
                )
            lines.append("")

    if rejected_items:
        lines += [
            "─" * 65,
            f"FILES NEEDING MANUAL LABELLING  ({len(rejected_items)} files)",
            "  YOLO found nothing usable here. Label these from scratch.",
            "─" * 65,
            "",
        ]
        for item in rejected_items:
            lines.append(f"  {item['stem']}")
        lines.append("")

    if unreadable:
        lines += [
            "─" * 65,
            f"UNREADABLE FILES  ({len(unreadable)} files)",
            "─" * 65,
            "",
        ]
        for item in unreadable:
            lines.append(f"  {item['stem']}")
        lines.append("")

    lines += [
        "─" * 65,
        "AFTER REVIEWING:",
        "  1. Save corrected labels in labels/review/",
        "  2. Merge labels/auto/ + labels/review/ + labels/rejected/",
        "     into dataset/real/labels/",
        "  3. python scripts/split_dataset.py \\",
        "         --img-dir dataset/real/images \\",
        "         --label-dir dataset/real/labels \\",
        "         --out dataset/split_real",
        "  4. python scripts/train.py   (or orch.train())",
        "  5. Run auto_label again on the next batch — acceptance rate",
        "     will improve each cycle.",
        "─" * 65,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Auto-label new images using trained YOLOv8 model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/auto_label.py --src dataset/real/images --dst dataset/real
  python scripts/auto_label.py --src dataset/real/images --dst dataset/real --debug
  python scripts/auto_label.py --src dataset/real/images --dst dataset/real \\
      --auto-accept 0.85 --review 0.50
  python scripts/auto_label.py --src dataset/real/images --dst dataset/real \\
      --weights runs/detect/vivarium_v2/weights/best.pt
        """,
    )
    ap.add_argument(
        "--src", type=Path, required=True,
        help="Folder of images to auto-label",
    )
    ap.add_argument(
        "--dst", type=Path, required=True,
        help="Output root folder (labels/ and debug/ created here)",
    )
    ap.add_argument(
        "--weights", type=str, default=None,
        help="Path to best.pt (default: YOLO_WEIGHTS from .env / config.py)",
    )
    ap.add_argument(
        "--auto-accept", type=float, default=AUTO_ACCEPT_THRESH,
        help=f"Confidence threshold for auto-accept (default: {AUTO_ACCEPT_THRESH})",
    )
    ap.add_argument(
        "--review", type=float, default=REVIEW_THRESH,
        help=f"Confidence threshold for review bucket (default: {REVIEW_THRESH})",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Save annotated debug images to dst/debug/",
    )
    args = ap.parse_args()

    if args.auto_accept <= args.review:
        ap.error(
            f"--auto-accept ({args.auto_accept}) must be greater than "
            f"--review ({args.review})"
        )

    main(
        src           = args.src,
        dst           = args.dst,
        weights       = args.weights,
        auto_thresh   = args.auto_accept,
        review_thresh = args.review,
        debug         = args.debug,
    )