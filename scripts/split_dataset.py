"""
scripts/split_dataset.py
========================
Splits images + labels into train/val sets.

Default mode (augmented dataset):
    python scripts/split_dataset.py

Direct from originals (no augmentation):
    python scripts/split_dataset.py \
        --img-dir   dataset/original \
        --label-dir dataset/original_labels_9class \
        --out       dataset/split
"""

from __future__ import annotations

import argparse
import shutil
import random
from pathlib import Path


def main(
    img_dir:   Path,
    label_dir: Path,
    out_dir:   Path,
    train_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)

    SKIP = {"classes.txt"}

    # Find all label files that have content and a matching image
    labelled = []
    for lbl in label_dir.glob("*.txt"):
        if lbl.name in SKIP:
            continue
        if lbl.stat().st_size == 0:
            continue
        # Find matching image — try .jpg first then .png
        img = img_dir / (lbl.stem + ".jpg")
        if not img.exists():
            img = img_dir / (lbl.stem + ".png")
        if not img.exists():
            img = img_dir / (lbl.stem + ".jpeg")
        if img.exists():
            labelled.append((img, lbl))

    if not labelled:
        print(f"[ERROR] No matched image+label pairs found.")
        print(f"  img_dir  : {img_dir}")
        print(f"  label_dir: {label_dir}")
        return

    random.shuffle(labelled)
    split_idx = int(len(labelled) * train_ratio)
    train_set = labelled[:split_idx]
    val_set   = labelled[split_idx:]

    # Ensure at least 1 in val
    if not val_set and train_set:
        val_set   = train_set[-1:]
        train_set = train_set[:-1]

    for split_name, split_files in [("train", train_set), ("val", val_set)]:
        (out_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in split_files:
            shutil.copy(img, out_dir / split_name / "images" / img.name)
            shutil.copy(lbl, out_dir / split_name / "labels" / lbl.name)

    print(f"Split complete")
    print(f"  Total pairs : {len(labelled)}")
    print(f"  Train       : {len(train_set)}")
    print(f"  Val         : {len(val_set)}")
    print(f"  Output      : {out_dir}")
    print(f"\nNext: python scripts/train.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir",    type=Path, default=Path("dataset/augmented/images"),
                    help="Folder of images (default: dataset/augmented/images)")
    ap.add_argument("--label-dir",  type=Path, default=Path("dataset/augmented/labels"),
                    help="Folder of labels (default: dataset/augmented/labels)")
    ap.add_argument("--out",        type=Path, default=Path("dataset/split"),
                    help="Output split folder (default: dataset/split)")
    ap.add_argument("--train-ratio",type=float, default=0.85)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()
    main(args.img_dir, args.label_dir, args.out, args.train_ratio, args.seed)