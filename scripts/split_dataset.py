# scripts/split_dataset.py
"""
Splits dataset/augmented → dataset/split/train and dataset/split/val.
Only copies images that have a non-empty label file (labelled images only).
"""
import random
import shutil
from pathlib import Path
from collections import Counter

SRC_IMGS   = Path("dataset/augmented/images")
SRC_LABELS = Path("dataset/augmented/labels")
DST        = Path("dataset/split")
TRAIN_RATIO = 0.85
SEED        = 42

CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

random.seed(SEED)

# Only use images that have a non-empty label file
labelled = [
    p for p in SRC_LABELS.glob("*.txt")
    if p.name != "classes.txt" and p.stat().st_size > 0
]
random.shuffle(labelled)

split_idx  = int(len(labelled) * TRAIN_RATIO)
train_set  = labelled[:split_idx]
val_set    = labelled[split_idx:]

print(f"Total labelled: {len(labelled)}  →  train={len(train_set)}  val={len(val_set)}")

# Verify class IDs before copying
counter = Counter()
bad = []
for lf in labelled:
    for line in lf.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            if cls not in range(13):
                bad.append((lf.name, cls))
            else:
                counter[cls] += 1

print("\nClass distribution going into split:")
for cls_id in range(13):
    print(f"  {cls_id:2d}  {CLASS_NAMES[cls_id]:<20} : {counter.get(cls_id, 0)}")

if bad:
    print(f"\n❌ {len(bad)} invalid class IDs found — fix before splitting!")
    raise SystemExit(1)
print("\n✅ Class IDs valid. Copying files...\n")

for split_name, split_files in [("train", train_set), ("val", val_set)]:
    img_out = DST / split_name / "images"
    lbl_out = DST / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for lf in split_files:
        img = SRC_IMGS / (lf.stem + ".jpg")
        if img.exists():
            shutil.copy(img, img_out / img.name)
            shutil.copy(lf,  lbl_out / lf.name)
            copied += 1
        else:
            print(f"  [WARN] Image not found for label: {lf.name}")
    print(f"  {split_name}: {copied} image-label pairs copied")

print("\n✅ Split complete. Proceed to Step 4.")