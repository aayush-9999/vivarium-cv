# scripts/split_dataset.py
import shutil, random
from pathlib import Path

IMG_DIR   = Path("dataset/augmented/images")
LABEL_DIR = Path("dataset/augmented/labels")
OUT_DIR   = Path("dataset/split")

TRAIN_RATIO = 0.85   # 85/15 split — fine for small datasets
SEED        = 42

random.seed(SEED)

labelled = [
    p for p in LABEL_DIR.glob("*.txt")
    if p.stat().st_size > 0         # skip empty label files
]
random.shuffle(labelled)

split_idx  = int(len(labelled) * TRAIN_RATIO)
train_set  = labelled[:split_idx]
val_set    = labelled[split_idx:]

for split_name, split_files in [("train", train_set), ("val", val_set)]:
    (OUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

    for lbl in split_files:
        img = IMG_DIR / (lbl.stem + ".jpg")
        if img.exists():
            shutil.copy(img, OUT_DIR / split_name / "images" / img.name)
            shutil.copy(lbl, OUT_DIR / split_name / "labels" / lbl.name)

print(f"Train: {len(train_set)} | Val: {len(val_set)}")
print(f"Output: {OUT_DIR}")