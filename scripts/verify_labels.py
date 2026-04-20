# scripts/verify_labels.py
from pathlib import Path
import cv2

IMG_DIR   = Path("dataset/augmented/images")
LABEL_DIR = Path("dataset/augmented/labels")

issues = []
for label_file in LABEL_DIR.glob("*.txt"):
    img_file = IMG_DIR / (label_file.stem + ".jpg")
    if not img_file.exists():
        issues.append(f"MISSING IMAGE: {label_file.name}")
        continue

    lines = label_file.read_text().strip().splitlines()
    if not lines:
        issues.append(f"EMPTY LABEL: {label_file.name}")
        continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append(f"BAD FORMAT: {label_file.name} → '{line}'")
            continue
        cls, cx, cy, w, h = parts
        if not (0 <= float(cx) <= 1 and 0 <= float(cy) <= 1):
            issues.append(f"OUT OF BOUNDS: {label_file.name}")

if issues:
    print(f"{len(issues)} issues found:")
    for i in issues: print(f"  {i}")
else:
    total = len(list(LABEL_DIR.glob("*.txt")))
    print(f"All {total} label files look clean.")