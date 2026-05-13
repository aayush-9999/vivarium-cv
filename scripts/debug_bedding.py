# scripts/check_class_balance.py
import json
from pathlib import Path
from collections import Counter

CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

data    = json.loads(Path("dataset/coco/annotations/train.json").read_text())
counter = Counter(a["category_id"] for a in data["annotations"])

print("Training set class distribution:")
for cls_id in range(13):
    count = counter.get(cls_id, 0)
    bar   = "█" * (count // 10)
    flag  = " ⚠️ LOW" if count < 50 else ""
    print(f"  {cls_id:2d}  {CLASS_NAMES[cls_id]:<20} : {count:4d}  {bar}{flag}")