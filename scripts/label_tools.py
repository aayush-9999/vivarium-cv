# # scripts/verify_source_labels.py
# """
# Quick sanity check on dataset/source/labels before augmentation.
# Prints class distribution and flags any out-of-range IDs.
# """
# from pathlib import Path
# from collections import Counter

# VALID_CLASS_IDS = set(range(13))
# CLASS_NAMES = [
#     "mouse",
#     "water_critical", "water_low", "water_ok", "water_full",
#     "food_critical",  "food_low",  "food_ok",  "food_full",
#     "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
# ]

# label_dir = Path("dataset/source/labels")
# label_files = list(label_dir.glob("*.txt"))
# print(f"Found {len(label_files)} label files in {label_dir}\n")

# counter = Counter()
# bad_files = []

# for lf in sorted(label_files):
#     for line in lf.read_text(encoding="utf-8").strip().splitlines():
#         parts = line.strip().split()
#         if len(parts) != 5:
#             continue
#         cls = int(parts[0])
#         if cls not in VALID_CLASS_IDS:
#             bad_files.append((lf.name, cls))
#         else:
#             counter[cls] += 1

# print("Class distribution in source/labels:")
# for cls_id in range(13):
#     print(f"  {cls_id:2d}  {CLASS_NAMES[cls_id]:<20} : {counter.get(cls_id, 0)} boxes")

# if bad_files:
#     print(f"\n❌ FOUND {len(bad_files)} INVALID CLASS IDs — fix before proceeding!")
#     for fname, cls in bad_files[:20]:
#         print(f"   {fname}: class {cls}")
# else:
#     print(f"\n✅ All class IDs valid (0–12). Proceed to Step 2.")


# # scripts/verify_augmented_labels.py
# from pathlib import Path
# from collections import Counter

# CLASS_NAMES = [
#     "mouse",
#     "water_critical", "water_low", "water_ok", "water_full",
#     "food_critical",  "food_low",  "food_ok",  "food_full",
#     "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
# ]

# label_dir = Path("dataset/augmented/labels")
# label_files = [f for f in label_dir.glob("*.txt") if f.name != "classes.txt"]
# print(f"Found {len(label_files)} augmented label files\n")

# counter = Counter()
# bad_files = []
# empty_files = 0

# for lf in label_files:
#     content = lf.read_text(encoding="utf-8").strip()
#     if not content:
#         empty_files += 1
#         continue
#     for line in content.splitlines():
#         parts = line.strip().split()
#         if len(parts) != 5:
#             continue
#         cls = int(parts[0])
#         if cls not in range(13):
#             bad_files.append((lf.name, cls))
#         else:
#             counter[cls] += 1

# print("Class distribution in augmented/labels:")
# for cls_id in range(13):
#     print(f"  {cls_id:2d}  {CLASS_NAMES[cls_id]:<20} : {counter.get(cls_id, 0)} boxes")

# print(f"\nEmpty label files (no annotations): {empty_files}")
# if bad_files:
#     print(f"❌ FOUND {len(bad_files)} INVALID CLASS IDs!")
#     for fname, cls in bad_files[:10]:
#         print(f"   {fname}: class {cls}")
# else:
#     print("✅ All augmented class IDs valid (0–12). Proceed to Step 3.")


# scripts/verify_coco.py
import json
from pathlib import Path
from collections import Counter

CLASS_NAMES = [
    "mouse",
    "water_critical", "water_low", "water_ok", "water_full",
    "food_critical",  "food_low",  "food_ok",  "food_full",
    "bedding_worst",  "bedding_bad", "bedding_ok", "bedding_perfect",
]

for split in ["train", "val"]:
    path = Path(f"dataset/coco/{split}.json")
    data = json.loads(path.read_text())

    categories = {c["id"]: c["name"] for c in data["categories"]}
    ann_counter = Counter(a["category_id"] for a in data["annotations"])

    print(f"\n{'='*50}")
    print(f"{split}.json")
    print(f"  Images      : {len(data['images'])}")
    print(f"  Annotations : {len(data['annotations'])}")
    print(f"  Categories  : {len(categories)}  (expected 13)")

    print(f"\n  Class distribution:")
    for cls_id in range(13):
        name  = categories.get(cls_id, "MISSING")
        count = ann_counter.get(cls_id, 0)
        flag  = "⚠️ " if count == 0 else "  "
        print(f"  {flag}{cls_id:2d}  {name:<20} : {count} boxes")

    # Check all 13 categories exist
    missing = [i for i in range(13) if i not in categories]
    if missing:
        print(f"\n❌ Missing category IDs: {missing}")
    else:
        print(f"\n✅ All 13 categories present.")
    
    # Check no out-of-range category IDs
    bad_cats = [a["category_id"] for a in data["annotations"] if a["category_id"] not in range(13)]
    if bad_cats:
        print(f"❌ Out-of-range category IDs in annotations: {set(bad_cats)}")
    else:
        print(f"✅ All annotation category IDs are in range 0–12.")