import os
from collections import Counter

# 👉 CHANGE THIS PATH
LABELS_DIR = "dataset\original_labels_9class"

# Class names (must match your classes.txt)
CLASS_NAMES = {
    0: "mouse",
    1: "water_critical",
    2: "water_low",
    3: "water_ok",
    4: "water_full",
    5: "food_critical",
    6: "food_low",
    7: "food_ok",
    8: "food_full",
}

def scan_labels():
    counter = Counter()
    total_files = 0

    for file in os.listdir(LABELS_DIR):
        if file.endswith(".txt"):
            total_files += 1
            path = os.path.join(LABELS_DIR, file)

            with open(path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    counter[class_id] += 1

    print("\n📊 CLASS DISTRIBUTION:\n")

    for i in range(9):
        print(f"{i} ({CLASS_NAMES[i]}): {counter[i]}")

    print("\n🚨 MISSING CLASSES:\n")

    missing = []
    for i in range(9):
        if counter[i] == 0:
            missing.append(CLASS_NAMES[i])

    if missing:
        for m in missing:
            print(f"❌ {m}")
    else:
        print("✅ No missing classes")

    print("\n🍽️ FOOD LABEL CHECK:\n")

    food_ids = [5, 6, 7, 8]
    food_count = sum(counter[i] for i in food_ids)

    if food_count == 0:
        print("❌ NO FOOD LABELS FOUND in dataset")
    else:
        print(f"✅ Food labels found: {food_count} instances")

    print(f"\n📁 Total label files scanned: {total_files}")

if __name__ == "__main__":
    scan_labels()