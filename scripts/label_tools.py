from pathlib import Path
from collections import Counter

def audit_labels(label_dir, name):
    txts = [f for f in Path(label_dir).glob("*.txt") if f.name != "classes.txt"]
    class_counts = Counter()
    for f in txts:
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    class_counts[int(line.split()[0])] += 1
                except ValueError:
                    pass
    print(f"\n{name} ({len(txts)} files):")
    for cls, cnt in sorted(class_counts.items()):
        names = ["mouse","water_critical","water_low","water_ok","water_full",
                 "food_critical","food_low","food_ok","food_full",
                 "bedding_worst","bedding_bad","bedding_ok","bedding_perfect"]
        label = names[cls] if cls < len(names) else "unknown"
        print(f"  class {cls:2d} ({label:20s}): {cnt:4d}")

audit_labels("dataset/source/labels", "source/labels")