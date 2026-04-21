"""
scripts/convert_labels_3to9.py
===============================
Converts 3-class GDINO labels → 9-class labels.

3-class (GDINO output):
    0 = mouse
    1 = water_container
    2 = food_area

9-class (training target):
    0 = mouse          (unchanged)
    1 = water_critical
    2 = water_low
    3 = water_ok       ← water_container maps here by default
    4 = water_full
    5 = food_critical
    6 = food_low
    7 = food_ok        ← food_area maps here by default
    8 = food_full

Since GDINO only detects container presence (not fill level), we default
water → water_ok (3) and food → food_ok (7).
This is fine because:
  - Your 27 real images likely show normal/ok fill levels anyway
  - The model will learn container position from these
  - Fill level classes will be learned from augmented data later
    (or you can manually relabel a few critical/low examples)

Usage:
    python scripts/convert_labels_3to9.py
    python scripts/convert_labels_3to9.py \
        --src dataset/original_labels \
        --dst dataset/original_labels_9class
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Default mapping: old class → new class
CLASS_MAP = {
    0: 0,   # mouse          → mouse
    1: 3,   # water_container → water_ok
    2: 7,   # food_area       → food_ok
}

CLASS_NAMES_9 = {
    0: "mouse",
    1: "water_critical", 2: "water_low", 3: "water_ok", 4: "water_full",
    5: "food_critical",  6: "food_low",  7: "food_ok",  8: "food_full",
}


def convert_file(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """Convert one label file. Returns (lines_in, lines_out)."""
    try:
        with open(str(src_path), "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
    except (FileNotFoundError, OSError):
        return 0, 0

    if not lines:
        dst_path.write_text("", encoding="utf-8")
        return 0, 0

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_cls = int(parts[0])
        new_cls = CLASS_MAP.get(old_cls, old_cls)  # pass through if unknown
        out_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    dst_path.write_text("\n".join(out_lines), encoding="utf-8")
    return len(lines), len(out_lines)


def main(src: Path, dst: Path, dry_run: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    # Write classes.txt
    (dst / "classes.txt").write_text(
        "\n".join(CLASS_NAMES_9[i] for i in range(9)) + "\n",
        encoding="utf-8"
    )

    label_files = [
        f for f in src.glob("*.txt")
        if f.name != "classes.txt"
    ]

    if not label_files:
        print(f"[ERROR] No .txt files found in {src}")
        return

    print(f"Converting {len(label_files)} label files")
    print(f"  water_container (1) → water_ok (3)")
    print(f"  food_area       (2) → food_ok  (7)")
    print(f"  mouse           (0) → mouse    (0)  (unchanged)")
    if dry_run:
        print("  [DRY RUN — no files written]\n")
    else:
        print(f"  Output → {dst}\n")

    total_in = total_out = 0
    for src_file in sorted(label_files):
        dst_file = dst / src_file.name
        n_in, n_out = convert_file(src_file, dst_file) if not dry_run \
                      else (sum(1 for l in src_file.read_text().splitlines() if l.strip()), 0)
        total_in  += n_in
        total_out += n_out
        print(f"  {src_file.name:<50}  {n_in} → {n_out} lines")

    print(f"""
{'─'*55}
{'DRY RUN' if dry_run else 'Done'}
  Files    : {len(label_files)}
  Lines in : {total_in}
  Lines out: {total_out}
  Output   : {dst}
{'─'*55}

Next steps:
  python scripts/split_dataset.py \\
      --img-dir  dataset/original \\
      --label-dir {dst} \\
      --out      dataset/split

  python scripts/train.py
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",     type=Path, default=Path("dataset/original_labels"),
                    help="Source 3-class label folder")
    ap.add_argument("--dst",     type=Path, default=Path("dataset/original_labels_9class"),
                    help="Output 9-class label folder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.src, args.dst, args.dry_run)  