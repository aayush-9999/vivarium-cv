# scripts/verify_labels.py
"""
Label verification — usable as a standalone script OR via the orchestrator.

    # As a script
    python scripts/verify_labels.py

    # Via orchestrator
    from pipeline.pipeline_factory import get_orchestrator
    issues = get_orchestrator().verify_labels()
"""

from pathlib import Path
import cv2

IMG_DIR   = Path("dataset/augmented/images")
LABEL_DIR = Path("dataset/augmented/labels")


def _collect_issues(
    img_dir:   Path = IMG_DIR,
    label_dir: Path = LABEL_DIR,
) -> list[str]:
    """
    Core verification logic.
    Returns a list of issue strings (empty list = all clean).
    """
    issues = []

    for label_file in label_dir.glob("*.txt"):
        img_file = img_dir / (label_file.stem + ".jpg")

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
            _, cx, cy, w, h = parts
            if not (0 <= float(cx) <= 1 and 0 <= float(cy) <= 1):
                issues.append(f"OUT OF BOUNDS: {label_file.name}")

    return issues


def main(
    img_dir:   Path = IMG_DIR,
    label_dir: Path = LABEL_DIR,
) -> None:
    issues = _collect_issues(img_dir, label_dir)

    if issues:
        print(f"{len(issues)} issues found:")
        for i in issues:
            print(f"  {i}")
    else:
        total = len(list(label_dir.glob("*.txt")))
        print(f"All {total} label files look clean.")


if __name__ == "__main__":
    main()