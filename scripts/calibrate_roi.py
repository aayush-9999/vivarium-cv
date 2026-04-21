"""
scripts/calibrate_roi.py
========================
Interactive ROI calibration tool.

Run this ONCE on a representative cage image.
Click and drag to define each zone. Prints the correct ROI_ZONES dict
for core/config.py when done.

Usage:
    python scripts/calibrate_roi.py --image dataset/original/image.png

Controls:
    - Click + drag to draw a rectangle for each zone
    - Press SPACE or ENTER to confirm current zone and move to next
    - Press R to redo current zone
    - Press Q to quit early
    - Zones collected in order: jug → hopper → floor

Output:
    Prints updated ROI_ZONES dict ready to paste into core/config.py
    Also saves a debug image: runs/roi_debug.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Zone definitions (order matters) ─────────────────────────────────────────
ZONES_TO_DEFINE = [
    ("jug",    (255, 180,  0),  "Draw rectangle around the WATER BOTTLE / JUG"),
    ("hopper", (0,   200, 80),  "Draw rectangle around the FOOD HOPPER"),
    ("floor",  (0,   120, 255), "Draw rectangle around the CAGE FLOOR (mouse area)"),
]

# ── Globals for mouse callback ────────────────────────────────────────────────
_drawing   = False
_ix, _iy   = -1, -1
_rect      = None   # (x, y, w, h) confirmed rectangle


def _mouse_cb(event, x, y, flags, param):
    global _drawing, _ix, _iy, _rect
    canvas: np.ndarray = param["canvas"]
    base:   np.ndarray = param["base"]

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _ix, _iy = x, y
        _rect    = None

    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        tmp = base.copy()
        cv2.rectangle(tmp, (_ix, _iy), (x, y), param["color"], 2)
        canvas[:] = tmp

    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        x1, y1 = min(_ix, x), min(_iy, y)
        x2, y2 = max(_ix, x), max(_iy, y)
        w = x2 - x1
        h = y2 - y1
        if w > 5 and h > 5:
            _rect = (x1, y1, w, h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), param["color"], 2)


def _letterbox(img: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Resize image to target×target with letterboxing. Returns (img, scale, pad_l, pad_t)."""
    h, w = img.shape[:2]
    scale  = target / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_t   = (target - new_h) // 2
    pad_l   = (target - new_w) // 2
    canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized
    return canvas, scale, pad_l, pad_t


def calibrate(image_path: Path) -> dict:
    global _rect

    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    # Resize to 640×640 — same as pipeline
    img_640, scale, pad_l, pad_t = _letterbox(img_raw, 640)

    results = {}
    debug   = img_640.copy()

    win = "ROI Calibrator  [drag to draw | SPACE=confirm | R=redo | Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 800)

    for zone_name, color, instruction in ZONES_TO_DEFINE:
        print(f"\n{'─'*60}")
        print(f"  Zone: {zone_name.upper()}")
        print(f"  → {instruction}")
        print(f"  Controls: drag to draw  |  SPACE/ENTER=confirm  |  R=redo")
        print(f"{'─'*60}")

        _rect    = None
        base     = debug.copy()

        # Overlay instruction text
        display = base.copy()
        cv2.putText(display, f"Zone: {zone_name.upper()} — {instruction}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display, f"Zone: {zone_name.upper()} — {instruction}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(display, "SPACE/ENTER = confirm   R = redo   Q = quit",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        canvas = display.copy()
        param  = {"canvas": canvas, "base": display.copy(), "color": color}
        cv2.setMouseCallback(win, _mouse_cb, param)

        while True:
            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (32, 13):   # SPACE or ENTER
                if _rect is not None:
                    x, y, w, h = _rect
                    results[zone_name] = (x, y, w, h)
                    print(f"  ✓ Confirmed: ({x}, {y}, {w}, {h})")

                    # Draw confirmed zone on debug overlay for next zones
                    cv2.rectangle(debug, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(debug, zone_name, (x+4, y+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break
                else:
                    print("  [WARN] No rectangle drawn yet. Draw first, then press SPACE.")

            elif key == ord('r') or key == ord('R'):
                _rect  = None
                canvas[:] = display.copy()
                print("  ↩ Redo — draw again.")

            elif key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quit early — partial results printed below.")
                cv2.destroyAllWindows()
                return results

    cv2.destroyAllWindows()
    return results


def print_config(zones: dict, cage_type: str = "default") -> None:
    print(f"\n{'═'*60}")
    print("  Paste this into core/config.py → ROI_ZONES")
    print(f"{'═'*60}\n")
    print("ROI_ZONES = {")
    print(f'    "{cage_type}": {{')
    for name, (x, y, w, h) in zones.items():
        print(f'        "{name}": ({x:4d}, {y:4d}, {w:4d}, {h:4d}),')
    print("    },")
    print("}\n")


def save_debug(image_path: Path, zones: dict) -> None:
    img_raw = cv2.imread(str(image_path))
    img_640, _, _, _ = _letterbox(img_raw, 640)
    colors = {
        "jug":    (255, 180,  0),
        "hopper": (0,   200, 80),
        "floor":  (0,   120, 255),
    }
    for name, (x, y, w, h) in zones.items():
        c = colors.get(name, (200, 200, 200))
        cv2.rectangle(img_640, (x, y), (x+w, y+h), c, 2)
        cv2.putText(img_640, name.upper(), (x+4, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

    out = Path("runs/roi_debug.jpg")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img_640, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Debug image saved → {out}")


def main():
    ap = argparse.ArgumentParser(description="Interactive ROI calibration for vivarium pipeline")
    ap.add_argument("--image", type=Path, required=True,
                    help="Path to a representative cage image (any resolution)")
    ap.add_argument("--cage-type", type=str, default="default",
                    help="Cage type key for ROI_ZONES dict (default: 'default')")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print(f"\nCalibrating ROI zones on: {args.image}")
    print("Image will be resized to 640×640 (same as pipeline)\n")

    zones = calibrate(args.image)

    if zones:
        print_config(zones, args.cage_type)
        save_debug(args.image, zones)
    else:
        print("[WARN] No zones were defined.")


if __name__ == "__main__":
    main()