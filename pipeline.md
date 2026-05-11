# Vivarium CV — Pipeline Guide

Complete reference for data preparation, training, and inference workflows.
All workflows are accessible via the Python orchestrator API or direct CLI scripts.

---

## Table of Contents

1. [Orchestrator Quick Reference](#orchestrator-quick-reference)
2. [Full Training Workflow](#full-training-workflow)
   - [Step 1 — Label Originals](#step-1--label-originals-with-grounding-dino)
   - [Step 2 — Augment](#step-2--augment-dataset)
   - [Step 3 — Clean + Dedup Labels](#step-3--clean--dedup-labels)
   - [Step 4 — Split Dataset](#step-4--split-dataset)
   - [Step 5 — Train YOLOX](#step-5--train-yolox)
   - [Step 6 — Generate PSPNet Masks](#step-6--generate-pspnet-masks)
   - [Step 7 — Train PSPNet](#step-7--train-pspnet)
3. [Inference](#inference)
4. [HTTP API](#http-api)
5. [Config Overrides](#config-overrides)
6. [Adding a New Cage Type](#adding-a-new-cage-type)
7. [Evaluation](#evaluation)
8. [Backends](#backends)

---

## Orchestrator Quick Reference

```python
from pipeline.pipeline_factory import get_orchestrator

orch = get_orchestrator()
```

| Method | What it does |
|--------|-------------|
| `orch.label_originals(propagate=True)` | Auto-label with Grounding DINO |
| `orch.augment(n=50)` | Generate augmented variants |
| `orch.clean_food_labels()` | Remove oversized food boxes (bedding FPs) |
| `orch.dedup_labels()` | NMS dedup + enforce 1 water box |
| `orch.split_dataset(train_ratio=0.85)` | Train/val split |
| `orch.train(epochs=100, device="0")` | Train YOLOX |
| `orch.validate()` | Validate on val split |
| `orch.infer(frame, cage_id)` | Single frame inference |
| `orch.infer_from_path(path, cage_id)` | Inference from file path |
| `orch.calibrate_roi(image, cage_type)` | Interactive ROI calibration |
| `orch.verify_labels()` | Check label/image integrity |
| `orch.run_data_pipeline(n_augments=50)` | label → augment → clean → dedup → split |
| `orch.run_full_pipeline(n_augments=50, epochs=100)` | Everything end-to-end |

---

## Full Training Workflow

### Step 1 — Label Originals with Grounding DINO

Runs on images in `dataset/original/`.
Writes YOLO `.txt` labels to `dataset/augmented/labels/`.
Saves debug visualisations to `dataset/augmented/debug_gdino_orig/`.

```python
orch.label_originals(propagate=True)
```

Or via CLI:
```powershell
python scripts/gdino_label_originals.py --propagate
```

Threshold tuning (lower = more detections, higher = fewer false positives):
```powershell
python scripts/gdino_label_originals.py `
    --propagate `
    --mouse-thresh 0.22 `
    --container-thresh 0.30 `
    --food-thresh 0.25
```

**Review `dataset/augmented/debug_gdino_orig/` before continuing.**
Coloured boxes = kept. Grey boxes = rejected by size filter.

---

### Step 2 — Augment Dataset

Generates N augmented variants per original image with synthetic fill overlays.
Water/food class labels are assigned from the sampled fill fraction — not from GDINO.

```python
orch.augment(n=50)
```

Or via CLI:
```powershell
python scripts/augment.py `
    --src dataset/original `
    --src-labels dataset/augmented/labels `
    --dst dataset/augmented `
    --n 50
```

**Augmenting empty/critical frames separately** (recommended when you have 4+ empty originals):
```powershell
# Copy your empty originals into dataset/original_empty/ first, then:
python scripts/augment.py `
    --src dataset/original_empty `
    --src-labels dataset/augmented/labels `
    --dst dataset/augmented `
    --n 150
```

Output structure:
```
dataset/augmented/
    images/     ← augmented JPEGs
    labels/     ← YOLO .txt labels (9-class)
    meta/       ← per-image augmentation params (JSON)
```

---

### Step 3 — Clean + Dedup Labels

**Clean** removes food boxes that are too large (bedding false positives):
```python
# Preview only — no writes
orch.clean_food_labels(dry_run=True)

# Apply
orch.clean_food_labels()
```

**Dedup** applies NMS and enforces exactly one water box in the top-right quadrant:
```python
orch.dedup_labels(dry_run=True)  # preview
orch.dedup_labels()              # apply
```

Via CLI:
```powershell
python scripts/label_tools.py clean `
    --label-dir dataset/augmented/labels `
    --dry-run

python scripts/label_tools.py dedup `
    --label-dir dataset/augmented/labels `
    --dry-run
```

**Verify** label integrity after cleaning:
```python
issues = orch.verify_labels()
for issue in issues:
    print(issue)
```

---

### Step 4 — Split Dataset

```python
orch.split_dataset(train_ratio=0.85)
```

Or:
```powershell
python scripts/split_dataset.py `
    --img-dir   dataset/augmented/images `
    --label-dir dataset/augmented/labels `
    --out       dataset/split `
    --train-ratio 0.85
```

Output:
```
dataset/split/
    train/
        images/
        labels/
    val/
        images/
        labels/
```

Before training YOLOX, convert to COCO format:
```powershell
python migration/convert_to_coco.py `
    --split-dir dataset/split `
    --output-dir dataset/coco
```

---

### Step 5 — Train YOLOX

```python
orch.train(epochs=100, device="0")   # device="cpu" if no GPU
```

Or directly via YOLOX trainer:
```powershell
python scripts/train.py `
    -f exps/vivarium_yolox_tiny.py `
    -d 1 `
    -b 16
```

Weights saved to `YOLOX_outputs/vivarium_yolox_tiny/`.
Copy best checkpoint to `models/yolo/best.pt` and update `.env`:
```
YOLO_WEIGHTS=models/yolo/best.pt
```

Validate:
```python
orch.validate()
```

---

### Step 6 — Generate PSPNet Masks

PSPNet requires pixel-level segmentation masks.
You must annotate originals in **LabelMe** using these label names exactly:

**Water bottle:**
- `bottle_wall` — outline the entire bottle
- `water_fill` — the liquid region (below meniscus)
- `empty_air` — the air gap (above meniscus)

**Food hopper:**
- `hopper_frame` — the wire frame
- `food_pellets` — the food region
- `empty_space` — empty hopper area

Once annotated, convert JSONs to masks:
```powershell
python scripts/labelme_to_mask.py
```

Output:
```
dataset/segmentation/
    water/
        images/    ← full 640×640 frames
        masks/     ← single-channel PNGs, pixel value = class ID
    food/
        images/    ← tight hopper crops (224×224)
        masks/
```

Verify mask class distribution before training:
```powershell
python -c "
import cv2, numpy as np
from pathlib import Path

for container in ['water', 'food']:
    print(f'\n=== {container} ===')
    mask_dir = Path(f'dataset/segmentation/{container}/masks')
    for p in sorted(mask_dir.glob('*.png')):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        fill_pct  = (m == 2).sum() / m.size * 100
        empty_pct = (m == 3).sum() / m.size * 100
        print(f'  {p.name}: fill={fill_pct:.1f}%  empty={empty_pct:.1f}%')
"
```

---

### Step 7 — Train PSPNet

Train water model:
```powershell
python segmentation/trainers/psp_trainer.py `
    --container water `
    --data-root dataset/segmentation `
    --output-dir runs/pspnet/water `
    --backbone resnet50 `
    --epochs 80 `
    --batch-size 4 `
    --device cuda
```

Train food model:
```powershell
python segmentation/trainers/psp_trainer.py `
    --container food `
    --data-root dataset/segmentation `
    --output-dir runs/pspnet/food `
    --backbone resnet50 `
    --epochs 80 `
    --batch-size 4 `
    --device cuda
```

Copy best weights and update `.env`:
```
PSP_WATER_WEIGHTS=runs/pspnet/water/best.pth
PSP_FOOD_WEIGHTS=runs/pspnet/food/best.pth
PSP_BACKBONE=resnet50
```

---

## Inference

### Python API

```python
from pipeline.pipeline_factory import get_orchestrator
import cv2

orch = get_orchestrator()

# From file
result = orch.infer_from_path("cage_frame.jpg", cage_id="cage_01")

# From frame array
frame = cv2.imread("cage_frame.jpg")
result = orch.infer(frame, cage_id="cage_01")

print(result.mouse_count)
print(result.water.status)   # OK | LOW | CRITICAL
print(result.water.pct)      # 0.0 – 100.0  (continuous, PSPNet)
print(result.food.status)
print(result.food.pct)
```

### Motion-gated inference (camera loop)

```python
import cv2
from pipeline.pipeline_factory import get_orchestrator

orch = get_orchestrator()
cap  = cv2.VideoCapture(0)

ret, ref_frame = cap.read()
orch.set_reference_frame(ref_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if orch.has_motion(frame):
        result = orch.infer(frame, cage_id="cage_01")
        print(result)
```

### Single-image smoke test

```powershell
python scripts/test_psp_inference.py --image dataset/original/frame.jpg

# Test PSPNet crop only
python scripts/test_psp_inference.py `
    --image dataset/segmentation/water/images/frame.jpg `
    --container water `
    --crop-only `
    --weights runs/pspnet/water/best.pth
```

---

## HTTP API

Start the server:
```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/infer` | Upload a frame, get `DetectionResult` |
| `GET` | `/cages` | Latest reading for every cage |
| `GET` | `/cages/critical` | Cages currently in CRITICAL state |
| `GET` | `/cages/{cage_id}` | Latest reading for one cage |
| `GET` | `/cages/{cage_id}/history` | Reading history (default last 50) |
| `GET` | `/alerts` | All unresolved alerts |
| `GET` | `/alerts/{cage_id}` | Alerts for one cage |
| `POST` | `/alerts/{cage_id}/resolve` | Resolve alerts after restocking |

### Example — upload frame

```bash
curl -X POST http://localhost:8000/infer \
  -F "cage_id=cage_01" \
  -F "frame=@cage_frame.jpg" \
  -F "save_flagged=true"
```

### Example — resolve alert after refilling water

```bash
curl -X POST "http://localhost:8000/alerts/cage_01/resolve?alert_type=water_critical"
```

---

## Config Overrides

```python
from pipeline.pipeline_factory import get_orchestrator
from pipeline.orchestrator import OrchestratorConfig
from pathlib import Path

orch = get_orchestrator(OrchestratorConfig(
    orig_dir      = Path("my_data/raw"),
    aug_dir       = Path("my_data/augmented"),
    aug_n         = 100,
    epochs        = 200,
    device        = "0",        # GPU index
    conf          = 0.40,
    mouse_thresh  = 0.20,
    train_ratio   = 0.85,
))
```

---

## Adding a New Cage Type

```python
orch = get_orchestrator()

# 1. Calibrate ROI zones interactively (OpenCV window)
zones = orch.calibrate_roi("new_cage.jpg", cage_type="type_b")
# Paste printed ROI_ZONES dict into core/config.py

# 2. Run data pipeline with new cage images
orch.run_data_pipeline(n_augments=60)

# 3. Train and validate
orch.train(epochs=120, device="0")
orch.validate()

# 4. Inference with new cage type
from pipeline.yolo_psp_pipeline import YOLOPSPPipeline
pipeline = YOLOPSPPipeline(cage_type="type_b")
result   = pipeline.run(frame, cage_id="cage_b_01")
```

---

## Evaluation

### YOLOX — validate on val split

```python
orch.validate()
# Results written to runs/val_test/results.txt
# Annotated images written to runs/val_test/annotated/
```

### PSPNet — evaluate against LabelMe ground truth

```powershell
# Both water and food
python scripts/psp_test.py

# One container only
python scripts/psp_test.py --only water
python scripts/psp_test.py --only food

# Override weights
python scripts/psp_test.py `
    --water-weights runs/pspnet/water/best.pth `
    --food-weights  runs/pspnet/food/best.pth
```

Output per run:
- `runs/psp_eval/results_water.csv`
- `runs/psp_eval/results_food.csv`
- `runs/psp_eval/visualizations/` — side-by-side GT vs prediction images

Target metrics:
| Metric | Water | Food |
|--------|-------|------|
| Status accuracy | > 90 % | > 85 % |
| MAE | < 10 % | < 15 % |

---

## Backends

Switch via `.env`:

```
BACKEND=yolo_psp   # default — YOLOX + PSPNet continuous levels
BACKEND=yolo       # YOLOX only — discrete 4-bucket levels
BACKEND=ssd        # SSD MobileNet (legacy)
```

**`yolo_psp` fallback chain (per frame):**

```
PSPNet estimate
    │
    ├─ NO_CONTAINER_SENTINEL (-1.0) → fall back to YOLOX class reading
    ├─ pct >= 97% AND YOLOX ≠ OK   → fall back to YOLOX class reading
    ├─ food pct > 40% AND YOLOX = CRITICAL → fall back to YOLOX (bedding confusion)
    └─ Exception                    → fall back to YOLOX class reading (if fallback_to_yolox=True)
```

This means the system is always safe — if PSPNet is not loaded or fails,
YOLOX discrete buckets are used automatically.