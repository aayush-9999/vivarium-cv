# Vivarium CV — Pipeline Guide

All workflows that previously required running individual scripts are now
unified under a single orchestrator. Scripts still exist as fallback,
but **everything is callable through the pipeline**.

---

## Python API (Recommended)

```python
from pipeline.pipeline_factory import get_orchestrator

orch = get_orchestrator()
```

### Full End-to-End

```python
# Data prep + train + validate in one call
orch.run_full_pipeline(n_augments=50, epochs=100)
```

### Step-by-Step

```python
# 1. Label original images with Grounding DINO
orch.label_originals(propagate=True)

# 2. Augment (50 variants per source image)
orch.augment(n=50)

# 3. Clean oversized food labels (bedding false positives)
orch.clean_food_labels()

# 4. Deduplicate labels
orch.dedup_labels()

# 5. Split into train/val
orch.split_dataset(train_ratio=0.85)

# 6. Train
orch.train(epochs=100, device="0")   # device="cpu" if no GPU

# 7. Validate
orch.validate()
```

### Single Frame Inference

```python
import cv2
result = orch.infer_from_path("cage_frame.jpg", cage_id="cage_01")
print(result.mouse_count)
print(result.water.status)   # OK | LOW | CRITICAL
print(result.food.pct)       # 0.0 – 100.0
```

### Label Utilities

```python
# Preview oversized food label removal (no writes)
orch.clean_food_labels(dry_run=True)

# Merge GDINO labels with existing hand-labelled ones
orch.merge_labels(
    existing_dir=Path("dataset/augmented/labels"),
    new_dir=Path("dataset/augmented/labels_gdino"),
    out_dir=Path("dataset/augmented/labels_merged"),
)

# Verify label/image pairing
issues = orch.verify_labels()

# Bootstrap labels with COCO pretrained YOLOv8n (faster, lower accuracy)
orch.auto_label(debug=True)

# One-time fix for datasets where all labels were class 0
orch.fix_labels(dry_run=True)
```

### ROI Calibration

```python
zones = orch.calibrate_roi("dataset/original/cage.png", cage_type="default")
# Interactive OpenCV window — paste output into core/config.py
```

### Motion Detection

```python
import cv2
cap = cv2.VideoCapture(0)
ret, ref_frame = cap.read()
orch.set_reference_frame(ref_frame)

while True:
    ret, frame = cap.read()
    if orch.has_motion(frame):
        result = orch.infer(frame, cage_id="cage_01")
        print(result)
```

---

## HTTP API

All operations are also exposed as REST endpoints (start the server with `uvicorn api.main:app`):

| Method | Endpoint                    | Description                            |
|--------|-----------------------------|----------------------------------------|
| POST   | `/infer`                    | Inference on uploaded frame            |
| POST   | `/pipeline/infer`           | Inference via orchestrator             |
| POST   | `/pipeline/label`           | GDINO labelling (background)           |
| POST   | `/pipeline/augment`         | Augment dataset (background)           |
| POST   | `/pipeline/clean`           | Clean food labels                      |
| POST   | `/pipeline/dedup`           | Dedup labels                           |
| POST   | `/pipeline/split`           | Split dataset                          |
| POST   | `/pipeline/train`           | Train YOLOv8 (background)              |
| POST   | `/pipeline/validate`        | Validate on val split (background)     |
| GET    | `/pipeline/verify-labels`   | Verify label integrity                 |
| POST   | `/pipeline/run-data`        | Full data prep pipeline (background)   |
| POST   | `/pipeline/run-full`        | End-to-end pipeline (background)       |

Interactive docs at `http://localhost:8000/docs`.

---

## Config Overrides

```python
from pipeline.pipeline_factory import get_orchestrator
from pipeline.orchestrator import OrchestratorConfig
from pathlib import Path

orch = get_orchestrator(OrchestratorConfig(
    orig_dir=Path("my_data/raw"),
    aug_dir=Path("my_data/augmented"),
    aug_n=100,
    epochs=200,
    device="0",        # GPU
    conf=0.40,
    mouse_thresh=0.20,
))
```

---

## Backends

Switch inference backend via `.env`:

```
BACKEND=yolo   # YOLOv8 (default)
BACKEND=ssd    # SSD MobileNet
```

---

## Workflow for a New Cage Type

```python
orch = get_orchestrator()

# 1. Calibrate ROI zones interactively
zones = orch.calibrate_roi("new_cage.jpg", cage_type="type_b")
# Paste printed ROI_ZONES into core/config.py

# 2. Run data pipeline
orch.run_data_pipeline(n_augments=60)

# 3. Train and validate
orch.train(epochs=120, device="0")
orch.validate()

# 4. Inference with new cage type
from pipeline.yolo_pipeline import YOLOPipeline
pipeline = YOLOPipeline(cage_type="type_b")
result = pipeline.run(frame, cage_id="cage_b_01")
```