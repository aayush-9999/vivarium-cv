# Vivarium CV

Production computer-vision system for automated vivarium cage monitoring —
mouse detection, water level tracking, and food level tracking via a
hybrid YOLOX + PSPNet inference pipeline.

---

## Architecture Overview

```
Camera Frame
     │
     ▼
YOLOPSPPipeline  (BACKEND=yolo_psp)   ← default, production
     ├── YOLOXDetector   → mouse count + container bounding boxes
     └── LevelEstimator  → PSPNet water % + PSPNet food %
             │
             ▼
       DetectionResult
             │
     ┌───────┴───────┐
     ▼               ▼
FastAPI REST      PostgreSQL
  /infer          cage_readings
  /cages              alerts
  /alerts
```

Backends are swappable via `.env` — see [Backends](#backends).

---

## Repository Layout

```
vivarium-cv/
│
├── api/                        # FastAPI application
│   ├── main.py                 # App factory, lifespan, router registration
│   ├── middleware.py           # CORS + request logging
│   └── routes/
│       ├── infer.py            # POST /infer  — frame upload → DetectionResult
│       └── cages.py            # GET/POST /cages, /alerts
│
├── core/                       # Shared contracts — never import upward
│   ├── schemas.py              # Pydantic: DetectionResult, LevelReading, BoundingBox
│   ├── config.py               # All constants: class map, ROI zones, thresholds
│   ├── exceptions.py           # Project exception hierarchy
│   ├── base_detector.py        # ABC for all detector backends
│   └── base_preprocessor.py   # ABC for all preprocessors
│
├── db/                         # Database layer
│   ├── models.py               # SQLAlchemy: CageReading, Alert
│   ├── crud.py                 # Async read/write helpers
│   └── session.py              # Engine, session factory, create_tables()
│
├── detectors/                  # One sub-package per detector backend
│   └── yolo/
│       ├── __init__.py
│       ├── yolo_detector.py    # YOLOXDetector — loads weights, runs inference
│       └── postprocessor.py    # Raw YOLOX output → DetectionResult
│
├── preprocessing/              # Frame preparation (shared across backends)
│   ├── frame_preprocessor.py  # Letterbox resize, normalize, blob
│   ├── background_subtractor.py# Reference-frame motion detection
│   └── roi_manager.py         # Named ROI zone cropping
│
├── pipeline/                   # Inference orchestration
│   ├── pipeline_factory.py    # get_pipeline() / get_orchestrator() — backend selector
│   ├── orchestrator.py        # VivariumOrchestrator — all workflow entry points
│   ├── yolo_pipeline.py       # Pure YOLOX pipeline (BACKEND=yolo)
│   └── yolo_psp_pipeline.py   # Hybrid YOLOX + PSPNet (BACKEND=yolo_psp)
│
├── segmentation/               # PSPNet level estimation
│   ├── models/
│   │   ├── pspnet.py           # PSPNet architecture (ResNet backbone + PPM)
│   │   └── level_estimator.py  # LevelEstimator — mask → fill %
│   ├── datasets/
│   │   └── level_dataset.py    # LevelSegDataset + make_dataloaders()
│   └── trainers/
│       └── psp_trainer.py      # PSPTrainer — train water or food model
│
├── scripts/                    # One-off and pipeline CLI tools
│   ├── gdino_label_originals.py# Auto-label originals with Grounding DINO
│   ├── augment.py              # Image + label augmentation (9-class)
│   ├── labelme_to_mask.py      # LabelMe JSON → PSPNet segmentation masks
│   ├── split_dataset.py        # Train/val split
│   ├── train.py                # YOLOX training entry point
│   ├── label_tools.py          # verify / clean / dedup / fix-classes
│   ├── psp_test.py             # Evaluate PSPNet vs LabelMe ground truth
│   └── test_psp_inference.py   # Single-image PSPNet smoke test
│
├── exps/
│   └── vivarium_yolox_tiny.py  # YOLOX experiment config (tiny model, 9 classes)
│
├── docker/
│   └── docker-compose.yml      # app + postgres services
│
├── dataset/                    # ← gitignored, local only
│   ├── original/               # Raw camera frames + LabelMe JSONs
│   ├── original_empty/         # Raw frames where hopper/bottle is empty/critical
│   ├── augmented/              # Output of scripts/augment.py
│   │   ├── images/
│   │   ├── labels/
│   │   └── meta/
│   ├── split/                  # Output of scripts/split_dataset.py
│   │   ├── train/
│   │   └── val/
│   ├── coco/                   # Output of migration/convert_to_coco.py
│   │   ├── train.json
│   │   ├── val.json
│   │   └── images/
│   └── segmentation/           # PSPNet training data
│       ├── water/
│       │   ├── images/         # 640×640 letterboxed frames
│       │   └── masks/          # Single-channel PNGs (class IDs 0-3)
│       └── food/
│           ├── images/         # Tight hopper crops (224×224)
│           └── masks/
│
├── models/                     # ← gitignored, local only
│   ├── yolo/
│   │   └── best.pt             # YOLOX trained weights
│   └── psp/
│       ├── water_best.pth      # PSPNet water model
│       └── food_best.pth       # PSPNet food model
│
├── runs/                       # ← gitignored, training outputs
│   ├── yolox/
│   └── pspnet/
│       ├── water/
│       └── food/
│
├── .env.example
├── requirements.txt
├── README.md                   # This file
└── pipeline.md                 # Full workflow + API reference
```

---

## Quick Start

### 1. Install

```bash
git clone <repo>
cd vivarium-cv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set DB_URL, YOLO_WEIGHTS, PSP_WATER_WEIGHTS, PSP_FOOD_WEIGHTS
```

### 3. Start services

```bash
docker compose -f docker/docker-compose.yml up -d   # PostgreSQL
uvicorn api.main:app --reload                        # API server
```

### 4. Test inference

```bash
curl -X POST http://localhost:8000/infer \
  -F "cage_id=cage_01" \
  -F "frame=@cage_frame.jpg"
```

Interactive API docs: `http://localhost:8000/docs`

---

## Backends

Set `BACKEND` in `.env`:

| Value | Description | Models needed |
|-------|-------------|---------------|
| `yolo` | Pure YOLOX — 4-bucket discrete levels | `YOLO_WEIGHTS` |
| `yolo_psp` | YOLOX bbox + PSPNet continuous % (**production default**) | `YOLO_WEIGHTS` + `PSP_WATER_WEIGHTS` + `PSP_FOOD_WEIGHTS` |
| `ssd` | SSD MobileNet (legacy) | `SSD_WEIGHTS` |

---

## Environment Variables

```bash
# .env.example

BACKEND=yolo_psp

# Database
DB_URL=postgresql+asyncpg://postgres:password@db:5432/vivarium

# YOLOX (9-class detector)
YOLO_WEIGHTS=models/yolo/best.pt
YOLO_DEVICE=cpu                     # or "0" for GPU

# PSPNet (level estimation — only needed for BACKEND=yolo_psp)
PSP_WATER_WEIGHTS=models/psp/water_best.pth
PSP_FOOD_WEIGHTS=models/psp/food_best.pth
PSP_BACKBONE=resnet50               # resnet18 | resnet50 | resnet101

# Camera
CAMERA_RTSP_URL=rtsp://camera-ip/stream

# Alerts
ALERT_WEBHOOK_URL=https://your-webhook-url
```

---

## 9-Class Detection Scheme

```
Class  Name              Type    Status range
─────  ────────────────  ──────  ────────────
  0    mouse             mouse   —
  1    water_critical    water   0 – 15 %
  2    water_low         water   15 – 35 %
  3    water_ok          water   35 – 80 %
  4    water_full        water   80 – 100 %
  5    food_critical     food    0 – 15 %
  6    food_low          food    15 – 35 %
  7    food_ok           food    35 – 80 %
  8    food_full         food    80 – 100 %
```

In `BACKEND=yolo_psp` mode the YOLOX class ID is used only for bbox location.
The actual fill percentage comes from PSPNet as a continuous 0–100 value.

---

## PSPNet Mask Classes

```
Water model:               Food model:
  0 = background             0 = background
  1 = bottle_wall            1 = hopper_frame
  2 = water_fill  ← measured 2 = food_pellets ← measured
  3 = empty_air   ← measured 3 = empty_space  ← measured
```

---

## Development

```bash
# Run all label integrity checks
python scripts/label_tools.py verify \
    --img-dir   dataset/original \
    --label-dir dataset/augmented/labels

# Evaluate PSPNet against LabelMe ground truth
python scripts/psp_test.py --only food

# Single frame smoke test
python scripts/test_psp_inference.py --image dataset/original/frame.jpg
```

See [pipeline.md](pipeline.md) for the full training and data-prep workflow.

---

## Files Intentionally Excluded from Version Control

| Path | Reason |
|------|--------|
| `dataset/` | Large binary files — store on shared drive or DVC |
| `models/` | Large binary files — store on shared drive or DVC |
| `runs/` | Training artifacts |
| `YOLOX_outputs/` | Training artifacts — use tensorboard locally |
| `migration/` | One-time conversion scripts — archived after use |