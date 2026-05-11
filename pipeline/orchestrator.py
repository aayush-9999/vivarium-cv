# pipeline/orchestrator.py
"""
VivariumOrchestrator — single entry-point for all Vivarium CV workflows.

The orchestrator owns two concerns:

1. **Inference** — delegates to InferencePipeline (backend selected by config).
2. **Data & training** — exposes every data-prep, labelling, augmentation,
   training, and validation step as a method.

API / camera code should use get_pipeline() directly.
Notebooks and CLI scripts should use get_orchestrator().

Backend selection
─────────────────
The active inference backend is taken from OrchestratorConfig.backend
(default: CONFIG["backend"] from .env).  Changing it at construction time
is enough — the orchestrator rebuilds the pipeline automatically.

    orch = get_orchestrator(OrchestratorConfig(backend="yolo_psp"))
    result = orch.infer(frame, "cage_01")
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config_loader import CONFIG
from core.exceptions import VivariumCVError
from core.schemas import DetectionResult


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """All tuneable parameters for data-prep and training workflows."""

    # ── Backend ───────────────────────────────────────────────────────────
    backend:      str   = field(default_factory=lambda: CONFIG["backend"])
    cage_type:    str   = "default"

    # ── Paths ─────────────────────────────────────────────────────────────
    orig_dir:     Path  = field(default_factory=lambda: Path("dataset/original"))
    aug_dir:      Path  = field(default_factory=lambda: Path("dataset/augmented"))
    split_dir:    Path  = field(default_factory=lambda: Path("dataset/split"))
    weights:      Path  = field(
        default_factory=lambda: Path(CONFIG["yolox"]["weights"])
    )
    device:       str   = field(default_factory=lambda: CONFIG["device"])

    # ── Augmentation ──────────────────────────────────────────────────────
    aug_n:        int   = 50
    aug_seed:     int   = field(default_factory=lambda: random.randint(0, 99_999))
    img_size:     int   = 640
    jpeg_quality: int   = 90

    # ── GDINO labelling ───────────────────────────────────────────────────
    mouse_thresh:     float = 0.22
    container_thresh: float = 0.30
    food_thresh:      float = 0.28

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int   = 100
    batch:        int   = 16
    train_ratio:  float = 0.85

    # ── Inference ─────────────────────────────────────────────────────────
    conf: float = 0.35
    iou:  float = 0.45

    # ── Label cleaning ────────────────────────────────────────────────────
    max_food_area: float = 0.12
    max_food_w:    float = 0.40
    max_food_h:    float = 0.50
    mouse_nms_iou: float = 0.45
    food_nms_iou:  float = 0.50


# ---------------------------------------------------------------------------
# VivariumOrchestrator
# ---------------------------------------------------------------------------

class VivariumOrchestrator:
    """
    Single entry-point for all Vivarium CV workflows.

    Inference is delegated to InferencePipeline (pipeline/pipeline.py).
    Data-prep and training methods call the relevant scripts directly.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.cfg = config or OrchestratorConfig()
        self._pipeline: Optional[object] = None   # lazy-loaded InferencePipeline

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def set_backend(self, backend: str) -> None:
        """
        Switch the inference backend at runtime.

        Valid values: "yolo", "yolo_psp", "ssd"
        Rebuilds the pipeline on next infer() call.
        """
        valid = {"yolo", "yolo_psp", "ssd"}
        if backend not in valid:
            raise ValueError(f"Unknown backend '{backend}'. Choose from {valid}.")
        self.cfg.backend = backend
        self._pipeline   = None   # force rebuild

    @property
    def active_backend(self) -> str:
        return self.cfg.backend

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
    ) -> DetectionResult:
        """Run inference on a BGR numpy frame."""
        return self._get_pipeline().run(
            frame=frame, cage_id=cage_id, save_flagged=save_flagged
        )

    def infer_from_path(
        self,
        image_path: str | Path,
        cage_id: str,
        save_flagged: bool = False,
    ) -> DetectionResult:
        """Load an image from disk and run inference."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise VivariumCVError(f"Cannot read image: {image_path}")
        return self.infer(img, cage_id=cage_id, save_flagged=save_flagged)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of the frame with all detections drawn."""
        return self._get_pipeline().debug_frame(frame)

    # ------------------------------------------------------------------
    # Motion / camera helpers
    # ------------------------------------------------------------------

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self._get_pipeline().set_reference_frame(frame)

    def has_motion(self, frame: np.ndarray) -> bool:
        return self._get_pipeline().has_motion(frame)

    # ------------------------------------------------------------------
    # Label tools
    # ------------------------------------------------------------------

    def verify_labels(
        self,
        img_dir: Optional[Path] = None,
        label_dir: Optional[Path] = None,
    ) -> list[str]:
        from scripts.label_tools import verify
        return verify(
            img_dir   = img_dir   or (self.cfg.aug_dir / "images"),
            label_dir = label_dir or (self.cfg.aug_dir / "labels"),
        )

    def clean_food_labels(
        self,
        label_dir: Optional[Path] = None,
        max_area: Optional[float] = None,
        max_w: Optional[float] = None,
        max_h: Optional[float] = None,
        dry_run: bool = False,
    ) -> dict:
        from scripts.label_tools import clean_food
        return clean_food(
            label_dir = label_dir or (self.cfg.aug_dir / "labels"),
            max_area  = max_area  or self.cfg.max_food_area,
            max_w     = max_w     or self.cfg.max_food_w,
            max_h     = max_h     or self.cfg.max_food_h,
            dry_run   = dry_run,
        )

    def dedup_labels(
        self,
        label_dir: Optional[Path] = None,
        mouse_iou: Optional[float] = None,
        food_iou: Optional[float] = None,
        dry_run: bool = False,
    ) -> dict:
        from scripts.label_tools import dedup
        return dedup(
            label_dir = label_dir or (self.cfg.aug_dir / "labels"),
            mouse_iou = mouse_iou or self.cfg.mouse_nms_iou,
            food_iou  = food_iou  or self.cfg.food_nms_iou,
            dry_run   = dry_run,
        )

    def fix_labels(
        self,
        label_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> dict:
        from scripts.label_tools import fix_classes
        return fix_classes(
            label_dir = label_dir or (self.cfg.aug_dir / "labels"),
            dry_run   = dry_run,
        )

    # ------------------------------------------------------------------
    # Auto-labelling
    # ------------------------------------------------------------------

    def label_originals(
        self,
        propagate: bool = False,
        mouse_thresh: Optional[float] = None,
        container_thresh: Optional[float] = None,
        food_thresh: Optional[float] = None,
    ) -> dict:
        from scripts.gdino_label_originals import load_model, label_originals as _label, propagate_labels
        thresholds = {
            0: mouse_thresh     or self.cfg.mouse_thresh,
            1: container_thresh or self.cfg.container_thresh,
            2: food_thresh      or self.cfg.food_thresh,
        }
        processor, model = load_model()
        stem_to_boxes = _label(processor, model, thresholds)
        if propagate:
            propagate_labels(stem_to_boxes)
        return stem_to_boxes

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        src: Optional[Path] = None,
        dst: Optional[Path] = None,
        src_labels: Optional[Path] = None,
        img_size: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> None:
        from scripts.augment import main as _augment
        _augment(
            src         = src          or self.cfg.orig_dir,
            dst         = dst          or self.cfg.aug_dir,
            src_labels  = src_labels,
            n           = n            or self.cfg.aug_n,
            seed        = seed         or self.cfg.aug_seed,
            img_size    = img_size     or self.cfg.img_size,
            quality     = jpeg_quality or self.cfg.jpeg_quality,
        )

    # ------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        train_ratio: Optional[float] = None,
        seed: int = 42,
        label_dir: Optional[Path] = None,
        img_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
    ) -> dict[str, int]:
        import shutil

        ratio    = train_ratio or self.cfg.train_ratio
        lbl_dir  = label_dir  or (self.cfg.aug_dir / "labels")
        src_imgs = img_dir    or (self.cfg.aug_dir / "images")
        dst      = out_dir    or self.cfg.split_dir

        random.seed(seed)
        labelled = [p for p in lbl_dir.glob("*.txt") if p.stat().st_size > 0]
        random.shuffle(labelled)
        split_idx  = int(len(labelled) * ratio)
        train_set  = labelled[:split_idx]
        val_set    = labelled[split_idx:]

        for split_name, split_files in [("train", train_set), ("val", val_set)]:
            (dst / split_name / "images").mkdir(parents=True, exist_ok=True)
            (dst / split_name / "labels").mkdir(parents=True, exist_ok=True)
            for lbl in split_files:
                img = src_imgs / (lbl.stem + ".jpg")
                if img.exists():
                    shutil.copy(img, dst / split_name / "images" / img.name)
                    shutil.copy(lbl, dst / split_name / "labels" / lbl.name)

        counts = {"train": len(train_set), "val": len(val_set)}
        print(f"Split → train={counts['train']}  val={counts['val']}  output: {dst}")
        return counts

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        yaml_path: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        device: Optional[str] = None,
        project: Optional[Path] = None,
        run_name: str = "vivarium_v1",
        resume: bool = False,
    ) -> Path:
        """Train the YOLOX detector.  Returns path to best.pt."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise VivariumCVError("ultralytics not installed — cannot train YOLOv8.")

        base_dir  = Path(__file__).resolve().parent.parent
        data_path = yaml_path or (base_dir / "dataset" / "vivarium.yaml")
        proj      = project   or (base_dir / "runs" / "detect")

        model = YOLO("yolov8n.pt")
        model.train(
            data      = str(data_path),
            epochs    = epochs or self.cfg.epochs,
            imgsz     = self.cfg.img_size,
            batch     = batch  or self.cfg.batch,
            device    = device or self.cfg.device,
            workers   = 0,
            cos_lr    = True,
            patience  = 20,
            hsv_h     = 0.015, hsv_s=0.4, hsv_v=0.3,
            fliplr    = 0.5,
            mosaic    = 0.8,
            project   = str(proj),
            name      = run_name,
            exist_ok  = True,
            resume    = resume,
            verbose   = True,
        )
        best_pt = proj / run_name / "weights" / "best.pt"
        print(f"\nTraining complete. Weights: {best_pt}")
        return best_pt

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        weights: Optional[Path] = None,
        val_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Path:
        from scripts.val_test import main as _val
        out = out_dir or Path("runs/val_test")
        _val(
            weights = str(weights or self.cfg.weights),
            val_dir = str(val_dir or (self.cfg.split_dir / "val" / "images")),
            out_dir = str(out),
            conf    = conf or self.cfg.conf,
            iou     = iou  or self.cfg.iou,
        )
        return out / "results.txt"

    # ------------------------------------------------------------------
    # ROI calibration
    # ------------------------------------------------------------------

    def calibrate_roi(
        self,
        image_path: str | Path,
        cage_type: str = "default",
    ) -> dict:
        from scripts.calibrate_roi import calibrate, print_config, save_debug
        zones = calibrate(Path(image_path))
        if zones:
            print_config(zones, cage_type)
            save_debug(Path(image_path), zones)
        return zones

    # ------------------------------------------------------------------
    # Pipeline workflow shortcuts
    # ------------------------------------------------------------------

    def run_data_pipeline(
        self,
        n_augments: int = 50,
        propagate: bool = True,
    ) -> None:
        """label → augment → clean → dedup → split"""
        steps = [
            ("Label originals (GDINO)", lambda: self.label_originals(propagate=False)),
            ("Augment dataset",         lambda: self.augment(n=n_augments)),
            ("Clean food labels",       self.clean_food_labels),
            ("Dedup labels",            self.dedup_labels),
            ("Split dataset",           self.split_dataset),
        ]
        for i, (title, fn) in enumerate(steps, 1):
            print(f"\n{'='*55}\n{i}/{len(steps)}  {title}\n{'='*55}")
            fn()
        print("\n✅ Data pipeline complete. Run orch.train() to start training.")

    def run_full_pipeline(
        self,
        n_augments: int = 50,
        epochs: int = 100,
    ) -> Path:
        """End-to-end: data prep → train → validate."""
        self.run_data_pipeline(n_augments=n_augments)
        best_pt = self.train(epochs=epochs)
        self.validate(weights=best_pt)
        return best_pt

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Lazy-load (or rebuild after set_backend) the InferencePipeline."""
        if self._pipeline is None:
            from pipeline.pipeline import InferencePipeline
            self._pipeline = InferencePipeline(
                cage_type = self.cfg.cage_type,
                backend   = self.cfg.backend,
            )
        return self._pipeline