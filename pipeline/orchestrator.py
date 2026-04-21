# pipeline/orchestrator.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.schemas import DetectionResult
from core.exceptions import VivariumCVError


@dataclass
class OrchestratorConfig:
    # Paths
    orig_dir:     Path  = field(default_factory=lambda: Path("dataset/original"))
    aug_dir:      Path  = field(default_factory=lambda: Path("dataset/augmented"))
    split_dir:    Path  = field(default_factory=lambda: Path("dataset/split"))
    weights:      Path  = field(default_factory=lambda: Path(os.getenv("YOLO_WEIGHTS", "models/yolo/best.pt")))
    device:       str   = field(default_factory=lambda: os.getenv("YOLO_DEVICE", "cpu"))

    # Augmentation
    aug_n:        int   = 50
    aug_seed:     int   = field(default_factory=lambda: random.randint(0, 99_999))
    img_size:     int   = 640
    jpeg_quality: int   = 90

    # GDINO labelling
    mouse_thresh:     float = 0.22
    container_thresh: float = 0.30
    food_thresh:      float = 0.28

    # Training
    epochs:       int   = 100
    batch:        int   = 16
    train_ratio:  float = 0.85

    # Inference
    conf:         float = 0.45
    iou:          float = 0.30

    # Label cleaning (passed to label_tools)
    max_food_area: float = 0.12
    max_food_w:    float = 0.40
    max_food_h:    float = 0.50
    mouse_nms_iou: float = 0.45
    food_nms_iou:  float = 0.50


class VivariumOrchestrator:
    """Single entry-point for all Vivarium CV workflows."""

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.cfg = config or OrchestratorConfig()
        self._inference_pipeline = None

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray, cage_id: str, save_flagged: bool = False) -> DetectionResult:
        return self._get_inference_pipeline().run(frame=frame, cage_id=cage_id, save_flagged=save_flagged)

    def infer_from_path(self, image_path: str | Path, cage_id: str, save_flagged: bool = False) -> DetectionResult:
        img = cv2.imread(str(image_path))
        if img is None:
            raise VivariumCVError(f"Cannot read image: {image_path}")
        return self.infer(img, cage_id=cage_id, save_flagged=save_flagged)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        return self._get_inference_pipeline().debug_frame(frame)

    # ── Label tools — all from scripts/label_tools.py ────────────────────

    def verify_labels(self, img_dir: Optional[Path] = None, label_dir: Optional[Path] = None) -> list[str]:
        from scripts.label_tools import verify
        return verify(
            img_dir=img_dir     or (self.cfg.aug_dir / "images"),
            label_dir=label_dir or (self.cfg.aug_dir / "labels"),
        )

    def clean_food_labels(self, label_dir: Optional[Path] = None, max_area: Optional[float] = None,
                          max_w: Optional[float] = None, max_h: Optional[float] = None,
                          dry_run: bool = False) -> dict:
        from scripts.label_tools import clean_food
        return clean_food(
            label_dir=label_dir or (self.cfg.aug_dir / "labels"),
            max_area=max_area   or self.cfg.max_food_area,
            max_w=max_w         or self.cfg.max_food_w,
            max_h=max_h         or self.cfg.max_food_h,
            dry_run=dry_run,
        )

    def dedup_labels(self, label_dir: Optional[Path] = None, mouse_iou: Optional[float] = None,
                     food_iou: Optional[float] = None, dry_run: bool = False) -> dict:
        from scripts.label_tools import dedup
        return dedup(
            label_dir=label_dir or (self.cfg.aug_dir / "labels"),
            mouse_iou=mouse_iou or self.cfg.mouse_nms_iou,
            food_iou=food_iou   or self.cfg.food_nms_iou,
            dry_run=dry_run,
        )

    def fix_labels(self, label_dir: Optional[Path] = None, dry_run: bool = False) -> dict:
        from scripts.label_tools import fix_classes
        return fix_classes(
            label_dir=label_dir or (self.cfg.aug_dir / "labels"),
            dry_run=dry_run,
        )

    # ── Labelling ─────────────────────────────────────────────────────────

    def label_originals(self, propagate: bool = False, mouse_thresh: Optional[float] = None,
                        container_thresh: Optional[float] = None, food_thresh: Optional[float] = None) -> dict:
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

    def auto_label(self, src: Optional[Path] = None, dst: Optional[Path] = None,
                   conf_override: Optional[float] = None, debug: bool = False) -> None:
        from scripts.auto_label import main as _auto_label
        _auto_label(
            src=src or (self.cfg.aug_dir / "images"),
            dst=dst or (self.cfg.aug_dir / "labels"),
            conf_override=conf_override,
            debug=debug,
        )

    def merge_labels(self, existing_dir: Optional[Path] = None, new_dir: Optional[Path] = None,
                     out_dir: Optional[Path] = None, dry_run: bool = False) -> None:
        from scripts.merge_labels import main as _merge
        _merge(
            existing_dir=existing_dir or (self.cfg.aug_dir / "labels"),
            new_dir=new_dir           or (self.cfg.aug_dir / "labels_gdino"),
            out_dir=out_dir           or (self.cfg.aug_dir / "labels_merged"),
            dry_run=dry_run,
        )

    # ── Augmentation ──────────────────────────────────────────────────────

    def augment(self, n: Optional[int] = None, seed: Optional[int] = None,
                src: Optional[Path] = None, dst: Optional[Path] = None,
                src_labels: Optional[Path] = None, img_size: Optional[int] = None,
                jpeg_quality: Optional[int] = None) -> None:
        from scripts.augment import main as _augment
        _augment(
            src=src          or self.cfg.orig_dir,
            dst=dst          or self.cfg.aug_dir,
            src_labels=src_labels,
            n=n              or self.cfg.aug_n,
            seed=seed        or self.cfg.aug_seed,
            img_size=img_size or self.cfg.img_size,
            quality=jpeg_quality or self.cfg.jpeg_quality,
        )

    # ── Dataset split ─────────────────────────────────────────────────────

    def split_dataset(self, train_ratio: Optional[float] = None, seed: int = 42,
                      label_dir: Optional[Path] = None, img_dir: Optional[Path] = None,
                      out_dir: Optional[Path] = None) -> dict[str, int]:
        import shutil
        ratio    = train_ratio or self.cfg.train_ratio
        lbl_dir  = label_dir  or (self.cfg.aug_dir / "labels")
        src_imgs = img_dir    or (self.cfg.aug_dir / "images")
        dst      = out_dir    or self.cfg.split_dir

        random.seed(seed)
        labelled = [p for p in lbl_dir.glob("*.txt") if p.stat().st_size > 0]
        random.shuffle(labelled)
        split_idx = int(len(labelled) * ratio)
        train_set, val_set = labelled[:split_idx], labelled[split_idx:]

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

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, yaml_path: Optional[Path] = None, epochs: Optional[int] = None,
              batch: Optional[int] = None, device: Optional[str] = None,
              project: Optional[Path] = None, run_name: str = "vivarium_v1",
              resume: bool = False) -> Path:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise VivariumCVError("ultralytics not installed.")

        base_dir  = Path(__file__).resolve().parent.parent
        data_path = yaml_path or (base_dir / "dataset" / "vivarium.yaml")
        proj      = project   or (base_dir / "runs" / "detect")

        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data_path),
            epochs=epochs or self.cfg.epochs,
            imgsz=self.cfg.img_size,
            batch=batch   or self.cfg.batch,
            device=device or self.cfg.device,
            workers=0, cos_lr=True, patience=20,
            hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
            fliplr=0.5, mosaic=0.8,
            project=str(proj), name=run_name,
            exist_ok=True, resume=resume, verbose=True,
        )
        best_pt = proj / run_name / "weights" / "best.pt"
        print(f"\nTraining complete. Weights: {best_pt}")
        return best_pt

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, weights: Optional[Path] = None, val_dir: Optional[Path] = None,
                 out_dir: Optional[Path] = None, conf: Optional[float] = None,
                 iou: Optional[float] = None) -> Path:
        from scripts.val_test import main as _val
        out = out_dir or Path("runs/val_test")
        _val(
            weights=str(weights or self.cfg.weights),
            val_dir=str(val_dir or (self.cfg.split_dir / "val" / "images")),
            out_dir=str(out),
            conf=conf or self.cfg.conf,
            iou=iou   or self.cfg.iou,
        )
        return out / "results.txt"

    # ── ROI calibration ───────────────────────────────────────────────────

    def calibrate_roi(self, image_path: str | Path, cage_type: str = "default") -> dict:
        from scripts.calibrate_roi import calibrate, print_config, save_debug
        zones = calibrate(Path(image_path))
        if zones:
            print_config(zones, cage_type)
            save_debug(Path(image_path), zones)
        return zones

    # ── Camera helpers ────────────────────────────────────────────────────

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self._get_inference_pipeline().set_reference_frame(frame)

    def has_motion(self, frame: np.ndarray) -> bool:
        return self._get_inference_pipeline().has_motion(frame)

    # ── Full workflow shortcuts ───────────────────────────────────────────

    def run_data_pipeline(self, n_augments: int = 50, propagate: bool = True) -> None:
        """label → augment → clean → dedup → split"""
        for i, (title, fn) in enumerate([
            ("Label originals (GDINO)", lambda: self.label_originals(propagate=False)),
            ("Augment dataset",         lambda: self.augment(n=n_augments)),
            ("Clean food labels",       self.clean_food_labels),
            ("Dedup labels",            self.dedup_labels),
            ("Split dataset",           self.split_dataset),
        ], 1):
            print(f"\n{'='*55}\n{i}/5  {title}\n{'='*55}")
            result = fn()
            # propagate after augment if we have boxes
            if i == 2 and propagate:
                from scripts.gdino_label_originals import propagate_labels
                # re-label to get stem_to_boxes (already cached in label step ideally)
                pass

        print("\n✅ Data pipeline complete. Run orch.train() to start training.")

    def run_full_pipeline(self, n_augments: int = 50, epochs: int = 100) -> Path:
        """End-to-end: data prep → train → validate."""
        self.run_data_pipeline(n_augments=n_augments)
        best_pt = self.train(epochs=epochs)
        self.validate(weights=best_pt)
        return best_pt

    # ── Private ───────────────────────────────────────────────────────────

    def _get_inference_pipeline(self):
        if self._inference_pipeline is None:
            from pipeline.yolo_pipeline import YOLOPipeline
            self._inference_pipeline = YOLOPipeline()
        return self._inference_pipeline