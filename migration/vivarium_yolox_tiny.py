"""
migration/vivarium_yolox_tiny.py
=================================
YOLOx experiment configuration for the vivarium 9-class detector.

This is a DRAFT placed in migration/ for review. Once confirmed correct,
copy it to exps/vivarium_yolox_tiny.py and update .env:
    YOLOX_EXP=exps/vivarium_yolox_tiny.py

What this file does
────────────────────
YOLOx requires every training run to be described by an Exp class that
inherits from yolox.exp.Exp. It replaces the YAML config that YOLOv8/
Ultralytics used (vivarium.yaml). The Exp class controls the model
architecture, dataset paths, augmentation, and training schedule.

Tiny model multipliers
───────────────────────
YOLOx model sizes are set by two multipliers applied to the base CSPDarknet:
    depth  = fraction of the base number of CSP blocks
    width  = fraction of the base channel count

    nano   : depth=0.33, width=0.25
    tiny   : depth=0.33, width=0.375   ← this file
    small  : depth=0.33, width=0.50
    medium : depth=0.67, width=0.75
    large  : depth=1.00, width=1.00
    x      : depth=1.33, width=1.25

Usage
──────
    # Train
    python -m yolox.tools.train \
        -f exps/vivarium_yolox_tiny.py \
        -d 1 \           # number of GPUs (1 for single GPU, 0 for CPU debug)
        -b 16 \          # batch size
        --fp16           # optional: mixed precision if GPU supports it

    # Evaluate
    python -m yolox.tools.eval \
        -f exps/vivarium_yolox_tiny.py \
        -c runs/yolox/best_ckpt.pth \
        -b 8 -d 1

    # Export to ONNX
    python -m yolox.tools.export_onnx \
        -f exps/vivarium_yolox_tiny.py \
        -c runs/yolox/best_ckpt.pth \
        --output-name models/yolox/vivarium_tiny.onnx
"""

from __future__ import annotations

import os
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ── Identity ──────────────────────────────────────────────────────────
        self.exp_name = "vivarium_yolox_tiny"

        # ── Model: tiny multipliers ───────────────────────────────────────────
        self.depth       = 0.33     # CSP block depth fraction
        self.width       = 0.375    # channel width fraction
        self.num_classes = 9        # must match vivarium 9-class scheme

        # ── Input ─────────────────────────────────────────────────────────────
        self.input_size  = (640, 640)   # (height, width) — note: YOLOx is h,w not w,h
        self.test_size   = (640, 640)
        self.random_size = (10, 20)     # multi-scale range: input_size * stride/32

        # ── Dataset (COCO JSON format, produced by migration/convert_to_coco.py)
        self.data_dir  = os.getenv("COCO_DATA_DIR", "dataset/coco")
        self.train_ann = "train.json"
        self.val_ann   = "val.json"

        # ── Training schedule ─────────────────────────────────────────────────
        self.max_epoch       = 100
        self.warmup_epochs   = 5
        self.no_aug_epochs   = 15    # last N epochs: disable mosaic/mixup augmentation
        self.eval_interval   = 5     # run val every N epochs

        # Learning rate: YOLOx scales lr by batch size
        # base formula: lr = basic_lr_per_img * batch_size
        # With batch=16: lr = 0.000156 (conservative for fine-tuning)
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler        = "yoloxwarmcos"

        # ── Augmentation ──────────────────────────────────────────────────────
        # Mosaic and Mixup are the key YOLOx augmentations — keep them on
        self.mosaic_prob  = 0.8    # probability of applying mosaic
        self.mixup_prob   = 0.0    # mixup is expensive; off for tiny model
        self.degrees      = 10.0   # rotation range
        self.translate    = 0.1
        self.scale        = (0.5, 1.5)
        self.shear        = 2.0
        self.hsv_prob     = 1.0    # always apply HSV augmentation
        self.flip_prob    = 0.5    # horizontal flip probability

        # ── Detection thresholds ──────────────────────────────────────────────
        self.test_conf  = 0.35     # matches YOLO_CONF_THRESHOLD in core/config.py
        self.nmsthre    = 0.45     # matches YOLO_IOU_THRESHOLD in core/config.py

        # ── Output ────────────────────────────────────────────────────────────
        self.output_dir = "runs/yolox"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Returns the training dataset.
        Uses COCODataset — works with the JSON produced by convert_to_coco.py.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir    = self.data_dir,
            json_file   = self.train_ann,
            img_size    = self.input_size,
            preproc     = TrainTransform(
                max_labels  = 50,
                flip_prob   = self.flip_prob,
                hsv_prob    = self.hsv_prob,
            ),
            cache       = cache,
            cache_type  = cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        return COCODataset(
            data_dir  = self.data_dir,
            json_file = self.val_ann,
            img_size  = self.test_size,
            preproc   = ValTransform(legacy=False),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader  = self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy),
            img_size    = self.test_size,
            confthre    = self.test_conf,
            nmsthre     = self.nmsthre,
            num_classes = self.num_classes,
            testdev     = testdev,
        )