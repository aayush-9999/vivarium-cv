# segmentation/trainers/psp_trainer.py
"""
Trainer for PSPNet water/food level segmentation.

Loss:
    Main loss    : weighted CrossEntropyLoss on final output
    Aux loss     : CrossEntropyLoss on layer3 output (weight=0.4)
    Total loss   = main_loss + 0.4 * aux_loss

    The auxiliary loss is PSPNet's training trick — it provides gradient
    signal deeper in the network and significantly improves convergence.

Metrics:
    - mIoU (mean Intersection over Union) — primary metric
    - Per-class IoU for fill and empty classes (the ones we care about)
    - Fill % MAE (how close is our pct estimate to ground truth pct)

Optimizer:
    SGD with poly learning rate schedule (standard for PSPNet)
    lr = base_lr * (1 - iter/max_iter) ^ 0.9

Usage:
    python segmentation/trainers/psp_trainer.py --container water
    python segmentation/trainers/psp_trainer.py --container food
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from segmentation.models.pspnet import PSPNet, build_water_model, build_food_model
from segmentation.models.level_estimator import (
    mask_to_fill_pct,
    WATER_FILL_CLASS, WATER_EMPTY_CLASS,
    FOOD_FILL_CLASS,  FOOD_EMPTY_CLASS,
)
from segmentation.datasets.level_dataset import make_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class PSPLoss(nn.Module):
    """
    Combined main + auxiliary segmentation loss.
    aux_weight=0.4 follows the original PSPNet paper.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        aux_weight:    float = 0.4,
        ignore_index:  int   = 255,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.criterion  = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits:  Optional[torch.Tensor],
        targets:     torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        main_loss = self.criterion(main_logits, targets)

        if aux_logits is not None:
            aux_loss = self.criterion(aux_logits, targets)
            total    = main_loss + self.aux_weight * aux_loss
        else:
            aux_loss = torch.tensor(0.0)
            total    = main_loss

        return total, {
            "total": total.item(),
            "main":  main_loss.item(),
            "aux":   aux_loss.item() if aux_logits is not None else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_miou(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> tuple[float, list[float]]:
    """
    Compute mean IoU and per-class IoU.

    Args:
        pred   : (B, H, W) predicted class IDs
        target : (B, H, W) ground truth class IDs

    Returns:
        (miou, per_class_iou_list)
    """
    ious = []
    pred   = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    valid = target != ignore_index
    pred   = pred[valid]
    target = target[valid]

    for cls in range(num_classes):
        pred_cls   = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum()
        union        = (pred_cls | target_cls).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection / union))

    valid_ious = [v for v in ious if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return miou, ious


def compute_fill_mae(
    pred_masks:  np.ndarray,
    true_masks:  np.ndarray,
    fill_class:  int,
    empty_class: int,
    use_height:  bool,
) -> float:
    """
    Mean absolute error between predicted and ground truth fill percentages.
    This is the metric that directly reflects how accurate our level readings are.
    """
    maes = []
    for pred, true in zip(pred_masks, true_masks):
        pred_pct = mask_to_fill_pct(pred, fill_class, empty_class, use_height)
        true_pct = mask_to_fill_pct(true, fill_class, empty_class, use_height)
        maes.append(abs(pred_pct - true_pct))

    return float(np.mean(maes)) if maes else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Poly LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

class PolyLR:
    """
    Polynomial learning rate decay.
    lr = base_lr * (1 - iter / max_iter) ^ power

    Standard for PSPNet — decays smoothly over training.
    """

    def __init__(
        self,
        optimizer:   optim.Optimizer,
        max_iters:   int,
        base_lr:     float,
        power:       float = 0.9,
        min_lr:      float = 1e-6,
    ):
        self.optimizer  = optimizer
        self.max_iters  = max_iters
        self.base_lr    = base_lr
        self.power      = power
        self.min_lr     = min_lr
        self._cur_iter  = 0

    def step(self) -> float:
        lr = self.base_lr * (1 - self._cur_iter / self.max_iters) ** self.power
        lr = max(lr, self.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._cur_iter += 1
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PSPTrainer:

    def __init__(
        self,
        container:     str,           # "water" | "food"
        data_root:     Path,
        output_dir:    Path,
        backbone:      str   = "resnet50",
        epochs:        int   = 60,
        batch_size:    int   = 8,
        base_lr:       float = 0.01,
        device:        str   = "cpu",
        val_interval:  int   = 5,
        resume:        Optional[str] = None,
    ):
        self.container    = container
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs       = epochs
        self.device       = device
        self.val_interval = val_interval

        # Fill/empty class IDs depend on container type
        if container == "water":
            self.fill_class  = WATER_FILL_CLASS
            self.empty_class = WATER_EMPTY_CLASS
            self.use_height  = True
            target_size      = (256, 128)
        else:
            self.fill_class  = FOOD_FILL_CLASS
            self.empty_class = FOOD_EMPTY_CLASS
            self.use_height  = False
            target_size      = (224, 224)

        # Data
        self.train_loader, self.val_loader, class_weights = make_dataloaders(
            data_root=data_root,
            container=container,
            target_size=target_size,
            batch_size=batch_size,
        )

        # Model
        build_fn = build_water_model if container == "water" else build_food_model
        self.model = build_fn(backbone=backbone, pretrained=True, device=device)
        self.model.train()

        # Loss
        self.criterion = PSPLoss(
            class_weights=class_weights.to(device),
            aux_weight=0.4,
        )

        # Optimizer — separate LR for backbone vs head (backbone uses 0.1x LR)
        backbone_params = list(self.model.backbone.parameters())
        head_params     = (
            list(self.model.ppm.parameters()) +
            list(self.model.bottleneck.parameters()) +
            list(self.model.classifier.parameters()) +
            list(self.model.aux_head.parameters())
        )
        self.optimizer = optim.SGD(
            [
                {"params": backbone_params, "lr": base_lr * 0.1},
                {"params": head_params,     "lr": base_lr},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )

        max_iters = epochs * len(self.train_loader)
        self.scheduler = PolyLR(self.optimizer, max_iters=max_iters, base_lr=base_lr)

        # Resume
        self.start_epoch   = 0
        self.best_miou     = 0.0
        self.history: list[dict] = []

        if resume:
            self._load_checkpoint(resume)

    # ── Training loop ──────────────────────────────────────────────────────

    def train(self) -> None:
        print(f"\n{'='*60}")
        print(f"Training PSPNet — {self.container} level segmentation")
        print(f"  Device   : {self.device}")
        print(f"  Epochs   : {self.epochs}")
        print(f"  Output   : {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)

            val_metrics = {}
            if (epoch + 1) % self.val_interval == 0 or epoch == self.epochs - 1:
                val_metrics = self._val_epoch(epoch)

                if val_metrics.get("miou", 0) > self.best_miou:
                    self.best_miou = val_metrics["miou"]
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  ★ New best mIoU: {self.best_miou:.4f}")

            elapsed = time.time() - t0
            log = {
                "epoch":  epoch + 1,
                "train":  train_metrics,
                "val":    val_metrics,
                "time_s": round(elapsed, 1),
            }
            self.history.append(log)
            self._save_checkpoint(epoch, is_best=False)
            self._print_epoch(log)
            self._save_history()

        print(f"\nTraining complete. Best mIoU: {self.best_miou:.4f}")
        print(f"Best weights: {self.output_dir / 'best.pth'}")

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        losses, mious = [], []

        for batch in self.train_loader:
            images  = batch["image"].to(self.device)
            masks   = batch["mask"].to(self.device)

            # Forward with auxiliary head
            main_logits, aux_logits = self.model(images, return_aux=True)
            loss, loss_parts = self.criterion(main_logits, aux_logits, masks)

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping helps stability with dilated convolutions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            with torch.no_grad():
                pred  = torch.argmax(main_logits, dim=1)
                miou, _ = compute_miou(pred, masks)
                losses.append(loss_parts["total"])
                mious.append(miou)

        return {
            "loss": float(np.mean(losses)),
            "miou": float(np.mean(mious)),
        }

    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        losses, mious, fill_maes = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                images  = batch["image"].to(self.device)
                masks   = batch["mask"].to(self.device)

                logits  = self.model(images, return_aux=False)
                loss, loss_parts = self.criterion(logits, None, masks)

                pred    = torch.argmax(logits, dim=1)
                miou, per_cls = compute_miou(pred, masks)

                # Fill MAE — how accurate is the actual percentage
                pred_np = pred.cpu().numpy()
                true_np = masks.cpu().numpy()
                mae = compute_fill_mae(
                    pred_np, true_np,
                    self.fill_class, self.empty_class,
                    self.use_height,
                )

                losses.append(loss_parts["total"])
                mious.append(miou)
                fill_maes.append(mae)

        return {
            "loss":     float(np.mean(losses)),
            "miou":     float(np.mean(mious)),
            "fill_mae": float(np.mean(fill_maes)),
        }

    # ── Checkpointing ──────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        ckpt = {
            "epoch":      epoch + 1,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "best_miou":  self.best_miou,
            "container":  self.container,
        }
        torch.save(ckpt, self.output_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best.pth")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"]
        self.best_miou   = ckpt.get("best_miou", 0.0)
        print(f"Resumed from {path} (epoch {self.start_epoch}, best mIoU {self.best_miou:.4f})")

    def _save_history(self) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

    def _print_epoch(self, log: dict) -> None:
        t = log["train"]
        v = log["val"]
        val_str = (
            f"  val_loss={v['loss']:.4f}  val_mIoU={v['miou']:.4f}  fill_MAE={v.get('fill_mae', 0):.2f}%"
            if v else ""
        )
        print(
            f"  Epoch {log['epoch']:3d}/{self.epochs}"
            f"  train_loss={t['loss']:.4f}  train_mIoU={t['miou']:.4f}"
            f"{val_str}"
            f"  ({log['time_s']}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train PSPNet for water/food level segmentation")
    ap.add_argument("--container",  required=True, choices=["water", "food"])
    ap.add_argument("--data-root",  type=Path, default=Path("dataset/segmentation"))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--backbone",   default="resnet50", choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--epochs",     type=int,   default=60)
    ap.add_argument("--batch-size", type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=0.01)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-every",  type=int,   default=5)
    ap.add_argument("--resume",     type=str,   default=None)
    args = ap.parse_args()

    out = args.output_dir or Path(f"runs/pspnet/{args.container}")

    trainer = PSPTrainer(
        container=args.container,
        data_root=args.data_root,
        output_dir=out,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        device=args.device,
        val_interval=args.val_every,
        resume=args.resume,
    )
    trainer.train()