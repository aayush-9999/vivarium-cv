# segmentation/datasets/level_dataset.py
"""
Dataset for PSPNet water/food level segmentation training.

Expected directory structure (you create this after annotation):

    dataset/
      segmentation/
        water/
          images/
            cage01_frame001.jpg
            cage01_frame002.jpg
            ...
          masks/
            cage01_frame001.png   ← PNG mask, pixel values = class IDs (0-3)
            cage01_frame002.png
            ...
        food/
          images/
            cage01_frame001.jpg
            ...
          masks/
            cage01_frame001.png
            ...

Mask pixel values:
    Water model:
        0 = background
        1 = bottle wall
        2 = water fill   ← what we measure
        3 = empty air    ← what we measure

    Food model:
        0 = background
        1 = hopper frame
        2 = food pellets ← what we measure
        3 = empty space  ← what we measure

IMPORTANT: Masks must be single-channel PNG (mode 'L' or 'P').
           NOT RGB. NOT RGBA. Each pixel value IS the class ID.
           Tools like CVAT, Label Studio, and LabelMe can export this format.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation transforms (geometry must be applied to both image AND mask)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationAugment:
    """
    Joint augmentation for image + mask pairs.
    All geometric transforms are applied identically to both.
    Color/noise transforms are applied to image only.
    """

    def __init__(
        self,
        flip_prob:      float = 0.5,
        rotate_range:   float = 10.0,
        scale_range:    tuple[float, float] = (0.8, 1.2),
        brightness:     float = 0.3,
        contrast:       float = 0.3,
        noise_sigma:    float = 10.0,
    ):
        self.flip_prob    = flip_prob
        self.rotate_range = rotate_range
        self.scale_range  = scale_range
        self.brightness   = brightness
        self.contrast     = contrast
        self.noise_sigma  = noise_sigma

    def __call__(
        self,
        image: np.ndarray,
        mask:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # ── Horizontal flip ────────────────────────────────────────────────
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask,  1)

        # ── Random rotation ────────────────────────────────────────────────
        angle = random.uniform(-self.rotate_range, self.rotate_range)
        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            mask  = cv2.warpAffine(mask,  M, (w, h),
                                   flags=cv2.INTER_NEAREST,   # no interpolation for masks
                                   borderMode=cv2.BORDER_REFLECT)

        # ── Random scale crop ──────────────────────────────────────────────
        scale = random.uniform(*self.scale_range)
        if abs(scale - 1.0) > 0.05:
            h, w   = image.shape[:2]
            new_h  = int(h * scale)
            new_w  = int(w * scale)
            image  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask   = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Crop or pad back to original size
            image  = _crop_or_pad(image, h, w)
            mask   = _crop_or_pad(mask,  h, w, fill_value=0)

        # ── Brightness / contrast (image only) ────────────────────────────
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta  = random.uniform(-self.brightness * 255, self.brightness * 255)
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # ── Gaussian noise (image only) ───────────────────────────────────
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # ── HSV shift (image only) ────────────────────────────────────────
        hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-10, 10), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30, 30), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image, mask


def _crop_or_pad(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    fill_value: int = 114,
) -> np.ndarray:
    """Center crop or pad array to target size."""
    h, w = arr.shape[:2]

    if len(arr.shape) == 2:
        out = np.full((target_h, target_w), fill_value, dtype=arr.dtype)
    else:
        out = np.full((target_h, target_w, arr.shape[2]), fill_value, dtype=arr.dtype)

    # Determine paste coords
    y_start = max(0, (h - target_h) // 2)
    x_start = max(0, (w - target_w) // 2)
    y_end   = min(h, y_start + target_h)
    x_end   = min(w, x_start + target_w)

    paste_y = max(0, (target_h - h) // 2)
    paste_x = max(0, (target_w - w) // 2)
    paste_h = y_end - y_start
    paste_w = x_end - x_start

    out[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = arr[y_start:y_end, x_start:x_end]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LevelSegDataset(Dataset):
    """
    PyTorch Dataset for level segmentation.

    Args:
        image_dir   : folder containing BGR images (.jpg / .png)
        mask_dir    : folder containing single-channel masks (.png)
                      pixel value = class ID (0, 1, 2, 3)
        target_size : (H, W) to resize to during loading
        augment     : if True apply random augmentation (use for training only)
        augmentor   : optional custom SegmentationAugment instance
    """

    def __init__(
        self,
        image_dir:   Path,
        mask_dir:    Path,
        target_size: tuple[int, int] = (256, 128),
        augment:     bool = False,
        augmentor:   Optional[SegmentationAugment] = None,
    ):
        self.image_dir   = Path(image_dir)
        self.mask_dir    = Path(mask_dir)
        self.target_size = target_size   # (H, W)
        self.augment     = augment
        self.augmentor   = augmentor or SegmentationAugment()

        # Find matched image-mask pairs
        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(
                f"No matched image-mask pairs found.\n"
                f"  images: {image_dir}\n"
                f"  masks:  {mask_dir}\n"
                f"Make sure mask filenames match image stems with .png extension."
            )

    def _find_pairs(self) -> list[tuple[Path, Path]]:
        img_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        pairs     = []
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        # Load image (BGR)
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Cannot read image: {img_path}")

        # Load mask (single channel — pixel values are class IDs)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Cannot read mask: {mask_path}")

        # Resize both to target size
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_NEAREST)

        # Augmentation (training only)
        if self.augment:
            image, mask = self.augmentor(image, mask)

        # Normalize image and convert to tensor
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        img_t  = torch.from_numpy(np.transpose(normed, (2, 0, 1)))   # CHW

        # Mask to long tensor (class IDs must be int64 for cross entropy)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "image":    img_t,
            "mask":     mask_t,
            "img_path": str(img_path),
        }

    def class_weights(self, num_classes: int = 4) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted cross entropy.
        Call once before training to handle class imbalance.
        (Background pixels usually dominate — this corrects for that.)
        """
        counts = torch.zeros(num_classes)
        for _, mask_path in self.pairs:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            for cls in range(num_classes):
                counts[cls] += int((mask == cls).sum())

        # Inverse frequency: rare classes get higher weight
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    data_root:    Path,
    container:    str,              # "water" or "food" — used only if images/ not directly in data_root
    target_size:  tuple[int, int],
    batch_size:   int   = 8,
    val_split:    float = 0.15,
    seed:         int   = 42,
    num_workers:  int   = 0,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build train and val DataLoaders from a segmentation dataset.

    Args:
        data_root   : root folder. Two layouts supported:
                        Layout A — data_root/images/ and data_root/masks/ exist directly
                                   (use when data_root=dataset/segmentation/water)
                        Layout B — data_root/water/images/ and data_root/water/masks/
                                   (use when data_root=dataset/segmentation)
        container   : "water" or "food" — only used for Layout B
        target_size : (H, W)
        batch_size  : training batch size
        val_split   : fraction of data for validation
        seed        : random seed for reproducible splits
        num_workers : DataLoader workers (0 = main process, safe on Windows)

    Returns:
        (train_loader, val_loader, class_weights)
    """
    import random as rng

    data_root = Path(data_root)

    # Auto-detect layout: if images/ exists directly under data_root use it,
    # otherwise fall back to data_root/container/images/
    if (data_root / "images").exists():
        img_dir  = data_root / "images"
        mask_dir = data_root / "masks"
    else:
        img_dir  = data_root / container / "images"
        mask_dir = data_root / container / "masks"

    full_dataset = LevelSegDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        target_size=target_size,
        augment=False,   # we split first, then set augment on train subset
    )

    # Split indices
    n     = len(full_dataset)
    idxs  = list(range(n))
    rng.seed(seed)
    rng.shuffle(idxs)
    split = int(n * (1 - val_split))
    train_idx = idxs[:split]
    val_idx   = idxs[split:]

    from torch.utils.data import Subset

    train_sub = Subset(full_dataset, train_idx)
    val_sub   = Subset(full_dataset, val_idx)

    # Enable augmentation on training set
    full_dataset.augment = False   # reset
    train_aug = LevelSegDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        target_size=target_size,
        augment=True,
    )
    train_aug_sub = Subset(train_aug, train_idx)

    train_loader = DataLoader(
        train_aug_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class weights from full dataset
    weights = full_dataset.class_weights(num_classes=4)

    print(f"\nDataset: {container}")
    print(f"  Total pairs : {n}")
    print(f"  Train       : {len(train_idx)}")
    print(f"  Val         : {len(val_idx)}")
    print(f"  Class weights: {weights.tolist()}")

    return train_loader, val_loader, weights