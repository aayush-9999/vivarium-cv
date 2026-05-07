# segmentation/models/pspnet.py
"""
PSPNet with ResNet backbone for water/food level segmentation.

Architecture:
    ResNet50/101 backbone (pretrained ImageNet)
        │
    Pyramid Pooling Module (1x1, 2x2, 3x3, 6x6)
        │
    Final Conv → Per-pixel class prediction

Classes per model:
    Water model:
        0 = background
        1 = bottle wall / container
        2 = water fill (below meniscus)
        3 = empty air  (above meniscus)

    Food model:
        0 = background
        1 = hopper frame / wire
        2 = food pellets (fill region)
        3 = empty hopper space
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Pyramid Pooling Module
# ─────────────────────────────────────────────────────────────────────────────

class PyramidPoolingModule(nn.Module):
    """
    PSPNet's core module. Pools features at 4 scales and concatenates
    them back to the original resolution. This gives the network both
    local detail and global scene context simultaneously.

    Pool sizes: [1, 2, 3, 6]
        1×1  = entire feature map (global average pool)
        2×2  = quadrant-level context
        3×3  = sub-region context
        6×6  = fine local context
    """

    def __init__(self, in_channels: int, pool_sizes: list[int] = [1, 2, 3, 6]):
        super().__init__()
        self.pool_sizes   = pool_sizes
        # Each branch reduces channels by 4x to keep concat manageable
        branch_channels   = in_channels // len(pool_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for pool_size in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w    = x.shape[2], x.shape[3]
        pooled  = [x]   # start with original features

        for branch in self.branches:
            out = branch(x)
            # Upsample back to original feature map size
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            pooled.append(out)

        return torch.cat(pooled, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet Backbone (extracts features, removes classification head)
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """
    ResNet50 or ResNet101 with:
    - Final classification layers removed
    - Dilated convolutions in layer3/layer4 to preserve spatial resolution
      (output stride 8 instead of 32 — keeps more spatial detail for segmentation)
    - Returns feature map at 1/8 of input resolution
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        if backbone == "resnet50":
            base   = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_channels = 2048
        elif backbone == "resnet101":
            base   = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_channels = 2048
        elif backbone == "resnet18":
            base   = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use resnet18/50/101")

        # Keep everything except avgpool and fc (classification head)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Apply dilation to layer3 and layer4 to maintain spatial resolution
        # This is the key trick in PSPNet — output stride becomes 8 instead of 32
        if backbone in ("resnet50", "resnet101"):
            self._apply_dilation(self.layer3, dilation=2, stride=1)
            self._apply_dilation(self.layer4, dilation=4, stride=1)

    def _apply_dilation(self, layer: nn.Module, dilation: int, stride: int) -> None:
        """Replace stride convolutions with dilated convolutions."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride  = (stride, stride)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding  = (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# PSPNet
# ─────────────────────────────────────────────────────────────────────────────

class PSPNet(nn.Module):
    """
    PSPNet for vivarium level segmentation.

    Args:
        num_classes : number of output classes (4 for water or food model)
        backbone    : "resnet50" | "resnet101" | "resnet18"
        pretrained  : load ImageNet weights for backbone
        dropout     : dropout rate before final classifier
        pool_sizes  : pyramid pooling scales

    Input:  (B, 3, H, W)  — RGB image, any size divisible by 8
    Output: (B, num_classes, H, W) — per-pixel logits (same size as input)
    """

    def __init__(
        self,
        num_classes: int  = 4,
        backbone:    str  = "resnet50",
        pretrained:  bool = True,
        dropout:     float = 0.1,
        pool_sizes:  list[int] = [1, 2, 3, 6],
    ):
        super().__init__()

        self.backbone    = ResNetBackbone(backbone=backbone, pretrained=pretrained)
        backbone_out_ch  = self.backbone.out_channels

        self.ppm         = PyramidPoolingModule(
            in_channels=backbone_out_ch,
            pool_sizes=pool_sizes,
        )

        # After PPM: backbone_out_ch + (backbone_out_ch // 4) * num_pools
        ppm_out_ch = backbone_out_ch + (backbone_out_ch // len(pool_sizes)) * len(pool_sizes)

        # Bottleneck conv after PPM
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_ch, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        # Final classifier
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        # Auxiliary loss head (helps training convergence — used only during training)
        # Applied at layer3 output, not layer4
        self.aux_head = nn.Sequential(
            nn.Conv2d(backbone_out_ch // 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_size = (x.shape[2], x.shape[3])

        # Backbone — save layer3 output for aux head
        feat0  = self.backbone.layer0(x)
        feat1  = self.backbone.layer1(feat0)
        feat2  = self.backbone.layer2(feat1)
        feat3  = self.backbone.layer3(feat2)   # aux head applied here
        feat4  = self.backbone.layer4(feat3)   # main head applied here

        # Main path: PPM → bottleneck → classifier
        ppm_out   = self.ppm(feat4)
        bottleneck = self.bottleneck(ppm_out)
        logits     = self.classifier(bottleneck)

        # Upsample back to input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)

        if return_aux and self.training:
            aux_logits = self.aux_head(feat3)
            aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=True)
            return logits, aux_logits

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        Returns argmax class map (B, H, W) — no gradients.
        """
        with torch.no_grad():
            logits = self.forward(x, return_aux=False)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probability map (B, num_classes, H, W).
        Useful for computing fill percentage with confidence weighting.
        """
        with torch.no_grad():
            logits = self.forward(x, return_aux=False)
            return F.softmax(logits, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_water_model(
    backbone:   str  = "resnet50",
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> PSPNet:
    """
    Build PSPNet for water bottle segmentation.
    4 classes: background / bottle wall / water fill / empty air
    """
    model = PSPNet(num_classes=4, backbone=backbone, pretrained=pretrained)
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
    return model.to(device).eval()


def build_food_model(
    backbone:   str  = "resnet50",
    pretrained: bool = True,
    weights_path: Optional[str] = None,
    device: str = "cpu",
) -> PSPNet:
    """
    Build PSPNet for food hopper segmentation.
    4 classes: background / hopper frame / food pellets / empty space
    """
    model = PSPNet(num_classes=4, backbone=backbone, pretrained=pretrained)
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
    return model.to(device).eval()