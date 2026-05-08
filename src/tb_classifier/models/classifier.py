"""Top-level TB classifier: backbone + linear head."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .backbone import build_backbone


class TBClassifier(nn.Module):
    def __init__(self, arch: str = "resnet50", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch=arch, pretrained=pretrained)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_classifier(cfg: ModelConfig) -> TBClassifier:
    return TBClassifier(arch=cfg.arch, num_classes=cfg.num_classes, pretrained=cfg.pretrained)
