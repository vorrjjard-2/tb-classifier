"""Feature-extracting backbones.

A backbone takes an image batch and returns pooled feature vectors of
shape ``(N, feat_dim)``. The classifier head sits on top and maps to
class logits.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def build_backbone(arch: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return ``(backbone, feature_dim)``.

    The backbone module emits pooled features ready for a linear head:
    its final classification layer is replaced with ``nn.Identity`` so
    callers don't have to know the original FC shape.
    """
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    raise ValueError(f"Unknown backbone arch: {arch!r}")
