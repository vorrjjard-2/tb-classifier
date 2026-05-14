"""Feature-extracting backbones.

A backbone takes an image batch and returns pooled feature vectors of
shape ``(N, feat_dim)``. The classifier head sits on top and maps to
class logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .blocks.flipr import FlipRBlock


def build_backbone(
    arch: str = "resnet50",
    pretrained: bool = True,
    keep_stages: int = 4,
    use_flipr: bool = False,
) -> tuple[nn.Module, int]:
    """Return ``(backbone, feature_dim)``.

    The backbone module emits pooled features ready for a linear head.

    When ``keep_stages == 4`` and ``use_flipr is False`` (the default),
    returns the torchvision ResNet with its final FC replaced by ``Identity``
    so existing checkpoints keep loading.

    Otherwise returns a :class:`FlipRResNet` that exposes stages explicitly,
    optionally inserts a :class:`FlipRBlock` after ``layer2``, and skips
    deeper stages beyond ``keep_stages``.
    """
    if not 1 <= keep_stages <= 4:
        raise ValueError(f"keep_stages must be in 1..4, got {keep_stages}")

    if arch not in {"resnet18", "resnet50"}:
        raise ValueError(f"Unknown backbone arch: {arch!r}")

    if keep_stages == 4 and not use_flipr:
        return _build_full_torchvision(arch, pretrained)

    model = FlipRResNet(
        arch=arch,
        pretrained=pretrained,
        keep_stages=keep_stages,
        use_flipr=use_flipr,
    )
    return model, model.feat_dim


def _build_full_torchvision(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class FlipRResNet(nn.Module):
    """ResNet18/50 with explicit stages, optional FlipR after layer2, and depth stripping.

    The deepest stages (``layer4``, then ``layer3``, then ``layer2``) are dropped
    when ``keep_stages`` is < 4. The resulting ``feat_dim`` matches the last kept
    stage's output channel count; the linear head adapts automatically.
    """

    def __init__(self, arch: str, pretrained: bool, keep_stages: int, use_flipr: bool):
        super().__init__()
        if arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            src = resnet50(weights=weights)
            last_conv_attr = "conv3"
        elif arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            src = resnet18(weights=weights)
            last_conv_attr = "conv2"
        else:
            raise ValueError(f"Unknown backbone arch: {arch!r}")

        self.keep_stages = keep_stages
        self.use_flipr = use_flipr

        self.conv1 = src.conv1
        self.bn1 = src.bn1
        self.relu = src.relu
        self.maxpool = src.maxpool

        stages = [src.layer1, src.layer2, src.layer3, src.layer4][:keep_stages]
        for i, stage in enumerate(stages, start=1):
            setattr(self, f"layer{i}", stage)

        if use_flipr:
            if keep_stages < 2:
                raise ValueError("use_flipr requires keep_stages >= 2 (FlipR sits after layer2)")
            layer2_out_channels = getattr(src.layer2[-1], last_conv_attr).out_channels
            self.flipr = FlipRBlock(in_channels=layer2_out_channels)
        else:
            self.flipr = None

        last_stage = stages[-1]
        self.feat_dim = getattr(last_stage[-1], last_conv_attr).out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.keep_stages >= 2:
            x = self.layer2(x)
            if self.flipr is not None:
                x = self.flipr(x)
        if self.keep_stages >= 3:
            x = self.layer3(x)
        if self.keep_stages >= 4:
            x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)
