"""Medical-relevant metrics for TB classification.

For a 3-class TB screening model, the most informative numbers are
macro AUROC and per-class recall (recall on the TB class is sensitivity
for TB; recall on healthy is the TPR for ruling out disease).
``build_metrics`` returns a :class:`torchmetrics.MetricCollection` and
:func:`flatten_metrics` expands per-class tensor metrics into scalar
log entries keyed by class name.
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassRecall,
    MulticlassSpecificity,
)

CLASS_NAMES: tuple[str, ...] = ("healthy", "sick-non-tb", "tb")


def build_metrics(num_classes: int = 3, prefix: str = "") -> MetricCollection:
    return MetricCollection(
        {
            "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "auroc_macro": MulticlassAUROC(num_classes=num_classes, average="macro"),
            "recall_per_class": MulticlassRecall(num_classes=num_classes, average=None),
            "specificity_per_class": MulticlassSpecificity(num_classes=num_classes, average=None),
        },
        prefix=prefix,
    )


def flatten_metrics(
    computed: dict[str, torch.Tensor],
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, torch.Tensor]:
    """Turn per-class tensor entries into one scalar entry per class.

    Keys ending in ``_per_class`` are split — e.g. ``val/recall_per_class``
    becomes ``val/recall/healthy``, ``val/recall/sick``, ``val/recall/tb``.
    """
    flat: dict[str, torch.Tensor] = {}
    for key, value in computed.items():
        if key.endswith("_per_class") and value.ndim == 1:
            stem = key[: -len("_per_class")]
            base, _, last = stem.rpartition("/")
            prefix = f"{base}/" if base else ""
            for cls_name, scalar in zip(class_names, value):
                flat[f"{prefix}{last}/{cls_name}"] = scalar
        else:
            flat[key] = value
    return flat
