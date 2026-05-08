"""TBX11K dataset for image-level TB classification with optional bboxes.

Image-level label scheme (3-class):
    0 = healthy        (file path starts with 'health/')
    1 = sick non-TB    (file path starts with 'sick/')
    2 = TB             (file path starts with 'tb/')

Bbox annotations come from the COCO-style JSONs under annotations/json/.
Only TB images carry bboxes (599 of 600 in train); other images get an
empty bbox list. Bbox category ids follow the COCO file:
    1 = ActiveTuberculosis
    2 = ObsoletePulmonaryTuberculosis
    3 = PulmonaryTuberculosis
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_HEALTHY = 0
LABEL_SICK = 1
LABEL_TB = 2

_PREFIX_TO_LABEL = {
    "health": LABEL_HEALTHY,
    "sick": LABEL_SICK,
    "tb": LABEL_TB,
}


def _label_from_path(file_name: str) -> int:
    prefix = file_name.split("/", 1)[0]
    if prefix not in _PREFIX_TO_LABEL:
        raise ValueError(f"Unexpected image folder prefix in {file_name!r}")
    return _PREFIX_TO_LABEL[prefix]


class TBDataset(Dataset):
    """COCO-backed TBX11K classification dataset.

    Parameters
    ----------
    root:
        Path to the TBX11K dataset root (the directory containing `imgs/`
        and `annotations/`).
    split:
        One of {"train", "val", "trainval"}. Picks the matching
        TBX11K_<split>.json file.
    transform:
        Albumentations Compose (or any callable) accepting
        ``image=np.ndarray, bboxes=list, bbox_labels=list`` and returning
        a dict with the same keys (with ``image`` typically a tensor).
        If None, the raw RGB uint8 array is returned.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "trainval"}:
            raise ValueError(f"split must be train/val/trainval, got {split!r}")

        self.root = Path(root)
        self.imgs_dir = self.root / "imgs"
        self.split = split
        self.transform = transform

        ann_path = self.root / "annotations" / "json" / f"TBX11K_{split}.json"
        with open(ann_path) as f:
            coco = json.load(f)

        boxes_by_image: dict[int, list[tuple[list[float], int]]] = defaultdict(list)
        for ann in coco["annotations"]:
            boxes_by_image[ann["image_id"]].append((list(ann["bbox"]), ann["category_id"]))

        self.records: list[dict] = []
        for img in coco["images"]:
            file_name = img["file_name"]
            boxes = boxes_by_image.get(img["id"], [])
            self.records.append(
                {
                    "file_name": file_name,
                    "label": _label_from_path(file_name),
                    "bboxes": [b for b, _ in boxes],
                    "bbox_labels": [c for _, c in boxes],
                    "width": img["width"],
                    "height": img["height"],
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        path = self.imgs_dir / rec["file_name"]

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [list(b) for b in rec["bboxes"]]
        bbox_labels = list(rec["bbox_labels"])

        if self.transform is not None:
            out = self.transform(image=image, bboxes=bboxes, bbox_labels=bbox_labels)
            image = out["image"]
            bboxes = out["bboxes"]
            bbox_labels = out["bbox_labels"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": int(rec["label"]),
            "bboxes": [list(map(float, b)) for b in bboxes],
            "bbox_labels": [int(c) for c in bbox_labels],
            "file_name": rec["file_name"],
        }

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for r in self.records:
            counts[r["label"]] += 1
        return counts
