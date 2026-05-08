"""Albumentations pipelines for TB chest X-ray classification.

Both pipelines are bbox-aware so the same dataset record (image + COCO bboxes
+ category labels) flows through unchanged. Bboxes use COCO format
(``[x, y, w, h]``) to match the source JSONs.

ImageNet mean/std is used so the pipeline is drop-in compatible with
ImageNet-pretrained backbones; swap to dataset stats if training from scratch.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        coord_format="coco",
        label_fields=["bbox_labels"],
        min_visibility=0.3,
        min_area=1.0,
    )


def get_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_range=(-0.15, 0.15),
                contrast_range=(-0.15, 0.15),
                p=0.5,
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )
