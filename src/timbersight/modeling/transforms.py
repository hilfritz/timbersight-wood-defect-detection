"""Bounding-box-aware augmentation pipelines."""

from __future__ import annotations

import torch
from torchvision.transforms import v2


def detection_transforms(training: bool) -> v2.Compose:
    transforms: list[object] = []
    if training:
        transforms.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.25, contrast=0.25)], p=0.5
                ),
            ]
        )
    transforms.extend([v2.ToDtype(torch.float32, scale=True), v2.ToPureTensor()])
    return v2.Compose(transforms)

