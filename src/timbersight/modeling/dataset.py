"""TorchVision dataset adapter for VNWoodKnot object detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from timbersight.data.validate import DatasetConfig


MODEL_CLASS_NAMES = {1: "live_knot", 2: "dead_knot"}


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    category_folder: str


class VNWoodKnotDetection(Dataset):
    """Return images and Faster R-CNN targets, including empty negative targets."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transforms: Callable | None = None,
    ) -> None:
        if split not in config.splits:
            raise ValueError(f"Unknown split {split!r}; expected one of {config.splits}")
        self.config = config
        self.split = split
        self.transforms = transforms
        self.samples = [
            DetectionSample(path, folder_id)
            for folder_id in config.image_categories
            for path in sorted((config.root / split / folder_id).glob("*.jpg"))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []
        detection_class = self.config.category_detection_class.get(
            sample.category_folder
        )
        if detection_class is not None:
            label_path = sample.image_path.with_suffix(".txt")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x_center, y_center, box_width, box_height = line.split()
                class_id_int = int(class_id)
                if class_id_int != detection_class:
                    raise ValueError(
                        f"Unexpected class {class_id_int} in {label_path}; "
                        f"expected {detection_class}"
                    )
                x_center_px = float(x_center) * width
                y_center_px = float(y_center) * height
                box_width_px = float(box_width) * width
                box_height_px = float(box_height) * height
                boxes.append(
                    [
                        x_center_px - box_width_px / 2,
                        y_center_px - box_height_px / 2,
                        x_center_px + box_width_px / 2,
                        y_center_px + box_height_px / 2,
                    ]
                )
                # Faster R-CNN reserves label 0 for background.
                labels.append(class_id_int + 1)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes_tensor)
            else torch.zeros((0,), dtype=torch.float32)
        )

        image_tensor = tv_tensors.Image(F.to_image(image))
        target: dict[str, Tensor] = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_tensor, format="XYXY", canvas_size=(height, width)
            ),
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target

    def get_height_and_width(self, index: int) -> tuple[int, int]:
        with Image.open(self.samples[index].image_path) as image:
            width, height = image.size
        return height, width


def collate_detection_batch(batch):
    return tuple(zip(*batch))


class BalancedDetectionBatchSampler:
    """Pair every negative with a positive to avoid background-only batches."""

    def __init__(self, dataset, batch_size: int, seed: int) -> None:
        if batch_size != 2:
            raise ValueError("BalancedDetectionBatchSampler currently requires batch_size=2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.negative_indices: list[int] = []
        self.positive_indices: list[int] = []

        for local_index in range(len(dataset)):
            sample = self._sample(local_index)
            base_dataset = self._base_dataset()
            if sample.category_folder in base_dataset.config.category_detection_class:
                self.positive_indices.append(local_index)
            else:
                self.negative_indices.append(local_index)

        if len(self.positive_indices) < len(self.negative_indices):
            raise ValueError("A positive partner is required for every negative image")

    def _base_dataset(self) -> VNWoodKnotDetection:
        return self.dataset.dataset if isinstance(self.dataset, torch.utils.data.Subset) else self.dataset

    def _sample(self, local_index: int) -> DetectionSample:
        if isinstance(self.dataset, torch.utils.data.Subset):
            return self.dataset.dataset.samples[self.dataset.indices[local_index]]
        return self.dataset.samples[local_index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        negatives = self.negative_indices.copy()
        positives = self.positive_indices.copy()
        generator.shuffle(negatives)
        generator.shuffle(positives)

        batches = [
            [negative, positives[index]] for index, negative in enumerate(negatives)
        ]
        remaining = positives[len(negatives) :]
        batches.extend(
            [remaining[index : index + self.batch_size] for index in range(0, len(remaining), self.batch_size)]
        )
        generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        remaining_positives = len(self.positive_indices) - len(self.negative_indices)
        positive_only_batches = (remaining_positives + self.batch_size - 1) // self.batch_size
        return len(self.negative_indices) + positive_only_batches
