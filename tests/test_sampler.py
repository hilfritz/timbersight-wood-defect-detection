from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from timbersight.data.validate import DatasetConfig
from timbersight.modeling.dataset import (
    BalancedDetectionBatchSampler,
    VNWoodKnotDetection,
)


def test_balanced_sampler_never_emits_an_all_negative_batch(tmp_path: Path) -> None:
    config = DatasetConfig(
        root=tmp_path,
        splits=("train",),
        image_categories={"0": "knot_free", "1": "live_knot", "2": "dead_knot"},
        detection_classes={0: "live_knot", 1: "dead_knot"},
        category_detection_class={"1": 0, "2": 1},
    )
    for folder, count in {"0": 3, "1": 4, "2": 2}.items():
        directory = tmp_path / "train" / folder
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{folder}_{index}.jpg").touch()

    dataset = VNWoodKnotDetection(config, "train")
    sampler = BalancedDetectionBatchSampler(dataset, batch_size=2, seed=42)
    batches = list(sampler)

    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(dataset)))
    for batch in batches:
        folders = {dataset.samples[index].category_folder for index in batch}
        assert folders != {"0"}

