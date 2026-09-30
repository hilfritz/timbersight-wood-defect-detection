from pathlib import Path

import pytest
import torch

from timbersight.inference import format_detections, resolve_checkpoint


def test_format_detections_filters_and_clamps_boxes():
    prediction = {
        "boxes": torch.tensor([[-2, 4, 120, 90], [1, 1, 3, 3]]),
        "labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.91, 0.20]),
    }

    result = format_detections(prediction, threshold=0.5, width=100, height=80)

    assert result == [
        {
            "class_id": 1,
            "label": "live knot",
            "confidence": pytest.approx(0.91),
            "box": [0.0, 4.0, 100.0, 80.0],
        }
    ]


def test_resolve_checkpoint_uses_explicit_file(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    assert resolve_checkpoint(checkpoint) == checkpoint.resolve()


def test_resolve_checkpoint_rejects_missing_explicit_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing.pt")

