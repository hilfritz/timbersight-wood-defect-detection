import pytest

torch = pytest.importorskip("torch")

from timbersight.modeling.metrics import detection_ap50


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _prediction(boxes, labels, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def test_perfect_predictions_have_perfect_map50() -> None:
    targets = [
        _target([[0, 0, 10, 10]], [1]),
        _target([[10, 10, 20, 20]], [2]),
    ]
    predictions = [
        _prediction([[0, 0, 10, 10]], [1], [0.9]),
        _prediction([[10, 10, 20, 20]], [2], [0.8]),
    ]

    metrics = detection_ap50(predictions, targets)

    assert metrics["map50"] == 1.0


def test_prediction_on_negative_image_is_false_positive() -> None:
    targets = [
        _target([], []),
        _target([[0, 0, 10, 10]], [1]),
    ]
    predictions = [
        _prediction([[0, 0, 10, 10]], [1], [0.95]),
        _prediction([[0, 0, 10, 10]], [1], [0.9]),
    ]

    metrics = detection_ap50(predictions, targets, class_ids=(1,))

    assert metrics["map50"] == 0.5

