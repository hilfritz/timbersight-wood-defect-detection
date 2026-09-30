"""Transparent AP50 metrics without an external evaluation service."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor


def box_iou(box: Tensor, boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(torch.float32).eps)


def _average_precision(recalls: Tensor, precisions: Tensor) -> float:
    padded_recalls = torch.cat(
        [torch.tensor([0.0]), recalls.cpu(), torch.tensor([1.0])]
    )
    padded_precisions = torch.cat(
        [torch.tensor([0.0]), precisions.cpu(), torch.tensor([0.0])]
    )
    for index in range(len(padded_precisions) - 2, -1, -1):
        padded_precisions[index] = torch.maximum(
            padded_precisions[index], padded_precisions[index + 1]
        )
    change_points = torch.where(padded_recalls[1:] != padded_recalls[:-1])[0]
    ap = torch.sum(
        (padded_recalls[change_points + 1] - padded_recalls[change_points])
        * padded_precisions[change_points + 1]
    )
    return float(ap)


def detection_ap50(
    predictions: Iterable[dict[str, Tensor]],
    targets: Iterable[dict[str, Tensor]],
    class_ids: tuple[int, ...] = (1, 2),
    score_threshold: float = 0.05,
) -> dict[str, float | int | dict[str, float | int]]:
    prediction_list = list(predictions)
    target_list = list(targets)
    if len(prediction_list) != len(target_list):
        raise ValueError("Predictions and targets must have the same length")

    class_metrics: dict[str, dict[str, float | int]] = {}
    aps: list[float] = []
    for class_id in class_ids:
        ground_truths: dict[int, Tensor] = {}
        matched: dict[int, set[int]] = defaultdict(set)
        total_ground_truths = 0
        detections: list[tuple[float, int, Tensor]] = []

        for image_index, (prediction, target) in enumerate(
            zip(prediction_list, target_list)
        ):
            target_mask = target["labels"].cpu() == class_id
            class_boxes = target["boxes"].cpu()[target_mask]
            ground_truths[image_index] = class_boxes
            total_ground_truths += len(class_boxes)

            prediction_mask = (
                (prediction["labels"].cpu() == class_id)
                & (prediction["scores"].cpu() >= score_threshold)
            )
            for score, box in zip(
                prediction["scores"].cpu()[prediction_mask],
                prediction["boxes"].cpu()[prediction_mask],
            ):
                detections.append((float(score), image_index, box))

        detections.sort(key=lambda item: item[0], reverse=True)
        true_positives = torch.zeros(len(detections), dtype=torch.float32)
        false_positives = torch.zeros(len(detections), dtype=torch.float32)
        for detection_index, (_, image_index, box) in enumerate(detections):
            ious = box_iou(box, ground_truths[image_index])
            if len(ious) == 0:
                false_positives[detection_index] = 1
                continue
            best_iou, best_index = torch.max(ious, dim=0)
            best_index_int = int(best_index)
            if best_iou >= 0.5 and best_index_int not in matched[image_index]:
                true_positives[detection_index] = 1
                matched[image_index].add(best_index_int)
            else:
                false_positives[detection_index] = 1

        cumulative_tp = torch.cumsum(true_positives, dim=0)
        cumulative_fp = torch.cumsum(false_positives, dim=0)
        recalls = cumulative_tp / max(total_ground_truths, 1)
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1)
        ap = _average_precision(recalls, precisions) if total_ground_truths else 0.0
        aps.append(ap)
        class_metrics[str(class_id)] = {
            "ap50": round(ap, 6),
            "ground_truths": total_ground_truths,
            "detections": len(detections),
        }

    return {
        "map50": round(sum(aps) / len(aps), 6) if aps else 0.0,
        "classes": class_metrics,
        "score_threshold": score_threshold,
    }

