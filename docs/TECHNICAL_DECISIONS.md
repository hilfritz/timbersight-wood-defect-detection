# Technical Decisions

## TD-001: Begin with one dataset

The first release uses VNWoodKnot only. Mixing datasets with different capture
conditions, taxonomies and annotation types would make the first experiment
harder to interpret.

## TD-002: Object detection before classification

The annotations contain knot bounding boxes, so the baseline must demonstrate
localization rather than only image-level classification. Knot-free images remain
in every split as background-only negatives.

## TD-003: Shift labels for TorchVision

VNWoodKnot uses YOLO object IDs `0` for live knots and `1` for dead knots.
TorchVision detection models reserve `0` for background. The adapter therefore
maps live knots to model label `1` and dead knots to model label `2`.

## TD-004: Lightweight transfer-learning baseline

The baseline is Faster R-CNN with a MobileNetV3-FPN backbone. It offers a clear,
well-documented transfer-learning path and is more practical for local iteration
than a heavier ResNet-FPN baseline. Accuracy and latency will both be reported.

## TD-005: AP50 implementation remains inspectable

The initial evaluation implements class-wise AP at IoU 0.50 directly. This makes
negative-image false positives and one-to-one matching behaviour easy to audit.
Later work may add COCO-style AP across multiple IoU thresholds.

## TD-006: Follow the reference detection optimizer recipe

The baseline uses SGD with momentum, weight decay and step learning-rate decay,
matching the optimizer family in the official TorchVision detection fine-tuning
tutorial. The first epoch linearly warms the learning rate from `1/1000` of its
configured value, following the reference engine's stabilization pattern.
Training fails immediately on non-finite losses or gradients instead of silently
recording an invalid experiment.

## TD-007: Retain negatives without background-only training batches

All 350 knot-free training images remain in the experiment. A deterministic
sampler pairs each negative with a positive image, then batches the remaining
positives together. This preserves false-positive learning while avoiding a
numerical instability observed in repeated background-only Faster R-CNN batches.
