"""Model construction for the TimberSight detection baseline."""

from __future__ import annotations

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(pretrained: bool = True):
    """Build a three-output Faster R-CNN: background, live knot, dead knot."""
    if pretrained:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        weights_backbone = None
    else:
        weights = None
        weights_backbone = None

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
    return model

