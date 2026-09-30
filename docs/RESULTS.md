# Baseline Results

## Experiment identity

- Run ID: `20260930T035750Z`
- Model: Faster R-CNN MobileNetV3-Large 320 FPN
- Initialization: TorchVision COCO pretrained weights
- Seed: `42`
- Optimizer: SGD with momentum, first-epoch warmup and step decay
- Training images: 1,060
- Validation images: 226
- Test images: 229
- Checkpoint SHA-256: `b8bfc20fd1ae1563205d6e2c9571e99f2fbab9f7a15c888826afd9e0dd8e4b26`

## Validation history

| Epoch | Learning rate | Mean training loss | Live-knot AP50 | Dead-knot AP50 | mAP50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0050 | 1.4203 | 0.6516 | 0.8725 | 0.7621 |
| 2 | 0.0050 | 1.4057 | 0.7883 | 0.9135 | 0.8509 |
| 3 | 0.0050 | 1.4099 | 0.7058 | 0.8822 | 0.7940 |
| 4 | 0.0005 | 1.0982 | 0.8468 | 0.9204 | 0.8836 |
| 5 | 0.0005 | 1.0677 | 0.8396 | 0.9286 | **0.8841** |

Epoch 5 was selected before opening the held-out test results because it had the
highest overall validation mAP50.

## Held-out test result

| Metric | Result |
| --- | ---: |
| Live-knot AP50 | **0.9031** |
| Dead-knot AP50 | **0.9050** |
| Overall mAP50 | **0.9040** |
| Mean CPU inference latency | **64.1 ms/image** |
| Evaluated images | **229** |

The AP implementation uses one-to-one matching at IoU 0.50 and includes false
positives from knot-free images. Predictions below score `0.05` are excluded.
These numbers are not COCO AP50:95 and should not be described as such.

## Interpretation

- Test performance is balanced across live and dead knots.
- The retained knot-free images make false-positive behaviour part of the metric.
- The test result is slightly higher than validation performance; this does not
  establish generalization to other wood species, cameras or production lines.
- CPU latency is a local benchmark, not a production throughput guarantee.

## Known gaps

- No robustness benchmark for lighting, blur, motion or perspective yet.
- No confidence calibration or threshold-selection analysis yet.
- No COCO AP50:95, small/medium/large AP or recall metrics yet.
- No external dataset or sawmill-site validation.
- No 3D geometry, calibration or point-cloud evaluation.

