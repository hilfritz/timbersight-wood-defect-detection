# TimberSight Baseline Model Card

## Model

The baseline is a TorchVision Faster R-CNN detector with a MobileNetV3-Large FPN
backbone. Its outputs are:

- `1`: live knot
- `2`: dead knot

Model label `0` is reserved for background. Knot-free images are retained during
training as negative examples.

## Intended use

This model is an experimental computer-vision prototype for studying reproducible
wood-knot detection. It may be used to demonstrate dataset validation,
transfer learning, evaluation and software integration.

It is not approved for production grading, structural assessment, worker-safety
decisions or autonomous equipment control.

## Dataset

VNWoodKnot contains 1,515 wood-surface images representing live knots, dead knots
and knot-free wood. The dataset is distributed under CC BY 4.0.

- Dataset DOI: https://doi.org/10.17632/vnst548g5n.1
- Data article DOI: https://doi.org/10.1016/j.dib.2025.112039
- Local profile: [docs/DATA_PROFILE.md](docs/DATA_PROFILE.md)

## Evaluation

The checkpoint was selected using overall validation AP50. The held-out test
split was evaluated once after selection.

| Metric | Test result |
| --- | ---: |
| Live-knot AP50 | 0.9031 |
| Dead-knot AP50 | 0.9050 |
| Overall mAP50 | 0.9040 |
| Mean CPU latency | 64.1 ms/image |

Full experiment details are in [docs/RESULTS.md](docs/RESULTS.md).

## Limitations

- Images come from a limited acquisition setup and may not represent other
  cameras, wood species, mills or environmental conditions.
- The model detects only two knot categories and does not assess structural
  strength or lumber grade.
- Robustness to motion blur, illumination shifts, occlusion and sensor noise has
  not yet been quantified.
- The score threshold has not been calibrated to a business cost function.
- Real production deployment requires additional validation, monitoring and a
  safe failure mode.

## Reproducibility

- Training seed: `42`
- Checkpoint SHA-256: `b8bfc20fd1ae1563205d6e2c9571e99f2fbab9f7a15c888826afd9e0dd8e4b26`
- Test metrics file: `artifacts/runs/20260930T035750Z/test-metrics.json`
