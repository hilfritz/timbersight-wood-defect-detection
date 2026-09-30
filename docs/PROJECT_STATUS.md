# Project Documentation

The repository is the source of truth. Published material must report only
results reproduced from versioned configurations and saved evaluation outputs.

## Published in this repository

| Asset | Status | Location |
| --- | --- | --- |
| Project overview | Complete | [`README.md`](../README.md) |
| Dataset profile | Complete | [`DATA_PROFILE.md`](DATA_PROFILE.md) |
| Held-out evaluation report | Complete | [`RESULTS.md`](RESULTS.md) |
| Model card | Complete | [`MODEL_CARD.md`](../MODEL_CARD.md) |
| Technical decision record | Complete | [`TECHNICAL_DECISIONS.md`](TECHNICAL_DECISIONS.md) |
| Camera and API guide | Complete | [`WEB_APP.md`](../WEB_APP.md) |
| Automated tests and CI | Complete | [`tests/`](../tests) and [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| Detection examples | Complete | [`demo_images/`](../demo_images) |
| Short demo video | Complete | [`demo_timbersight.mp4`](../demo_images/demo_timbersight.mp4) |

## Future additions

1. A robustness report covering lighting, blur, motion, noise and perspective.
2. Confidence-threshold selection tied to false-positive and false-negative
   costs.
3. External-camera or production-site validation.
4. A C#/.NET integration client and persistent inspection records.
5. A concise technical presentation.

## Release rule

Accuracy and performance claims must be traceable to saved evaluation outputs.
The demo may illustrate workflow, but it must not imply production validation,
structural grading or performance on untested cameras and sawmill conditions.
