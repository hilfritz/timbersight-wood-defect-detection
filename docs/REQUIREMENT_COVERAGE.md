# Role Requirement Coverage

Evidence mode: **Bridge**. Completed items are supported by code, tests or
measured artifacts in this repository. Planned items identify transferable
engineering direction and must not be presented as completed experience.

| Role emphasis | Project evidence | Status |
| --- | --- | --- |
| Prepare and document image datasets | Dataset provenance, schema, validator, QA checks and generated profile | Implemented |
| Repeatable experiments with defined metrics | Versioned configuration, fixed splits, AP50 evaluation and run artifacts | Implemented and measured |
| Python and computer vision | PyTorch/TorchVision Faster R-CNN training and inference pipeline | Implemented and trained |
| Critical evaluation and reproducibility | Held-out test evaluation, checkpoint hash, fixed seed, fail-fast numerical guards and documented limitations | Implemented; robustness work remains |
| Data-quality checks | Image, label, coordinate, bounds, class and split-leakage checks | Implemented and tested |
| Changing lighting and motion | Brightness, contrast, blur and motion robustness evaluation | Planned |
| Application and API development | FastAPI image endpoint plus upload, camera capture and continuous browser inspection | Implemented |
| Monitoring and debugging | Health endpoint, device reporting, class counts, confidence scores, latency and live frame rate | Prototype implemented |
| C#/.NET product integration | Proposed .NET client consuming the Python inference API | Planned; supported by professional background, not this repository |
| Relational databases | Proposed storage for inspections and experiment metadata | Planned; not implemented here |
| Documentation and review | README, technical decisions, results report, model card, tests and CI | Implemented |
| 3D geometry and point clouds | Not covered by the VNWoodKnot release | Explicit gap |
| Industrial site validation | No site access, production hardware or representative mill data | Explicit gap |

## Honest interview framing

TimberSight demonstrates the transferable workflow: validate data, run a
controlled experiment, evaluate an untouched split, package inference behind an
API and expose observable results in an application. It does not establish that
the model is ready for lumber grading or sawmill deployment. The next engineering
step is testing with representative cameras, species, lighting, motion and
operational cost thresholds.
