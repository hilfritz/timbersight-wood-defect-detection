# Data

Raw dataset files are intentionally excluded from Git.

## Source

- **Dataset:** VNWoodKnot: A benchmark image dataset for wood knot detection and classification
- **DOI:** https://doi.org/10.17632/vnst548g5n.1
- **License:** CC BY 4.0

Download and extract the dataset beside this repository, or update
`config/dataset.toml` to point to another local location.

## Confirmed archive mapping

The extracted archive uses category folders and YOLO detection labels differently:

| Image folder | Meaning | YOLO object class |
| --- | --- | --- |
| `0` | Knot-free wood | None (negative image) |
| `1` | Live knot | `0` |
| `2` | Dead knot | `1` |

Do not invent an empty bounding box for knot-free images. They are retained as
background-only negatives for the detector.

