# VNWoodKnot Data Profile

Generated from the local extracted dataset by `timbersight-profile`.

## Dataset composition

- Total images: **1515**
- Positive images: **1015**
- Knot-free negative images: **500**

| Split | Knot-free | Live knot | Dead knot |
| --- | ---: | ---: | ---: |
| train | 350 | 363 | 347 |
| validation | 75 | 77 | 74 |
| test | 75 | 79 | 75 |

## Detection objects

| Class | Objects |
| --- | ---: |
| dead_knot | 502 |
| live_knot | 519 |

## Geometry and appearance

- Median bounding-box area: **10.03%** of the image.
- Bounding-box area interquartile range: **6.57%–15.45%**.
- Median bounding-box aspect ratio: **1.013**.
- Mean normalized luminance: **0.592**.
- Luminance range: **0.364–0.811**.
- Mean within-image luminance contrast: **0.141**.

## Experimental implications

- Preserve knot-free images as negative detector examples.
- Report per-class metrics because live and dead knots are different operational cases.
- Include brightness and contrast perturbations in robustness evaluation.
- Evaluate small and large boxes separately because knot scale varies materially.
- Use the provided test split only for final evaluation.
