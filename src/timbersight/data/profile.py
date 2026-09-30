"""Create a reproducible exploratory profile for VNWoodKnot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat

from timbersight.data.validate import DatasetConfig, load_config


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    items = sorted(values)
    if not items:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        position = fraction * (len(items) - 1)
        lower = int(position)
        upper = min(lower + 1, len(items) - 1)
        weight = position - lower
        return items[lower] * (1 - weight) + items[upper] * weight

    return {
        "count": len(items),
        "min": round(items[0], 6),
        "p25": round(percentile(0.25), 6),
        "median": round(statistics.median(items), 6),
        "mean": round(statistics.fmean(items), 6),
        "p75": round(percentile(0.75), 6),
        "max": round(items[-1], 6),
    }


def _read_yolo_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append(
            (
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
        )
    return boxes


def build_profile(config: DatasetConfig) -> dict[str, Any]:
    split_category_counts: dict[str, dict[str, int]] = {}
    split_object_counts: dict[str, Counter[str]] = {}
    object_counts: Counter[str] = Counter()
    objects_per_image: list[float] = []
    box_area_fractions: list[float] = []
    box_aspect_ratios: list[float] = []
    luminance_means: list[float] = []
    luminance_contrasts: list[float] = []
    negative_images = 0
    total_images = 0

    for split in config.splits:
        category_counts: dict[str, int] = {}
        split_objects: Counter[str] = Counter()
        for folder_id, category_name in config.image_categories.items():
            category_dir = config.root / split / folder_id
            images = sorted(category_dir.glob("*.jpg"))
            category_counts[category_name] = len(images)
            total_images += len(images)

            for image_path in images:
                with Image.open(image_path) as image:
                    grayscale = image.convert("L").resize((64, 64))
                    stats = ImageStat.Stat(grayscale)
                    luminance_means.append(stats.mean[0] / 255.0)
                    luminance_contrasts.append(stats.stddev[0] / 255.0)

                detection_class = config.category_detection_class.get(folder_id)
                if detection_class is None:
                    negative_images += 1
                    objects_per_image.append(0.0)
                    continue

                boxes = _read_yolo_boxes(image_path.with_suffix(".txt"))
                objects_per_image.append(float(len(boxes)))
                for class_id, _, _, width, height in boxes:
                    class_name = config.detection_classes[class_id]
                    object_counts[class_name] += 1
                    split_objects[class_name] += 1
                    box_area_fractions.append(width * height)
                    box_aspect_ratios.append(width / height)

        split_category_counts[split] = category_counts
        split_object_counts[split] = split_objects

    return {
        "dataset_root": str(config.root),
        "total_images": total_images,
        "negative_images": negative_images,
        "positive_images": total_images - negative_images,
        "split_category_counts": split_category_counts,
        "split_object_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in split_object_counts.items()
        },
        "object_counts": dict(sorted(object_counts.items())),
        "objects_per_image": _summary(objects_per_image),
        "box_area_fraction": _summary(box_area_fractions),
        "box_aspect_ratio": _summary(box_aspect_ratios),
        "normalized_luminance": _summary(luminance_means),
        "normalized_luminance_contrast": _summary(luminance_contrasts),
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# VNWoodKnot Data Profile",
        "",
        "Generated from the local extracted dataset by `timbersight-profile`.",
        "",
        "## Dataset composition",
        "",
        f"- Total images: **{profile['total_images']}**",
        f"- Positive images: **{profile['positive_images']}**",
        f"- Knot-free negative images: **{profile['negative_images']}**",
        "",
        "| Split | Knot-free | Live knot | Dead knot |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, counts in profile["split_category_counts"].items():
        lines.append(
            f"| {split} | {counts['knot_free']} | "
            f"{counts['live_knot']} | {counts['dead_knot']} |"
        )

    lines.extend(
        [
            "",
            "## Detection objects",
            "",
            "| Class | Objects |",
            "| --- | ---: |",
        ]
    )
    for name, count in profile["object_counts"].items():
        lines.append(f"| {name} | {count} |")

    area = profile["box_area_fraction"]
    aspect = profile["box_aspect_ratio"]
    luminance = profile["normalized_luminance"]
    contrast = profile["normalized_luminance_contrast"]
    lines.extend(
        [
            "",
            "## Geometry and appearance",
            "",
            f"- Median bounding-box area: **{area['median']:.2%}** of the image.",
            f"- Bounding-box area interquartile range: **{area['p25']:.2%}–{area['p75']:.2%}**.",
            f"- Median bounding-box aspect ratio: **{aspect['median']:.3f}**.",
            f"- Mean normalized luminance: **{luminance['mean']:.3f}**.",
            f"- Luminance range: **{luminance['min']:.3f}–{luminance['max']:.3f}**.",
            f"- Mean within-image luminance contrast: **{contrast['mean']:.3f}**.",
            "",
            "## Experimental implications",
            "",
            "- Preserve knot-free images as negative detector examples.",
            "- Report per-class metrics because live and dead knots are different operational cases.",
            "- Include brightness and contrast perturbations in robustness evaluation.",
            "- Evaluate small and large boxes separately because knot scale varies materially.",
            "- Use the provided test split only for final evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new_or_confirmed(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/dataset.toml"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("artifacts/data-profile.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("docs/DATA_PROFILE.md")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    profile = build_profile(config)
    _write_new_or_confirmed(
        args.json_output,
        json.dumps(profile, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    _write_new_or_confirmed(
        args.markdown_output,
        render_markdown(profile),
        overwrite=args.overwrite,
    )
    print(f"Profiled {profile['total_images']} images")
    print(f"JSON: {args.json_output}")
    print(f"Markdown: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

