"""Validate the extracted VNWoodKnot dataset before model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    splits: tuple[str, ...]
    image_categories: dict[str, str]
    detection_classes: dict[int, str]
    category_detection_class: dict[str, int]


@dataclass
class ValidationReport:
    dataset_root: str
    image_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    annotation_counts: dict[str, int] = field(default_factory=dict)
    image_sizes: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(sum(counts.values()) for counts in self.image_counts.values())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid"] = self.valid
        payload["total_images"] = self.total_images
        return payload


def load_config(path: Path) -> DatasetConfig:
    """Load and normalize the dataset configuration."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    root = (path.parent / raw["dataset_root"]).resolve()
    return DatasetConfig(
        root=root,
        splits=tuple(raw["splits"]),
        image_categories=dict(raw["image_categories"]),
        detection_classes={
            int(key): value for key, value in raw["detection_classes"].items()
        },
        category_detection_class={
            key: int(value)
            for key, value in raw["category_detection_class"].items()
        },
    )


def _validate_label(
    label_path: Path,
    expected_class: int,
    report: ValidationReport,
) -> int:
    """Validate one YOLO label file and return its object count."""
    object_count = 0
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        location = f"{label_path}:{line_number}"
        if len(fields) != 5:
            report.errors.append(f"{location}: expected 5 YOLO values, found {len(fields)}")
            continue

        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            report.errors.append(f"{location}: label contains a non-numeric value")
            continue

        if class_id != expected_class:
            report.errors.append(
                f"{location}: expected class {expected_class}, found {class_id}"
            )

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            report.errors.append(f"{location}: box centre is outside [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            report.errors.append(f"{location}: box size must be inside (0, 1]")

        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2
        tolerance = 1e-6
        if min(left, top) < -tolerance or max(right, bottom) > 1.0 + tolerance:
            report.errors.append(f"{location}: box extends beyond image bounds")

        object_count += 1

    if object_count == 0:
        report.errors.append(f"{label_path}: annotation file contains no objects")
    return object_count


def validate_dataset(config: DatasetConfig) -> ValidationReport:
    """Run structural, image-integrity, and annotation checks."""
    report = ValidationReport(dataset_root=str(config.root))
    if not config.root.is_dir():
        report.errors.append(f"Dataset root does not exist: {config.root}")
        return report

    stems_by_split: dict[str, set[str]] = defaultdict(set)
    content_locations: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    annotation_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()

    for split in config.splits:
        split_counts: dict[str, int] = {}
        for folder_id, category_name in config.image_categories.items():
            category_dir = config.root / split / folder_id
            if not category_dir.is_dir():
                report.errors.append(f"Missing category directory: {category_dir}")
                split_counts[category_name] = 0
                continue

            images = sorted(category_dir.glob("*.jpg"))
            labels = {path.stem: path for path in category_dir.glob("*.txt")}
            split_counts[category_name] = len(images)
            image_stems = {path.stem for path in images}

            orphan_labels = sorted(set(labels) - image_stems)
            for stem in orphan_labels:
                report.errors.append(f"Orphan annotation: {labels[stem]}")

            expected_detection_class = config.category_detection_class.get(folder_id)
            for image_path in images:
                stems_by_split[split].add(image_path.stem)
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image.verify()
                    size_counts[f"{width}x{height}"] += 1
                    with image_path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    content_locations[digest].append((split, image_path))
                except (OSError, UnidentifiedImageError) as exc:
                    report.errors.append(f"Unreadable image {image_path}: {exc}")

                label_path = labels.get(image_path.stem)
                if expected_detection_class is None:
                    if label_path is not None:
                        report.errors.append(
                            f"Negative image unexpectedly has an annotation: {label_path}"
                        )
                    continue

                if label_path is None:
                    report.errors.append(f"Missing annotation for {image_path}")
                    continue

                objects = _validate_label(
                    label_path=label_path,
                    expected_class=expected_detection_class,
                    report=report,
                )
                class_name = config.detection_classes[expected_detection_class]
                annotation_counts[class_name] += objects

        report.image_counts[split] = split_counts

    for index, first_split in enumerate(config.splits):
        for second_split in config.splits[index + 1 :]:
            overlap = stems_by_split[first_split] & stems_by_split[second_split]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                report.warnings.append(
                    f"Filename overlap between {first_split} and {second_split}: "
                    f"{len(overlap)} files ({preview}); content hashes are checked separately"
                )

    for digest, locations in content_locations.items():
        represented_splits = {split for split, _ in locations}
        if len(represented_splits) <= 1:
            continue
        paths = ", ".join(str(path) for _, path in locations[:3])
        report.errors.append(
            "Duplicate image content appears across dataset splits "
            f"(sha256={digest[:12]}...): {paths}"
        )

    report.annotation_counts = dict(sorted(annotation_counts.items()))
    report.image_sizes = dict(sorted(size_counts.items()))
    if len(report.image_sizes) > 1:
        report.warnings.append(
            f"Dataset contains {len(report.image_sizes)} distinct image dimensions"
        )
    return report


def _print_report(report: ValidationReport) -> None:
    status = "PASS" if report.valid else "FAIL"
    print(f"Dataset validation: {status}")
    print(f"Root: {report.dataset_root}")
    print(f"Images: {report.total_images}")
    for split, counts in report.image_counts.items():
        summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"  {split}: {summary}")
    print("Objects: " + ", ".join(
        f"{name}={count}" for name, count in report.annotation_counts.items()
    ))
    print("Image sizes: " + ", ".join(
        f"{size}={count}" for size, count in report.image_sizes.items()
    ))
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset.toml"),
        help="Path to dataset configuration (default: config/dataset.toml)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable validation report",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    report = validate_dataset(config)
    _print_report(report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
