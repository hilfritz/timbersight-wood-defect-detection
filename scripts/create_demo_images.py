"""Select and annotate representative held-out TimberSight examples."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from timbersight.data.validate import load_config
from timbersight.inference import TimberSightDetector


@dataclass
class Candidate:
    path: Path
    category: str
    target_count: int
    result: dict


def target_count(path: Path, category: str) -> int:
    if category == "0":
        return 0
    return sum(bool(line.strip()) for line in path.with_suffix(".txt").read_text().splitlines())


def best_single(records: list[Candidate], category: str, class_id: int) -> Candidate:
    candidates = [
        record
        for record in records
        if record.category == category
        and record.target_count == 1
        and record.result["total"] == 1
        and record.result["detections"][0]["class_id"] == class_id
    ]
    return max(candidates, key=lambda record: record.result["detections"][0]["confidence"])


def most_challenging_correct(records: list[Candidate]) -> Candidate:
    candidates = [
        record
        for record in records
        if record.category in {"1", "2"}
        and record.target_count == 1
        and record.result["total"] == 1
        and record.result["detections"][0]["class_id"] == int(record.category)
    ]
    return min(
        candidates,
        key=lambda record: record.result["detections"][0]["confidence"],
    )


def font(size: int):
    locations = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for location in locations:
        if Path(location).is_file():
            return ImageFont.truetype(location, size)
    return ImageFont.load_default()


def annotate(source: Path, destination: Path, result: dict) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1.0, image.width / 1000)
    line_width = max(4, round(5 * scale))
    label_font = font(max(24, round(30 * scale)))

    for detection in result["detections"]:
        x1, y1, x2, y2 = detection["box"]
        colour = "#31d27c" if detection["class_id"] == 1 else "#ff6259"
        label = f'{detection["label"]}  {detection["confidence"]:.0%}'
        draw.rounded_rectangle(
            (x1, y1, x2, y2), radius=10, outline=colour, width=line_width
        )
        text_box = draw.textbbox((x1, y1), label, font=label_font, stroke_width=1)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        label_top = max(0, y1 - text_height - 18)
        draw.rounded_rectangle(
            (x1, label_top, x1 + text_width + 20, label_top + text_height + 14),
            radius=7,
            fill=colour,
        )
        draw.text(
            (x1 + 10, label_top + 5),
            label,
            fill="#101b15",
            font=label_font,
            stroke_width=1,
        )

    if not result["detections"]:
        message = "No knot detected"
        draw.rounded_rectangle((30, 30, 385, 92), radius=10, fill="#31d27c")
        draw.text((48, 43), message, fill="#101b15", font=label_font)

    image.save(destination, quality=92, optimize=True)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output = project / "demo_images"
    config = load_config(project / "config/dataset.toml")
    detector = TimberSightDetector(
        project / "artifacts/runs/20260930T035750Z/model.pt",
        device="cpu",
    )

    records: list[Candidate] = []
    for category in config.image_categories:
        for image_path in sorted((config.root / "test" / category).glob("*.jpg")):
            with Image.open(image_path) as source:
                result = detector.predict(source, threshold=0.5)
            records.append(
                Candidate(
                    path=image_path,
                    category=category,
                    target_count=target_count(image_path, category),
                    result=result,
                )
            )

    negatives = [
        record for record in records if record.category == "0" and record.result["total"] == 0
    ]
    selected = {
        "01_live_knot": best_single(records, category="1", class_id=1),
        "02_dead_knot": best_single(records, category="2", class_id=2),
        "03_challenging_knot": most_challenging_correct(records),
        "04_knot_free": negatives[len(negatives) // 2],
    }

    manifest = {}
    for name, record in selected.items():
        original = output / f"{name}_original.jpg"
        evaluated = output / f"{name}_evaluated.jpg"
        if original.exists() or evaluated.exists():
            raise FileExistsError(f"Refusing to overwrite demo output for {name}")
        shutil.copy2(record.path, original)
        annotate(record.path, evaluated, record.result)
        manifest[name] = {
            "source": str(record.path),
            "dataset_split": "test",
            "ground_truth_objects": record.target_count,
            **record.result,
        }

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
