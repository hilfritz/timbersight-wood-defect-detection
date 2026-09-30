from pathlib import Path

from PIL import Image

from timbersight.data.profile import build_profile, render_markdown
from timbersight.data.validate import DatasetConfig


def _dataset(root: Path) -> DatasetConfig:
    config = DatasetConfig(
        root=root,
        splits=("train", "validation", "test"),
        image_categories={"0": "knot_free", "1": "live_knot", "2": "dead_knot"},
        detection_classes={0: "live_knot", 1: "dead_knot"},
        category_detection_class={"1": 0, "2": 1},
    )
    for split in config.splits:
        for folder in ("0", "1", "2"):
            image = root / split / folder / f"{split}_{folder}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), color=(100, 120, 140)).save(image)
            if folder in {"1", "2"}:
                class_id = 0 if folder == "1" else 1
                image.with_suffix(".txt").write_text(
                    f"{class_id} 0.5 0.5 0.4 0.2\n", encoding="utf-8"
                )
    return config


def test_profile_counts_images_objects_and_negatives(tmp_path: Path) -> None:
    profile = build_profile(_dataset(tmp_path))

    assert profile["total_images"] == 9
    assert profile["negative_images"] == 3
    assert profile["object_counts"] == {"dead_knot": 3, "live_knot": 3}
    assert profile["box_area_fraction"]["median"] == 0.08


def test_markdown_contains_reproducible_counts(tmp_path: Path) -> None:
    markdown = render_markdown(build_profile(_dataset(tmp_path)))

    assert "Total images: **9**" in markdown
    assert "| live_knot | 3 |" in markdown

