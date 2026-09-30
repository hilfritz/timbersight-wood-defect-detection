import hashlib
from pathlib import Path

from PIL import Image

from timbersight.data.validate import DatasetConfig, validate_dataset


def _build_config(root: Path) -> DatasetConfig:
    return DatasetConfig(
        root=root,
        splits=("train", "validation", "test"),
        image_categories={"0": "knot_free", "1": "live_knot", "2": "dead_knot"},
        detection_classes={0: "live_knot", 1: "dead_knot"},
        category_detection_class={"1": 0, "2": 1},
    )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).digest()
    Image.new("RGB", (32, 32), color=tuple(digest[:3])).save(path)


def _build_valid_fixture(root: Path) -> DatasetConfig:
    config = _build_config(root)
    for split_index, split in enumerate(config.splits):
        clear = root / split / "0" / f"clear_{split_index}.jpg"
        live = root / split / "1" / f"live_{split_index}.jpg"
        dead = root / split / "2" / f"dead_{split_index}.jpg"
        for path in (clear, live, dead):
            _write_image(path)
        live.with_suffix(".txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        dead.with_suffix(".txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return config


def test_valid_dataset_passes(tmp_path: Path) -> None:
    report = validate_dataset(_build_valid_fixture(tmp_path))

    assert report.valid
    assert report.total_images == 9
    assert report.annotation_counts == {"dead_knot": 3, "live_knot": 3}


def test_wrong_detection_class_fails(tmp_path: Path) -> None:
    config = _build_valid_fixture(tmp_path)
    (tmp_path / "train" / "1" / "live_0.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )

    report = validate_dataset(config)

    assert not report.valid
    assert any("expected class 0, found 1" in error for error in report.errors)


def test_out_of_bounds_box_fails(tmp_path: Path) -> None:
    config = _build_valid_fixture(tmp_path)
    (tmp_path / "test" / "2" / "dead_2.txt").write_text(
        "1 0.9 0.5 0.4 0.4\n", encoding="utf-8"
    )

    report = validate_dataset(config)

    assert not report.valid
    assert any("extends beyond image bounds" in error for error in report.errors)


def test_duplicate_content_across_splits_fails(tmp_path: Path) -> None:
    config = _build_valid_fixture(tmp_path)
    source = tmp_path / "train" / "0" / "clear_0.jpg"
    duplicate = tmp_path / "test" / "0" / "clear_2.jpg"
    duplicate.write_bytes(source.read_bytes())

    report = validate_dataset(config)

    assert not report.valid
    assert any("Duplicate image content" in error for error in report.errors)
