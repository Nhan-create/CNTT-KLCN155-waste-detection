"""Detection manifests preserve holdouts and reject data that would corrupt a run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.detection_manifest import load_detection_manifest, prepare_detection_manifest


def write_image(source: Path, split: str, name: str, color: str) -> Path:
    path = source / "images" / split / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)
    label = source / "labels" / split / f"{name}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("7 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return path


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    for split, color in zip(("train", "val", "test"), ("red", "green", "blue"), strict=True):
        write_image(source, split, f"oi_{split}", color)
    return source


def test_preparation_uses_all_train_images_and_authoritative_holdouts(tmp_path):
    source = make_source(tmp_path)
    first = write_image(source, "train", "taco_0001", "yellow")
    selected_val = source / "images" / "val" / "oi_val.png"
    write_image(source, "val", "excluded_val", "purple")
    manifests = tmp_path / "published_splits"
    manifests.mkdir()
    for split in ("train", "val", "test"):
        paths = sorted((source / "images" / split).glob("*.png"))
        if split == "val":
            paths = [selected_val]
        (manifests / f"{split}.txt").write_text("".join(f"{p.resolve()}\n" for p in paths))

    output = tmp_path / "prepared"
    result = prepare_detection_manifest(source, output, splits_source=manifests)
    loaded = load_detection_manifest(output / "dataset.json")

    assert loaded == result
    assert len(loaded["splits"]["train"]) == 2
    assert loaded["splits"]["val"][0]["image"] == str(selected_val.resolve())
    assert str(first.resolve()) in (output / "train.txt").read_text()
    assert loaded["counts"]["train"]["boxes_by_class"]["plastic"] == 2
    assert loaded["audit"]["manual_instance_review_complete"] is False
    assert (output / "dataset.yaml").is_file()
    assert json.loads((output / "audit.json").read_text())["exact_cross_split_duplicates"] == 0


def test_cross_split_duplicate_is_rejected_before_output_is_published(tmp_path):
    source = make_source(tmp_path)
    train = source / "images" / "train" / "oi_train.png"
    (source / "images" / "val" / "oi_val.png").write_bytes(train.read_bytes())

    with pytest.raises(ValueError, match="duplicate.*train.*val"):
        prepare_detection_manifest(source, tmp_path / "prepared")

    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("row", ["7 nan .5 .2 .2", "7.5 .5 .5 .2 .2", "10 .5 .5 .2 .2", "7 .05 .5 .4 .2", "7 .5 .5 1e-30 .2"])
def test_invalid_label_is_rejected_without_modifying_source(tmp_path, row):
    source = make_source(tmp_path)
    label = source / "labels" / "train" / "oi_train.txt"
    label.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match="label"):
        prepare_detection_manifest(source, tmp_path / "prepared")

    assert label.read_text() == row
    assert not (tmp_path / "prepared").exists()


def test_changed_annotation_invalidates_existing_manifest(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "prepared"
    prepare_detection_manifest(source, output)
    (source / "labels" / "train" / "oi_train.txt").write_text("4 .5 .5 .5 .5\n")

    with pytest.raises(ValueError, match="changed.*labels"):
        load_detection_manifest(output / "dataset.json")


def test_existing_output_is_never_overwritten(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("previous run")

    with pytest.raises(FileExistsError):
        prepare_detection_manifest(source, output)

    assert marker.read_text() == "previous run"


def test_missing_annotation_requires_explicit_background_authorization(tmp_path):
    source = make_source(tmp_path)
    label = source / "labels" / "train" / "oi_train.txt"
    label.unlink()
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="Missing label"):
        prepare_detection_manifest(source, output)

    manifest = prepare_detection_manifest(source, output, allow_background=True)
    background = manifest["splits"]["train"][0]
    assert manifest["counts"]["train"]["background_images"] == 1
    assert Path(background["labels"]).read_text() == ""
    assert not label.exists()
    assert load_detection_manifest(output / "dataset.json") == manifest


def test_oriented_annotation_requires_review_instead_of_guessing_boxes(tmp_path):
    source = make_source(tmp_path)
    image = source / "images" / "train" / "oi_train.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 60), "red").save(image, exif=exif)

    with pytest.raises(ValueError, match="EXIF-oriented labels require annotation review"):
        prepare_detection_manifest(source, tmp_path / "prepared")


def test_cli_publishes_loadable_manifest(tmp_path, capsys):
    from scripts.prepare_detection_data import main

    source = make_source(tmp_path)
    output = tmp_path / "prepared"

    assert main(["--source", str(source), "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["manifest"] == str(output / "dataset.json")
    assert load_detection_manifest(Path(result["manifest"]))["counts"]["test"]["images"] == 1


def write_review_package(tmp_path: Path, source: Path, *, split: str = "train", bounds=(0, 0, 20, 15)) -> Path:
    parent = source / "images" / split / f"oi_{split}.png"
    image = tmp_path / "reviewed" / "images" / "train" / "background.png"
    image.parent.mkdir(parents=True)
    with Image.open(parent) as opened:
        opened.crop(bounds).save(image)
    label = tmp_path / "reviewed" / "labels" / "train" / "background.txt"
    label.parent.mkdir(parents=True)
    label.write_text("")
    evidence = tmp_path / "review.png"
    Image.new("RGB", (1, 1), "white").save(evidence)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"backgrounds": [{
        "image": str(image), "labels": str(label), "parent_image": str(parent), "bounds": list(bounds),
        "reviewer": "test visual review", "reason": "Known empty corner of a labelled fixture",
        "review_evidence": str(evidence),
    }], "exclude_train": []}))
    return review


def test_reviewed_background_and_quarantine_preserve_holdout_membership(tmp_path):
    source = make_source(tmp_path)
    excluded = write_image(source, "train", "taco_group", "yellow")
    review = write_review_package(tmp_path, source)
    payload = json.loads(review.read_text())
    payload["exclude_train"] = [{
        "image": str(excluded), "reviewer": "test review", "reason": "Known group annotation fixture",
        "review_evidence": str(tmp_path / "review.png"),
    }]
    review.write_text(json.dumps(payload))

    manifest = prepare_detection_manifest(source, tmp_path / "prepared", extra_train=review)

    assert manifest["counts"]["train"]["images"] == 2
    assert manifest["counts"]["train"]["background_images"] == 1
    assert manifest["counts"]["val"]["images"] == 1
    assert manifest["counts"]["test"]["images"] == 1
    assert str(excluded) not in [row["image"] for row in manifest["splits"]["train"]]
    assert manifest["audit"]["quarantined_training_images"][0]["image"] == str(excluded)
    assert manifest["splits"]["train"][-1]["group"] == manifest["splits"]["train"][0]["group"]
    assert load_detection_manifest(tmp_path / "prepared" / "dataset.json") == manifest


@pytest.mark.parametrize("split,bounds", [("test", (0, 0, 20, 15)), ("train", (20, 15, 60, 45))])
def test_background_review_rejects_holdout_parents_and_label_intersections(tmp_path, split, bounds):
    source = make_source(tmp_path)
    review = write_review_package(tmp_path, source, split=split, bounds=bounds)

    with pytest.raises(ValueError, match="training parent|intersects"):
        prepare_detection_manifest(source, tmp_path / "prepared", extra_train=review)
