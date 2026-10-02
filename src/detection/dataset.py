"""Strict validation for YOLO-format multi-object waste datasets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.data.schema import VALID_IMAGE_EXTENSIONS
from src.detection.manifest import DetectionRecord, image_fingerprints, load_manifest
from src.detection.schema import CLASS_NAMES, DetectionDatasetError, SPLITS, map_source_label


@dataclass(frozen=True)
class DetectionDatasetReport:
    images_by_split: dict[str, int]
    boxes_by_split: dict[str, int]
    boxes_by_class: dict[str, int]
    boxes_by_split_and_class: dict[str, dict[str, int]]
    empty_images_by_split: dict[str, int]


def _ordered_names(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, dict):
        if set(value) != set(range(len(value))):
            raise DetectionDatasetError("Dataset class indices must be contiguous integers from zero")
        try:
            return tuple(str(value[index]) for index in range(len(value)))
        except KeyError as error:
            raise DetectionDatasetError(
                "Dataset class indices must be contiguous from zero"
            ) from error
    raise DetectionDatasetError("Dataset YAML 'names' must be a list or index mapping")


def load_detection_dataset_yaml(path: Path) -> tuple[dict[str, object], Path]:
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DetectionDatasetError(f"Cannot read dataset YAML: {path}") from error
    if not isinstance(payload, dict):
        raise DetectionDatasetError("Dataset YAML root must be a mapping")
    names = _ordered_names(payload.get("names"))
    if names != CLASS_NAMES:
        raise DetectionDatasetError(
            f"Dataset class order is {names}; expected exactly {CLASS_NAMES}"
        )
    if "nc" in payload and payload["nc"] != len(CLASS_NAMES):
        raise DetectionDatasetError(f"Dataset nc must be {len(CLASS_NAMES)}")
    root_value = Path(str(payload.get("path", ".")))
    root = root_value if root_value.is_absolute() else path.parent / root_value
    return payload, root.resolve()


def _split_directory(payload: dict[str, object], root: Path, split: str) -> Path:
    value = payload.get(split)
    if not isinstance(value, str) or not value.strip():
        raise DetectionDatasetError(f"Dataset YAML requires a string '{split}' path")
    path = Path(value)
    result = (path if path.is_absolute() else root / path).resolve()
    if not result.is_relative_to(root):
        raise DetectionDatasetError(f"{split} images are outside dataset root")
    return result


def _label_path(image_path: Path, root: Path) -> Path:
    try:
        relative = image_path.relative_to(root)
    except ValueError as error:
        raise DetectionDatasetError(
            f"Image is outside dataset root: {image_path}"
        ) from error
    parts = list(relative.parts)
    try:
        images_index = parts.index("images")
    except ValueError as error:
        raise DetectionDatasetError(
            f"Image path must be below an 'images' directory: {image_path}"
        ) from error
    parts[images_index] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def read_yolo_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        raise DetectionDatasetError(f"Missing label file: {path}")
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DetectionDatasetError(
                f"{path}:{line_number} must contain class x_center y_center width height"
            )
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise DetectionDatasetError(
                f"{path}:{line_number} contains a non-numeric value"
            ) from error
        if not all(math.isfinite(value) for value in [class_value, *values]):
            raise DetectionDatasetError(f"{path}:{line_number} contains a non-finite value")
        class_index = int(class_value)
        if class_value != class_index or not 0 <= class_index < len(CLASS_NAMES):
            raise DetectionDatasetError(
                f"{path}:{line_number} has invalid class index {fields[0]}"
            )
        x_center, y_center, width, height = values
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise DetectionDatasetError(
                f"{path}:{line_number} center must be within [0,1]"
            )
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise DetectionDatasetError(
                f"{path}:{line_number} width and height must be within (0,1]"
            )
        epsilon = 1e-6
        if (
            x_center - width / 2 < -epsilon
            or x_center + width / 2 > 1.0 + epsilon
            or y_center - height / 2 < -epsilon
            or y_center + height / 2 > 1.0 + epsilon
        ):
            raise DetectionDatasetError(
                f"{path}:{line_number} bounding box extends outside the image"
            )
        boxes.append((class_index, x_center, y_center, width, height))
    return boxes


def _validate_label_file(path: Path) -> list[int]:
    return [box[0] for box in read_yolo_boxes(path)]


def validate_record_mapping(record: DetectionRecord, boxes: list[tuple]) -> None:
    if not record.source_label:
        if record.source_dataset in {"garbage_v2", "vn_trash", "garbage_classification_v2", "vn_trash_classification"} and not record.augmented_from:
            raise DetectionDatasetError(f"{record.image_id}: original source_label is required for classification-source audit")
        return
    expected = map_source_label(record.source_dataset, record.source_label, manual_class=record.material_class, reviewer=record.material_reviewer)
    if any(CLASS_NAMES[box[0]] != expected for box in boxes):
        raise DetectionDatasetError(f"{record.image_id}: reviewed boxes disagree with source material mapping {expected}")


def validate_scene_metadata(records: list[DetectionRecord]) -> None:
    scenes: dict[str, str] = {}
    by_id = {record.image_id: record for record in records}
    for record in records:
        if not record.split:
            raise DetectionDatasetError(f"{record.image_id}: exported records require a split")
        previous = scenes.setdefault(record.scene_id, record.split)
        if previous != record.split:
            raise DetectionDatasetError(f"Scene leakage: {record.scene_id} occurs in {previous} and {record.split}")
        if not record.is_real and record.split != "train":
            raise DetectionDatasetError(f"{record.image_id}: synthetic data is allowed only in train")
        for parent in record.augmented_from:
            if parent not in by_id or by_id[parent].split != "train":
                raise DetectionDatasetError(f"{record.image_id}: augmentation parent must exist in train: {parent}")


def validate_detection_dataset(path: Path, *, splits: tuple[str, ...] = SPLITS) -> DetectionDatasetReport:
    """Read only selected split files; globally check manifest scene isolation.

    Training uses splits=('train', 'val') so held-out test contents remain sealed.
    """

    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise DetectionDatasetError(f"Invalid validation splits: {splits}")
    payload, root = load_detection_dataset_yaml(path)
    manifest_value = payload.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise DetectionDatasetError("Dataset requires a reviewed JSONL manifest; image-folder class labels are not bounding boxes")
    manifest_path = Path(manifest_value)
    manifest_path = (manifest_path if manifest_path.is_absolute() else root / manifest_path).resolve()
    records = load_manifest(manifest_path)
    validate_scene_metadata(records)
    by_path: dict[Path, DetectionRecord] = {}
    for record in records:
        image_path = record.resolve("image_path", manifest_path.parent)
        if image_path in by_path:
            raise DetectionDatasetError(f"Image referenced twice in manifest: {image_path}")
        by_path[image_path] = record
    fingerprints: dict[str, tuple[str, str]] = {}
    images_by_split: dict[str, int] = {}
    boxes_by_split: dict[str, int] = {}
    empty_by_split: dict[str, int] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    split_class_counts: dict[str, dict[str, int]] = {}
    for split in splits:
        split_target = _split_directory(payload, root, split)
        if split_target.is_file() and split_target.suffix.lower() == ".txt":
            images = []
            for line in split_target.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                images.append((p if p.is_absolute() else root / p).resolve())
            images = sorted(images)
        elif split_target.is_dir():
            images = sorted(
                path.resolve()
                for path in split_target.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
        else:
            raise DetectionDatasetError(f"Missing {split} image directory or split file: {split_target}")
        if not images:
            raise DetectionDatasetError(f"No images found in {split_target}")
        expected = {p for p, record in by_path.items() if record.split == split}
        if set(images) != expected:
            raise DetectionDatasetError(f"{split}: manifest and image directory differ (unlisted or missing images)")
        box_count = 0
        empty_count = 0
        current_class_counts = {name: 0 for name in CLASS_NAMES}
        for image_path in images:
            record = by_path[image_path]
            label = record.resolve("label_path", manifest_path.parent)
            if label != _label_path(image_path, root).resolve():
                raise DetectionDatasetError(f"{record.image_id}: label path does not match YOLO images/labels layout")
            boxes = read_yolo_boxes(label)
            validate_record_mapping(record, boxes)
            indices = [box[0] for box in boxes]
            for digest in image_fingerprints(image_path):
                previous = fingerprints.setdefault(digest, (split, record.image_id))
                if previous[0] != split:
                    raise DetectionDatasetError(f"Duplicate-image leakage: {record.image_id} and {previous[1]}")
            if not indices:
                empty_count += 1
            for class_index in indices:
                class_counts[CLASS_NAMES[class_index]] += 1
                current_class_counts[CLASS_NAMES[class_index]] += 1
            box_count += len(indices)
        images_by_split[split] = len(images)
        boxes_by_split[split] = box_count
        empty_by_split[split] = empty_count
        split_class_counts[split] = current_class_counts
        metadata = payload.get("metadata", {})
        allowed_missing: set[str] = set()
        if isinstance(metadata, dict):
            allowed_missing.update(metadata.get("classes_blocked", []))
            if split in ("val", "test"):
                allowed_missing.update(metadata.get("classes_limited_evaluable", []))
        missing_classes = [
            name for name, count in current_class_counts.items() if count == 0 and name not in allowed_missing
        ]
        if missing_classes:
            raise DetectionDatasetError(
                f"Classes with no boxes in {split}: {missing_classes}"
            )
    return DetectionDatasetReport(
        images_by_split=images_by_split,
        boxes_by_split=boxes_by_split,
        boxes_by_class=class_counts,
        boxes_by_split_and_class=split_class_counts,
        empty_images_by_split=empty_by_split,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    arguments = parser.parse_args()
    try:
        report = validate_detection_dataset(arguments.data, splits=tuple(arguments.splits))
    except DetectionDatasetError as error:
        print(f"Detection dataset validation failed: {error}")
        return 2
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
