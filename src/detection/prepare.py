"""Export reviewed boxes into a fixed-seed, scene-isolated 70/15/15 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from src.detection.dataset import read_yolo_boxes, validate_detection_dataset, validate_record_mapping, validate_scene_metadata
from src.detection.manifest import DetectionRecord, image_fingerprints, load_manifest, write_manifest
from src.detection.schema import CLASS_NAMES, DetectionDatasetError, SPLITS, SPLIT_RATIOS, SPLIT_SEED


def split_records(records: list[DetectionRecord], manifest_base: Path, *, seed: int = SPLIT_SEED) -> list[DetectionRecord]:
    """Group scenes and byte/pixel duplicates before splitting, never after.

    Ratios are approximate because a scene is indivisible. Every split must
    cover all six classes. Synthetic scenes are training-only. Explicit splits
    are preserved, but still checked for leakage and coverage during export.
    """
    if not records:
        raise DetectionDatasetError("No reviewed detection records")
    if any(record.augmented_from for record in records):
        raise DetectionDatasetError("Split original reviewed scenes before generating augmentation")
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        parents[find(left)] = find(right)

    keys: dict[str, int] = {}
    classes: list[set[int]] = []
    for index, record in enumerate(records):
        boxes = read_yolo_boxes(record.resolve("label_path", manifest_base))
        validate_record_mapping(record, boxes)
        classes.append({box[0] for box in boxes})
        fingerprints = image_fingerprints(record.resolve("image_path", manifest_base))
        for key in (f"scene:{record.scene_id}", *(f"digest:{digest}" for digest in fingerprints)):
            if key in keys:
                union(index, keys[key])
            else:
                keys[key] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)
    groups = sorted(grouped.values(), key=lambda group: min(records[i].image_id for i in group))
    explicit = [bool(record.split) for record in records]
    if any(explicit):
        if not all(explicit):
            raise DetectionDatasetError("Either set split for every record or leave all splits empty")
        for group in groups:
            if len({records[i].split for i in group}) != 1:
                raise DetectionDatasetError("Scene/duplicate leakage in manually assigned splits")
        validate_scene_metadata(records)
        return records
    eligible = [index for index, group in enumerate(groups) if all(records[i].is_real for i in group)]
    if len(eligible) < 3:
        raise DetectionDatasetError("At least three independent real scene groups are required for train/val/test")
    group_classes = [set().union(*(classes[i] for i in group)) for group in groups]
    targets = [len(records) * ratio for ratio in SPLIT_RATIOS]
    rng = random.Random(seed)
    best: tuple[float, dict[int, str]] | None = None
    for _ in range(512):
        order = list(eligible)
        rng.shuffle(order)
        assignment = {index: "train" for index in range(len(groups))}
        remaining = set(order)
        # Greedily cover rare classes, then approach the scene-count target.
        for split, target in (("test", targets[2]), ("val", targets[1])):
            selected: list[int] = []
            covered: set[int] = set()
            count = 0
            while remaining and (len(covered) < len(CLASS_NAMES) or count < target):
                candidates = [index for index in order if index in remaining]
                index = min(candidates, key=lambda item: (
                    -len(group_classes[item] - covered),
                    abs(target - count - len(groups[item])),
                ))
                if len(covered) == len(CLASS_NAMES) and abs(count - target) <= abs(count + len(groups[index]) - target):
                    break
                selected.append(index)
                covered.update(group_classes[index])
                count += len(groups[index])
                remaining.remove(index)
                assignment[index] = split
        counts = {split: 0 for split in SPLITS}
        coverage = {split: set() for split in SPLITS}
        for index, split in assignment.items():
            counts[split] += len(groups[index])
            coverage[split].update(group_classes[index])
        missing = sum(len(CLASS_NAMES) - len(coverage[split]) for split in SPLITS)
        score = missing * 10000 + sum(abs(counts[split] - targets[i]) for i, split in enumerate(SPLITS))
        if best is None or score < best[0]:
            best = score, assignment
    assert best is not None
    if best[0] >= 10000:
        raise DetectionDatasetError("Insufficient independent scenes to cover all six classes in train/val/test; collect more reviewed scenes")
    assigned = {}
    for group_index, split in best[1].items():
        for index in groups[group_index]:
            assigned[index] = replace(records[index], split=split)
    result = [assigned[index] for index in range(len(records))]
    validate_scene_metadata(result)
    return result


def write_dataset_yaml(output: Path, *, seed: int = SPLIT_SEED, variant: str = "none") -> Path:
    payload = {
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
        "manifest": "manifest.jsonl", "names": dict(enumerate(CLASS_NAMES)),
        "split_seed": seed, "split_ratios": list(SPLIT_RATIOS), "augmentation_variant": variant,
    }
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def copy_record(record: DetectionRecord, base: Path, output: Path) -> DetectionRecord:
    # Fixed hashes avoid path traversal, Windows reserved names, and collisions.
    stem = hashlib.sha256(record.image_id.encode()).hexdigest()[:24]
    source = record.resolve("image_path", base)
    image_relative = Path("images") / record.split / f"{stem}{source.suffix.lower()}"
    label_relative = Path("labels") / record.split / f"{stem}.txt"
    for relative in (image_relative, label_relative):
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / image_relative)
    shutil.copy2(record.resolve("label_path", base), output / label_relative)
    masks = []
    for index, mask in enumerate(record.foreground_masks):
        mask_relative = Path("masks") / record.split / f"{stem}-{index}.png"
        (output / mask_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((base / mask).resolve(), output / mask_relative)
        masks.append(mask_relative.as_posix())
    return replace(record, image_path=image_relative.as_posix(), label_path=label_relative.as_posix(), foreground_masks=tuple(masks))


def prepare_dataset(manifest: Path, output: Path, *, seed: int = SPLIT_SEED) -> Path:
    manifest, output = manifest.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise DetectionDatasetError(f"Output must be a new or empty directory; existing data is never overwritten: {output}")
    records = split_records(load_manifest(manifest), manifest.parent, seed=seed)
    # Source labels and images have all been validated before creating output.
    output.mkdir(parents=True, exist_ok=True)
    exported = [copy_record(record, manifest.parent, output) for record in records]
    write_manifest(output / "manifest.jsonl", exported)
    data = write_dataset_yaml(output, seed=seed)
    report = validate_detection_dataset(data)
    (output / "data_report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed bounding-box JSONL, not a classification manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    try:
        path = prepare_dataset(args.manifest, args.output, seed=args.seed)
    except (DetectionDatasetError, OSError) as error:
        print(f"Detection data preparation failed: {error}")
        return 2
    print(json.dumps({"dataset": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
