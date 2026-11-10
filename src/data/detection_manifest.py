"""Prepare reproducible detection splits without creating or guessing annotations."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps
import yaml

from src.detection.schema import DETECTION_CLASS_NAMES, SPLITS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROUNDING_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    """Hash the exact bytes used by a run, including annotation files."""
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def read_yolo_labels(path: Path, *, allow_missing: bool = False) -> list[tuple[int, float, float, float, float]]:
    """Validate zero-based normalized YOLO boxes and clip rounding at image edges."""
    path = Path(path)
    if not path.is_file():
        if allow_missing:
            return []
        raise ValueError(f"Missing label file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError("expected five finite values")
            class_id, x, y, width, height = values
            if not class_id.is_integer() or not 0 <= class_id < len(DETECTION_CLASS_NAMES):
                raise ValueError("invalid class index")
            if not 0 < width <= 1 or not 0 < height <= 1 or not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("invalid normalized box")
            left, top, right, bottom = x - width / 2, y - height / 2, x + width / 2, y + height / 2
            if min(left, top) < -ROUNDING_TOLERANCE or max(right, bottom) > 1 + ROUNDING_TOLERANCE:
                raise ValueError("box extends outside image")
            left, top, right, bottom = max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom)
            if right <= left or bottom <= top:
                raise ValueError("box collapses at numeric precision")
            rows.append((int(class_id), (left + right) / 2, (top + bottom) / 2, right - left, bottom - top))
        except ValueError as error:
            raise ValueError(f"Invalid label {path}:{line_number}: {error}") from error
    return rows


def _split_images(source: Path, split: str, splits_source: Path | None) -> list[Path]:
    image_dir = source / "images" / split
    manifest = (splits_source or source / "splits") / f"{split}.txt"
    if splits_source is not None and not manifest.is_file():
        raise ValueError(f"Missing authoritative split manifest: {manifest}")
    if manifest.is_file():
        images = [(manifest.parent / line.strip()).resolve()
                  for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        images = sorted(path.resolve() for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"Empty detection split: {split}")
    if len(set(images)) != len(images):
        raise ValueError(f"Repeated image in {split} split")
    for path in images:
        if not path.is_file() or not path.is_relative_to(image_dir):
            raise ValueError(f"Split {split} image must exist under {image_dir}: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem.startswith("multi_"):
            raise ValueError(f"Unsupported or synthetic image in real split {split}: {path}")
    return images


def _source_and_group(image: Path) -> tuple[str, str]:
    stem = image.stem
    source = stem.split("_")[0]
    if source == "oi":
        source = "open_images"
    if source.startswith("wave"):
        source = "_".join(stem.split("_")[:2])
    group = re.sub(r"_(?:aug|crop)\d+$", "", stem)
    return source, f"{source}/{group}"


def _fingerprint(splits: dict) -> str:
    content = {"class_names": DETECTION_CLASS_NAMES, "splits": splits}
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _check_leakage(splits: dict) -> None:
    images: dict[str, str] = {}
    groups: dict[str, str] = {}
    for split in SPLITS:
        for row in splits[split]:
            digest, group = row["image_sha256"], row["group"]
            previous = images.get(digest)
            if previous is not None and previous != split:
                raise ValueError(f"Exact image duplicate across {previous} and {split}: {row['image']}")
            previous_group = groups.get(group)
            if previous_group is not None and previous_group != split:
                raise ValueError(f"Source image group shared across {previous_group} and {split}: {group}")
            images[digest], groups[group] = split, split


def _apply_training_review(inspected_splits: dict, review_path: Path | None) -> tuple[list, list]:
    if review_path is None:
        return [], []
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or not set(review).issubset({"backgrounds", "exclude_train"}):
        raise ValueError("Training review must contain backgrounds and/or exclude_train lists")
    originals = {record["image"]: (record, rows) for record, rows in inspected_splits["train"]}
    quarantined, backgrounds = [], []
    for action in ("exclude_train", "backgrounds"):
        entries = review.get(action, [])
        if not isinstance(entries, list):
            raise ValueError(f"Training review {action} must be a list")
        for entry in entries:
            if not entry.get("reviewer", "").strip() or not entry.get("reason", "").strip():
                raise ValueError("Training changes require a recorded reviewer and reason")
            if not Path(entry.get("review_evidence", "")).is_file():
                raise ValueError("Training changes require existing visual review evidence")
            if action == "exclude_train":
                image = str(Path(entry["image"]).resolve())
                if image not in originals or any(row["image"] == image for row in quarantined):
                    raise ValueError(f"Quarantine image must occur once in authoritative training split: {image}")
                quarantined.append({**entry, "image": image})
                continue
            parent_image = str(Path(entry["parent_image"]).resolve())
            if parent_image not in originals or any(row["image"] == parent_image for row in quarantined):
                raise ValueError(f"Background crop requires a retained authoritative training parent: {parent_image}")
            parent, parent_labels = originals[parent_image]
            image, label = Path(entry["image"]).resolve(), Path(entry["labels"]).resolve()
            canonical_label = image.parents[2] / "labels" / image.parent.name / image.with_suffix(".txt").name
            if image.parent.name != "train" or image.parent.parent.name != "images" or label != canonical_label:
                raise ValueError("Reviewed backgrounds require matching images/train and labels/train paths")
            if read_yolo_labels(label):
                raise ValueError(f"Reviewed background must have an explicitly empty label: {label}")
            bounds = entry.get("bounds", [])
            if len(bounds) != 4 or any(not isinstance(value, int) for value in bounds):
                raise ValueError("Background crop bounds require four integer pixel coordinates")
            left, top, right, bottom = bounds
            if not 0 <= left < right <= parent["width"] or not 0 <= top < bottom <= parent["height"]:
                raise ValueError("Background crop lies outside its training parent")
            for _, x, y, width, height in parent_labels:
                x1, y1 = (x - width / 2) * parent["width"], (y - height / 2) * parent["height"]
                x2, y2 = (x + width / 2) * parent["width"], (y + height / 2) * parent["height"]
                if min(right, x2) > max(left, x1) and min(bottom, y2) > max(top, y1):
                    raise ValueError(f"Reviewed background intersects a parent annotation: {image}")
            with Image.open(image) as opened:
                prepared = ImageOps.exif_transpose(opened)
                prepared.load()
                if prepared.size != (right - left, bottom - top):
                    raise ValueError("Background crop dimensions differ from recorded bounds")
            record = {
                "image": str(image), "labels": str(label), "source": parent["source"], "group": parent["group"],
                "image_sha256": sha256_file(image), "labels_sha256": sha256_file(label),
                "width": right - left, "height": bottom - top, "object_count": 0,
                "parent_image": parent_image, "bounds": bounds, "reviewer": entry["reviewer"],
                "review_reason": entry["reason"], "review_evidence": entry["review_evidence"],
            }
            backgrounds.append(record)
    excluded = {row["image"] for row in quarantined}
    inspected_splits["train"] = [(record, rows) for record, rows in inspected_splits["train"]
                                  if record["image"] not in excluded]
    inspected_splits["train"].extend((record, []) for record in backgrounds)
    return quarantined, backgrounds


def prepare_detection_manifest(
    source: Path,
    output: Path,
    *,
    splits_source: Path | None = None,
    allow_background: bool = False,
    extra_train: Path | None = None,
) -> dict:
    """Publish a new, audited real-data manifest while preserving existing split membership.

    Existing output directories are rejected. Original images and annotations are
    read-only. Missing labels require explicit background authorization and are
    represented by newly created empty label files in the output directory.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    splits_source = Path(splits_source).resolve() if splits_source is not None else None
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("Prepared output must be outside the original dataset")
    paths = {split: _split_images(source, split, splits_source) for split in SPLITS}
    splits, counts = {}, {}
    large_boxes, backgrounds = [], []
    generated_labels = []

    def inspect_image(task: tuple[str, Path]) -> tuple[dict, list]:
        split, image = task
        relative = image.relative_to(source / "images" / split)
        label = source / "labels" / split / relative.with_suffix(".txt")
        rows = read_yolo_labels(label, allow_missing=allow_background)
        missing = not label.is_file()
        recorded_label = output / "background_labels" / split / relative.with_suffix(".txt") if missing else label
        with Image.open(image) as opened:
            orientation = opened.getexif().get(274, 1)
            if orientation not in (None, 1) and rows:
                raise ValueError(f"EXIF-oriented labels require annotation review before training: {image}")
            prepared = ImageOps.exif_transpose(opened)
            prepared.load()
            width, height = prepared.size
        source_name, group = _source_and_group(image)
        record = {
            "image": str(image), "labels": str(recorded_label), "source": source_name,
            "group": group, "image_sha256": sha256_file(image),
            "labels_sha256": sha256(b"").hexdigest() if missing else sha256_file(label),
            "width": width, "height": height, "object_count": len(rows),
        }
        return record, rows

    with ThreadPoolExecutor(max_workers=8) as pool:
        inspected_splits = {
            split: list(pool.map(inspect_image, ((split, image) for image in paths[split])))
            for split in SPLITS
        }
    quarantined, reviewed_backgrounds = _apply_training_review(inspected_splits, extra_train)
    for split in SPLITS:
        inspected = inspected_splits[split]
        splits[split] = [record for record, _ in inspected]
        boxes_by_class = Counter()
        images_by_source = Counter()
        small_boxes = 0
        for record, rows in inspected:
            images_by_source[record["source"]] += 1
            if not rows:
                backgrounds.append(record["image"])
            if Path(record["labels"]).is_relative_to(output):
                generated_labels.append(Path(record["labels"]).relative_to(output))
            for class_id, _, _, width, height in rows:
                boxes_by_class[DETECTION_CLASS_NAMES[class_id]] += 1
                small_boxes += width * record["width"] * height * record["height"] < 32 * 32
                if width * height >= 0.65:
                    large_boxes.append({"split": split, "image": record["image"], "class_id": class_id})
        counts[split] = {
            "images": len(inspected), "boxes": sum(boxes_by_class.values()),
            "boxes_by_class": {name: boxes_by_class[name] for name in DETECTION_CLASS_NAMES},
            "images_by_source": dict(sorted(images_by_source.items())),
            "small_boxes": small_boxes,
            "background_images": sum(not rows for _, rows in inspected),
        }
    _check_leakage(splits)
    audit = {
        "exact_cross_split_duplicates": 0,
        "manual_instance_review_complete": False,
        "near_duplicate_review_complete": False,
        "scene_group_review_complete": False,
        "group_method": "source and filename, with terminal crop/augmentation index removed",
        "split_selection": str(splits_source or source / "splits"),
        "synthetic_images_included": 0,
        "large_box_review_candidates": large_boxes,
        "background_images": backgrounds,
        "quarantined_training_images": quarantined,
        "reviewed_background_crops": reviewed_backgrounds,
        "notes": [
            "A valid label format does not establish one box per physical object.",
            "Large boxes are review candidates, not automatic annotation errors.",
            "Existing holdouts are preserved and are not a new blind test set.",
        ],
    }
    manifest = {"format_version": 1, "class_names": list(DETECTION_CLASS_NAMES),
                "source": str(source), "splits": splits, "fingerprint": _fingerprint(splits),
                "counts": counts, "audit": audit}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        for label in generated_labels:
            target = staging / label
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
        for split in SPLITS:
            (staging / f"{split}.txt").write_text("".join(f"{row['image']}\n" for row in splits[split]), encoding="utf-8")
        config = {"path": output.as_posix(), "train": "train.txt", "val": "val.txt", "test": "test.txt",
                  "names": list(DETECTION_CLASS_NAMES)}
        (staging / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for name, payload in (("dataset.json", manifest), ("audit.json", audit)):
            (staging / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        staging.rename(output)
    return manifest


def load_detection_manifest(path: Path, *, verify_files: bool = True) -> dict:
    """Read a manifest and fail if its split contract or recorded input bytes changed."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or tuple(manifest.get("class_names", ())) != DETECTION_CLASS_NAMES:
        raise ValueError("Invalid detection manifest version or class order")
    splits = manifest.get("splits", {})
    if set(splits) != set(SPLITS) or not all(isinstance(splits[split], list) and splits[split] for split in SPLITS):
        raise ValueError("Detection manifest requires nonempty train, val and test splits")
    if manifest.get("fingerprint") != _fingerprint(splits):
        raise ValueError("Detection manifest fingerprint changed")
    _check_leakage(splits)
    if verify_files:
        def verify_record(record: dict) -> None:
            for field in ("image", "labels"):
                target = Path(record[field])
                if not target.is_absolute() or not target.is_file() or sha256_file(target) != record[f"{field}_sha256"]:
                    raise ValueError(f"Input changed for {field}: {target}")
            if len(read_yolo_labels(Path(record["labels"]))) != record["object_count"]:
                raise ValueError(f"Annotation count changed: {record['labels']}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify_record, (record for split in SPLITS for record in splits[split])))
    return manifest
