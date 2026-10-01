"""Reviewed per-image annotation provenance for phase-two detection data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from src.detection.schema import DetectionDatasetError, HARD_CONDITIONS, SPLITS, map_source_label


@dataclass(frozen=True)
class DetectionRecord:
    image_id: str
    image_path: str
    label_path: str
    source_dataset: str
    scene_id: str
    is_real: bool
    reviewed: bool
    reviewer: str
    split: str = ""
    source_label: str = ""
    material_class: str = ""
    material_reviewer: str = ""
    foreground_masks: tuple[str, ...] = ()
    mask_reviewed: bool = False
    mask_reviewer: str = ""
    augmented_from: tuple[str, ...] = ()
    augmentation: str = "none"
    conditions: tuple[str, ...] = ()

    def resolve(self, field: str, base: Path) -> Path:
        value = Path(getattr(self, field))
        return (value if value.is_absolute() else base / value).resolve()


def load_manifest(path: Path) -> list[DetectionRecord]:
    """Load strict JSONL; image and label paths are relative to this manifest."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DetectionDatasetError(f"Cannot read reviewed manifest: {path}") from error
    records: list[DetectionRecord] = []
    identifiers: set[str] = set()
    allowed = {field.name for field in fields(DetectionRecord)}
    required = {"image_id", "image_path", "label_path", "source_dataset", "scene_id", "is_real", "reviewed", "reviewer"}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DetectionDatasetError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(raw, dict) or not required.issubset(raw) or set(raw) - allowed:
            raise DetectionDatasetError(f"{path}:{number}: missing or unknown manifest fields")
        for field in ("foreground_masks", "augmented_from", "conditions"):
            values = raw.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise DetectionDatasetError(f"{path}:{number}: {field} must be a list of strings")
            raw[field] = tuple(values)
        record = DetectionRecord(**raw)
        for field in required - {"is_real", "reviewed"}:
            if not isinstance(getattr(record, field), str) or not getattr(record, field).strip():
                raise DetectionDatasetError(f"{path}:{number}: {field} must be non-empty text")
        for field in ("split", "source_label", "material_class", "material_reviewer", "mask_reviewer", "augmentation"):
            if not isinstance(getattr(record, field), str):
                raise DetectionDatasetError(f"{path}:{number}: {field} must be text")
        if type(record.is_real) is not bool or type(record.reviewed) is not bool or type(record.mask_reviewed) is not bool:
            raise DetectionDatasetError(f"{path}:{number}: real/review flags must be JSON booleans")
        if not record.reviewed:
            raise DetectionDatasetError(f"{record.image_id}: bounding boxes have not been reviewed")
        if record.split and record.split not in SPLITS:
            raise DetectionDatasetError(f"{record.image_id}: invalid split {record.split!r}")
        if record.split == "test" and not record.is_real:
            raise DetectionDatasetError(f"{record.image_id}: official test must contain only real images")
        if record.augmented_from and (record.is_real or record.split not in {"", "train"}):
            raise DetectionDatasetError(f"{record.image_id}: augmented images must be synthetic training data")
        if record.foreground_masks and (not record.mask_reviewed or not record.mask_reviewer.strip()):
            raise DetectionDatasetError(f"{record.image_id}: foreground masks need a named human reviewer")
        if set(record.conditions) - set(HARD_CONDITIONS):
            raise DetectionDatasetError(f"{record.image_id}: unsupported hard-condition tag")
        if record.augmentation not in {"none", "geometric", "photometric", "combined"}:
            raise DetectionDatasetError(f"{record.image_id}: unknown augmentation variant")
        if record.source_label:
            map_source_label(record.source_dataset, record.source_label, manual_class=record.material_class, reviewer=record.material_reviewer)
        if record.image_id in identifiers:
            raise DetectionDatasetError(f"Duplicate image_id: {record.image_id}")
        identifiers.add(record.image_id)
        records.append(record)
    if not records:
        raise DetectionDatasetError(f"Reviewed manifest is empty: {path}")
    return records


def write_manifest(path: Path, records: list[DetectionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def load_reviewed_manifest(data_path: Path) -> list[dict]:
    """Return YAML manifest records with absolute image/label/mask paths."""
    from src.detection.dataset import load_detection_dataset_yaml
    payload, root = load_detection_dataset_yaml(data_path)
    value = payload.get("manifest")
    if not isinstance(value, str) or not value.strip():
        raise DetectionDatasetError("Dataset requires a reviewed JSONL manifest")
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    result = []
    for record in load_manifest(path):
        raw = asdict(record)
        for field in ("image_path", "label_path"):
            raw[field] = str(record.resolve(field, path.parent))
        raw["foreground_masks"] = [str((path.parent / mask).resolve()) for mask in record.foreground_masks]
        result.append(raw)
    return result


def image_fingerprints(path: Path) -> tuple[str, str]:
    """Exact bytes and decoded pixels catch renamed/re-encoded identical images."""
    try:
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                raise DetectionDatasetError(f"Normalize EXIF orientation before reviewing boxes: {path}")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            pixels = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    except (OSError, UnidentifiedImageError) as error:
        raise DetectionDatasetError(f"Cannot decode image: {path}") from error
    return file_digest, pixels
