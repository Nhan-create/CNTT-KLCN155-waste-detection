# Mô tả: Tái lập kết quả kiểm toán nguồn gốc dữ liệu TACO và OpenImages ở mức ưu tiên P0 (Giai đoạn 3).
"""Reproduce Phase 3 P0 TACO and Open Images provenance findings.

This audit is read-only. It never changes labels, mappings, manifests, or splits.
It uses the repository's cached official TACO COCO JSON and the local Open Images
V7 validation CSV whose SHA-256 is recorded in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_real_detection_v2 import coco_to_yolo  # noqa: E402


TACO_CASES = (
    {
        "image_id": 112,
        "annotation_id": 387,
        "label_line": 12,
        "class_id": 7,
        "before_line": "7 0.398125 0.128564 0.092250 0.258224",
        "before_label_sha256": "1921d73b34f75abb5da777be0fe29cd311d61c52bb02aa2fb239df316dc68aa7",
    },
    {
        "image_id": 457,
        "annotation_id": 1621,
        "label_line": 5,
        "class_id": 6,
        "before_line": "6 0.334936 0.057933 0.157692 0.116346",
        "before_label_sha256": "06acb9489e3f5becf1ba15876caa7d09b43543c38b12df1db72f7e914adf2e43",
    },
)

MIRROR_CONSUMERS = {
    "data/detection/labels_10cls_backup/real": {
        "kind": "backup",
        "consumer": None,
        "evidence": "No repository config or code reference found.",
    },
    "data/detection/labels_agnostic/real": {
        "kind": "agnostic_generator_source",
        "consumer": "scripts/build_real_v2_splits.py and tests/detection/test_split_real_v2.py",
        "evidence": "Generated one-class source mirror; copied into data/detection_agnostic.",
    },
    "data/detection_agnostic/labels/real": {
        "kind": "active_agnostic_dataset",
        "consumer": "configs/detection_dataset_real_v2_agnostic.yaml",
        "evidence": "Ultralytics resolves /images/ paths to this /labels/ tree.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo(path: Path) -> list[tuple[int, list[float]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split()
            rows.append((int(parts[0]), [float(value) for value in parts[1:5]]))
    return rows


def bounds(box: list[float] | tuple[float, ...]) -> list[float]:
    xc, yc, width, height = box
    return [xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2]


def oi_to_yolo(row: dict[str, str]) -> list[float]:
    xmin, xmax = float(row["XMin"]), float(row["XMax"])
    ymin, ymax = float(row["YMin"]), float(row["YMax"])
    return [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]


def split_membership() -> tuple[dict[str, str], dict[str, list[str]]]:
    by_id: dict[str, str] = {}
    by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = ROOT / "data/detection/splits_real_v2" / f"{split}.txt"
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stem = Path(line).stem
            if stem.startswith("oi_"):
                image_id = stem[3:]
                ids.append(image_id)
                by_id[image_id] = split
        by_split[split] = ids
    return by_id, by_split


def taco_audit(raw: dict) -> dict:
    images = {item["id"]: item for item in raw["images"]}
    annotations = {item["id"]: item for item in raw["annotations"]}
    categories = {item["id"]: item["name"] for item in raw["categories"]}
    cases = []
    for spec in TACO_CASES:
        image_id = spec["image_id"]
        annotation = annotations[spec["annotation_id"]]
        meta = images[image_id]
        label_path = ROOT / f"data/detection/labels/real/taco_{image_id:04d}.txt"
        image_path = ROOT / f"data/detection/images/real/taco_{image_id:04d}.jpg"
        current_line = label_path.read_text(encoding="utf-8").splitlines()[spec["label_line"] - 1]
        current_parts = current_line.split()
        expected = coco_to_yolo(annotation["bbox"], meta["width"], meta["height"])
        x, y, width, height = annotation["bbox"]
        clipped = [
            max(0.0, x),
            max(0.0, y),
            min(float(meta["width"]), x + width) - max(0.0, x),
            min(float(meta["height"]), y + height) - max(0.0, y),
        ]
        with Image.open(image_path) as opened:
            stored_size = list(opened.size)
        before_box = [float(value) for value in spec["before_line"].split()[1:]]
        current_box = [float(value) for value in current_parts[1:]]
        cases.append(
            {
                **spec,
                "source_file": meta["file_name"],
                "source_category_id": annotation["category_id"],
                "source_category_name": categories[annotation["category_id"]],
                "source_bbox_xywh": annotation["bbox"],
                "source_segmentation_has_negative_y": any(
                    point < 0 for polygon in annotation.get("segmentation", []) for point in polygon[1::2]
                ),
                "source_image_size": [meta["width"], meta["height"]],
                "stored_image_size": stored_size,
                "clipped_bbox_xywh": clipped,
                "expected_yolo": [round(value, 6) for value in expected],
                "before_bounds_xyxy": [round(value, 9) for value in bounds(before_box)],
                "after_line": current_line,
                "after_bounds_xyxy": [round(value, 9) for value in bounds(current_box)],
                "after_matches_converter_6dp": current_box == [round(value, 6) for value in expected],
                "after_label_sha256": sha256(label_path),
                "image_sha256": sha256(image_path),
            }
        )
    return {
        "official_cache": "data/audit/taco_annotations_raw.json",
        "official_cache_sha256": sha256(ROOT / "data/audit/taco_annotations_raw.json"),
        "converter": "scripts/collect_real_detection_v2.py::coco_to_yolo",
        "cases": cases,
    }


def mirror_audit() -> list[dict]:
    result = []
    split_for = {112: "train", 457: "test"}
    for directory, consumer in MIRROR_CONSUMERS.items():
        for spec in TACO_CASES:
            path = ROOT / directory / f"taco_{spec['image_id']:04d}.txt"
            line = path.read_text(encoding="utf-8").splitlines()[spec["label_line"] - 1]
            box = [float(value) for value in line.split()[1:]]
            xyxy = bounds(box)
            result.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": spec["label_line"],
                    "value": line,
                    "bounds_xyxy": [round(value, 9) for value in xyxy],
                    "out_of_bounds": any(value < 0.0 or value > 1.0 for value in xyxy),
                    "split": split_for[spec["image_id"]],
                    **consumer,
                }
            )
    return result


def openimages_audit(oi_boxes_path: Path, oi_classes_path: Path, config: dict) -> dict:
    split_for, ids_by_split = split_membership()
    active_ids = set(split_for)
    mapping = config["sources"]["openimages_v7"]["class_mapping"]
    mapped = {mid: item for mid, item in mapping.items() if item.get("class_id") is not None}
    descriptions = {}
    with oi_classes_path.open(newline="", encoding="utf-8") as handle:
        descriptions = {row["LabelName"]: row["DisplayName"] for row in csv.DictReader(handle)}

    source_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with oi_boxes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] in active_ids:
                source_rows[row["ImageID"]].append(row)

    manifest_path = ROOT / "data/detection/real_v2_source_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest_fields = list(reader.fieldnames or [])
        manifest = {
            row["source_image_id"]: row
            for row in reader
            if row["source"] == "openimages_v7"
        }

    missing = []
    for image_id in sorted(active_ids - set(manifest)):
        label_path = ROOT / f"data/detection/labels/real/oi_{image_id}.txt"
        matches = []
        for class_id, label_box in read_yolo(label_path):
            candidates = []
            for row in source_rows[image_id]:
                source_box = oi_to_yolo(row)
                candidates.append((max(abs(a - b) for a, b in zip(label_box, source_box)), row, source_box))
            distance, row, source_box = min(candidates, key=lambda item: item[0])
            matches.append(
                {
                    "class_id": class_id,
                    "label_yolo": label_box,
                    "source_mid": row["LabelName"],
                    "source_display_name": descriptions.get(row["LabelName"]),
                    "source_yolo": source_box,
                    "max_abs_coordinate_delta": distance,
                    "exact_to_8dp": distance <= 5e-8,
                    "mapping_config_entry": mapping.get(row["LabelName"]),
                }
            )
        missing.append(
            {
                "image_id": image_id,
                "split": split_for[image_id],
                "image_sha256": sha256(ROOT / f"data/detection/images/real/oi_{image_id}.jpg"),
                "label_sha256": sha256(label_path),
                "source_box_matches": matches,
                "finding": "Source rows can be coordinate-matched, but the MIDs are absent from the approved mapping and the image is absent from the v2 provenance manifest.",
                "action": "BLOCK_AND_REVIEW; do not infer approval from coordinate match.",
            }
        )

    mismatches = []
    for image_id in sorted(active_ids & set(manifest)):
        label_path = ROOT / f"data/detection/labels/real/oi_{image_id}.txt"
        actual = read_yolo(label_path)
        mapped_rows = [row for row in source_rows[image_id] if row["LabelName"] in mapped]
        if len(actual) == len(mapped_rows):
            continue
        actual_matches = []
        for class_id, label_box in actual:
            candidates = []
            for row in source_rows[image_id]:
                source_box = oi_to_yolo(row)
                candidates.append((max(abs(a - b) for a, b in zip(label_box, source_box)), row))
            distance, row = min(candidates, key=lambda item: item[0])
            actual_matches.append(
                {
                    "class_id": class_id,
                    "source_mid": row["LabelName"],
                    "source_display_name": descriptions.get(row["LabelName"]),
                    "max_abs_coordinate_delta": distance,
                    "mapping_status": mapping.get(row["LabelName"], {}).get("status", "UNMAPPED"),
                }
            )
        mismatches.append(
            {
                "image_id": image_id,
                "split": split_for[image_id],
                "stored_box_count": len(actual),
                "current_config_mapped_source_box_count": len(mapped_rows),
                "mapped_source_mids": [row["LabelName"] for row in mapped_rows],
                "stored_box_source_matches": actual_matches,
                "finding": "Count differs under the current mapping; hierarchy duplicates or a provisional ambiguous material decision are present.",
                "action": "OBJECT_LEVEL_REVIEW; do not add, remove, or relabel boxes automatically.",
            }
        )

    box_counts: dict[str, Counter] = {}
    image_counts: dict[str, Counter] = {}
    for split, image_ids in ids_by_split.items():
        boxes, images = Counter(), Counter()
        for image_id in image_ids:
            present = set()
            for class_id, _ in read_yolo(ROOT / f"data/detection/labels/real/oi_{image_id}.txt"):
                boxes[class_id] += 1
                present.add(class_id)
            images.update(present)
        box_counts[split] = boxes
        image_counts[split] = images

    requires_review = {mid: item for mid, item in mapped.items() if item.get("status") == "REQUIRES_REVIEW"}
    confirmed = {mid: item for mid, item in mapped.items() if item.get("status") == "CONFIRMED"}
    return {
        "source_boxes_csv": str(oi_boxes_path),
        "source_boxes_sha256": sha256(oi_boxes_path),
        "source_classes_csv": str(oi_classes_path),
        "source_classes_sha256": sha256(oi_classes_path),
        "v2_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "v2_manifest_sha256": sha256(manifest_path),
        "v2_manifest_fields": manifest_fields,
        "license_attribution_status": (
            "BLOCKED: the v2 manifest records a generic URL but has no per-image license or attribution fields."
        ),
        "active_images_by_split": {key: len(value) for key, value in ids_by_split.items()},
        "missing_v2_provenance": missing,
        "active_count_mismatches": mismatches,
        "mapping_review": {
            "confirmed_non_null_mids": confirmed,
            "requires_review_non_null_mids": requires_review,
            "collector_behavior": "collect_real_detection_v2.py selects every non-null class_id; status is not checked.",
            "active_openimages_boxes_by_split": {
                split: {str(key): value for key, value in sorted(counts.items())}
                for split, counts in box_counts.items()
            },
            "active_openimages_images_by_split_and_class": {
                split: {str(key): value for key, value in sorted(counts.items())}
                for split, counts in image_counts.items()
            },
            "clothes_boxes_class_3": sum(counts[3] for counts in box_counts.values()),
            "shoes_boxes_class_8": sum(counts[8] for counts in box_counts.values()),
            "status": "BLOCKED_PENDING_SCOPE_AND_OBJECT_REVIEW",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oi-boxes-csv",
        type=Path,
        default=(Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "waste-training"))) / "openimages-validation-boxes.csv"),
    )
    parser.add_argument(
        "--oi-classes-csv",
        type=Path,
        default=(Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "waste-training"))) / "openimages-boxable-classes.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/PHASE3_B_DATA_P0_AUDIT_2026-10-11.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/detection_source_mapping.yaml").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "data/audit/taco_annotations_raw.json").read_text(encoding="utf-8"))
    report = {
        "scope": "Phase 3B data P0 provenance; no GT, mapping, OI label, split, or raw image mutation",
        "taco": taco_audit(raw),
        "remaining_taco_mirrors": mirror_audit(),
        "openimages": openimages_audit(args.oi_boxes_csv, args.oi_classes_csv, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "taco_cases": len(report["taco"]["cases"]),
        "remaining_oob_mirrors": sum(item["out_of_bounds"] for item in report["remaining_taco_mirrors"]),
        "oi_missing_provenance": len(report["openimages"]["missing_v2_provenance"]),
        "oi_count_mismatches": len(report["openimages"]["active_count_mismatches"]),
        "oi_clothes_boxes": report["openimages"]["mapping_review"]["clothes_boxes_class_3"],
        "oi_shoes_boxes": report["openimages"]["mapping_review"]["shoes_boxes_class_8"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
