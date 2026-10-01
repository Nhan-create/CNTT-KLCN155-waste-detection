"""COCO bbox evaluation and fixed-threshold precision/recall for phase 2."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageOps

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return intersection / (a.area + b.area - intersection)


def operating_metrics(predictions, targets, *, confidence=0.35, iou_threshold=0.5) -> dict:
    """Greedy confidence-ordered, same-class, one-to-one matching per image."""
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts differ")
    if not (0 <= confidence <= 1 and 0 < iou_threshold <= 1):
        raise ValueError("Invalid operating thresholds")
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in DETECTION_CLASS_NAMES}
    for result, ground_truth in zip(predictions, targets, strict=True):
        matched = set()
        for detection in sorted(result.detections, key=lambda item: -item.confidence):
            if detection.confidence < confidence:
                continue
            candidates = [(box_iou(detection.box, target.box), index)
                          for index, target in enumerate(ground_truth)
                          if index not in matched and detection.class_index == target.class_index]
            best_iou, index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                matched.add(index)
                counts[detection.class_id]["tp"] += 1
            else:
                counts[detection.class_id]["fp"] += 1
        for index, target in enumerate(ground_truth):
            if index not in matched:
                counts[target.class_id]["fn"] += 1

    def scores(values):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        return dict(values, precision=tp / (tp + fp) if tp + fp else 0.0,
                    recall=tp / (tp + fn) if tp + fn else 0.0,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)

    total = {key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn")}
    return {"confidence": confidence, "matching_iou": iou_threshold,
            "micro": scores(total), "per_class": {name: scores(v) for name, v in counts.items()}}


def coco_metrics(
    predictions: Sequence[DetectionResult], targets: Sequence[Sequence[Detection]],
) -> dict:
    """Official pycocotools COCO bbox AP; absent classes yield null, not fake zero AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions or len(predictions) != len(targets):
        raise ValueError("Need aligned, nonempty image predictions and targets")
    images, annotations, detections = [], [], []
    for image_id, (result, truth) in enumerate(zip(predictions, targets, strict=True), start=1):
        images.append({"id": image_id, "width": result.image_width, "height": result.image_height})
        for target in truth:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": target.class_index,
                                "bbox": [target.box.x1, target.box.y1, target.box.width, target.box.height],
                                "area": target.box.area, "iscrowd": 0})
        for row in result.detections:
            detections.append({"image_id": image_id, "category_id": row.class_index,
                               "bbox": [row.box.x1, row.box.y1, row.box.width, row.box.height],
                               "score": row.confidence})
    with redirect_stdout(io.StringIO()):
        truth_coco = COCO()
        truth_coco.dataset = {"images": images, "annotations": annotations, "info": {},
                              "categories": [{"id": i, "name": n} for i, n in enumerate(DETECTION_CLASS_NAMES)]}
        truth_coco.createIndex()
        if detections:
            predicted_coco = truth_coco.loadRes(detections)
        else:
            predicted_coco = COCO()
            predicted_coco.dataset = dict(truth_coco.dataset, annotations=[])
            predicted_coco.createIndex()
        evaluator = COCOeval(truth_coco, predicted_coco, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    def valid_mean(values):
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDet
    return {
        "map50_95": valid_mean(precision[:, :, :, 0, -1]),
        "map50": valid_mean(precision[0, :, :, 0, -1]),
        "ap_by_class": {
            name: {"ap50_95": valid_mean(precision[:, :, index, 0, -1]),
                   "ap50": valid_mean(precision[0, :, index, 0, -1]),
                   "instances": sum(row["category_id"] == index for row in annotations)}
            for index, name in enumerate(DETECTION_CLASS_NAMES)
        },
        "protocol": {"implementation": "pycocotools.COCOeval", "iou": "0.50:0.05:0.95",
                     "recall_points": 101, "max_detections": 100, "area": "all", "iscrowd": 0},
    }


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_split(data_path: Path, split: str):
    from src.detection.dataset import load_detection_dataset_yaml, validate_detection_dataset

    validate_detection_dataset(data_path)
    payload, root = load_detection_dataset_yaml(data_path)
    manifest_path = Path(str(payload["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No images in split {split}")
    for row in selected:
        for key in ("image_path", "label_path"):
            value = Path(row[key])
            row[key] = str(value if value.is_absolute() else root / value)
    fingerprint = hashlib.sha256()
    # All image/label bytes are bound to the selection, not just a filename list.
    for row in sorted(selected, key=lambda item: item["image_id"]):
        identity = {key: row.get(key) for key in ("image_id", "scene_id", "source_dataset", "is_real", "reviewed", "reviewer", "conditions")}
        fingerprint.update(json.dumps(identity, sort_keys=True).encode())
        for key in ("image_path", "label_path"):
            fingerprint.update(file_hash(row[key]).encode())
    return selected, fingerprint.hexdigest()


def image_targets(row) -> tuple[Image.Image, list[Detection]]:
    with Image.open(row["image_path"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    targets = []
    for line in Path(row["label_path"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, x, y, w, h = map(float, line.split())
        index = int(label)
        targets.append(Detection(index, DETECTION_CLASS_NAMES[index], 1.0,
                                 BoundingBox((x - w / 2) * width, (y - h / 2) * height,
                                             (x + w / 2) * width, (y + h / 2) * height)))
    return image, targets


def collect_predictions(detector, rows, *, warmup: int = 3):
    import torch

    first_image, _ = image_targets(rows[0])
    for _ in range(warmup):
        detector.detect_pil(first_image)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions, targets, timings = [], [], []
    for row in rows:
        image, truth = image_targets(row)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = detector.detect_pil(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(result)
        targets.append(truth)
    mean_ms = float(np.mean(timings))
    benchmark = {"batch_size": 1, "warmup_images": warmup, "images": len(rows),
                 "latency_mean_ms": mean_ms, "latency_p50_ms": float(np.median(timings)),
                 "latency_p95_ms": float(np.percentile(timings, 95)),
                 "fps": 1000 / mean_ms if mean_ms else None,
                 "scope": "PIL input through inference and postprocessing; excludes file decoding and UI",
                 "python": platform.python_version(), "platform": platform.platform(),
                 "cuda_available": torch.cuda.is_available(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None}
    return predictions, targets, benchmark


def evaluate_subsets(predictions, targets, rows, confidence, matching_iou):
    reports = {}
    conditions = sorted({condition for row in rows for condition in row.get("conditions", [])})
    for condition in conditions:
        indices = [i for i, row in enumerate(rows) if condition in row.get("conditions", [])]
        subset_predictions = [predictions[i] for i in indices]
        subset_targets = [targets[i] for i in indices]
        reports[condition] = {"images": len(indices), **coco_metrics(subset_predictions, subset_targets),
                              "operating": operating_metrics(subset_predictions, subset_targets,
                                                            confidence=confidence, iou_threshold=matching_iou)}
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/detection_dataset.yaml"))
    parser.add_argument("--selection", type=Path, required=True, help="Frozen validation selection JSON from tune")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    from src.detection.factory import create_detector

    if args.output.exists():
        raise ValueError("Evaluation output already exists; choose a new path")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("selected_on") != "val":
        raise ValueError("Selection must have been made on validation only")
    _, val_fingerprint = load_evaluation_split(args.data, "val")
    if selection["validation_fingerprint"] != val_fingerprint:
        raise ValueError("Validation data changed since tuning; selection is stale")
    for key, digest in selection["model_hashes"].items():
        if file_hash(selection["models"][key]) != digest:
            raise ValueError("Model changed since validation selection")
    rows, fingerprint = load_evaluation_split(args.data, args.split)
    options = dict(selection["detector_options"])
    options["device"] = args.device
    options["confidence_threshold"] = selection["ap_score_floor"]
    backend = selection["backend"]
    detector = create_detector(backend, selection["models"]["primary"],
                               secondary_model_path=selection["models"].get("secondary"), **options)
    predictions, targets, benchmark = collect_predictions(detector, rows)
    operating = selection["operating_confidence"]
    matching_iou = selection["matching_iou"]
    report = {"split": args.split, "dataset_fingerprint": fingerprint,
              "selection": selection, "metrics": coco_metrics(predictions, targets),
              "operating": operating_metrics(predictions, targets, confidence=operating, iou_threshold=matching_iou),
              "conditions": evaluate_subsets(predictions, targets, rows, operating, matching_iou),
              "benchmark": dict(benchmark, model_bytes=sum(Path(path).stat().st_size for path in selection["models"].values())),
              "predictions": [{"image_id": row["image_id"], "detections": [
                  {"class_id": d.class_id, "confidence": d.confidence,
                   "xyxy": [d.box.x1, d.box.y1, d.box.x2, d.box.y2]} for d in result.detections]}
                  for row, result in zip(rows, predictions, strict=True)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
