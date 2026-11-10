"""Measure instance detection and tune operating thresholds on validation images."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from src.detection.fusion import box_iou
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import Detection, DetectionResult


@dataclass(frozen=True)
class ImagePrediction:
    image: Path
    source: str
    ground_truth: tuple[Detection, ...]
    result: DetectionResult


def pipeline_identity(settings) -> dict:
    """Bind calibration to model bytes and the operations that produce scores."""
    from src.data.detection_manifest import sha256_file

    paths = {"primary": settings.model_path}
    if settings.secondary_model_path is not None:
        paths["secondary"] = settings.secondary_model_path
    if settings.enable_two_stage:
        paths["classifier"] = settings.classifier_path
    identity = {
        "model_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "models": {name: str(path.resolve()) for name, path in paths.items()},
        "options": {
            name: getattr(settings, name)
            for name in (
                "backend", "iou_threshold", "image_size", "max_detections",
                "enable_tiling", "tile_size", "tile_overlap", "tiling_min_side",
                "enable_two_stage", "two_stage_scoring_mode", "two_stage_padding",
                "two_stage_blend_weight", "two_stage_min_proposal_gate",
                "two_stage_proposal_threshold", "fusion_weights",
                "fusion_iou_threshold", "fusion_input_threshold",
            )
        },
    }
    return json.loads(json.dumps(identity))


def match_predictions(
    predictions: Sequence[Detection], targets: Sequence[Detection], *, iou_threshold: float = .5,
) -> list[bool]:
    """Match each target at most once, processing predictions in score order."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    used: set[int] = set()
    matches = [False] * len(predictions)
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence):
        prediction = predictions[index]
        candidates = [
            (box_iou(prediction.box, target.box), target_index)
            for target_index, target in enumerate(targets)
            if target_index not in used and prediction.class_index == target.class_index
        ]
        overlap, target_index = max(candidates, key=lambda item: item[0], default=(0.0, -1))
        if overlap >= iou_threshold:
            used.add(target_index)
            matches[index] = True
    return matches


def _metrics(counts: Counter) -> dict:
    true = counts["true_positives"]
    predicted = counts["predictions"]
    objects = counts["objects"]
    precision = true / predicted if predicted else 0.0
    recall = true / objects if objects else 0.0
    return {
        "images": counts["images"], "objects": objects, "predictions": predicted,
        "true_positives": true, "false_positives": predicted - true,
        "false_negatives": objects - true, "precision": precision, "recall": recall,
        "f1": 2 * true / (predicted + objects) if predicted + objects else 0.0,
    }


def summarize_predictions(
    samples: Sequence[ImagePrediction], *, default_threshold: float = .29,
    per_class_thresholds: dict[str, float] | None = None,
    disabled_classes: Sequence[str] = (),
) -> dict:
    thresholds = per_class_thresholds or {}
    if any(not 0 <= value <= 1 for value in [default_threshold, *thresholds.values()]):
        raise ValueError("Score thresholds must be in [0, 1]")
    totals = Counter()
    by_class = {name: Counter() for name in DETECTION_CLASS_NAMES}
    by_source: dict[str, Counter] = {}
    for sample in samples:
        source = by_source.setdefault(sample.source, Counter())
        for counts in (totals, source):
            counts["images"] += 1
            counts["objects"] += len(sample.ground_truth)
        for target in sample.ground_truth:
            by_class[target.class_id]["objects"] += 1
        for name in {target.class_id for target in sample.ground_truth}:
            by_class[name]["images"] += 1
        kept = [
            row for row in sample.result.detections
            if row.class_id not in disabled_classes
            and row.confidence >= thresholds.get(row.class_id, default_threshold)
        ]
        for prediction, correct in zip(kept, match_predictions(kept, sample.ground_truth), strict=True):
            for counts in (totals, source, by_class[prediction.class_id]):
                counts["predictions"] += 1
                counts["true_positives"] += int(correct)
    return {
        **_metrics(totals), "iou_threshold": .5,
        "per_class": {name: _metrics(counts) for name, counts in by_class.items()},
        "per_source": {name: _metrics(counts) for name, counts in by_source.items()},
    }


def calibrate_thresholds(
    samples: Sequence[ImagePrediction], *, split: str, min_precision: float = .8,
) -> dict:
    """Choose the best recall satisfying an empirical precision target per class.

    Unsupported classes use a threshold of 1 and carry an explicit status. The
    observed precision is an operating point on this validation set, not a
    probability guarantee for new scenes.
    """
    if split != "val":
        raise ValueError("Threshold calibration requires the validation split")
    if not 0 < min_precision <= 1:
        raise ValueError("Minimum precision must be in (0, 1]")
    classes = {}
    thresholds = {}
    for name in DETECTION_CLASS_NAMES:
        support = sum(target.class_id == name for sample in samples for target in sample.ground_truth)
        scored = []
        for sample in samples:
            predictions = [row for row in sample.result.detections if row.class_id == name]
            scored.extend(zip(
                (row.confidence for row in predictions),
                match_predictions(predictions, sample.ground_truth), strict=True,
            ))
        grouped: dict[float, Counter] = {}
        for score, correct in scored:
            counts = grouped.setdefault(score, Counter())
            counts["predictions"] += 1
            counts["true_positives"] += int(correct)
        cumulative = Counter(objects=support)
        operating_points = []
        for score in sorted(grouped, reverse=True):
            cumulative.update(grouped[score])
            metrics = _metrics(cumulative)
            if metrics["precision"] >= min_precision and metrics["true_positives"]:
                operating_points.append({"threshold": score, **metrics})
        if support and operating_points:
            selected = max(operating_points, key=lambda point: (point["recall"], point["precision"]))
            classes[name] = {"status": "selected_on_validation", **selected}
            thresholds[name] = selected["threshold"]
        else:
            thresholds[name] = 1.0
            classes[name] = {
                "status": "no_validation_targets" if not support else "precision_unattainable",
                "threshold": 1.0, "objects": support,
            }
    return {
        "selected_on": "val", "minimum_empirical_precision": min_precision,
        "per_class_thresholds": thresholds, "classes": classes,
        "disabled_classes": [name for name, row in classes.items() if row["status"] != "selected_on_validation"],
    }
