"""Evaluate the application detection pipeline on a fingerprinted real split."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageOps

from src.app.detection_logic import build_detection_pipeline
from src.app.detection_settings import detector_input_threshold, parse_detection_settings
from src.data.detection_manifest import load_detection_manifest, read_yolo_labels
from src.detection.evaluation import (
    ImagePrediction, calibrate_thresholds, pipeline_identity, summarize_predictions,
)
from src.detection.factory import create_detector
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection


def collect_predictions(records: list[dict], settings, score_floor: float) -> list[ImagePrediction]:
    collection = settings if settings.enable_tiling else replace(
        settings, confidence_threshold=score_floor, per_class_thresholds=(),
        disabled_classes=(), thresholds_path=None,
    )
    base = create_detector(
        collection.backend, collection.model_path,
        secondary_model_path=collection.secondary_model_path, device=collection.device,
        confidence_threshold=detector_input_threshold(collection),
        iou_threshold=collection.iou_threshold, image_size=collection.image_size,
        max_detections=collection.max_detections, fusion_weights=collection.fusion_weights,
        fusion_iou_threshold=collection.fusion_iou_threshold,
        fusion_input_threshold=collection.fusion_input_threshold,
    )
    pipeline = build_detection_pipeline(base, collection)
    samples = []
    for record in records:
        path = Path(record["image"])
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        targets = tuple(
            Detection(
                cls, DETECTION_CLASS_NAMES[cls], 1.0,
                BoundingBox(
                    (cx - bw / 2) * width, (cy - bh / 2) * height,
                    (cx + bw / 2) * width, (cy + bh / 2) * height,
                ),
            )
            for cls, cx, cy, bw, bh in read_yolo_labels(Path(record["labels"]))
        )
        samples.append(ImagePrediction(path, record["source"], targets, pipeline.detect_pil(image)))
        if len(samples) % 50 == 0:
            print(f"Evaluated {len(samples)}/{len(records)} images", flush=True)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--minimum-precision", type=float, default=.8)
    parser.add_argument("--score-floor", type=float, default=.05)
    parser.add_argument("--thresholds-profile", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args, detector_args = parser.parse_known_args(argv)
    if args.calibrate and args.split != "val":
        parser.error("Calibration requires the validation split")
    if args.calibrate and args.thresholds_profile:
        parser.error("Calibrate and apply a profile in separate evaluations")
    if not 0 < args.score_floor <= 1 or args.threads < 1:
        parser.error("Score floor must be in (0, 1] and threads must be positive")
    settings = parse_detection_settings(detector_args, os.environ)
    if settings.enable_tiling and args.calibrate:
        parser.error("Calibrate whole-image inference first; tile suppression depends on the input cutoff")
    if args.score_floor > settings.confidence_threshold:
        parser.error("Score floor must not exceed the operating threshold")
    dataset = load_detection_manifest(args.dataset)
    identity = pipeline_identity(settings)
    thresholds = dict(settings.per_class_thresholds)
    disabled_classes = settings.disabled_classes
    if args.thresholds_profile:
        profile = json.loads(args.thresholds_profile.read_text(encoding="utf-8"))
        if profile.get("selected_on") != "val" or profile.get("pipeline") != identity:
            parser.error("Threshold profile does not match this checkpoint and pipeline")
        if profile.get("dataset_fingerprint") != dataset["fingerprint"]:
            parser.error("Threshold profile belongs to another dataset")
        if profile.get("score_floor") != args.score_floor:
            parser.error("Threshold profile was collected with a different score floor")
        thresholds = profile["per_class_thresholds"]
        disabled_classes = tuple(profile.get("disabled_classes", ()))
        settings = replace(settings, per_class_thresholds=tuple(thresholds.items()), disabled_classes=disabled_classes)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(args.threads)
    samples = collect_predictions(dataset["splits"][args.split], settings, args.score_floor)
    report = {
        "split": args.split, "dataset_fingerprint": dataset["fingerprint"],
        "pipeline": identity, "score_floor": args.score_floor,
        "default_threshold": settings.confidence_threshold,
        "metrics": summarize_predictions(
            samples, default_threshold=settings.confidence_threshold, per_class_thresholds=thresholds,
            disabled_classes=disabled_classes,
        ),
    }
    if args.calibrate:
        profile = {
            **calibrate_thresholds(samples, split=args.split, min_precision=args.minimum_precision),
            "dataset_fingerprint": dataset["fingerprint"], "pipeline": identity,
            "score_floor": args.score_floor,
        }
        (args.output / "thresholds.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        report["calibrated_metrics"] = summarize_predictions(
            samples, per_class_thresholds=profile["per_class_thresholds"],
            disabled_classes=profile["disabled_classes"],
        )
    for name, value in (
        ("metrics.json", report),
        ("predictions.json", [
            {
                "image": str(sample.image), "source": sample.source,
                "ground_truth": [asdict(row) for row in sample.ground_truth],
                "predictions": [asdict(row) for row in sample.result.detections],
            }
            for sample in samples
        ]),
    ):
        temporary = args.output / (name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(args.output / name)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
