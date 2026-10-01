"""Select detector/WBF inference settings using validation images only."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

from src.detection.evaluate import (
    coco_metrics, collect_predictions, file_hash, load_evaluation_split, operating_metrics,
)
from src.detection.factory import create_detector
from src.detection.fusion import fuse_detections


def tune(data_path, backend, primary, secondary, config_path, output, device="auto"):
    output = Path(output)
    if output.exists():
        raise ValueError("Selection output already exists; choose a new path")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows, fingerprint = load_evaluation_split(Path(data_path), "val")
    models = {"primary": str(Path(primary).resolve())}
    if backend == "wbf":
        if secondary is None:
            raise ValueError("WBF tuning needs a YOLOv8n secondary checkpoint")
        models["secondary"] = str(Path(secondary).resolve())
    floor = float(config["ap_score_floor"])
    if not 0 < floor <= 0.01:
        raise ValueError("Use a low, positive AP proposal score floor, at most 0.01")
    common = dict(image_size=int(config["image_size"]), max_detections=100)
    trials, best_predictions, best_trial, targets = [], None, None, None
    for nms_iou in config["nms_iou_grid"]:
        options = dict(common, iou_threshold=float(nms_iou), confidence_threshold=floor, device=device)
        first_backend = "ssdlite" if backend == "wbf" else backend
        predictions, targets, _ = collect_predictions(create_detector(first_backend, primary, **options), rows)
        second_predictions = None
        if backend == "wbf":
            second_predictions, _, _ = collect_predictions(create_detector("yolov8n", secondary, **options), rows)
            candidate_grid = itertools.product(config["fusion_iou_grid"], config["fusion_weights_grid"], config["fusion_input_confidence_grid"])
        else:
            candidate_grid = [(None, None, None)]
        for fusion_iou, weights, input_confidence in candidate_grid:
            candidate_options = dict(common, iou_threshold=float(nms_iou))
            candidate_predictions = predictions
            if backend == "wbf":
                if float(input_confidence) < floor:
                    raise ValueError("Fusion input confidence cannot be below the cached AP floor")
                candidate_predictions = [fuse_detections(
                    (first, second), weights=weights, iou_threshold=float(fusion_iou),
                    confidence_threshold=float(input_confidence), output_threshold=floor,
                ) for first, second in zip(predictions, second_predictions, strict=True)]
                candidate_options.update(fusion_iou_threshold=float(fusion_iou),
                                         fusion_weights=list(weights), fusion_input_threshold=float(input_confidence))
            metrics = coco_metrics(candidate_predictions, targets)
            score = metrics["map50_95"]
            if score is None:
                raise ValueError("Validation split has no annotated positives; cannot tune AP")
            trial = {"detector_options": candidate_options, "map50_95": score, "map50": metrics["map50"]}
            trials.append(trial)
            # Stable first-in-grid tie break; never inspect test to break ties.
            if best_trial is None or score > best_trial["map50_95"]:
                best_trial, best_predictions = trial, candidate_predictions
    if best_trial is None or best_predictions is None:
        raise ValueError("Tuning grid is empty")
    matching_iou = float(config["matching_iou"])
    operating_trials = [operating_metrics(best_predictions, targets, confidence=float(conf), iou_threshold=matching_iou)
                        for conf in config["operating_confidence_grid"]]
    if not operating_trials:
        raise ValueError("Operating confidence grid is empty")
    operating = max(operating_trials, key=lambda row: row["micro"]["f1"])
    selection = {"schema_version": 1, "selected_on": "val", "backend": backend,
                 "validation_fingerprint": fingerprint, "models": models,
                 "model_hashes": {key: file_hash(path) for key, path in models.items()},
                 "objective": "COCO mAP@0.5:0.95", "ap_score_floor": floor,
                 "matching_iou": matching_iou, "detector_options": best_trial["detector_options"],
                 "operating_confidence": operating["confidence"],
                 "operating_objective": "micro F1 on validation at fixed matching IoU",
                 "best_validation_metrics": coco_metrics(best_predictions, targets),
                 "grid": config, "trials": trials, "operating_trials": operating_trials}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/detection_dataset.yaml"))
    parser.add_argument("--backend", choices=("ssdlite", "yolov8n", "wbf"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--secondary-model", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/detection_fusion.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    selection = tune(args.data, args.backend, args.model, args.secondary_model, args.config, args.output, args.device)
    print(json.dumps({"selection": str(args.output), "validation_map50_95": selection["best_validation_metrics"]["map50_95"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
