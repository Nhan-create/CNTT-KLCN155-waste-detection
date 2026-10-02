"""Train phase-2 SSDLite/YOLOv8n on reviewed train/val data; never run test here."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.detection.dataset import (
    DetectionDatasetError,
    load_detection_dataset_yaml,
    validate_detection_dataset,
)
from src.detection.training_common import run_metadata, write_json


def _load_training_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DetectionDatasetError(f"Training config not found: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("train"), dict):
        raise DetectionDatasetError("Training config requires a 'train' mapping")
    return payload


def _resolved_dataset_file(data_path: Path, output_root: Path, run_name: str) -> Path:
    payload, root = load_detection_dataset_yaml(data_path)
    resolved = dict(payload)
    resolved["path"] = str(root)
    # Ultralytics must never follow download hooks or access the held-out test split.
    resolved.pop("download", None)
    resolved.pop("test", None)
    output_root.mkdir(parents=True, exist_ok=True)
    for split_key in ("train", "val"):
        split_val = resolved.get(split_key)
        if isinstance(split_val, str) and split_val.strip():
            split_p = Path(split_val)
            split_full = split_p if split_p.is_absolute() else root / split_p
            if split_full.is_file() and split_full.suffix.lower() == ".txt":
                lines = split_full.read_text(encoding="utf-8").splitlines()
                abs_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    lp = Path(line)
                    abs_lines.append(str((lp if lp.is_absolute() else root / lp).resolve()).replace("\\", "/"))
                dest_split_file = output_root / f"{run_name}-{split_key}-abs.txt"
                dest_split_file.write_text("\n".join(abs_lines) + "\n", encoding="utf-8")
                resolved[split_key] = str(dest_split_file).replace("\\", "/")
    destination = output_root / f"{run_name}-resolved-dataset.yaml"
    destination.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def _serialize_test_metrics(metrics: Any) -> dict[str, Any]:
    """Serialize metrics (legacy name kept for callers; training only passes val)."""

    overall = {key: float(value) for key, value in metrics.results_dict.items()}
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", {})
    if box is None or not isinstance(names, dict):
        return {"overall": overall, "per_class": {}}

    metric_arrays = {
        "precision": getattr(box, "p", []),
        "recall": getattr(box, "r", []),
        "f1": getattr(box, "f1", []),
        "map50": getattr(box, "ap50", []),
        "map50_95": getattr(box, "maps", []),
    }
    per_class: dict[str, dict[str, float]] = {}
    observed_ids = list(getattr(box, "ap_class_index", sorted(names)))
    for class_index, class_name in sorted(names.items()):
        values: dict[str, float] = {}
        for metric_name, metric_values in metric_arrays.items():
            try:
                # p/r/f1/ap50 contain observed classes only, maps is indexed globally.
                index = class_index if metric_name == "map50_95" else observed_ids.index(class_index)
                values[metric_name] = float(metric_values[index])
            except (IndexError, KeyError, TypeError, ValueError):
                continue
        per_class[str(class_name)] = values
    return {"overall": overall, "per_class": per_class}


def _best_f1_confidence(metrics: Any) -> dict[str, float]:
    """Choose the operating confidence from the validation F1-confidence curve."""

    box = getattr(metrics, "box", None)
    for x_values, y_values, x_title, y_title in getattr(box, "curves_results", []):
        if x_title != "Confidence" or y_title != "F1":
            continue
        confidence = np.asarray(x_values, dtype=float)
        f1_by_class = np.asarray(y_values, dtype=float)
        if confidence.size == 0 or f1_by_class.size == 0:
            break
        mean_f1 = np.nanmean(f1_by_class, axis=0)
        best_index = int(np.nanargmax(mean_f1))
        return {
            "confidence": float(confidence[best_index]),
            "mean_f1": float(mean_f1[best_index]),
        }
    return {}


class AblationAugmentationTransform:
    """Applies official Phase-2 augmentation ablation transforms within YOLO dataset pipeline."""

    def __init__(self, strategy: str = "none", donor_bank: Any = None, seed: int | None = None):
        self.strategy = strategy.lower().strip()
        self.donor_bank = donor_bank
        self.seed = seed

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.strategy == "none":
            return labels
        img = labels.get("img")
        bboxes = labels.get("bboxes", [])
        cls = labels.get("cls", [])
        if img is None or len(bboxes) == 0:
            return labels

        boxes_list = [list(b) for b in bboxes]
        cats_list = [int(c[0]) if hasattr(c, "__len__") else int(c) for c in cls]

        from src.detection.augmentation import apply_augmentation
        t_img, t_boxes, t_cats = apply_augmentation(
            image=img,
            boxes=boxes_list,
            category_ids=cats_list,
            strategy=self.strategy,
            donor_bank=self.donor_bank,
            seed=self.seed,
        )
        labels["img"] = t_img
        labels["bboxes"] = np.array(t_boxes, dtype=np.float32).reshape(-1, 4)
        labels["cls"] = np.array(t_cats, dtype=np.float32).reshape(-1, 1)
        return labels


def _make_yolo_trainer(metadata: dict[str, Any]):
    """Remove hidden augmentation and explicitly select checkpoints by val mAP."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Phase2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model.phase2_metadata = dict(metadata, trained=True)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode=mode, batch=batch)
            dataset.augment = False
            dataset.transforms = dataset.build_transforms(self.args)
            strategy = str(metadata.get("augmentation_variant", "none")).lower().strip()
            if mode == "train" and strategy != "none":
                from src.detection.copy_paste import DonorBank
                donor_bank = DonorBank.from_dataset_root(".") if strategy in ("combined", "copy_paste") else None
                ablation_transform = AblationAugmentationTransform(strategy=strategy, donor_bank=donor_bank)
                if hasattr(dataset.transforms, "transforms") and isinstance(dataset.transforms.transforms, list):
                    dataset.transforms.transforms.insert(0, ablation_transform)
            return dataset

        def validate(self):
            metrics = self.validator(self)
            if metrics is None:
                raise RuntimeError("Validation must run every epoch for phase-2 selection")
            metrics.pop("fitness", None)
            fitness = float(metrics["metrics/mAP50-95(B)"])
            if not np.isfinite(fitness):
                raise RuntimeError("Non-finite YOLOv8n validation mAP")
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_fitness = fitness
            return metrics, fitness

    return Phase2Trainer


def _train_yolo(data_path: Path, config: dict[str, Any], run_directory: Path,
                metadata: dict[str, Any]) -> Path:
    from ultralytics import YOLO
    from src.detection.yolo import validate_yolov8n_architecture, validate_yolo_metadata

    model = YOLO(str(config.get("model", "yolov8n.pt")), task="detect")
    validate_yolov8n_architecture(model)
    args = dict(config["train"])
    args.pop("freeze_bn", None)
    args.pop("clip_grad_norm", None)
    max_hours = args.pop("max_hours", None)
    if max_hours is not None:
        args["time"] = float(max_hours)
    if str(args.get("device", "auto")) == "auto":
        args.pop("device", None)
    args.update(project=str(run_directory.parent), name=run_directory.name, exist_ok=True,
                val=True, split="val", max_det=100, conf=0.001,
                augment=False, multi_scale=False, close_mosaic=0, mosaic=0.0,
                mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
                scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
                auto_augment=None, erasing=0.0)
    # Disable expensive AMP capability self-checks that can download another YOLO model.
    # CUDA autocast remains optional for SSDLite; YOLO comparison uses full precision.
    args["amp"] = False
    write_json(run_directory / "effective_train_args.json", args)
    started = time.monotonic()
    model.train(data=str(data_path), trainer=_make_yolo_trainer(metadata), **args)
    best_path = Path(model.trainer.best)
    if not best_path.is_file():
        raise RuntimeError(f"Training completed without best detector: {best_path}")
    best_model = YOLO(str(best_path), task="detect")
    validate_yolo_metadata(best_model)
    validation_metrics = best_model.val(data=str(data_path), split="val",
        imgsz=int(args.get("imgsz", 320)), device=args.get("device"), max_det=100,
        conf=0.001, plots=True, project=str(run_directory), name="validation", exist_ok=True)
    write_json(run_directory / "training_summary.json", {
        **metadata, "trained": True, "best_checkpoint": str(best_path),
        "effective_train_args": args, "elapsed_seconds": time.monotonic() - started,
        "epochs_completed": int(model.trainer.epoch) + 1,
        "validation_metrics": _serialize_test_metrics(validation_metrics),
        "recommended_threshold_from_val": _best_f1_confidence(validation_metrics),
    })
    return best_path


def train_detector(
    data_path: Path,
    config_path: Path,
    *,
    model_override: str | None = None,
    device_override: str | None = None,
    variant_override: str | None = None,
    seed_override: int | None = None,
) -> Path:
    """Validate before downloads; keep test locked until final evaluation."""

    config = deepcopy(_load_training_config(config_path))
    backend = str(config.get("backend", "ssdlite"))
    if backend not in {"ssdlite", "yolov8n"}:
        raise ValueError("Phase 2 supports only ssdlite and yolov8n")
    config["backend"] = backend
    if model_override is not None:
        config["model"] = model_override
    if backend == "yolov8n" and not (
        Path(str(config.get("model", "yolov8n.pt"))).is_file()
        or str(config.get("model", "yolov8n.pt")) in {"yolov8n.pt", "yolov8n.yaml"}
    ):
        raise ValueError("Only YOLOv8n pretrained weights/config are allowed")
    if backend == "ssdlite" and config.get("model", "ssdlite320_mobilenet_v3_large") != "ssdlite320_mobilenet_v3_large":
        raise ValueError("SSDLite requires model: ssdlite320_mobilenet_v3_large")
    args = config["train"]
    if device_override is not None:
        args["device"] = device_override
    if seed_override is not None:
        args["seed"] = seed_override
    if int(args.get("epochs", 120)) < 1:
        raise ValueError("epochs must be positive")
    if args.get("max_hours") is not None and float(args["max_hours"]) <= 0:
        raise ValueError("max_hours must be positive")
    report = validate_detection_dataset(data_path, splits=("train", "val"))
    payload, _ = load_detection_dataset_yaml(data_path)
    variant = variant_override or str(config.get("augmentation_variant", "none"))
    if variant not in {"none", "geometric", "photometric", "combined"}:
        raise ValueError("Unknown offline augmentation variant")
    if payload.get("augmentation_variant", "none") != variant:
        raise DetectionDatasetError("Requested augmentation variant does not match dataset YAML")
    config["augmentation_variant"] = variant
    project = Path(str(config.get("project", "artifacts/detection"))).resolve()
    default_name = "waste-ssdlite320-mobilenetv3" if backend == "ssdlite" else "waste-yolov8n"
    run_name = str(config.get("name", default_name))
    if variant != "none" or seed_override is not None:
        run_name += f"-{variant}-seed{int(args.get('seed', 42))}"
    if Path(run_name).name != run_name:
        raise ValueError("Run name must be a single directory name")
    run_directory = project / run_name
    if run_directory.exists():
        raise ValueError(f"Run already exists; choose a new config name: {run_directory}")
    metadata = run_metadata(data_path, config)
    metadata["dataset_report"] = asdict(report)
    run_directory.mkdir(parents=True)
    write_json(run_directory / "run_metadata.json", metadata)
    resolved_data = _resolved_dataset_file(data_path, run_directory, "train-val")
    # SSDLite uses original YAML for reviewed manifest resolution; YOLO sees only train/val.
    try:
        if backend == "ssdlite":
            from src.detection.ssdlite_train import train_ssdlite
            return train_ssdlite(data_path, config, run_directory, metadata)
        return _train_yolo(resolved_data, config, run_directory, metadata)
    except Exception as error:
        write_json(run_directory / "failure.json", {"error": str(error), "test_evaluated": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("configs/detection_dataset.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/detection_training.yaml")
    )
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--variant", choices=("none", "geometric", "photometric", "combined"))
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    try:
        best_path = train_detector(
            arguments.data,
            arguments.config,
            model_override=arguments.model,
            device_override=arguments.device,
            variant_override=arguments.variant,
            seed_override=arguments.seed,
        )
    except (DetectionDatasetError, RuntimeError, ValueError, ImportError) as error:
        print(f"Detection training failed: {error}")
        return 2
    print(json.dumps({"best_detector": str(best_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
