"""Train and evaluate SSDLite on versioned, real-scene detection manifests."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import random
import time
from types import MethodType
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from src.detection.evaluation import match_predictions
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite import ARCHITECTURE, build_ssdlite, resolve_device, tensor_detections
from src.detection.types import BoundingBox, Detection


class DetectionTrainingDataset(Dataset):
    """Keep image/box coordinates identical to the runtime detector input."""

    def __init__(self, records: list[dict], *, augment: bool = False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from src.data.detection_manifest import read_yolo_labels

        record = self.records[index]
        with Image.open(record["image"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255)
        width, height = image.size
        boxes, labels = [], []
        for class_id, x, y, box_width, box_height in read_yolo_labels(Path(record["labels"])):
            boxes.append([
                (x - box_width / 2) * width, (y - box_height / 2) * height,
                (x + box_width / 2) * width, (y + box_height / 2) * height,
            ])
            labels.append(class_id + 1)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and torch.rand(()).item() < 0.5:
            tensor = tensor.flip((2,))
            boxes_tensor[:, [0, 2]] = width - boxes_tensor[:, [2, 0]]
        return tensor, {"boxes": boxes_tensor, "labels": torch.tensor(labels, dtype=torch.int64)}, record


def ssd_loss_with_background(model, targets, head_outputs, anchors, matched_idxs):
    """TorchVision SSD loss with a hard-negative quota on empty-target images.

    Positive images retain SSD's original negative-to-positive sampling rule.
    Empty frames contribute negative anchors instead of silently having zero loss.
    """
    regression = head_outputs["bbox_regression"]
    logits = head_outputs["cls_logits"]
    box_losses, class_targets = [], []
    foreground_count = 0
    for target, per_image_regression, per_image_logits, per_image_anchors, matches in zip(
        targets, regression, logits, anchors, matched_idxs, strict=True
    ):
        foreground = torch.where(matches >= 0)[0]
        selected_targets = matches[foreground]
        foreground_count += selected_targets.numel()
        encoded = model.box_coder.encode_single(target["boxes"][selected_targets], per_image_anchors[foreground])
        box_losses.append(F.smooth_l1_loss(per_image_regression[foreground], encoded, reduction="sum"))
        classes = torch.zeros(
            per_image_logits.shape[0], dtype=target["labels"].dtype, device=target["labels"].device
        )
        classes[foreground] = target["labels"][selected_targets]
        class_targets.append(classes)
    class_targets = torch.stack(class_targets)
    class_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), class_targets.reshape(-1), reduction="none"
    ).reshape(class_targets.shape)
    foreground = class_targets > 0
    positive_counts = foreground.sum(1, keepdim=True)
    negative_quota = torch.where(
        positive_counts > 0,
        model.neg_to_pos_ratio * positive_counts,
        torch.full_like(positive_counts, model.minimum_background_samples),
    )
    ranks = class_loss.detach().masked_fill(foreground, -float("inf")).argsort(1, descending=True).argsort(1)
    background = (ranks < negative_quota) & ~foreground
    # An empty frame receives the same scale as SSD's usual negative:positive ratio.
    empty_negatives = (background & (positive_counts == 0)).sum().item()
    normalizer = max(1.0, foreground_count + empty_negatives / model.neg_to_pos_ratio)
    return {
        "bbox_regression": torch.stack(box_losses).sum() / max(1, foreground_count),
        "classification": (class_loss[foreground].sum() + class_loss[background].sum()) / normalizer,
    }


def configure_background_loss(model, minimum_background_samples: int = 32):
    if minimum_background_samples < 1:
        raise ValueError("minimum_background_samples must be positive")
    model.minimum_background_samples = minimum_background_samples
    model.compute_loss = MethodType(ssd_loss_with_background, model)
    return model


def collate_detection_batch(rows):
    images, targets, records = zip(*rows, strict=True)
    return list(images), list(targets), list(records)


def _with_rates(counts):
    counts = dict(counts)
    true = counts.get("true_positives", 0)
    counts["precision"] = true / max(1, counts.get("predictions", 0))
    counts["recall"] = true / max(1, counts.get("objects", 0))
    return counts


def _operating_counts(output, target, confidence_threshold, width, height):
    decoded = tensor_detections(output, width, height, confidence_threshold=confidence_threshold)
    labels = target["labels"].tolist()
    targets = tuple(
        Detection(label - 1, DETECTION_CLASS_NAMES[label - 1], 1.0, BoundingBox(*box))
        for box, label in zip(target["boxes"].tolist(), labels, strict=True)
    )
    counts = {index: Counter(objects=labels.count(index), predictions=0, true_positives=0)
              for index in range(1, len(DETECTION_CLASS_NAMES) + 1)}
    for prediction, correct in zip(decoded.detections, match_predictions(decoded.detections, targets), strict=True):
        category = prediction.class_index + 1
        counts[category]["predictions"] += 1
        counts[category]["true_positives"] += int(correct)
    return counts


@torch.inference_mode()
def evaluate_detector(model, loader, device, *, confidence_threshold: float = 0.29):
    """COCO AP and fixed-cutoff, one-to-one class-aware IoU>=.50 metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    images, annotations, predictions = [], [], []
    per_class = {index: Counter(objects=0, predictions=0, true_positives=0)
                 for index in range(1, len(DETECTION_CLASS_NAMES) + 1)}
    per_source = {}
    for tensors, targets, records in loader:
        outputs = model([tensor.to(device) for tensor in tensors])
        for tensor, target, output, record in zip(tensors, targets, outputs, records, strict=True):
            image_id = len(images) + 1
            images.append({"id": image_id, "width": tensor.shape[2], "height": tensor.shape[1]})
            output = {key: value.detach().cpu() for key, value in output.items()}
            for box, label in zip(target["boxes"].tolist(), target["labels"].tolist(), strict=True):
                x1, y1, x2, y2 = box
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": label,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
            for box, label, score in zip(
                output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist(), strict=True
            ):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": image_id, "category_id": label, "score": score,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
            source = per_source.setdefault(record["source"], Counter(objects=0, predictions=0, true_positives=0))
            for category, counts in _operating_counts(
                output, target, confidence_threshold, tensor.shape[2], tensor.shape[1],
            ).items():
                per_class[category].update(counts)
                source.update(counts)
    if not images:
        raise ValueError("Evaluation split must contain images")
    report = {
        "images": len(images), "confidence_threshold": confidence_threshold, "matching_iou": 0.5,
        "map50": 0.0, "map50_95": 0.0,
        "per_class": {name: _with_rates(per_class[index]) for index, name in enumerate(DETECTION_CLASS_NAMES, 1)},
        "per_source": {name: _with_rates(counts) for name, counts in per_source.items()},
    }
    total = Counter(objects=0, predictions=0, true_positives=0)
    for counts in per_class.values():
        total.update(counts)
    report.update(_with_rates(total))
    if predictions and annotations:
        with redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = {
                "info": {}, "images": images, "annotations": annotations,
                "categories": [{"id": index, "name": name} for index, name in enumerate(DETECTION_CLASS_NAMES, 1)],
            }
            truth.createIndex()
            evaluation = COCOeval(truth, truth.loadRes(predictions), "bbox")
            evaluation.params.imgIds = list(range(1, len(images) + 1))
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        report["map50_95"] = max(0.0, float(evaluation.stats[0]))
        report["map50"] = max(0.0, float(evaluation.stats[1]))
        precision = evaluation.eval["precision"]
        for index, name in enumerate(DETECTION_CLASS_NAMES):
            values = precision[:, :, index, 0, -1]
            valid = values[values >= 0]
            report["per_class"][name]["map50_95"] = float(valid.mean()) if len(valid) else None
    return report


@dataclass(frozen=True)
class TrainingConfig:
    dataset: Path
    output: Path
    initial_weights: Path
    epochs: int = 8
    learning_rate: float = 0.0003
    batch_size: int = 4
    patience: int = 3
    workers: int = 0
    device: str = "cpu"
    seed: int = 42
    minimum_background_samples: int = 32
    confidence_threshold: float = 0.29
    threads: int = 2
    resume: bool = False

    def validate(self):
        if min(self.epochs, self.batch_size, self.patience, self.threads, self.minimum_background_samples) < 1:
            raise ValueError("epochs, batch_size, patience, threads and background quota must be positive")
        if self.workers < 0 or not 0 <= self.seed < 2 ** 32:
            raise ValueError("workers must be nonnegative and seed must be in [0, 2**32)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(directory):
    # OS-held lock is released after a crash, so --resume does not need to guess
    # whether a PID in a stale text file still belongs to a training process.
    with (directory / ".training.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _rng_state(generator, device):
    numpy_rng = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "loader": generator.get_state(), "python": random.getstate(),
        "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def _restore_rng(state, generator, device):
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    random.setstate(state["python"])
    numpy_rng = state["numpy"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    if state["cuda"] and device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_epoch(model, loader, optimizer, device):
    model.train()
    # Stable fine-tuning with a small final batch and no lost training samples.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    totals = Counter()
    image_count = 0
    for batch_index, (images, targets, _) in enumerate(loader, 1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in batch {batch_index}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        image_count += len(images)
        for name, value in losses.items():
            totals[name] += float(value.detach()) * len(images)
        if batch_index % 25 == 0:
            print(json.dumps({"phase": "train", "batch": batch_index, "images": image_count}), flush=True)
    return {name: value / image_count for name, value in totals.items()}


def _configuration_record(config):
    values = asdict(config)
    for key in ("dataset", "output", "initial_weights"):
        values[key] = str(Path(values[key]).resolve())
    values.pop("resume")
    return values


def run_training(config: TrainingConfig):
    """Fine-tune into a new run and choose the model using real validation only."""
    from src.data.detection_manifest import load_detection_manifest, sha256_file

    config.validate()
    config = TrainingConfig(**{
        **asdict(config), "dataset": Path(config.dataset).resolve(), "output": Path(config.output).resolve(),
        "initial_weights": Path(config.initial_weights).resolve(),
    })
    manifest = load_detection_manifest(config.dataset, verify_files=True)
    if any(not manifest["splits"][split] for split in ("train", "val", "test")):
        raise ValueError("Training requires nonempty train, val and test splits")
    if any(not any(record["object_count"] for record in manifest["splits"][split]) for split in ("train", "val")):
        raise ValueError("Train and validation splits each need annotated objects")
    if not config.initial_weights.is_file():
        raise FileNotFoundError(config.initial_weights)
    if config.resume:
        if not (config.output / "weights" / "last.pt").is_file():
            raise ValueError("--resume requires weights/last.pt in the same run")
    else:
        config.output.mkdir(parents=True, exist_ok=False)
        (config.output / "weights").mkdir()
    with _run_lock(config.output):
        try:
            return _run_training(config, manifest, sha256_file)
        except Exception as error:
            _atomic_json(config.output / "failure.json", {
                "error": type(error).__name__, "message": str(error), "time": time.time(),
            })
            raise


def _run_training(config, manifest, sha256_file):
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    initial = torch.load(config.initial_weights, map_location="cpu", weights_only=True)
    metadata = initial.get("metadata", {})
    if (
        metadata.get("architecture") != ARCHITECTURE
        or tuple(metadata.get("class_names", ())) != DETECTION_CLASS_NAMES
        or metadata.get("backend") != "ssdlite"
        or metadata.get("background_index") != 0
        or metadata.get("trained") is not True
        or int(initial.get("epoch", 0)) < 1
    ):
        raise ValueError("Initial checkpoint architecture or class order does not match SSDLite")
    initial_hash = sha256_file(config.initial_weights)
    model = configure_background_loss(build_ssdlite(pretrained=False), config.minimum_background_samples).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    model.score_thresh, model.nms_thresh, model.detections_per_img = 0.001, 0.55, 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 20
    )
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            DetectionTrainingDataset(manifest["splits"][split], augment=split == "train"),
            batch_size=config.batch_size, shuffle=split == "train", num_workers=config.workers,
            collate_fn=collate_detection_batch, generator=generator if split == "train" else None,
            pin_memory=device.type == "cuda",
        ) for split in ("train", "val", "test")
    }
    run_metadata = {
        "architecture": ARCHITECTURE, "backend": "ssdlite", "class_names": list(DETECTION_CLASS_NAMES),
        "background_index": 0, "trained": True, "image_size": 320,
        "dataset_manifest": str(config.dataset), "dataset_fingerprint": manifest["fingerprint"],
        "initial_checkpoint": str(config.initial_weights), "initial_checkpoint_sha256": initial_hash,
        "preprocessing": "EXIF-corrected RGB float32 /255 at original size; model transform fixed 320",
        "validation_domain": "real", "selection_metric": "map50_95", "seed": config.seed,
        "minimum_background_samples": config.minimum_background_samples, "freeze_batch_norm_statistics": True,
    }
    configuration = _configuration_record(config)
    history, best_epoch, last_epoch, stale = [], 0, 0, 0
    best_score = -math.inf
    if config.resume:
        saved_run = json.loads((config.output / "run.json").read_text(encoding="utf-8"))
        stable_keys = set(configuration) - {"device", "workers", "threads"}
        if any(saved_run["config"][key] != configuration[key] for key in stable_keys):
            raise ValueError("Resume configuration differs from the original run")
        if saved_run["metadata"] != run_metadata:
            raise ValueError("Resume dataset or initial checkpoint changed")
        saved = torch.load(config.output / "weights" / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best_epoch, last_epoch = saved["history"], saved["best_training_epoch"], saved["training_epoch"]
        best_score, stale = saved["best_score"], saved["epochs_without_improvement"]
        best_payload = saved["best_checkpoint"]
        best_path = config.output / "weights" / "best.pt"
        if not best_path.is_file() or sha256_file(best_path) != saved["best_checkpoint_sha256"]:
            # last.pt is the committed training state. A crash between writing a
            # new best.pt and last.pt must not change the selected checkpoint.
            _atomic_checkpoint(best_path, best_payload)
            saved["best_checkpoint_sha256"] = sha256_file(best_path)
            _atomic_checkpoint(config.output / "weights" / "last.pt", saved)
        _restore_rng(saved["rng"], generator, device)
    else:
        _atomic_json(config.output / "run.json", {"config": configuration, "metadata": run_metadata})
        _atomic_json(config.output / "dataset.json", manifest)
        baseline = evaluate_detector(model, loaders["val"], device, confidence_threshold=config.confidence_threshold)
        best_score = baseline["map50_95"]
        history.append({"epoch": 0, "validation": baseline})
        best_payload = {
            "epoch": max(1, int(initial.get("epoch", 1))), "training_epoch": 0,
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": run_metadata, "validation": baseline,
        }
        _atomic_checkpoint(config.output / "weights" / "best.pt", best_payload)
        _atomic_checkpoint(config.output / "weights" / "last.pt", {
            **best_payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _rng_state(generator, device), "history": history, "best_training_epoch": 0,
            "best_score": best_score, "epochs_without_improvement": 0,
            "best_checkpoint": best_payload,
            "best_checkpoint_sha256": sha256_file(config.output / "weights" / "best.pt"),
        })
        _atomic_json(config.output / "history.json", history)
        _atomic_json(config.output / "progress.json", {"phase": "initial_validation", "epoch": 0})
        print(json.dumps({"phase": "initial_validation", **baseline}), flush=True)
    for epoch in range(last_epoch + 1, config.epochs + 1):
        if stale >= config.patience:
            break
        started = time.perf_counter()
        train_stats = _train_epoch(model, loaders["train"], optimizer, device)
        validation = evaluate_detector(model, loaders["val"], device, confidence_threshold=config.confidence_threshold)
        scheduler.step()
        row = {"epoch": epoch, "seconds": time.perf_counter() - started, "train": train_stats, "validation": validation}
        history.append(row)
        score = validation["map50_95"]
        payload = {
            "epoch": epoch, "training_epoch": epoch, "model_state_dict": model.state_dict(),
            "metadata": run_metadata, "validation": validation,
        }
        stale += 1
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_payload = {
                **payload,
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
            _atomic_checkpoint(config.output / "weights" / "best.pt", best_payload)
        _atomic_checkpoint(config.output / "weights" / "last.pt", {
            **payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _rng_state(generator, device), "history": history, "best_training_epoch": best_epoch,
            "best_score": best_score, "epochs_without_improvement": stale,
            "best_checkpoint": best_payload,
            "best_checkpoint_sha256": sha256_file(config.output / "weights" / "best.pt"),
        })
        _atomic_json(config.output / "history.json", history)
        _atomic_json(config.output / "progress.json", {
            "phase": "training", "epoch": epoch, "target_epochs": config.epochs,
            "best_training_epoch": best_epoch, "best_map50_95": best_score,
        })
        print(json.dumps(row, allow_nan=False), flush=True)
    best_path = config.output / "weights" / "best.pt"
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    # The test split participates only after checkpoint selection is finished.
    report = {
        "best_training_epoch": best_epoch, "checkpoint": str(best_path), "checkpoint_sha256": sha256_file(best_path),
        "dataset_fingerprint": manifest["fingerprint"], "validation": selected["validation"],
        "test": evaluate_detector(model, loaders["test"], device, confidence_threshold=config.confidence_threshold),
    }
    _atomic_json(config.output / "test_metrics.json", report)
    _atomic_json(config.output / "progress.json", {
        "phase": "complete", "epochs_completed": len(history) - 1, "best_training_epoch": best_epoch,
    })
    (config.output / "failure.json").unlink(missing_ok=True)
    return report
