"""Dataset provenance and reproducibility shared by both phase-2 detectors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import VALID_IMAGE_EXTENSIONS
from src.detection.dataset import _label_path, _split_directory, load_detection_dataset_yaml
from src.detection.schema import DETECTION_CLASS_NAMES


def split_images(data_path: Path, split: str) -> tuple[list[Path], Path]:
    payload, root = load_detection_dataset_yaml(data_path)
    target = _split_directory(payload, root, split)
    if target.is_file() and target.suffix.lower() == ".txt":
        images = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            images.append((p if p.is_absolute() else root / p).resolve())
        return sorted(images), root
    return sorted(path for path in target.rglob("*")
                  if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS), root


def dataset_fingerprint(data_path: Path) -> str:
    """Hash bytes of train/val images+labels, never read test pixels or labels."""
    digest = hashlib.sha256()
    digest.update(json.dumps(DETECTION_CLASS_NAMES).encode())
    for split in ("train", "val"):
        images, root = split_images(data_path, split)
        for image in images:
            for path in (image, _label_path(image, root)):
                digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import os
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def run_metadata(data_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import torch

    payload, _ = load_detection_dataset_yaml(data_path)
    versions = {}
    for package in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 2,
        "backend": config["backend"],
        "class_names": list(DETECTION_CLASS_NAMES),
        "trained": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path.resolve()),
        "dataset_fingerprint_train_val_sha256": dataset_fingerprint(data_path),
        "augmentation_variant": payload.get("augmentation_variant", "none"),
        "online_augmentation": False,
        "selection_split": "val",
        "selection_metric": "map50_95",
        "test_evaluated": False,
        "seed": int(config["train"].get("seed", 42)),
        "config": config,
        "budget": {key: config["train"].get(key) for key in
                   ("epochs", "batch", "imgsz", "patience", "max_hours")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": versions, "torch_threads": torch.get_num_threads(),
                        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                        "gpus": [{"name": torch.cuda.get_device_name(index),
                                  "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                                 for index in range(torch.cuda.device_count())]},
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
