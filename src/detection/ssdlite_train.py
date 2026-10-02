"""Train SSDLite on reviewed bounding boxes; use validation only for model selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.detection.dataset import _label_path
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite import ARCHITECTURE, build_ssdlite, resolve_device, tensor_detections
from src.detection.training_common import seed_everything, split_images, write_json
from src.detection.types import BoundingBox, Detection


class YoloBoxDataset:
    """Consume the identical offline variant used by YOLO; no random transforms."""

    def __init__(self, data_path: Path, split: str):
        self.images, self.root = split_images(data_path, split)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        image_path = self.images[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        boxes, labels = [], []
        for line in _label_path(image_path, self.root).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_value, cx, cy, bw, bh = (float(value) for value in line.split())
            boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                          (cx + bw / 2) * width, (cy + bh / 2) * height])
            labels.append(int(class_value) + 1)  # zero is reserved for background
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return pil_to_tensor(image).float() / 255.0, {
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }


def collate_detection_batch(batch):
    return tuple(zip(*batch))


def evaluate_validation(model, loader, device) -> dict[str, Any]:
    import torch
    from src.detection.evaluate import coco_metrics

    predictions, targets = [], []
    model.eval()
    with torch.inference_mode():
        for images, annotations in loader:
            outputs = model([image.to(device) for image in images])
            for image, output, annotation in zip(images, outputs, annotations, strict=True):
                height, width = image.shape[-2:]
                predictions.append(tensor_detections(output, width, height, max_detections=100))
                rows = []
                for box, label in zip(annotation["boxes"], annotation["labels"], strict=True):
                    class_index = int(label) - 1
                    rows.append(Detection(class_index, DETECTION_CLASS_NAMES[class_index], 1.0,
                                          BoundingBox(*(float(value) for value in box))))
                targets.append(rows)
    return coco_metrics(predictions, targets)


def train_ssdlite(data_path: Path, config: dict[str, Any], run_directory: Path,
                  metadata: dict[str, Any]) -> Path:
    import torch
    from torch.utils.data import DataLoader

    args = config["train"]
    device = resolve_device(str(args.get("device", "auto")))
    seed_everything(int(args.get("seed", 42)), bool(args.get("deterministic", True)))
    train_data = YoloBoxDataset(data_path, "train")
    val_data = YoloBoxDataset(data_path, "val")
    batch_size = int(args.get("batch", 4))
    if batch_size <= 0 or not train_data or not val_data:
        raise ValueError("SSDLite requires positive batch size and nonempty train/val data")
    if int(args.get("imgsz", 320)) != 320:
        raise ValueError("SSDLite320 requires imgsz: 320")
    generator = torch.Generator().manual_seed(int(args.get("seed", 42)))
    loader_args = {"batch_size": batch_size, "num_workers": int(args.get("workers", 0)),
                   "collate_fn": collate_detection_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    # pycocotools must exist before the optional COCO download/model initialization.
    import pycocotools.cocoeval  # noqa: F401

    model = build_ssdlite(
        pretrained=bool(config.get("pretrained", True)),
        backbone_weights=config.get("backbone_weights"),
    ).to(device)
    model.score_thresh = 0.001
    model.detections_per_img = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.get("lr0", 0.001)),
                                 weight_decay=float(args.get("weight_decay", 0.0005)))
    epochs = int(args.get("epochs", 120))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(args.get("lr0", 0.001)) * float(args.get("lrf", 0.01)))
    use_amp = bool(args.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_map, best_epoch, history = -1.0, 0, []
    started = time.monotonic()
    weights_directory = run_directory / "weights"
    weights_directory.mkdir(parents=True, exist_ok=False)
    best_path = weights_directory / "best.pt"
    metadata = dict(
        metadata,
        architecture=ARCHITECTURE,
        background_index=0,
        backbone_weights=str(config.get("backbone_weights", "")),
    )
    max_hours = args.get("max_hours")
    for epoch in range(1, epochs + 1):
        model.train()
        if args.get("freeze_bn", True):
            # Pooled 1x1 features cannot estimate BN statistics in single-item microbatches.
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        loss_total = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = sum(model(images, targets).values())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite SSDLite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.get("clip_grad_norm", 10.0)))
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach())
        metrics = evaluate_validation(model, val_loader, device)
        map_value = float(metrics["map50_95"])
        if not np.isfinite(map_value) or map_value < 0:
            raise RuntimeError("Validation did not return a valid mAP50:95")
        row = {"epoch": epoch, "loss": loss_total / len(train_loader),
               "learning_rate": optimizer.param_groups[0]["lr"], "validation": metrics,
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        scheduler.step()
        trained_metadata = dict(metadata, trained=True)
        checkpoint = {"metadata": trained_metadata, "epoch": epoch,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(), "validation": metrics}
        torch.save(checkpoint, weights_directory / "last.pt")
        if map_value > best_map:
            best_map, best_epoch = map_value, epoch
            torch.save(checkpoint, best_path)
        write_json(run_directory / "history.json", history)
        print(f"epoch {epoch}/{epochs}: loss={row['loss']:.4f} val_mAP50:95={map_value:.4f}", flush=True)
        patience = int(args.get("patience", 30))
        if patience > 0 and epoch - best_epoch >= patience:
            break
        if max_hours is not None and (time.monotonic() - started) / 3600 >= float(max_hours):
            break
    write_json(run_directory / "training_summary.json", {
        **metadata, "trained": True, "best_checkpoint": str(best_path), "best_epoch": best_epoch,
        "best_validation_map50_95": best_map, "epochs_completed": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    })
    return best_path
