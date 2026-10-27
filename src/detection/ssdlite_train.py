# Mô tả: Huấn luyện SSDLite320-MobileNetV3 trên tập dữ liệu định dạng YOLO (tự động cộng +1 vào class ID khi nạp target để dành 0 cho Background), đánh giá Validation sau mỗi epoch và lưu best.pt / last.pt.
"""Train SSDLite on reviewed bounding boxes; use validation only for model selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.detection.augmentation import apply_augmentation
from src.detection.copy_paste import DonorBank
from src.detection.dataset import _label_path
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite import ARCHITECTURE, build_ssdlite, resolve_device, tensor_detections
from src.detection.training_common import seed_everything, split_images, write_json
from src.detection.types import BoundingBox, Detection


SSD_FILL_RGB = (124, 116, 104)  # ImageNet mean, as in torchvision's SSD reference recipe


def dataset_class_names(data_path: Path) -> tuple[str, ...]:
    """Class names declared by the dataset YAML (10-class or custom, e.g. 1-class agnostic)."""
    from src.detection.dataset import _ordered_names, load_detection_dataset_yaml

    payload, _ = load_detection_dataset_yaml(Path(data_path), allow_custom_names=True)
    return _ordered_names(payload["names"])


def build_ssd_augmentation(*, photometric: bool = True, zoom_out_max: float = 2.5):
    """SSD recipe on tv_tensors: boxes are transformed together with pixels."""
    from torchvision.transforms import v2

    steps = []
    if photometric:
        steps.append(v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.7))
    steps += [
        v2.RandomZoomOut(fill=SSD_FILL_RGB, side_range=(1.0, zoom_out_max), p=0.5),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.SanitizeBoundingBoxes(min_size=1.0),
    ]
    return v2.Compose(steps)


class YoloBoxDataset:
    """Consume reviewed bounding boxes and apply configured augmentation strategy on train split."""

    def __init__(
        self,
        data_path: Path,
        split: str,
        augmentation_strategy: str = "none",
        donor_bank: DonorBank | None = None,
    ):
        self.images, self.root = split_images(data_path, split, allow_custom_names=True)
        self.class_names = dataset_class_names(data_path)
        self.split = split
        self.strategy = augmentation_strategy.lower().strip()
        self.donor_bank = donor_bank
        self.telemetry_history: list[dict[str, Any]] = []
        self._ssd_augment = build_ssd_augmentation() if self.strategy == "ssd" else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from src.detection.augmentation import apply_augmentation_with_telemetry

        image_path = self.images[index]
        with Image.open(image_path) as opened:
            # Labels follow the displayed (EXIF-corrected) frame, as in Ultralytics/inference.
            transposed = ImageOps.exif_transpose(opened)
            # Cap max dimension to 640 to prevent massive memory spikes on 4000x3000 images during SSD augmentation
            max_dim = max(transposed.width, transposed.height)
            if max_dim > 640:
                scale = 640.0 / max_dim
                new_w = max(1, round(transposed.width * scale))
                new_h = max(1, round(transposed.height * scale))
                image = transposed.resize((new_w, new_h), Image.Resampling.BILINEAR).convert("RGB")
            else:
                image = transposed.convert("RGB")
            del transposed
        raw_boxes, raw_classes = [], []
        for line in _label_path(image_path, self.root).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_value, cx, cy, bw, bh = (float(value) for value in line.split())
            if not 0 <= int(class_value) < len(self.class_names):
                raise ValueError(f"{image_path}: class {int(class_value)} outside dataset names")
            raw_boxes.append([cx, cy, bw, bh])
            raw_classes.append(int(class_value))

        # Apply data augmentation if in training split
        if self.split == "train" and self.strategy not in ("none", "ssd"):
            t_img, t_boxes, t_classes, telem = apply_augmentation_with_telemetry(
                image,
                raw_boxes,
                raw_classes,
                strategy=self.strategy,
                donor_bank=self.donor_bank,
            )
            if len(self.telemetry_history) < 100:
                self.telemetry_history.append(telem)
        else:
            t_img = np.array(image)
            t_boxes = raw_boxes
            t_classes = raw_classes

        # Convert YOLO normalized cx, cy, bw, bh to pixel [x1, y1, x2, y2]
        h, w = t_img.shape[:2]
        boxes, labels = [], []
        for b, c in zip(t_boxes, t_classes):
            cx, cy, bw, bh = b
            x1 = max(0.0, (cx - bw / 2.0) * w)
            y1 = max(0.0, (cy - bh / 2.0) * h)
            x2 = min(float(w), (cx + bw / 2.0) * w)
            y2 = min(float(h), (cy + bh / 2.0) * h)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c) + 1)  # zero is reserved for background in SSDLite

        image_uint8 = torch.from_numpy(np.ascontiguousarray(t_img)).permute(2, 0, 1)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        if self.split == "train" and self._ssd_augment is not None and len(boxes) > 0:
            from torchvision import tv_tensors

            sample = self._ssd_augment({
                "image": tv_tensors.Image(image_uint8),
                "boxes": tv_tensors.BoundingBoxes(box_tensor, format="XYXY", canvas_size=(h, w)),
                "labels": label_tensor,
            })
            image_uint8 = sample["image"].as_subclass(torch.Tensor)
            box_tensor = sample["boxes"].as_subclass(torch.Tensor).reshape(-1, 4)
            label_tensor = sample["labels"]
        img_tensor = image_uint8.float() / 255.0
        return img_tensor, {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros(len(box_tensor), dtype=torch.int64),
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
                  metadata: dict[str, Any], resume: bool = False) -> Path:
    import json
    import torch
    from torch.utils.data import DataLoader
    from src.detection.preflight_guard import enforce_preflight_guard

    args = config["train"]
    device = resolve_device(str(args.get("device", "auto")))

    # Mandatory Preflight hardware safety guard (Zero bypass allowed)
    min_free_ram = float(args.get("min_free_ram_mb", 1200.0))
    min_free_vram = float(args.get("min_free_vram_mb", 500.0))
    enforce_preflight_guard(min_free_ram_mb=min_free_ram, min_free_vram_mb=min_free_vram, device=device.type)

    seed_everything(int(args.get("seed", 42)), bool(args.get("deterministic", True)))
    aug_variant = str(args.get("augmentation_variant", config.get("augmentation_variant", metadata.get("augmentation_variant", "none"))))
    donor_bank = None
    if aug_variant.lower().strip() in ("combined", "copy_paste"):
        donor_bank = DonorBank.from_dataset_root(".")
    train_data = YoloBoxDataset(data_path, "train", augmentation_strategy=aug_variant, donor_bank=donor_bank)
    val_data = YoloBoxDataset(data_path, "val", augmentation_strategy="none")
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weights_directory = run_directory / "weights"
    weights_directory.mkdir(parents=True, exist_ok=True)
    best_path = weights_directory / "best.pt"
    last_path = weights_directory / "last.pt"

    start_epoch = 1
    best_map, best_epoch, history = -1.0, 0, []
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_file = run_directory / "history.json"
        if history_file.is_file():
            try:
                history = json.loads(history_file.read_text(encoding="utf-8"))
                for row in history:
                    v_map = float(row.get("validation", {}).get("map50_95", -1.0))
                    if v_map > best_map:
                        best_map = v_map
                        best_epoch = int(row.get("epoch", 0))
            except Exception:
                pass
        print(f"Resuming SSDLite from epoch {start_epoch} (best_epoch={best_epoch}, best_map={best_map:.4f})", flush=True)

    started = time.monotonic()
    metadata = dict(
        metadata,
        architecture=ARCHITECTURE,
        background_index=0,
        backbone_weights=str(config.get("backbone_weights", "")),
    )
    max_hours = args.get("max_hours")
    for epoch in range(start_epoch, epochs + 1):
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
            with torch.amp.autocast("cuda", enabled=use_amp):
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
        if hasattr(train_data, "telemetry_history") and train_data.telemetry_history:
            write_json(run_directory / "augmentation_telemetry.json", train_data.telemetry_history)
        print(f"epoch {epoch}/{epochs}: loss={row['loss']:.4f} val_mAP50:95={map_value:.4f}", flush=True)
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
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
