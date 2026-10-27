# Mô tả: Trình chạy huấn luyện và đánh giá các mô hình phát hiện trên bộ dữ liệu thực tế detection_real_v2.
"""Phase-3 (data v2, real-only) detector training/evaluation runner.

Sequential GPU usage: run ONE command at a time.

  python scripts/train_detection_v2.py yolo    --run E1_yolov8n_10cls --data configs/detection_dataset_real_v2.yaml
  python scripts/train_detection_v2.py yolo    --run E3_yolov8n_agnostic --data configs/detection_dataset_real_v2_agnostic.yaml --agnostic
  python scripts/train_detection_v2.py ssdlite --run E2_ssdlite320_10cls --data configs/detection_dataset_real_v2.yaml
  python scripts/train_detection_v2.py post    --run E1_yolov8n_10cls

Training never reads the test split or data/holdout. ``post`` evaluates val+test,
calibrates per-class thresholds on VAL only and exports raw QA predictions (conf>=0.001).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.schema import DETECTION_CLASS_NAMES  # noqa: E402

OUT_ROOT = ROOT / "artifacts" / "detection_v2"
QA_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
PRED_DIR = ROOT / "reports" / "model_v2"
SEED = 42

YOLO_HYP = dict(
    epochs=80, patience=25, imgsz=640, batch=16, workers=0, optimizer="AdamW", lr0=0.001, lrf=0.01,
    weight_decay=0.0005, warmup_epochs=3.0, cos_lr=True, mosaic=1.0, close_mosaic=10, scale=0.5,
    translate=0.1, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0,
    mixup=0.1, cutmix=0.0, copy_paste=0.0, amp=True, seed=SEED, deterministic=True, plots=True,
    max_det=300,
)


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_dataset(data_yaml: Path) -> dict[str, Any]:
    """Resolve a v2 dataset YAML (root-relative lists) without class-count assumptions."""
    payload = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(str(payload.get("path", ".")))
    root = (root if root.is_absolute() else Path(data_yaml).resolve().parent / root).resolve()
    names = payload.get("names")
    if names is None and isinstance(payload.get("taxonomy"), dict):
        names = payload["taxonomy"].get("names")
    if names is not None:
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    else:
        names = []
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        value = payload.get(split)
        if not value and isinstance(payload.get("paths"), dict):
            value = payload["paths"].get(f"{split}_split")
        if not value:
            continue
        list_file = (root / value).resolve()
        if not list_file.is_file():
            splits[split] = {"list_file": str(list_file), "sha256": None, "images": []}
            continue
        lines = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        images = [((list_file.parent / ln) if ln.startswith(".") else (root / ln)).resolve() for ln in lines]
        splits[split] = {"list_file": str(list_file), "sha256": sha256_file(list_file), "images": images}
    return {"yaml": str(Path(data_yaml).resolve()), "root": root, "names": names, "payload": payload, "splits": splits}


def label_path(image: Path, root: Path) -> Path:
    parts = list(image.relative_to(root).parts)
    parts[parts.index("images")] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def environment() -> dict[str, Any]:
    import importlib.metadata as md

    import torch

    pkgs = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "pycocotools", "albumentations"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def base_metadata(run: str, backend: str, ds: dict[str, Any], agnostic: bool, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run, "backend": backend, "agnostic": agnostic, "class_names": ds["names"],
        "dataset_yaml": ds["yaml"], "dataset_root": str(ds["root"]),
        "split_sha256": {k: v["sha256"] for k, v in ds["splits"].items()},
        "split_sha256_declared": ds["payload"].get("metadata", {}).get("split_sha256"),
        "split_images": {k: len(v["images"]) for k, v in ds["splits"].items()},
        "seed": SEED, "config": config, "selection_split": "val", "selection_metric": "map50_95",
        "test_used_for_training_or_selection": False, "qa_images_used_for_training": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
    }


def verify_training_dataset_safety(data_yaml: Path, ds: dict[str, Any]) -> None:
    """
    Mandatory safety verification before training (both fresh and resume):
    1. Cross-checks authoritative test and holdout splits to ensure zero data leakage.
    2. Binds dataset verification to dataset version / hash and enforces quality gate verdict.
    """
    from src.detection.export_train_manifest import load_authoritative_splits
    test_set, val_set, train_set, holdout_set = load_authoritative_splits()

    train_images = ds["splits"].get("train", {}).get("images", [])
    for img_path in train_images:
        fn = Path(img_path).name
        if fn in test_set:
            raise SystemExit(f"[TRAINING BLOCKED] Split contamination: {fn} from authoritative TEST split is present in train dataset!")
        if fn in holdout_set:
            raise SystemExit(f"[TRAINING BLOCKED] Split contamination: {fn} from official HOLDOUT manifest is present in train dataset!")

    yaml_str = str(data_yaml).lower()
    payload = ds.get("payload", {}) if isinstance(ds, dict) else {}
    version = str(payload.get("metadata", {}).get("version", "")).lower() if isinstance(payload, dict) else ""
    is_candidate = "candidate" in yaml_str or "pilot" in yaml_str or "v3" in yaml_str or "v3" in version or "candidate" in version

    if is_candidate:
        gate_report_path = ROOT / "reports" / "data_v2" / "pilot_quality_gates_report.json"
        if not gate_report_path.is_file():
            raise SystemExit("[TRAINING BLOCKED] Candidate dataset requires verified pilot_quality_gates_report.json, but none was found.")
        report_data = json.loads(gate_report_path.read_text(encoding="utf-8"))
        if report_data.get("verdict") != "TRAINING_ELIGIBLE":
            raise SystemExit(
                f"[TRAINING BLOCKED] Candidate dataset quality gate status is '{report_data.get('verdict')}'. "
                f"Training on unapproved candidate data is strictly forbidden."
            )
        # Verify cryptographic split hash consistency
        expected_train_sha = ds["splits"].get("train", {}).get("sha256")
        reported_sha = report_data.get("gates", {}).get("split_safety_gate", {}).get("train_split_sha256")
        if reported_sha and expected_train_sha and reported_sha != expected_train_sha:
            raise SystemExit(
                f"[TRAINING BLOCKED] Dataset train split hash mismatch! "
                f"Gate report was computed for {reported_sha[:8]}..., but active dataset has {expected_train_sha[:8]}..."
            )


# --------------------------------------------------------------------------- YOLO
def train_yolo(run: str, data_yaml: Path | None, agnostic: bool, overrides: dict[str, Any], resume: bool = False) -> None:
    from ultralytics import YOLO
    from src.detection.preflight_guard import enforce_preflight_guard

    # Mandatory Preflight Hardware Guard (zero bypass allowed)
    enforce_preflight_guard(
        min_free_ram_mb=float(overrides.get("min_free_ram_mb", 1200.0)),
        min_free_vram_mb=float(overrides.get("min_free_vram_mb", 500.0)),
        device="cuda" if str(overrides.get("device", "0")) != "cpu" else "cpu"
    )

    run_dir = OUT_ROOT / run
    started = time.monotonic()
    if resume:
        if not run_dir.exists():
            raise SystemExit(f"Cannot resume, missing run dir: {run_dir}")
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            raise SystemExit(f"Cannot resume, missing {last_pt}")
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        # Verify dataset safety in resume mode as well
        resume_yaml = Path(data_yaml or metadata.get("dataset_yaml"))
        ds = load_dataset(resume_yaml)
        verify_training_dataset_safety(resume_yaml, ds)
        expected = metadata["class_names"]
        agnostic = metadata["agnostic"]
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        if data_yaml is None:
            raise SystemExit("Missing --data argument for fresh training run")
        ds = load_dataset(data_yaml)
        verify_training_dataset_safety(data_yaml, ds)
        expected = ["waste"] if agnostic else list(DETECTION_CLASS_NAMES)
        if ds["names"] != expected:
            raise SystemExit(f"Unexpected class names {ds['names']}")
        if run_dir.exists():
            raise SystemExit(f"Run exists: {run_dir}")
        run_dir.mkdir(parents=True)
        # Absolute train/val lists only (test sealed during training).
        lists = {}
        for split in ("train", "val"):
            dest = run_dir / f"{split}_abs.txt"
            dest.write_text("\n".join(p.as_posix() for p in ds["splits"][split]["images"]) + "\n", encoding="utf-8")
            lists[split] = dest.as_posix()
        resolved = {"path": ds["root"].as_posix(), "train": lists["train"], "val": lists["val"],
                    "nc": len(expected), "names": {i: n for i, n in enumerate(expected)}}
        resolved_yaml = run_dir / "dataset_train_val.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        hyp = dict(YOLO_HYP, **overrides)
        metadata = base_metadata(run, "yolov8n", ds, agnostic, {"model": "yolov8n.pt", **hyp})
        write_json(run_dir / "run_metadata.json", metadata)
        model = YOLO(str(ROOT / "yolov8n.pt"), task="detect")
        model.train(data=str(resolved_yaml), project=str(OUT_ROOT), name=run, exist_ok=True, device=0,
                    val=True, **hyp)
    best = Path(model.trainer.best)
    # Inject runtime metadata so src.detection.yolo.WasteDetector accepts the 10-class model.
    import torch

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    meta = {"backend": "yolov8n", "trained": True, "class_names": expected, "agnostic": agnostic,
            "run_name": run, "split_sha256": metadata["split_sha256"]}
    for key in ("model", "ema"):
        if ckpt.get(key) is not None:
            ckpt[key].phase2_metadata = meta
            ckpt[key].names = {i: n for i, n in enumerate(expected)}
    ckpt["phase2_metadata"] = meta
    torch.save(ckpt, best)
    epochs_completed = int(model.trainer.epoch) + 1
    best_epoch = None
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        import csv

        rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
        key = next(k for k in rows[0] if "mAP50-95" in k)
        best_row = max(rows, key=lambda r: float(r[key]))
        best_epoch = int(float(best_row["epoch"]))
    metadata.update(trained=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, epochs_completed=epochs_completed,
                    best_epoch=best_epoch, best_checkpoint=str(best), checkpoint_sha256=sha256_file(best))
    write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("best_checkpoint", "checkpoint_sha256", "best_epoch",
                                               "epochs_completed", "elapsed_seconds")}, indent=1))


# --------------------------------------------------------------------------- SSDLite
def train_ssd(run: str, data_yaml: Path | None, agnostic: bool, overrides: dict[str, Any], resume: bool = False) -> None:
    from src.detection.ssdlite_train import train_ssdlite
    from src.detection.preflight_guard import enforce_preflight_guard

    # Mandatory Preflight Hardware Guard (zero bypass allowed)
    enforce_preflight_guard(
        min_free_ram_mb=float(overrides.get("min_free_ram_mb", 1200.0)),
        min_free_vram_mb=float(overrides.get("min_free_vram_mb", 500.0)),
        device="cuda" if str(overrides.get("device", "cuda")) != "cpu" else "cpu"
    )

    run_dir = OUT_ROOT / run
    started = time.monotonic()
    if resume:
        if not run_dir.exists():
            raise SystemExit(f"Cannot resume, missing run dir: {run_dir}")
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            raise SystemExit(f"Cannot resume, missing {last_pt}")
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        data_yaml = Path(data_yaml or metadata["dataset_yaml"])
        ds = load_dataset(data_yaml)
        verify_training_dataset_safety(data_yaml, ds)
        config = metadata["config"]
        config["train"].update(overrides)
    else:
        if data_yaml is None:
            raise SystemExit("Missing --data argument for fresh training run")
        ds = load_dataset(data_yaml)
        verify_training_dataset_safety(data_yaml, ds)
        if run_dir.exists():
            raise SystemExit(f"Run exists: {run_dir}")
        run_dir.mkdir(parents=True)
        train_args = dict(epochs=100, patience=25, imgsz=320, batch=16, workers=0, optimizer="AdamW", lr0=0.001,
                          lrf=0.01, weight_decay=0.0005, warmup_epochs=3, clip_grad_norm=10.0, freeze_bn=False,
                          amp=True, seed=SEED, deterministic=True, device="cuda", augmentation_variant="ssd")
        train_args.update(overrides)
        config = {"backend": "ssdlite", "model": "ssdlite320_mobilenet_v3_large", "pretrained": True,
                  "backbone_weights": None, "train": train_args, "augmentation_variant": "ssd"}
        metadata = base_metadata(run, "ssdlite", ds, agnostic, config)
        write_json(run_dir / "run_metadata.json", metadata)

    best = train_ssdlite(Path(data_yaml), config, run_dir, dict(metadata, backend="ssdlite",
                                                                 class_names=ds["names"]), resume=resume)
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    metadata.update(trained=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, epochs_completed=summary["epochs_completed"],
                    best_epoch=summary["best_epoch"], best_val_map50_95=summary["best_validation_map50_95"],
                    best_checkpoint=str(best), checkpoint_sha256=sha256_file(best))
    write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("best_checkpoint", "checkpoint_sha256", "best_epoch",
                                               "epochs_completed", "elapsed_seconds")}, indent=1))


# --------------------------------------------------------------------------- evaluation
def load_image(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def read_targets(image: Path, root: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    rows = []
    for line in label_path(image, root).read_text(encoding="utf-8").splitlines():
        if line.strip():
            c, x, y, w, h = map(float, line.split())
            rows.append((int(c), [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]))
    return rows


class Predictor:
    """Raw (conf>=0.001) predictions in EXIF-transposed original-pixel xyxy, 0-based class ids."""

    def __init__(self, meta: dict[str, Any], device: str = "cuda") -> None:
        self.backend = meta["backend"]
        self.ckpt = Path(meta["best_checkpoint"])
        self.imgsz = int(meta["config"].get("imgsz", meta["config"].get("train", {}).get("imgsz", 320)))
        self.device = device
        if self.backend == "yolov8n":
            from ultralytics import YOLO

            self.model = YOLO(str(self.ckpt), task="detect")
        else:
            import torch

            from src.detection.ssdlite import build_ssdlite

            checkpoint = torch.load(self.ckpt, map_location="cpu", weights_only=False)
            names = checkpoint["metadata"].get("class_names", list(DETECTION_CLASS_NAMES))
            model = build_ssdlite(pretrained=False, num_classes=len(names))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.score_thresh, model.nms_thresh, model.detections_per_img = 0.001, 0.55, 300
            self.model = model.to(device).eval()

    def __call__(self, pil) -> list[dict[str, Any]]:
        import numpy as np

        if self.backend == "yolov8n":
            r = self.model.predict(source=np.asarray(pil)[:, :, ::-1].copy(), imgsz=self.imgsz, conf=0.001,
                                   iou=0.7, max_det=300, device=0 if self.device == "cuda" else "cpu",
                                   verbose=False)[0]
            return [{"class_id": int(c), "confidence": float(s), "bbox_xyxy": [float(v) for v in b]}
                    for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        tensor = pil_to_tensor(pil).to(self.device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            out = self.model([tensor])[0]
        w, h = pil.size
        rows = []
        for b, s, lab in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
            x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])
            if x2 > x1 and y2 > y1:
                rows.append({"class_id": int(lab) - 1, "confidence": float(s), "bbox_xyxy": [x1, y1, x2, y2]})
        return rows


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def per_class_f1_thresholds(preds_per_image, targets_per_image, names, iou_thr=0.5) -> dict[str, Any]:
    """Greedy same-class matching at IoU>=0.5; choose the confidence that maximizes F1 per class."""
    out = {}
    for ci, name in enumerate(names):
        scored = []  # (confidence, is_tp)
        n_gt = 0
        for preds, gts in zip(preds_per_image, targets_per_image):
            g = [b for c, b in gts if c == ci]
            n_gt += len(g)
            used = set()
            for p in sorted((p for p in preds if p["class_id"] == ci), key=lambda p: -p["confidence"]):
                best, bi = 0.0, -1
                for j, gb in enumerate(g):
                    if j not in used:
                        v = _iou(p["bbox_xyxy"], gb)
                        if v > best:
                            best, bi = v, j
                if best >= iou_thr:
                    used.add(bi)
                    scored.append((p["confidence"], 1))
                else:
                    scored.append((p["confidence"], 0))
        scored.sort(key=lambda t: -t[0])
        best = {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        tp = fp = 0
        for k, (conf, hit) in enumerate(scored):
            tp += hit
            fp += 1 - hit
            if k + 1 < len(scored) and scored[k + 1][0] == conf:
                continue
            prec, rec = tp / (tp + fp), (tp / n_gt if n_gt else 0.0)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best["f1"]:
                best = {"threshold": round(float(conf), 4), "f1": f1, "precision": prec, "recall": rec}
        best["val_instances"] = n_gt
        out[name] = best
    return out


def evaluate_split(predictor: Predictor, ds, split: str, names) -> tuple[dict[str, Any], list, list]:
    from src.detection.evaluate import coco_metrics
    from src.detection.types import BoundingBox, Detection, DetectionResult

    preds_all, gts_all, results, targets = [], [], [], []
    for image in ds["splits"][split]["images"]:
        pil = load_image(image)
        w, h = pil.size
        preds = predictor(pil)
        gts = read_targets(image, ds["root"], w, h)
        preds_all.append(preds)
        gts_all.append(gts)
        dets = []
        for p in preds:
            box = BoundingBox(*p["bbox_xyxy"])
            if box.width > 0 and box.height > 0:
                dets.append(Detection(p["class_id"], names[p["class_id"]], min(1.0, p["confidence"]), box))
        results.append(DetectionResult(tuple(sorted(dets, key=lambda d: -d.confidence)), w, h))
        targets.append([Detection(c, names[c], 1.0, BoundingBox(*b)) for c, b in gts if b[2] > b[0] and b[3] > b[1]])
    metrics = coco_metrics(results, targets)
    if len(names) != len(DETECTION_CLASS_NAMES):  # agnostic: coco_metrics labels category ids by the 10-class list
        metrics["ap_by_class"] = {names[0]: metrics["ap_by_class"][DETECTION_CLASS_NAMES[0]]}
    metrics["images"] = len(results)
    return metrics, preds_all, gts_all


def post(run: str, device: str) -> None:
    run_dir = OUT_ROOT / run
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    ds = load_dataset(Path(meta["dataset_yaml"]))
    names = meta["class_names"]
    if sha256_file(Path(meta["best_checkpoint"])) != meta["checkpoint_sha256"]:
        raise SystemExit("Checkpoint changed since training")
    predictor = Predictor(meta, device=device)
    report: dict[str, Any] = {"run": run, "checkpoint_sha256": meta["checkpoint_sha256"], "imgsz": predictor.imgsz,
                              "protocol": "pycocotools COCOeval bbox on EXIF-transposed images, conf>=0.001"}
    val_metrics, val_preds, val_gts = evaluate_split(predictor, ds, "val", names)
    report["val"] = val_metrics
    thresholds = per_class_f1_thresholds(val_preds, val_gts, names)
    write_json(run_dir / "per_class_thresholds.json", {
        "per_class_thresholds": {n: v["threshold"] for n, v in thresholds.items() if v["threshold"] is not None},
        "calibration": {"split": "val", "criterion": "max F1, same-class greedy matching IoU>=0.5",
                        "details": thresholds, "checkpoint_sha256": meta["checkpoint_sha256"]},
    })
    test_metrics, _, _ = evaluate_split(predictor, ds, "test", names)
    report["test"] = test_metrics
    if predictor.backend == "yolov8n":  # Ultralytics' own val numbers for cross-checking
        report["ultralytics_val"] = ultralytics_metrics(predictor, ds, run_dir, names)
    write_json(run_dir / "eval_val_test.json", report)
    # QA export: 27 images, raw predictions only (no thresholds/selection from these images).
    manifest = json.loads(QA_MANIFEST.read_text(encoding="utf-8"))
    preds = {}
    for item in manifest["files"]:
        preds[item["filename"]] = predictor(load_image(Path(item["absolute_path"])))
    payload = {"predictions": preds, "confidence_threshold": 0.001, "backend": predictor.backend,
               "checkpoint_sha256": meta["checkpoint_sha256"], "run": run, "imgsz": predictor.imgsz,
               "agnostic": bool(meta.get("agnostic")), "class_names": names,
               "coordinates": "pixel xyxy in original image after EXIF transpose"}
    write_json(PRED_DIR / f"predictions_{run}.json", payload)
    print(json.dumps({"val": {k: val_metrics[k] for k in ("map50", "map50_95")},
                      "test": {k: test_metrics[k] for k in ("map50", "map50_95")}}, indent=1))


def ultralytics_metrics(predictor: Predictor, ds, run_dir: Path, names) -> dict[str, Any]:
    out = {}
    for split in ("val", "test"):
        lst = run_dir / f"eval_{split}_abs.txt"
        lst.write_text("\n".join(p.as_posix() for p in ds["splits"][split]["images"]) + "\n", encoding="utf-8")
        yml = run_dir / f"eval_{split}.yaml"
        yml.write_text(yaml.safe_dump({"path": ds["root"].as_posix(), "train": lst.as_posix(), "val": lst.as_posix(),
                                       "nc": len(names), "names": {i: n for i, n in enumerate(names)}}), encoding="utf-8")
        m = predictor.model.val(data=str(yml), split="val", imgsz=predictor.imgsz, conf=0.001, iou=0.7, max_det=300,
                                device=0, plots=False, project=str(run_dir), name=f"ultra_{split}", exist_ok=True,
                                verbose=False)
        per_class = {}
        for idx, ci in enumerate(m.box.ap_class_index.tolist()):
            per_class[names[int(ci)]] = {"ap50": float(m.box.ap50[idx]), "ap50_95": float(m.box.ap[idx])}
        out[split] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("yolo", "ssdlite", "post"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--agnostic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", nargs="*", default=[], help="hyper-parameter overrides key=value (YAML-parsed)")
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    if args.command == "yolo":
        train_yolo(args.run, args.data, args.agnostic, overrides, resume=args.resume)
    elif args.command == "ssdlite":
        train_ssd(args.run, args.data, args.agnostic, overrides, resume=args.resume)
    else:
        post(args.run, args.device)


if __name__ == "__main__":
    main()
