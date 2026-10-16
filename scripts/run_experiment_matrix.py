"""Automated Execution and Evaluation Orchestrator for Phase-2 Experiment Matrix.

Covers the 2x4 Matrix (SSDLite320-MobileNetV3 and YOLOv8n across 4 Augmentations:
none, geometric, photometric, combined), followed by Validation WBF Grid Search
and final evaluation on the held-out Test split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = [
    {
        "exp_id": "EXP-SSD-01",
        "model": "ssdlite",
        "variant": "none",
        "config_template": "configs/detection_training.yaml",
        "desc": "SSDLite320 Baseline (No Augmentation)",
    },
    {
        "exp_id": "EXP-SSD-02",
        "model": "ssdlite",
        "variant": "geometric",
        "config_template": "configs/detection_training.yaml",
        "desc": "SSDLite320 + Geometric Augmentation",
    },
    {
        "exp_id": "EXP-SSD-03",
        "model": "ssdlite",
        "variant": "photometric",
        "config_template": "configs/detection_training.yaml",
        "desc": "SSDLite320 + Photometric Augmentation",
    },
    {
        "exp_id": "EXP-SSD-04",
        "model": "ssdlite",
        "variant": "combined",
        "config_template": "configs/detection_training.yaml",
        "desc": "SSDLite320 + Combined (Geometric + Photometric + Copy-Paste)",
    },
    {
        "exp_id": "EXP-YOLO-01",
        "model": "yolov8n",
        "variant": "none",
        "config_template": "configs/detection_training_yolov8n.yaml",
        "desc": "YOLOv8n Baseline (No Augmentation)",
    },
    {
        "exp_id": "EXP-YOLO-02",
        "model": "yolov8n",
        "variant": "geometric",
        "config_template": "configs/detection_training_yolov8n.yaml",
        "desc": "YOLOv8n + Geometric Augmentation",
    },
    {
        "exp_id": "EXP-YOLO-03",
        "model": "yolov8n",
        "variant": "photometric",
        "config_template": "configs/detection_training_yolov8n.yaml",
        "desc": "YOLOv8n + Photometric Augmentation",
    },
    {
        "exp_id": "EXP-YOLO-04",
        "model": "yolov8n",
        "variant": "combined",
        "config_template": "configs/detection_training_yolov8n.yaml",
        "desc": "YOLOv8n + Combined (Geometric + Photometric + Copy-Paste)",
    },
]


def resolve_run_dir(base_project: Path, config_name: str, variant: str, seed: int) -> Path:
    run_name = config_name
    if variant != "none" or seed != 42:
        run_name += f"-{variant}-seed{seed}"
    return base_project / run_name


def run_experiment(
    exp: dict[str, Any],
    data_path: Path,
    epochs: int | None,
    batch_size: int | None,
    patience: int | None,
    device: str,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    from src.detection.train import train_detector

    exp_id = exp["exp_id"]
    model_type = exp["model"]
    variant = exp["variant"]
    template_path = PROJECT_ROOT / exp["config_template"]

    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    base_project = Path(config.get("project", "artifacts/detection"))
    if not base_project.is_absolute():
        base_project = PROJECT_ROOT / base_project

    config_name = config.get("name", f"waste-{model_type}")
    run_dir = resolve_run_dir(base_project, config_name, variant, seed)
    best_weight = run_dir / "weights" / "best.pt"

    print(f"\n{'='*70}")
    print(f"[{exp_id}] Starting {exp['desc']}")
    print(f"  Model: {model_type} | Variant: {variant} | Seed: {seed}")
    print(f"  Target Run Directory: {run_dir}")
    print(f"{'='*70}")

    if best_weight.is_file() and not overwrite:
        print(f"  -> Checkpoint already exists at: {best_weight}")
        print("  -> SKIPPING training (use --overwrite to force re-training)")
        return {
            "exp_id": exp_id,
            "status": "CACHED_EXISTING",
            "best_checkpoint": str(best_weight),
            "run_dir": str(run_dir),
            "variant": variant,
            "model": model_type,
        }

    # Clean existing run dir if overwriting
    if run_dir.exists() and overwrite:
        print(f"  -> Removing previous run dir: {run_dir}")
        shutil.rmtree(run_dir)

    # Prepare custom config YAML if overrides were given
    temp_config_path = PROJECT_ROOT / "configs" / f"temp_{exp_id.lower()}.yaml"
    train_args = config.get("train", {})
    if epochs is not None:
        train_args["epochs"] = epochs
    if batch_size is not None:
        train_args["batch"] = batch_size
    if patience is not None:
        train_args["patience"] = patience
    train_args["seed"] = seed
    train_args["device"] = device
    config["train"] = train_args
    config["augmentation_variant"] = variant

    temp_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    start_time = time.time()
    try:
        best_path = train_detector(
            data_path=data_path,
            config_path=temp_config_path,
            device_override=device,
            variant_override=variant,
            seed_override=seed,
        )
        elapsed = time.time() - start_time
        print(f"  -> Training COMPLETED in {elapsed:.1f}s")
        print(f"  -> Best model: {best_path}")
        return {
            "exp_id": exp_id,
            "status": "COMPLETED",
            "best_checkpoint": str(best_path),
            "run_dir": str(run_dir),
            "elapsed_seconds": round(elapsed, 2),
            "variant": variant,
            "model": model_type,
        }
    except Exception as error:
        elapsed = time.time() - start_time
        print(f"  -> ERROR in {exp_id}: {error}", file=sys.stderr)
        return {
            "exp_id": exp_id,
            "status": "FAILED",
            "error": str(error),
            "run_dir": str(run_dir),
            "elapsed_seconds": round(elapsed, 2),
            "variant": variant,
            "model": model_type,
        }
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()


def run_single_model_tuning_and_eval(
    name: str,
    backend: str,
    primary_ckpt: Path,
    data_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    from src.detection.evaluate import coco_metrics, collect_predictions, load_evaluation_split, operating_metrics
    from src.detection.factory import create_detector
    from src.detection.tune import tune

    print(f"\n--- Tuning & Evaluating Single Model: {name} ({backend}) ---")
    selection_file = output_dir / f"selection_{name}.json"
    fusion_cfg = PROJECT_ROOT / "configs" / "detection_fusion.yaml"

    if selection_file.exists():
        selection_file.unlink()

    tune(
        data_path=data_path,
        backend=backend,
        primary=primary_ckpt,
        secondary=None,
        config_path=fusion_cfg,
        output=selection_file,
        device=device,
    )
    print(f"  Saved validation selection: {selection_file}")

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    test_rows, test_fingerprint = load_evaluation_split(data_path, "test")

    options = dict(selection["detector_options"])
    options["device"] = device
    options["confidence_threshold"] = selection["ap_score_floor"]
    detector = create_detector(backend, str(primary_ckpt), **options)

    predictions, targets, benchmark = collect_predictions(detector, test_rows)
    metrics = coco_metrics(predictions, targets)
    operating = operating_metrics(
        predictions,
        targets,
        confidence=selection["operating_confidence"],
        iou_threshold=selection["matching_iou"],
    )

    eval_result = {
        "model_name": name,
        "backend": backend,
        "checkpoint": str(primary_ckpt),
        "test_mAP50": metrics["map50"],
        "test_mAP50_95": metrics["map50_95"],
        "precision": operating["micro"]["precision"],
        "recall": operating["micro"]["recall"],
        "f1": operating["micro"]["f1"],
        "latency_p50_ms": benchmark["latency_p50_ms"],
        "latency_p95_ms": benchmark["latency_p95_ms"],
        "fps": benchmark["fps"],
    }
    print(f"  Test Results: mAP50={eval_result['test_mAP50']:.4f}, mAP50-95={eval_result['test_mAP50_95']:.4f}, F1={eval_result['f1']:.4f}")
    return eval_result


def run_wbf_tuning_and_eval(
    ssd_ckpt: Path,
    yolo_ckpt: Path,
    data_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    from src.detection.evaluate import coco_metrics, collect_predictions, load_evaluation_split, operating_metrics
    from src.detection.factory import create_detector
    from src.detection.fusion import fuse_detections
    from src.detection.tune import tune

    print(f"\n--- Tuning & Evaluating WBF Ensemble (SSDLite + YOLOv8n) ---")
    selection_file = output_dir / "selection_wbf_ensemble.json"
    fusion_cfg = PROJECT_ROOT / "configs" / "detection_fusion.yaml"

    if selection_file.exists():
        selection_file.unlink()

    tune(
        data_path=data_path,
        backend="wbf",
        primary=ssd_ckpt,
        secondary=yolo_ckpt,
        config_path=fusion_cfg,
        output=selection_file,
        device=device,
    )
    print(f"  Saved WBF selection: {selection_file}")

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    test_rows, _ = load_evaluation_split(data_path, "test")

    options = dict(selection["detector_options"])
    options["device"] = device
    options["confidence_threshold"] = selection["ap_score_floor"]

    detector_ssd = create_detector("ssdlite", str(ssd_ckpt), **options)
    detector_yolo = create_detector("yolov8n", str(yolo_ckpt), **options)

    t0 = time.perf_counter()
    preds_ssd, targets, bm_ssd = collect_predictions(detector_ssd, test_rows)
    preds_yolo, _, bm_yolo = collect_predictions(detector_yolo, test_rows)

    fused_predictions = [
        fuse_detections(
            (p1, p2),
            weights=selection["detector_options"]["fusion_weights"],
            iou_threshold=selection["detector_options"]["fusion_iou_threshold"],
            confidence_threshold=selection["detector_options"]["fusion_input_threshold"],
            output_threshold=selection["ap_score_floor"],
        )
        for p1, p2 in zip(preds_ssd, preds_yolo, strict=True)
    ]
    t_fused = (time.perf_counter() - t0) / max(len(test_rows), 1)

    metrics = coco_metrics(fused_predictions, targets)
    operating = operating_metrics(
        fused_predictions,
        targets,
        confidence=selection["operating_confidence"],
        iou_threshold=selection["matching_iou"],
    )

    eval_result = {
        "model_name": "WBF Ensemble (Best SSDLite + Best YOLOv8n)",
        "backend": "wbf",
        "primary_checkpoint": str(ssd_ckpt),
        "secondary_checkpoint": str(yolo_ckpt),
        "fusion_weights": selection["detector_options"]["fusion_weights"],
        "fusion_iou": selection["detector_options"]["fusion_iou_threshold"],
        "test_mAP50": metrics["map50"],
        "test_mAP50_95": metrics["map50_95"],
        "precision": operating["micro"]["precision"],
        "recall": operating["micro"]["recall"],
        "f1": operating["micro"]["f1"],
        "latency_p50_ms": (bm_ssd["latency_p50_ms"] + bm_yolo["latency_p50_ms"]),
        "latency_p95_ms": (bm_ssd["latency_p95_ms"] + bm_yolo["latency_p95_ms"]),
        "fps": 1.0 / t_fused if t_fused > 0 else 0.0,
    }
    print(f"  WBF Test Results: mAP50={eval_result['test_mAP50']:.4f}, mAP50-95={eval_result['test_mAP50_95']:.4f}, F1={eval_result['f1']:.4f}")
    return eval_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 2x4 Detection Experiment Matrix and WBF Evaluation")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "configs" / "detection_dataset.yaml")
    parser.add_argument("--models", choices=("all", "ssdlite", "yolo", "yolov8n"), default="all")
    parser.add_argument("--variants", choices=("all", "none", "geometric", "photometric", "combined"), default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch budget (e.g. 120)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing runs")
    parser.add_argument("--skip-eval", action="store_true", help="Skip WBF and final test evaluation")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "artifacts" / "detection" / "matrix_summary.json")
    args = parser.parse_args()

    # Filter experiments
    selected_exps = []
    for exp in EXPERIMENTS:
        if args.models != "all":
            target_model = "yolov8n" if args.models in ("yolo", "yolov8n") else args.models
            if exp["model"] != target_model:
                continue
        if args.variants != "all" and exp["variant"] != args.variants:
            continue
        selected_exps.append(exp)

    print("=" * 70)
    print("PHASE 2: 2x4 OBJECT DETECTION EXPERIMENT MATRIX")
    print(f"Total experiments selected: {len(selected_exps)}")
    print(f"Device: {args.device} | Base Seed: {args.seed}")
    print("=" * 70)

    results = []
    best_ckpts: dict[str, dict[str, Any]] = {"ssdlite": {}, "yolov8n": {}}

    for exp in selected_exps:
        res = run_experiment(
            exp=exp,
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        results.append(res)
        if res.get("status") in ("COMPLETED", "CACHED_EXISTING"):
            model = res["model"]
            variant = res["variant"]
            best_ckpts[model][variant] = Path(res["best_checkpoint"])

    # Summary table
    print("\n" + "=" * 70)
    print("EXPERIMENT MATRIX TRAINING SUMMARY")
    print("=" * 70)
    for r in results:
        status_symbol = "OK" if r.get("status") in ("COMPLETED", "CACHED_EXISTING") else "FAIL"
        print(f"[{status_symbol}] {r['exp_id']} ({r['model']} - {r['variant']}): {r['status']}")
        if "best_checkpoint" in r:
            print(f"    Checkpoint: {r['best_checkpoint']}")

    # Final Tuning & Test Evaluation
    eval_reports = []
    out_dir = PROJECT_ROOT / "artifacts" / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        print("\n" + "=" * 70)
        print("PROCEEDING TO VALIDATION TUNING & HELD-OUT TEST EVALUATION")
        print("=" * 70)

        # Evaluate best SSDLite (pick 'combined' or 'none' if available)
        ssd_candidate = (
            best_ckpts["ssdlite"].get("combined")
            or best_ckpts["ssdlite"].get("geometric")
            or best_ckpts["ssdlite"].get("none")
        )
        yolo_candidate = (
            best_ckpts["yolov8n"].get("combined")
            or best_ckpts["yolov8n"].get("geometric")
            or best_ckpts["yolov8n"].get("none")
        )

        if ssd_candidate and ssd_candidate.is_file():
            try:
                rep_ssd = run_single_model_tuning_and_eval(
                    name="ssdlite320_best",
                    backend="ssdlite",
                    primary_ckpt=ssd_candidate,
                    data_path=args.data,
                    output_dir=out_dir,
                    device=args.device,
                )
                eval_reports.append(rep_ssd)
            except Exception as e:
                print(f"Error evaluating SSDLite: {e}", file=sys.stderr)

        if yolo_candidate and yolo_candidate.is_file():
            try:
                rep_yolo = run_single_model_tuning_and_eval(
                    name="yolov8n_best",
                    backend="yolov8n",
                    primary_ckpt=yolo_candidate,
                    data_path=args.data,
                    output_dir=out_dir,
                    device=args.device,
                )
                eval_reports.append(rep_yolo)
            except Exception as e:
                print(f"Error evaluating YOLOv8n: {e}", file=sys.stderr)

        if ssd_candidate and yolo_candidate and ssd_candidate.is_file() and yolo_candidate.is_file():
            try:
                rep_wbf = run_wbf_tuning_and_eval(
                    ssd_ckpt=ssd_candidate,
                    yolo_ckpt=yolo_candidate,
                    data_path=args.data,
                    output_dir=out_dir,
                    device=args.device,
                )
                eval_reports.append(rep_wbf)
            except Exception as e:
                print(f"Error evaluating WBF Ensemble: {e}", file=sys.stderr)

    # Save full matrix summary
    final_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_experiments": len(selected_exps),
        "results": results,
        "evaluations": eval_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFinal summary written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
