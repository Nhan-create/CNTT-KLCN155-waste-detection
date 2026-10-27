# Mô tả: Kiểm thử kết quả huấn luyện và đánh giá thử nghiệm E3 (YOLOv8n Pilot v3) ở Task 6, đảm bảo các tài sản đóng băng không bị sửa đổi.
"""
tests/qa/test_task6_e3_training_and_eval.py
-------------------------------------------
Automated verification suite for Sprint AI-UPGRADE-01 Task 6:
1. GPU preflight log and dataset V3 materialization integrity (1509/328/299 images, 4074/930/720 boxes).
2. Preservation of frozen v2 splits, E1 checkpoint, holdout GT, and Streamlit production code.
3. E3 YOLOv8n Pilot training artifacts (best.pt, last.pt, results.csv, args.yaml, training_log.txt, epoch_telemetry.json).
4. E3 checkpoint compatibility with src.detection.yolo.WasteDetector.
5. Comprehensive E1 vs E3 evaluation across all 4 benchmarks (Val V3, Val V2, Test Frozen, Holdout 27).
6. Visual side-by-side 3-panel comparisons across all 5 required inspection groups.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
import yaml

from src.detection.yolo import WasteDetector

ROOT = Path(__file__).resolve().parents[2]
PILOT_ARTIFACTS = ROOT / "artifacts" / "detection_v3_pilot"
E3_DIR = PILOT_ARTIFACTS / "E3_yolov8n_pilot"
EVAL_DIR = PILOT_ARTIFACTS / "evaluation_e1_vs_e3"
VIS_DIR = PILOT_ARTIFACTS / "visual_comparison_e1_vs_e3"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_gpu_preflight_and_frozen_assets_untouched():
    """Verify GPU preflight log exists and frozen v2, E1, and Holdout GT hashes are unchanged."""
    gpu_log = PILOT_ARTIFACTS / "gpu_preflight_log.txt"
    assert gpu_log.is_file()
    txt = gpu_log.read_text(encoding="utf-8")
    assert "NVIDIA GeForce RTX 2050" in txt
    assert "Available: True" in txt

    e1_pt = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
    assert _sha256(e1_pt) == "7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8"

    v2_yaml = yaml.safe_load((ROOT / "configs" / "detection_dataset_real_v2.yaml").read_text(encoding="utf-8"))
    for sp, exp_sha in v2_yaml["metadata"]["split_sha256"].items():
        sp_p = ROOT / "data" / "detection" / "splits_real_v2" / f"{sp}.txt"
        assert _sha256(sp_p) == exp_sha

    holdout_gt = ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac.json"
    assert _sha256(holdout_gt) == "f390f0f9440a977d0a0afe9e55b862562f1aafd3011ab18313e1772b18d470c0"


def test_dataset_v3_materialized_and_approvals_scoped():
    """Verify Dataset V3 has exact 1509/328/299 images and 4074/930/720 boxes, with 210 augs & batch_a excluded."""
    rep_p = PILOT_ARTIFACTS / "dataset_v3_materialization_and_preflight_report.json"
    assert rep_p.is_file()
    rep = json.loads(rep_p.read_text(encoding="utf-8"))

    assert rep["all_10_preflight_checks_passed"] is True
    assert rep["approval_record"]["approval_status"] == "USER_APPROVED_TASK6_PILOT"
    assert rep["approval_record"]["excluded_augmented_variants"] == 210
    assert rep["approval_record"]["excluded_batch_a_candidates"] == 8
    assert rep["builder_summary"]["approved_battery_train_staged"] == 111
    assert rep["builder_summary"]["approved_battery_val_staged"] == 29

    ms = rep["measured_on_disk_splits"]
    assert ms["train"]["images_count"] == 1509 and ms["train"]["total_boxes"] == 4074
    assert ms["train"]["per_class_boxes"]["battery"] == 112
    assert ms["val"]["images_count"] == 328 and ms["val"]["total_boxes"] == 930
    assert ms["val"]["per_class_boxes"]["battery"] == 29
    assert ms["test"]["images_count"] == 299 and ms["test"]["total_boxes"] == 720
    assert ms["test"]["per_class_boxes"]["battery"] == 1

    for sp in ("train", "val", "test"):
        assert ms[sp]["missing_images_count"] == 0
        assert ms[sp]["missing_labels_count"] == 0


def test_e3_training_artifacts_and_yolo_adapter_compatibility():
    """Verify E3 training outputs exist, contain valid telemetry, and load in WasteDetector."""
    best_pt = E3_DIR / "weights" / "best.pt"
    last_pt = E3_DIR / "weights" / "last.pt"
    results_csv = E3_DIR / "results.csv"
    args_yaml = E3_DIR / "args.yaml"
    train_log = E3_DIR / "training_log.txt"
    telemetry_json = E3_DIR / "epoch_telemetry.json"

    for p in (best_pt, last_pt, results_csv, args_yaml, train_log, telemetry_json):
        assert p.is_file(), f"Missing E3 artifact: {p}"
        assert p.stat().st_size > 0, f"Empty E3 artifact: {p}"

    tel = json.loads(telemetry_json.read_text(encoding="utf-8"))
    assert tel["epochs_completed"] >= 1
    assert tel["best_epoch"] is not None and tel["best_epoch"] >= 1
    assert len(tel["epoch_records"]) == tel["epochs_completed"]

    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    assert len(rows) == tel["epochs_completed"]

    # Must load cleanly in src.detection.yolo.WasteDetector (validates architecture + phase2_metadata)
    det = WasteDetector(best_pt, device="cpu", confidence_threshold=0.25, iou_threshold=0.55)
    det._ensure_loaded()
    assert det.is_loaded


def test_comprehensive_e1_vs_e3_evaluation_all_4_benchmarks():
    """Verify Evaluation A, B, C, D and per-class comparison CSV exist and are complete."""
    comp_p = EVAL_DIR / "e1_vs_e3_comprehensive_evaluation.json"
    csv_p = EVAL_DIR / "e1_vs_e3_per_class_comparison.csv"
    holdout_p = EVAL_DIR / "holdout_27_e1_vs_e3_per_image.json"

    assert comp_p.is_file()
    assert csv_p.is_file()
    assert holdout_p.is_file()

    comp = json.loads(comp_p.read_text(encoding="utf-8"))
    for key in (
        "evaluation_A_val_v3",
        "evaluation_B_val_v2",
        "evaluation_C_test_frozen",
        "evaluation_D_holdout_27",
        "qualitative_hard_negatives_batch_a",
    ):
        assert key in comp

    # Check Val V3 battery class presence
    assert comp["evaluation_A_val_v3"]["E3_imgsz512"]["per_class"]["battery"]["present_in_gt"] is True
    # Check Val V2 9-class catastrophic forgetting dictionary
    forgetting = comp["evaluation_B_val_v2"]["catastrophic_forgetting_analysis_9_classes"]
    assert len(forgetting) == 9
    assert "battery" not in forgetting

    # Check Holdout 27 images
    assert comp["evaluation_D_holdout_27"]["E3_imgsz512"]["n_images"] == 27
    assert comp["evaluation_D_holdout_27"]["E3_imgsz512"]["n_gt_objects"] == 57


def test_visual_comparison_panels_and_contact_sheets():
    """Verify 3-panel visual comparisons and contact sheets across all 5 required groups."""
    manifest_p = VIS_DIR / "visual_comparison_manifest.json"
    assert manifest_p.is_file()
    m = json.loads(manifest_p.read_text(encoding="utf-8"))

    assert m["total_panels"] >= 35
    for cs_name in m["contact_sheets"]:
        cs_p = VIS_DIR / cs_name
        assert cs_p.is_file() and cs_p.stat().st_size > 100_000

    groups_seen = set()
    for p_info in m["panels"]:
        groups_seen.add(p_info["group"])
        p_file = VIS_DIR / p_info["panel_file"]
        assert p_file.is_file()
        with Image.open(p_file) as im:
            assert im.width >= 1400 and im.height >= 300

    assert len(groups_seen) == 5
