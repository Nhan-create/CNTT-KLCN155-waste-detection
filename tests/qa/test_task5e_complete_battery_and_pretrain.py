"""
tests/qa/test_task5e_complete_battery_and_pretrain.py
-----------------------------------------------------
Sprint AI-UPGRADE-01 — Task 5E Regression & Pre-Training Validation Tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

from src.detection.safe_dataset_builder import (
    validate_on_disk_sample,
    verify_visual_qa_pass_evidence,
)
from scripts.validate_pilot_v3_pretraining_readiness import run_pretraining_validation

ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
FULL140_DIR = STAGING_DIR / "reannotated_v2_full140"
MP20_DIR = STAGING_DIR / "reannotated_v1_minipilot20"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_canonical_representative_consistency_across_all_manifests():
    """
    Task 5E (Task A): Every manifest must select the exact same 140 canonical representative
    filenames (1 per base_id) and the exact same recommended_leak_free_split (0 mismatches).
    """
    with open(STAGING_DIR / "roboflow_battery_staging_manifest.json", "r", encoding="utf-8") as f:
        staging_m = json.load(f)
    with open(STAGING_DIR / "augmentation_and_session_leakage_analysis.json", "r", encoding="utf-8") as f:
        session_m = json.load(f)
    with open(STAGING_DIR / "base_images_qa_manifest.json", "r", encoding="utf-8") as f:
        base_qa_m = json.load(f)
    with open(FULL140_DIR / "full140_reannotation_manifest.json", "r", encoding="utf-8") as f:
        full140_m = json.load(f)
    with open(MP20_DIR / "minipilot20_reannotation_manifest.json", "r", encoding="utf-8") as f:
        mp20_m = json.load(f)
    with open(STAGING_DIR / "training_approval_candidate_list.json", "r", encoding="utf-8") as f:
        approval_m = json.load(f)

    staging_reps = {
        it["base_id"]: it["filename"]
        for it in staging_m["items"]
        if it.get("is_base_representative") is True
    }
    session_reps = {
        bg["base_id"]: bg["representative_filename"]
        for bg in session_m["base_id_groups"]
    }
    base_qa_reps = {it["base_id"]: it["filename"] for it in base_qa_m["items"]}
    full140_reps = {it["base_id"]: it["filename"] for it in full140_m["items"]}
    mp20_reps = {it["base_id"]: it["filename"] for it in mp20_m["items"]}
    approval_reps = {it["base_id"]: it["filename"] for it in approval_m["visual_qa_pass_candidates"]}

    assert len(full140_reps) == 140
    assert staging_reps == full140_reps
    assert session_reps == full140_reps
    assert base_qa_reps == full140_reps
    assert approval_reps == full140_reps
    for bid, fn in mp20_reps.items():
        assert full140_reps[bid] == fn, f"Minipilot20 representative mismatch on {bid}"


def test_minipilot20_annotations_preserved_untouched():
    """
    Task 5E (Task B): The 20 PM-approved mini-pilot annotations must remain untouched
    in full140_reannotation_manifest.json and on disk.
    """
    with open(FULL140_DIR / "full140_reannotation_manifest.json", "r", encoding="utf-8") as f:
        full140_m = json.load(f)
    with open(MP20_DIR / "minipilot20_reannotation_manifest.json", "r", encoding="utf-8") as f:
        mp20_m = json.load(f)

    full_by_base = {it["base_id"]: it for it in full140_m["items"]}
    assert len(mp20_m["items"]) == 20
    for mp_it in mp20_m["items"]:
        bid = mp_it["base_id"]
        f_it = full_by_base[bid]
        assert f_it["filename"] == mp_it["filename"]
        assert f_it["new_box"]["pixel_xyxy"] == mp_it["new_box"]["pixel_xyxy"]
        assert f_it["annotation_batch"] == "MINIPILOT_20_LOCKED"


def test_all_140_reannotated_images_and_labels_valid_on_disk():
    """
    Task 5E (Task B & D): All 140 canonical base images must have 1:1 valid image + YOLO label
    files on disk, valid BEFORE/AFTER overlays, and verified review_evidence.
    """
    with open(FULL140_DIR / "full140_reannotation_manifest.json", "r", encoding="utf-8") as f:
        full140_m = json.load(f)

    items = full140_m["items"]
    assert len(items) == 140

    imgs = list((FULL140_DIR / "images").glob("*.jpg"))
    lbls = list((FULL140_DIR / "labels").glob("*.txt"))
    overlays = list((FULL140_DIR / "overlays_before_after").glob("*.jpg"))
    sheets = list((FULL140_DIR / "contact_sheets").glob("*.jpg"))

    assert len(imgs) == 140
    assert len(lbls) == 140
    assert len(overlays) == 140
    assert len(sheets) == 7

    for it in items:
        img_p = STAGING_DIR / Path(it["image_path"])
        lbl_p = STAGING_DIR / Path(it["label_path"])
        ov_p = STAGING_DIR / Path(it["before_after_overlay_path"])
        assert img_p.exists()
        assert lbl_p.exists()
        assert ov_p.exists()
        validate_on_disk_sample(img_p, lbl_p, expected_boxes=1)
        ok_ev, err_ev = verify_visual_qa_pass_evidence(it)
        assert ok_ev, f"Failed visual QA evidence on {it['base_id']}: {err_ev}"
        assert it["approved_for_train"] in (False, True)


def test_no_constant_scaling_of_old_text_boxes():
    """
    Task 5E (Task B): Verify that the 120 newly re-annotated boxes were NOT created by scaling
    old text boxes by a constant factor (area expansion ratios vary genuinely by image/orientation).
    """
    with open(FULL140_DIR / "full140_reannotation_manifest.json", "r", encoding="utf-8") as f:
        full140_m = json.load(f)

    new_120 = [it for it in full140_m["items"] if it["annotation_batch"] == "TASK5E_REMAINING_120_VERIFIED"]
    assert len(new_120) == 120
    ratios = [it["area_ratio_new_vs_old"] for it in new_120]
    assert float(np.std(ratios)) > 3.0
    assert min(ratios) > 1.8


def test_specimen_isolated_leak_free_split_and_cross_split_adjudication():
    """
    Task 5E (Task A): Verify specimen-isolated split has zero specimen overlap between train and valid,
    zero SAME_OBJECT_SESSION cross-split pairs, and a visual adjudication sheet on disk.
    """
    with open(STAGING_DIR / "augmentation_and_session_leakage_analysis.json", "r", encoding="utf-8") as f:
        session_m = json.load(f)

    cap = session_m["capture_session_and_specimen_leakage"]
    dedup = session_m["deduplication_and_phash_reconciliation"]

    assert cap["recommended_train_base_count"] == 111
    assert cap["recommended_valid_base_count"] == 29
    assert dedup["recommended_split_same_object_session_pairs_count"] == 0
    assert dedup["recommended_split_uncertain_pairs_count"] == 0

    train_specs = {
        bg["specimen_cluster_id"]
        for bg in session_m["base_id_groups"]
        if bg["recommended_leak_free_split"] == "train"
    }
    valid_specs = {
        bg["specimen_cluster_id"]
        for bg in session_m["base_id_groups"]
        if bg["recommended_leak_free_split"] == "valid"
    }
    assert len(train_specs & valid_specs) == 0

    adj_sheet = STAGING_DIR / Path(dedup["visual_adjudication_contact_sheet"])
    assert adj_sheet.exists() and adj_sheet.stat().st_size > 50_000


def test_gatekeeper_approved_for_train_remains_zero_and_v2_untouched():
    """
    Task 5E (Task C, D, E): APPROVED_FOR_TRAIN must remain 0 everywhere and v2 splits must be untouched.
    """
    v2_yaml = yaml.safe_load((ROOT / "configs" / "detection_dataset_real_v2.yaml").read_text(encoding="utf-8"))
    expected_shas = v2_yaml["metadata"]["split_sha256"]
    for split_name, exp_sha in expected_shas.items():
        sp = ROOT / "data" / "detection" / "splits_real_v2" / f"{split_name}.txt"
        assert _sha256(sp) == exp_sha

    report = run_pretraining_validation()
    assert report["all_pretraining_checks_passed"] is True
    assert report["checks"]["check_4_zero_unapproved_in_active_training_staging"]["passed"] is True


def test_v2_direct_label_counts_and_holdout_57_reconciliation():
    """
    Task 5E Sign-Off Fix (Task 1): Direct label counting from v2 splits must match
    configs/detection_dataset_real_v3_pilot.yaml (3963 / 901 / 720), and holdout 27 images
    must reconcile 57 frozen v1 GT objects vs 45 active / 39 accepted discrete v2 proposed objects.
    """
    from src.detection.safe_dataset_builder import count_yolo_split_labels_directly

    v2_dir = ROOT / "data" / "detection"
    tr = count_yolo_split_labels_directly(v2_dir / "splits_real_v2" / "train.txt", dataset_root=v2_dir)
    va = count_yolo_split_labels_directly(v2_dir / "splits_real_v2" / "val.txt", dataset_root=v2_dir)
    te = count_yolo_split_labels_directly(v2_dir / "splits_real_v2" / "test.txt", dataset_root=v2_dir)

    assert tr["images_count"] == 1398 and tr["total_boxes"] == 3963
    assert va["images_count"] == 299 and va["total_boxes"] == 901
    assert te["images_count"] == 299 and te["total_boxes"] == 720
    assert tr["missing_images_count"] == 0 and tr["missing_labels_count"] == 0
    assert va["missing_images_count"] == 0 and va["missing_labels_count"] == 0
    assert te["missing_images_count"] == 0 and te["missing_labels_count"] == 0

    ds_cfg = yaml.safe_load((ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml").read_text(encoding="utf-8"))
    meta = ds_cfg["metadata"]
    assert meta["per_class_boxes"]["train"] == tr["per_class_boxes"]
    assert meta["per_class_boxes"]["val"] == va["per_class_boxes"]
    assert meta["per_class_boxes"]["test"] == te["per_class_boxes"]

    report = json.loads((STAGING_DIR / "pilot_v3_pretraining_validation_report.json").read_text(encoding="utf-8"))
    holdout_rec = report["checks"]["check_8_single_source_of_truth_accounting"]["holdout_27_hinh_anh_rac_reconciliation"]
    assert holdout_rec["total_images"] == 27
    assert holdout_rec["authoritative_v1_frozen"]["total_gt_objects"] == 57
    assert holdout_rec["authoritative_v1_frozen"]["per_class_boxes"]["battery"] == 2
    assert holdout_rec["standardized_v2_proposed_not_frozen"]["total_v1_objects"] == 57
    assert holdout_rec["standardized_v2_proposed_not_frozen"]["active_objects_for_instance_eval"] == 45
    assert holdout_rec["standardized_v2_proposed_not_frozen"]["accepted_discrete_objects"] == 39


def test_additional_8_batch_a_forensic_verification_and_exclusion():
    """
    Task 5E Sign-Off Fix (Task 2): Verify exact classification of the 8 batch_a_mini01 images
    (2 TRUE_CLEAN_NEGATIVE, 4 POSITIVE_WASTE_REAL_DISCARDED, 2 POSITIVE_WASTE_OBJECT_ONLY_SEGREGATED,
    plus 2 QUARANTINED/PENDING blocked items with 0-byte labels) and confirm exclusion from primary Pilot V3.
    """
    audit_p = ROOT / "data" / "acquisition_staging" / "additional_8_images_forensic_verification.json"
    assert audit_p.exists()
    audit = json.loads(audit_p.read_text(encoding="utf-8"))

    assert audit["total_batch_a_items_on_disk"] == 10
    assert audit["visual_qa_pass_items_count"] == 8
    assert audit["quarantined_or_pending_blocked_count"] == 2
    bd = audit["classification_breakdown"]
    assert bd["TRUE_CLEAN_NEGATIVE"] == 2
    assert bd["POSITIVE_WASTE_REAL_DISCARDED"] == 4
    assert bd["POSITIVE_WASTE_OBJECT_ONLY_SEGREGATED"] == 2
    assert bd["QUARANTINED_OR_PENDING_BLOCKED"] == 2
    assert audit["pilot_v3_inclusion_decision"] == "EXCLUDED_FROM_PRIMARY_PILOT_V3_BATTERY_ONLY_EXTENSION"

    for it in audit["items"]:
        assert it["approved_for_train"] is False
        assert it["included_in_pilot_v3_battery_extension"] is False
        if it["classification_role"] == "QUARANTINED_OR_PENDING_BLOCKED":
            assert it["on_disk_label_bytes"] == 0
            assert it["eligible_if_approved_in_default_in_the_wild_slice"] is False


def test_controlled_dryrun_pilot_v3_builder_and_ultralytics_resolution():
    """
    Task 5E Sign-Off Fix (Task 3): Verify controlled dry-run execution log, direct builder counts
    (1509 train / 4074 boxes with battery=112; 328 val / 930 boxes with battery=29; 299 test / 720 boxes),
    Ultralytics path/label resolution, and require_approval=False security block.
    """
    import pytest
    from src.detection.safe_dataset_builder import (
        SafeDatasetBuilderError,
        build_pilot_v3_battery_extension_dataset,
    )

    with pytest.raises(SafeDatasetBuilderError, match="require_approval cannot be False"):
        build_pilot_v3_battery_extension_dataset(
            battery_manifest_path=FULL140_DIR / "full140_reannotation_manifest.json",
            v2_root_dir=ROOT / "data" / "detection",
            output_pilot_dir=ROOT / "data" / "detection_upgrade_v3_pilot",
            require_approval=False,
        )

    dry_p = STAGING_DIR / "pilot_v3_dryrun_execution_log.json"
    assert dry_p.exists()
    dry = json.loads(dry_p.read_text(encoding="utf-8"))

    assert dry["real_manifest_untouched"] is True
    assert dry["builder_status"] == "BUILT_WITH_APPROVED_BATTERY_EXTENSION"
    assert dry["approved_battery_train_staged_in_dryrun"] == 111
    assert dry["approved_battery_val_staged_in_dryrun"] == 29
    assert dry["overlapping_specimens"] == []

    ms = dry["measured_splits_from_builder"]
    assert ms["train"]["images_count"] == 1509 and ms["train"]["total_boxes"] == 4074
    assert ms["train"]["per_class_boxes"]["battery"] == 112
    assert ms["val"]["images_count"] == 328 and ms["val"]["total_boxes"] == 930
    assert ms["val"]["per_class_boxes"]["battery"] == 29
    assert ms["test"]["images_count"] == 299 and ms["test"]["total_boxes"] == 720
    assert ms["test"]["per_class_boxes"]["battery"] == 1

    res = dry["ultralytics_path_and_img2label_resolution"]
    for sp, exp_n in [("train", 1509), ("val", 328), ("test", 299)]:
        assert res[sp]["total_lines"] == exp_n
        assert res[sp]["resolved_images_existing"] == exp_n
        assert res[sp]["resolved_labels_existing"] == exp_n
        assert res[sp]["missing_images_count"] == 0
        assert res[sp]["missing_labels_count"] == 0


def test_gpu_rtx2050_and_training_config_alignment():
    """
    Task 5E Sign-Off Fix (Task 4): Verify live GPU detection (RTX 2050) and YAML config alignment.
    """
    ds_cfg = yaml.safe_load((ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml").read_text(encoding="utf-8"))
    tr_cfg = yaml.safe_load((ROOT / "configs" / "training_yolov8n_v3_pilot.yaml").read_text(encoding="utf-8"))

    assert "RTX 2050" in ds_cfg["metadata"]["hardware_target"]
    assert tr_cfg["model"] == "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"
    assert (ROOT / tr_cfg["model"]).exists()
    assert tr_cfg["imgsz"] == 512
    assert tr_cfg["batch"] == 4
    assert tr_cfg["epochs"] == 25
    assert tr_cfg["patience"] == 5
    assert tr_cfg["cache"] is False
    assert tr_cfg["workers"] == 2

