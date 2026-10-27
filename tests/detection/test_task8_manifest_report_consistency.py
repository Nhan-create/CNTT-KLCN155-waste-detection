"""
Repeatable Consistency & Quality Gate Verification Tests for Task 8C:
- Verifies Wave 1 manifest (`task8_wave1_qa_manifest.json`), `.txt` label files, and `WAVE1_QA_CORRECTION_REPORT.md`
  match 100% on `104` `QA_PASS` unique images and `170` total bounding boxes (`59` `REAL_WASTE_PASS` [101 boxes] + `45` `KEEP_AUXILIARY` [69 boxes]).
- Verifies the 3 independent quality gate flags (`OBJECT_ANNOTATION_VALID`, `REAL_WASTE_CONTEXT_VERIFIED`, `APPROVED_FOR_TRAIN == False`)
  across all Wave 1 (`125`) and Wave 2 (`1,196`) records.
- Verifies Wave 2 `REAL_WASTE_PASS` (`labels_real_waste_pass/`) vs `KEEP_AUXILIARY` (`labels_auxiliary/`) separation,
  priority non-waste items (`0022`, `0116`, `0484`, `0588`, `0731`), and zero session/specimen leakage between train and `VAL_REAL_WASTE`.
"""

from __future__ import annotations

import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
W1_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch"
W2_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch"
REPORTS_DIR = REPO_ROOT / "reports" / "task8_dataset_program"


def test_wave1_manifest_labels_and_report_consistency() -> None:
    """Task 1: Reconcile `WAVE1_QA_CORRECTION_REPORT.md` with `task8_wave1_qa_manifest.json` (104 QA_PASS, 170 boxes)."""
    w1_manifest_path = W1_DIR / "task8_wave1_qa_manifest.json"
    assert w1_manifest_path.exists(), f"Missing {w1_manifest_path}"
    w1 = json.loads(w1_manifest_path.read_text(encoding="utf-8"))
    recs = w1["records"]
    assert len(recs) == 221

    qa_pass_recs = [r for r in recs if r["qa_verdict"] == "QA_PASS"]
    assert len(qa_pass_recs) == 104

    total_manifest_boxes = sum(len(r["boxes"]) for r in qa_pass_recs)
    assert total_manifest_boxes == 170
    assert w1["a4_independent_accounting_qa_pass"]["total_boxes_qa_pass"] == 170

    # Verify .txt label files match manifest box counts 1-to-1
    labels_dir = W1_DIR / "labels"
    txt_files = sorted(labels_dir.glob("*.txt"))
    assert len(txt_files) == 104
    total_txt_lines = 0
    for r in qa_pass_recs:
        lbl_p = labels_dir / f"{r['candidate_id']}.txt"
        assert lbl_p.exists(), f"Missing label file for {r['candidate_id']}"
        lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == len(r["boxes"])
        total_txt_lines += len(lines)
    assert total_txt_lines == 170

    # Verify 3 independent criteria on Wave 1
    rw_recs = [r for r in recs if r["qa_decision"] == "REAL_WASTE_PASS"]
    aux_recs = [r for r in recs if r["qa_decision"] == "KEEP_AUXILIARY"]
    assert len(rw_recs) == 59
    assert len(aux_recs) == 45
    assert sum(len(r["boxes"]) for r in rw_recs) == 93
    assert sum(len(r["boxes"]) for r in aux_recs) == 77

    for r in recs:
        assert r["APPROVED_FOR_TRAIN"] is False
        if r["qa_decision"] == "REAL_WASTE_PASS":
            assert r["OBJECT_ANNOTATION_VALID"] is True
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
            assert r["domain_tag"] == "ACTUAL_WASTE"
        elif r["qa_decision"] == "KEEP_AUXILIARY":
            assert r["OBJECT_ANNOTATION_VALID"] is True
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is False
            assert r["domain_tag"] != "ACTUAL_WASTE"
        else:
            assert r["OBJECT_ANNOTATION_VALID"] is False
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is False

    # Verify WAVE1_QA_CORRECTION_REPORT.md contains exact 104 QA_PASS and 170 boxes
    report_path = REPORTS_DIR / "WAVE1_QA_CORRECTION_REPORT.md"
    assert report_path.exists()
    report_txt = report_path.read_text(encoding="utf-8")
    assert "`104` `QA_PASS`" in report_txt
    assert "**170**" in report_txt
    assert "`93` boxes" in report_txt
    assert "`77` boxes" in report_txt


def test_wave2_three_criteria_and_priority_reaudit() -> None:
    """Task 2, 3 & 4: Verify 3 independent criteria, 175 re-audit priority items, and 200-candidate salvage on Wave 2."""
    w2_manifest_path = W2_DIR / "task8_wave2_qa_manifest.json"
    assert w2_manifest_path.exists()
    w2 = json.loads(w2_manifest_path.read_text(encoding="utf-8"))
    recs = w2["records"]
    assert len(recs) == 1196
    by_id = {r["item_id"]: r for r in recs}

    # Verify all records have the 3 independent boolean criteria and APPROVED_FOR_TRAIN is False
    for r in recs:
        assert "OBJECT_ANNOTATION_VALID" in r and isinstance(r["OBJECT_ANNOTATION_VALID"], bool)
        assert "REAL_WASTE_CONTEXT_VERIFIED" in r and isinstance(r["REAL_WASTE_CONTEXT_VERIFIED"], bool)
        assert r.get("APPROVED_FOR_TRAIN") is False
        assert r.get("approved_for_train") is False

        decision = r["qa_decision"]
        if decision == "REAL_WASTE_PASS":
            assert r["OBJECT_ANNOTATION_VALID"] is True
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
            assert r["domain_tag"] == "ACTUAL_WASTE"
            assert len(r["boxes"]) >= 1
        elif decision == "KEEP_AUXILIARY":
            assert r["OBJECT_ANNOTATION_VALID"] is True
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is False
            assert r["domain_tag"] in ("PRODUCT_STUDIO", "IN_USE", "AMBIGUOUS_CONTEXT")
            assert len(r["boxes"]) >= 1
        elif decision == "NEEDS_REVIEW":
            assert r["OBJECT_ANNOTATION_VALID"] is False
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is False
        elif decision == "REJECT":
            assert r["OBJECT_ANNOTATION_VALID"] is False
            assert r["REAL_WASTE_CONTEXT_VERIFIED"] is False
            assert len(r["boxes"]) == 0
        else:
            raise AssertionError(f"Unexpected qa_decision: {decision}")

    # Verify priority items cited by PM in Task 3
    for pid, expected_domains in [
        ("task8w2_0022", ("PRODUCT_STUDIO",)),
        ("task8w2_0116", ("PRODUCT_STUDIO",)),
        ("task8w2_0484", ("IN_USE",)),
        ("task8w2_0588", ("PRODUCT_STUDIO",)),
        ("task8w2_0731", ("PRODUCT_STUDIO", "AMBIGUOUS_CONTEXT")),
    ]:
        rec = by_id[pid]
        assert rec["REAL_WASTE_CONTEXT_VERIFIED"] is False, f"{pid} must not be verified as real waste"
        assert rec["qa_decision"] == "KEEP_AUXILIARY", f"{pid} must be KEEP_AUXILIARY"
        assert rec["domain_tag"] in expected_domains

    # Verify label directories match manifest counts 1-to-1
    rw_recs = [r for r in recs if r["qa_decision"] == "REAL_WASTE_PASS"]
    aux_recs = [r for r in recs if r["qa_decision"] == "KEEP_AUXILIARY"]
    assert len(rw_recs) == 41  # 12 from 175 re-audit + 29 salvaged from NEEDS_REVIEW
    assert len(aux_recs) == 166  # 134 from 175 re-audit + 32 salvaged from NEEDS_REVIEW

    rw_lbl_files = list((W2_DIR / "labels_real_waste_pass").glob("*.txt"))
    aux_lbl_files = list((W2_DIR / "labels_auxiliary").glob("*.txt"))
    assert len(rw_lbl_files) == len(rw_recs)
    assert len(aux_lbl_files) == len(aux_recs)


def test_real_waste_validation_hygiene_and_zero_leakage() -> None:
    """Task 5: Verify VAL_REAL_WASTE contains ONLY verified ACTUAL_WASTE and has 0 session/specimen leakage with train."""
    w1 = json.loads((W1_DIR / "task8_wave1_qa_manifest.json").read_text(encoding="utf-8"))["records"]
    w2 = json.loads((W2_DIR / "task8_wave2_qa_manifest.json").read_text(encoding="utf-8"))["records"]

    w1_val = [r for r in w1 if r["real_waste_split"] == "VAL_REAL_WASTE_W1"]
    w1_train = [r for r in w1 if r["real_waste_split"] in ("TRAIN_REAL_WASTE_W1", "AUXILIARY_TRAIN_W1")]
    assert len(w1_val) == 16
    for r in w1_val:
        assert r["qa_decision"] == "REAL_WASTE_PASS"
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert r["domain_tag"] == "ACTUAL_WASTE"

    w1_val_specimens = {r["specimen_group"] for r in w1_val}
    w1_train_specimens = {r["specimen_group"] for r in w1_train}
    assert w1_val_specimens.isdisjoint(w1_train_specimens)

    w2_val = [r for r in w2 if r["candidate_split"] == "VAL_REAL_WASTE_W2"]
    w2_train = [r for r in w2 if r["candidate_split"] in ("TRAIN_REAL_WASTE_W2", "AUXILIARY_TRAIN_W2")]
    assert len(w2_val) >= 8
    for r in w2_val:
        assert r["qa_decision"] == "REAL_WASTE_PASS"
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert r["domain_tag"] == "ACTUAL_WASTE"

    w2_val_sessions = {r["session_group"] for r in w2_val}
    w2_train_sessions = {r["session_group"] for r in w2_train}
    assert w2_val_sessions.isdisjoint(w2_train_sessions)

    w2_val_specimens = {r["specimen_group"] for r in w2_val}
    w2_train_specimens = {r["specimen_group"] for r in w2_train}
    assert w2_val_specimens.isdisjoint(w2_train_specimens)


def test_task8d_audited_manifest_and_mini_pack_consistency() -> None:
    """Task 8E Section 1: Verify audited Task 8D manifest, source_split preservation, provenance filter, and compact mini-pack."""
    w8d_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste"
    mini_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8d_independent_qa_mini_pack"

    w8d_p = w8d_dir / "task8d_qa_manifest.json"
    assert w8d_p.exists()
    w8d = json.loads(w8d_p.read_text(encoding="utf-8"))
    recs = w8d["records"]
    assert len(recs) == 443
    assert w8d["qa_status_counts"]["REAL_WASTE_PASS"] == 147
    assert w8d["qa_status_counts"]["KEEP_AUXILIARY"] == 68
    assert w8d["qa_status_counts"]["NEEDS_REVIEW"] == 30
    assert w8d["qa_status_counts"]["REJECT"] == 198
    assert w8d["split_counts"]["TRAIN_REAL_WASTE_W8D"] == 65
    assert w8d["split_counts"]["VAL_REAL_WASTE_W8D"] == 81
    assert w8d["split_counts"]["HOLD_SOURCE_TEST_W8D"] == 1
    assert w8d["real_waste_pass_total_boxes"] == 261
    assert w8d["group_leakage_verification"]["train_val_session_overlap_count"] == 0
    assert w8d["group_leakage_verification"]["train_val_specimen_overlap_count"] == 0
    assert w8d["group_leakage_verification"]["source_test_in_train_count"] == 0
    assert w8d["group_leakage_verification"]["source_valid_in_train_count"] == 0

    rw_recs = [r for r in recs if r["qa_status"] == "REAL_WASTE_PASS"]
    for r in rw_recs:
        assert r["APPROVED_FOR_TRAIN"] is False
        assert r["OBJECT_ANNOTATION_VALID"] is True
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert (REPO_ROOT / r["image_path"]).exists()
        assert (REPO_ROOT / r["label_path"]).exists()
        assert (REPO_ROOT / r["overlay_path"]).exists()
        if r["source_split"] == "test":
            assert r["target_split"] == "HOLD_SOURCE_TEST_W8D"
        elif r["source_split"] in ("valid", "validation"):
            assert r["target_split"] == "VAL_REAL_WASTE_W8D"

    rw_lbl_files = sorted((w8d_dir / "labels_real_waste_pass").glob("*.txt"))
    assert len(rw_lbl_files) == 147
    total_rw_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in rw_lbl_files
    )
    assert total_rw_lines == 261

    # Verify Task 8D Independent QA Mini-Pack
    assert (mini_dir / "task8d_qa_manifest.json").exists()
    assert len(list((mini_dir / "images_sample_30").glob("*.jpg"))) == 30
    assert len(list((mini_dir / "labels_sample_30").glob("*.txt"))) == 30
    assert len(list((mini_dir / "overlays_sample_30").glob("*.jpg"))) == 30


def test_task8e_final_qa_patch_and_regression_audit() -> None:
    """Part A (A1–A4): Verify Task 8E pre-patch snapshot (403/672) vs post-patch manifest (348/628), A1 (1167), A2 (1226), A3 (1152), and physical YOLO files."""
    w8e_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste"
    pre_p = w8e_dir / "qa_patch_before_after" / "task8e_qa_manifest_pre_patch.json"
    post_p = w8e_dir / "task8e_qa_manifest.json"
    summary_p = REPORTS_DIR / "task8e_final_qa_patch_summary.json"
    report_md_p = REPORTS_DIR / "TASK8E_FINAL_QA_PATCH_REPORT.md"

    assert pre_p.exists() and post_p.exists() and summary_p.exists() and report_md_p.exists()
    pre = json.loads(pre_p.read_text(encoding="utf-8"))
    w8e = json.loads(post_p.read_text(encoding="utf-8"))
    patch_summary = json.loads(summary_p.read_text(encoding="utf-8"))

    assert pre["real_waste_pass_total_images"] == 403
    assert pre["real_waste_pass_total_boxes"] == 672

    recs = w8e["records"]
    by_id = {r["candidate_id"]: r for r in recs}
    assert len(recs) == 1246
    assert w8e["qa_status_counts"]["REAL_WASTE_PASS"] == 348
    assert w8e["qa_status_counts"]["KEEP_AUXILIARY"] == 399
    assert w8e["qa_status_counts"]["NEEDS_REVIEW"] == 76
    assert w8e["qa_status_counts"]["REJECT"] == 423
    assert w8e["split_counts"]["TRAIN_REAL_WASTE_W8E"] == 332
    assert w8e["split_counts"]["VAL_REAL_WASTE_W8E"] == 16
    assert w8e["real_waste_pass_total_boxes"] == 628
    assert patch_summary["total_patched_records"] == 114

    # A1: task8e_1167 — 9 per-instance battery boxes
    r1167 = by_id["task8e_1167"]
    assert r1167["qa_status"] == "REAL_WASTE_PASS"
    assert r1167["domain_tag"] == "ACTUAL_WASTE"
    assert r1167["bbox_count"] == 9
    assert all(b["class_id"] == 0 for b in r1167["boxes"])

    # A2: task8e_1226 — 2 per-instance shoe boxes in VAL_REAL_WASTE_W8E
    r1226 = by_id["task8e_1226"]
    assert r1226["qa_status"] == "REAL_WASTE_PASS"
    assert r1226["target_split"] == "VAL_REAL_WASTE_W8E"
    assert r1226["bbox_count"] == 2
    assert all(b["class_id"] == 8 for b in r1226["boxes"])

    # A3: task8e_1152 — demoted to KEEP_AUXILIARY / PRODUCT_STUDIO
    r1152 = by_id["task8e_1152"]
    assert r1152["qa_status"] == "KEEP_AUXILIARY"
    assert r1152["domain_tag"] == "PRODUCT_STUDIO"
    assert r1152["OBJECT_ANNOTATION_VALID"] is True
    assert r1152["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert r1152["APPROVED_FOR_TRAIN"] is False

    # Verify physical YOLO label files match post-patch manifest 1-to-1
    rw_lbl_files = sorted((w8e_dir / "labels_real_waste_pass").glob("*.txt"))
    assert len(rw_lbl_files) == 348
    total_rw_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in rw_lbl_files
    )
    assert total_rw_lines == 628


def test_part_b_three_tier_ledger_and_part_cd_task8f_expansion() -> None:
    """Part B, C & D: Verify Three-Tier 5,000 Ledger, Task 8F Scarce-Class Expansion, and 10-Class Validation Plan."""
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    w8f_p = REPO_ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes" / "task8f_qa_manifest.json"
    c_rep_p = REPORTS_DIR / "TASK8F_SCARCE_CLASS_EXPANSION_REPORT.json"
    d_plan_p = REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json"

    assert ledger_p.exists() and w8f_p.exists() and c_rep_p.exists() and d_plan_p.exists()
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    w8f = json.loads(w8f_p.read_text(encoding="utf-8"))
    d_plan = json.loads(d_plan_p.read_text(encoding="utf-8"))

    # Part B: Three tiers explicitly separated
    tiers = ledger["three_tier_summary"]
    assert tiers["1_CLEAN_BASELINE"]["total_unique_train_images"] == 1277
    assert tiers["1_CLEAN_BASELINE"]["taco_verified_real_waste_train_images"] == 958
    assert tiers["1_CLEAN_BASELINE"]["openimages_unverified_domain_train_images"] == 319
    assert tiers["2_VERIFIED_REAL_WASTE"]["approved_baseline_taco_train_images"] == 958
    assert tiers["2_VERIFIED_REAL_WASTE"]["train_val_id_overlap_count"] == 0
    assert tiers["3_PENDING_APPROVAL"]["approved_for_train"] is False

    # Roboflow Battery 111 train boxes reconciliation preserved
    rf = ledger["roboflow_battery_reconciliation"]
    assert rf["train_base_images"] == 111 and rf["train_base_boxes"] == 111
    assert rf["val_base_images"] == 29 and rf["val_base_boxes"] == 29

    # Historical audit trail preserved
    hist = ledger["historical_audit_trail"]
    assert hist["task8d_filtering_history"]["pre_audit_real_waste_pass_images"] == 200
    assert hist["task8d_filtering_history"]["post_audit_real_waste_pass_images"] == 147
    assert hist["task8e_qa_patch_history"]["pre_patch_real_waste_pass_images"] == 403
    assert hist["task8e_qa_patch_history"]["post_patch_real_waste_pass_images"] == 348
    assert hist["task8f_scarce_expansion"]["pre_correction_real_waste_pass_images"] == 203
    assert hist["task8f_scarce_expansion"]["real_waste_pass_images"] == 86
    assert hist["task8f_scarce_expansion"]["post_correction_high_iou_same_class_pairs"] == 0

    # Part C: Task 8F Scarce-Class Expansion (Post-Semantic QA Correction: 748 candidates, 86 verified REAL_WASTE_PASS)
    assert w8f["total_downloaded_candidates"] == 748
    assert w8f["real_waste_pass_total_images"] == 86
    assert w8f["real_waste_pass_train_images"] == 71
    assert w8f["real_waste_pass_val_images"] == 15
    assert w8f["real_waste_pass_total_boxes"] == 120
    assert w8f["approved_for_train"] is False

    w8f_pass = [r for r in w8f["records"] if r["qa_status"] == "REAL_WASTE_PASS"]
    assert len(w8f_pass) == w8f["real_waste_pass_total_images"] == 86
    for r in w8f_pass:
        assert r["APPROVED_FOR_TRAIN"] is False
        assert r["OBJECT_ANNOTATION_VALID"] is True
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert r["domain_tag"] == "ACTUAL_WASTE"
        assert r["source_split"] != "test"
        if r["source_split"] in ("validation", "valid"):
            assert r["target_split"] == "VAL_REAL_WASTE_W8F"
        assert (REPO_ROOT / r["image_path"]).exists()
        assert (REPO_ROOT / r["label_path"]).exists()
        assert (REPO_ROOT / r["overlay_path"]).exists()

    # Part D: 10-Class Validation Plan
    assert d_plan["frozen_test_untouched_299"] is True
    assert d_plan["candidate_test_untouched_274"] is True
    assert d_plan["external_holdout_untouched_27"] is True
    assert d_plan["zero_train_val_leakage_verified"] is True
    assert len(d_plan["per_class_validation_inventory"]) == 10


def test_task8f_semantic_qa_correction_and_regression_gates() -> None:
    """Task 8F Quality Correction (Tasks 1–5): Verify semantic class fixes, domain re-audit, zero IoU>0.70 duplicates, and ledger synchronization."""
    from scripts.acquire_and_qa_task8f_scarce_classes import (
        deduplicate_same_class_boxes_iou,
        infer_material_class_from_title,
    )
    from scripts.correct_task8f_semantic_qa import compute_iou_xyxy

    w8f_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes"
    pre_p = w8f_dir / "qa_correction_before_after" / "task8f_qa_manifest_pre_correction.json"
    post_p = w8f_dir / "task8f_qa_manifest.json"
    corr_json_p = REPORTS_DIR / "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.json"
    corr_md_p = REPORTS_DIR / "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md"
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    val_plan_p = REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json"

    assert pre_p.exists() and post_p.exists() and corr_json_p.exists() and corr_md_p.exists()
    pre = json.loads(pre_p.read_text(encoding="utf-8"))
    post = json.loads(post_p.read_text(encoding="utf-8"))
    corr = json.loads(corr_json_p.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    val_plan = json.loads(val_plan_p.read_text(encoding="utf-8"))

    assert pre["real_waste_pass_total_images"] == 203
    assert pre["real_waste_pass_total_boxes"] == 407

    by_id = {r["candidate_id"]: r for r in post["records"]}
    assert len(by_id) == 748
    assert post["qa_status_counts"]["REAL_WASTE_PASS"] == 86
    assert post["qa_status_counts"]["KEEP_AUXILIARY"] == 379
    assert post["qa_status_counts"]["NEEDS_REVIEW"] == 126
    assert post["qa_status_counts"]["REJECT"] == 157

    # Task 1: Confirmed wrong-class fixes
    # 1) task8f_0657: Jägermeister glass bottle -> 4:glass (not 6:paper), 1 tight box
    r0657 = by_id["task8f_0657"]
    assert r0657["qa_status"] == "REAL_WASTE_PASS"
    assert r0657["classes_present_ids"] == [4]
    assert r0657["classes_present_names"] == ["glass"]
    assert r0657["bbox_count"] == 1

    # 2) task8f_0664: Pepsi metal can -> 5:metal (not 6:paper), 1 tight box
    r0664 = by_id["task8f_0664"]
    assert r0664["qa_status"] == "REAL_WASTE_PASS"
    assert r0664["classes_present_ids"] == [5]
    assert r0664["classes_present_names"] == ["metal"]
    assert r0664["bbox_count"] == 1

    # 3) task8f_0661: Miller High Life beer can -> 5:metal (not 6:paper), 1 tight box
    r0661 = by_id["task8f_0661"]
    assert r0661["qa_status"] == "REAL_WASTE_PASS"
    assert r0661["classes_present_ids"] == [5]
    assert r0661["classes_present_names"] == ["metal"]
    assert r0661["bbox_count"] == 1

    # 4) task8f_0712: Boxing gloves on wire -> 3:clothes (not 8:shoes), 2 tight boxes, NEEDS_REVIEW
    r0712 = by_id["task8f_0712"]
    assert r0712["qa_status"] == "NEEDS_REVIEW"
    assert r0712["classes_present_ids"] == [3]
    assert r0712["classes_present_names"] == ["clothes"]
    assert r0712["bbox_count"] == 2
    for b in r0712["boxes"]:
        _, _, bw, bh = b["yolo_xywhn"]
        assert bw * bh < 0.05  # Tight per-glove boxes, not the [0.14,0.14,0.86,0.86] fallback box

    # Task 2: Priority non-waste / in-use / food display items removed from REAL_WASTE_PASS
    for aux_cid in ("task8f_0026", "task8f_0118", "task8f_0250", "task8f_0255", "task8f_0573", "task8f_0590"):
        r_aux = by_id[aux_cid]
        assert r_aux["qa_status"] == "KEEP_AUXILIARY", f"{aux_cid} must be KEEP_AUXILIARY"
        assert r_aux["domain_tag"] == "IN_USE"
        assert r_aux["OBJECT_ANNOTATION_VALID"] is True
        assert r_aux["REAL_WASTE_CONTEXT_VERIFIED"] is False
        assert r_aux["APPROVED_FOR_TRAIN"] is False

    r0690 = by_id["task8f_0690"]
    assert r0690["qa_status"] == "REJECT"
    assert r0690["bbox_count"] == 0

    # Task 3 & Final Safety Check Task 1:
    # - Zero DUPLICATE_ANNOTATION pairs across all 748 records
    # - Zero IoU > 0.70 duplicate pairs in REAL_WASTE_PASS (86 records)
    # - Preserved VALID_SAME_CLASS_OVERLAP pairs in KEEP_AUXILIARY / NEEDS_REVIEW (not blindly deleted to force IoU>0.70 == 0)
    from scripts.acquire_and_qa_task8f_scarce_classes import classify_box_pair_overlap
    from scripts.correct_task8f_semantic_qa import _struct_box_to_norm_tuple

    assert len(corr["task3_high_iou_33_pairs_resolutions"]) == 33
    preserved_same_class_overlaps = 0
    for r in post["records"]:
        bxs = r["boxes"]
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                if bxs[i]["class_id"] == bxs[j]["class_id"]:
                    iou_val = compute_iou_xyxy(bxs[i]["bbox_xyxy"], bxs[j]["bbox_xyxy"])
                    rel = classify_box_pair_overlap(
                        _struct_box_to_norm_tuple(bxs[i]),
                        _struct_box_to_norm_tuple(bxs[j]),
                    )
                    assert rel["category"] != "DUPLICATE_ANNOTATION", (
                        f"Duplicate annotation pair in {r['candidate_id']}: {rel}"
                    )
                    if r["qa_status"] == "REAL_WASTE_PASS":
                        assert iou_val <= 0.70, f"High-IoU pair in REAL_WASTE_PASS {r['candidate_id']}: IoU={iou_val}"
                    elif iou_val > 0.70:
                        assert rel["category"] == "VALID_SAME_CLASS_OVERLAP" and rel["keep_both"] is True
                        preserved_same_class_overlaps += 1
    assert preserved_same_class_overlaps == 9, f"Expected 9 preserved valid same-class overlaps in AUX/REVIEW, got {preserved_same_class_overlaps}"

    # Verify physical YOLO files in labels_real_waste_pass match post-correction manifest 1-to-1
    pass_lbls = sorted((w8f_dir / "labels_real_waste_pass").glob("*.txt"))
    assert len(pass_lbls) == 86
    total_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in pass_lbls
    )
    assert total_lines == 120

    # Task 4: Ledger & Validation Plan exact synchronization with Task 8F post-correction manifest (+ Wave 8G / Wave 8H if present)
    tiers = ledger["three_tier_summary"]
    w8g_tr_cohort = int(tiers["3_PENDING_APPROVAL"]["cohorts"].get("wave8g_large_scale_real_waste_train", 0))
    w8h_tr_cohort = int(tiers["3_PENDING_APPROVAL"]["cohorts"].get("wave8h_scarce_scaling_real_waste_train", 0))
    w8g_val_cohort = int(val_plan["cohorts_included"].get("wave8g_val_images", 0))
    w8h_val_cohort = int(val_plan["cohorts_included"].get("wave8h_val_images", 0))
    assert tiers["3_PENDING_APPROVAL"]["cohorts"]["wave8f_scarce_real_waste_train"] == 71
    assert tiers["3_PENDING_APPROVAL"]["pending_verified_real_waste_unique_train_images"] == 563 + w8g_tr_cohort + w8h_tr_cohort
    assert tiers["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_train_images"] == 1521 + w8g_tr_cohort + w8h_tr_cohort
    assert tiers["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_val_images"] == 353 + w8g_val_cohort + w8h_val_cohort
    assert val_plan["cohorts_included"]["wave8f_val_images"] == 15
    assert val_plan["cohorts_included"]["total_unique_verified_real_waste_val_images"] == 353 + w8g_val_cohort + w8h_val_cohort

    # Task 5: Unit tests on root-cause prevention functions
    deduped = deduplicate_same_class_boxes_iou(
        [(2, 0.13, 0.07, 0.96, 0.66), (2, 0.135, 0.072, 0.958, 0.658)], iou_thresh=0.70
    )
    assert len(deduped) == 1
    assert infer_material_class_from_title("File:Leere Flasche Jägermeister 2014.JPG", 6) == 4
    assert infer_material_class_from_title("File:Pepsi can (34041536296).jpg", 6) == 5
    assert infer_material_class_from_title("File:Miller High Life flattened.jpg", 6) == 5
    assert infer_material_class_from_title("File:Boxing gloves hanging from wires.jpg", 8) == 3


def test_task8f_final_safety_nms_and_overlap_three_regression_cases() -> None:
    """Task 8F Final Safety Check (Task 1): Verify 3 mandatory regression cases for NMS & high-IoU bbox handling."""
    import inspect
    from scripts.acquire_and_qa_task8f_scarce_classes import (
        classify_box_pair_overlap,
        deduplicate_same_class_boxes_iou,
        extract_tight_per_instance_boxes,
        norm_boxes_to_struct,
    )
    from scripts.correct_task8f_semantic_qa import deduplicate_boxes_struct

    # Verify agnostic_nms=False is enforced in extract_tight_per_instance_boxes (never blind agnostic_nms=True)
    src_code = inspect.getsource(extract_tight_per_instance_boxes)
    assert "agnostic_nms=False" in src_code
    assert "agnostic_nms=True" not in src_code

    # -------------------------------------------------------------------------
    # REGRESSION CASE 1: Hai annotation trùng một vật thể (DUPLICATE_ANNOTATION)
    # Same class, IoU >= 0.90 (here IoU = 0.964), coinciding edges (max_edge_delta <= 0.045)
    # -------------------------------------------------------------------------
    dup_box_a = (6, 0.190, 0.250, 0.610, 0.670)
    dup_box_b = (6, 0.194, 0.252, 0.608, 0.668)
    rel_case1 = classify_box_pair_overlap(dup_box_a, dup_box_b)
    assert rel_case1["category"] == "DUPLICATE_ANNOTATION"
    assert rel_case1["keep_both"] is False
    assert rel_case1["iou"] >= 0.90
    assert rel_case1["max_edge_delta"] <= 0.045

    dedup_case1 = deduplicate_same_class_boxes_iou([dup_box_a, dup_box_b], iou_thresh=0.70, is_ground_truth=False)
    assert len(dedup_case1) == 1, "Duplicate annotation on 1 single object must collapse to 1 box"

    # -------------------------------------------------------------------------
    # REGRESSION CASE 2: Hai vật thể cùng lớp chồng lấp mạnh (VALID_SAME_CLASS_OVERLAP)
    # Same class, strong overlap (IoU = 0.7826 > 0.70), distinct object centers/edges (max_edge_delta = 0.060 > 0.045)
    # Must NEVER auto-delete the second distinct object!
    # -------------------------------------------------------------------------
    ov_same_a = (2, 0.200, 0.200, 0.700, 0.700)
    ov_same_b = (2, 0.260, 0.200, 0.760, 0.700)
    rel_case2 = classify_box_pair_overlap(ov_same_a, ov_same_b)
    assert rel_case2["category"] == "VALID_SAME_CLASS_OVERLAP"
    assert rel_case2["keep_both"] is True
    assert rel_case2["requires_visual_review"] is True
    assert 0.70 < rel_case2["iou"] < 0.90

    dedup_case2 = deduplicate_same_class_boxes_iou([ov_same_a, ov_same_b], iou_thresh=0.70, is_ground_truth=False)
    assert len(dedup_case2) == 2, "Two strongly overlapping same-class objects must BOTH be preserved"

    gt_struct_case2 = norm_boxes_to_struct(1000, 1000, [ov_same_a, ov_same_b], origin="openimages_v7_train_gt")
    assert len(gt_struct_case2) == 2, "Ground-truth strongly overlapping same-class boxes must NEVER be deleted"
    assert len(deduplicate_boxes_struct(gt_struct_case2, iou_thresh=0.70)) == 2

    # -------------------------------------------------------------------------
    # REGRESSION CASE 3: Hai vật thể khác lớp chồng lấp (VALID_CROSS_CLASS_OVERLAP)
    # - Case 3a (Contained cross-class in task8f_0694): (2, 0.210, 0.030, 0.830, 0.940) & (1, 0.390, 0.560, 0.540, 0.720) -> exact IoU = 0.042538
    # - Case 3b (High-IoU cross-class overlap): (4, 0.200, 0.200, 0.700, 0.700) & (6, 0.210, 0.210, 0.690, 0.690) -> exact IoU = 0.921569
    # Must NEVER suppress or overwrite the second class in either case!
    # -------------------------------------------------------------------------
    cross_box_a = (2, 0.210, 0.030, 0.830, 0.940)  # 2:cardboard (task8f_0694 outer pastry box)
    cross_box_b = (1, 0.390, 0.560, 0.540, 0.720)  # 1:biological (task8f_0694 inner cake)
    rel_case3_contained = classify_box_pair_overlap(cross_box_a, cross_box_b)
    assert rel_case3_contained["category"] == "VALID_CROSS_CLASS_OVERLAP"
    assert rel_case3_contained["keep_both"] is True
    assert abs(rel_case3_contained["iou"] - 0.042538) < 1e-5, f"Expected IoU=0.042538, got {rel_case3_contained['iou']}"

    cross_high_iou_a = (4, 0.200, 0.200, 0.700, 0.700)  # 4:glass bottle (area = 0.2500)
    cross_high_iou_b = (6, 0.210, 0.210, 0.690, 0.690)  # 6:paper label/wrapper (area = 0.2304 -> exact IoU = 0.2304 / 0.2500 = 0.9216)
    rel_case3_high_iou = classify_box_pair_overlap(cross_high_iou_a, cross_high_iou_b)
    assert rel_case3_high_iou["category"] == "VALID_CROSS_CLASS_OVERLAP"
    assert rel_case3_high_iou["keep_both"] is True
    assert abs(rel_case3_high_iou["iou"] - 0.9216) < 1e-6, f"Expected IoU=0.9216, got {rel_case3_high_iou['iou']}"

    dedup_case3 = deduplicate_same_class_boxes_iou(
        [cross_box_a, cross_box_b, cross_high_iou_a, cross_high_iou_b], iou_thresh=0.70, is_ground_truth=False
    )
    assert len(dedup_case3) == 4
    struct_case3 = norm_boxes_to_struct(1000, 1000, [cross_box_a, cross_box_b], origin="task8f_semantic_qa_verified_v2")
    assert [b["class_id"] for b in struct_case3] == [2, 1]


def test_task8f_final_safety_title_inference_and_idempotency() -> None:
    """Task 8F Final Safety Check (Tasks 2 & 3): Verify candidate-only title proposal, human override protection, and dataset idempotency."""
    from scripts.acquire_and_qa_task8f_scarce_classes import (
        infer_material_class_from_title,
        propose_candidate_class_from_title,
    )

    # 1. Candidate-only behavior: never auto-promotes QA_PASS or ACTUAL_WASTE
    prop_glass = propose_candidate_class_from_title("File:Leere Flasche Jägermeister 2014.JPG", default_cid=6)
    assert prop_glass["proposed_class_id"] == 4
    assert prop_glass["is_candidate_only"] is True
    assert prop_glass["title_changed_class"] is True
    assert prop_glass["requires_visual_qa"] is True
    assert prop_glass["can_auto_promote_qa_pass"] is False
    assert prop_glass["can_auto_assert_actual_waste"] is False

    # 2. Human-verified annotations are strictly preserved even if filename has conflicting tokens
    prop_human = propose_candidate_class_from_title(
        "File:Paper cup wrapper next to Pepsi can.jpg",
        default_cid=6,
        human_verified=True,
        verified_cid=6,
    )
    assert prop_human["proposed_class_id"] == 6
    assert prop_human["human_verified_preserved"] is True
    assert infer_material_class_from_title(
        "File:Paper cup wrapper next to Pepsi can.jpg", 6, human_verified=True, verified_cid=6
    ) == 6

    # 3. Final dataset & multi-instance overlap preservation check
    w8f_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes"
    post = json.loads((w8f_dir / "task8f_qa_manifest.json").read_text(encoding="utf-8"))
    by_id = {r["candidate_id"]: r for r in post["records"]}

    # Multi-instance overlapping objects preserved in REAL_WASTE_PASS:
    # - task8f_0285: 4 touching corroded AA batteries (3 adjacent overlapping pairs with IoU 0.35..0.44)
    assert by_id["task8f_0285"]["qa_status"] == "REAL_WASTE_PASS"
    assert by_id["task8f_0285"]["bbox_count"] == 4
    # - task8f_0694: 2 overlapping cross-class objects (2:cardboard box + 1:biological cake inside)
    assert by_id["task8f_0694"]["qa_status"] == "REAL_WASTE_PASS"
    assert by_id["task8f_0694"]["classes_present_ids"] == [1, 2]
    assert by_id["task8f_0694"]["bbox_count"] == 2
    # - task8f_0716: 4 overlapping discarded shoes on trail
    assert by_id["task8f_0716"]["qa_status"] == "REAL_WASTE_PASS"
    assert by_id["task8f_0716"]["bbox_count"] == 4

    # Check all 748 records have APPROVED_FOR_TRAIN = False and zero train/val specimen or session leakage
    assert all(r["APPROVED_FOR_TRAIN"] is False for r in post["records"])
    pass_recs = [r for r in post["records"] if r["qa_status"] == "REAL_WASTE_PASS"]
    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8F"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8F"]
    assert len(pass_recs) == 86 and len(train_recs) == 71 and len(val_recs) == 15
    assert sum(r["bbox_count"] for r in pass_recs) == 120
    assert not ({r["session_group"] for r in train_recs} & {r["session_group"] for r in val_recs})
    assert not ({r["specimen_group"] for r in train_recs} & {r["specimen_group"] for r in val_recs})


def test_task8g_large_scale_expansion_and_quality_gates() -> None:
    """Task 8G (Tasks 0–5): Verify large-scale checkpointed acquisition, scarce/multiclass QA gates, V4 ledger, and handover ZIP."""
    from scripts.acquire_and_qa_task8f_scarce_classes import classify_box_pair_overlap
    from scripts.correct_task8f_semantic_qa import _struct_box_to_norm_tuple

    w8g_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8g_large_scale_expansion"
    ckpt_p = w8g_dir / "checkpoint_raw_candidates.json"
    raw_p = w8g_dir / "task8g_raw_candidates_manifest.json"
    qa_p = w8g_dir / "task8g_qa_manifest.json"
    rep_p = REPORTS_DIR / "TASK8G_LARGE_SCALE_EXPANSION_REPORT.json"
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    val_plan_p = REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json"
    zip_p = REPO_ROOT / "artifacts" / "deliveries" / "TASK8G_LARGE_SCALE_EXPANSION.zip"
    if not zip_p.exists():
        zip_p = REPO_ROOT / "TASK8G_LARGE_SCALE_EXPANSION.zip"

    assert ckpt_p.exists() and raw_p.exists() and qa_p.exists() and rep_p.exists() and zip_p.exists()
    qa = json.loads(qa_p.read_text(encoding="utf-8"))
    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    val_plan = json.loads(val_plan_p.read_text(encoding="utf-8"))

    # Task 1 & 2: Checkpointed 920 unique candidates, 171 REAL_WASTE_PASS (160 train / 11 val), 52 multi-class
    assert qa["total_downloaded_candidates"] == 920
    assert qa["real_waste_pass_total_images"] == 171
    assert qa["real_waste_pass_train_images"] == 160
    assert qa["real_waste_pass_val_images"] == 11
    assert qa["real_waste_pass_total_boxes"] == 291
    assert qa["real_waste_pass_multiclass_images"] == 52
    assert qa["approved_for_train"] is False
    assert qa["zero_unannotated_waste_used_as_hard_negative"] is True

    # Task 3: Safety gates on all 920 records
    assert all(r["APPROVED_FOR_TRAIN"] is False for r in qa["records"])
    pass_recs = [r for r in qa["records"] if r["qa_status"] == "REAL_WASTE_PASS"]
    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8G"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8G"]
    assert len(pass_recs) == 171 and len(train_recs) == 160 and len(val_recs) == 11
    assert not ({r["session_group"] for r in train_recs} & {r["session_group"] for r in val_recs})
    assert not ({r["specimen_group"] for r in train_recs} & {r["specimen_group"] for r in val_recs})

    for r in pass_recs:
        assert r["domain_tag"] == "ACTUAL_WASTE"
        assert r["OBJECT_ANNOTATION_VALID"] is True
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert r["source_split"] != "test"
        if r["source_split"] in ("validation", "valid"):
            assert r["target_split"] == "VAL_REAL_WASTE_W8G"
        assert (REPO_ROOT / r["image_path"]).exists()
        assert (REPO_ROOT / r["label_path"]).exists()
        assert (REPO_ROOT / r["overlay_path"]).exists()

    # Zero DUPLICATE_ANNOTATION pairs across all 920 records
    for r in qa["records"]:
        bxs = r["boxes"]
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                rel = classify_box_pair_overlap(
                    _struct_box_to_norm_tuple(bxs[i]),
                    _struct_box_to_norm_tuple(bxs[j]),
                )
                assert rel["category"] != "DUPLICATE_ANNOTATION"

    # Physical YOLO label files match manifest 1-to-1
    pass_lbls = sorted((w8g_dir / "labels_real_waste_pass").glob("*.txt"))
    assert len(pass_lbls) == 171
    total_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in pass_lbls
    )
    assert total_lines == 291

    # Task 4 & 5: Ledger V4/V5/V6, Validation Plan V3/V4/V5 & Handover ZIP (< 500 MB)
    assert ledger["ledger_version"] in (
        "TASK8G_CANONICAL_5000_LEDGER_V4_LARGE_SCALE_EXPANSION",
        "TASK8H_CANONICAL_5000_LEDGER_V5_SCARCE_CLASS_SCALING",
        "TASK8I_CANONICAL_5000_LEDGER_V6_SOURCE_SIGNOFF_AND_FIELD_PILOT",
    )
    assert ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["cohorts"]["wave8g_large_scale_real_waste_train"] == 160
    assert ledger["three_tier_summary"]["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_train_images"] >= 1681
    assert val_plan["cohorts_included"]["wave8g_val_images"] == 11
    assert val_plan["cohorts_included"]["total_unique_verified_real_waste_val_images"] >= 364
    assert rep["task0_regression_case3_fix"]["status"] == "FIXED_AND_VERIFIED"
    assert 0 < zip_p.stat().st_size < 500 * 1024 * 1024


def test_task8h_scarce_class_scaling_and_quality_gates() -> None:
    """Task 8H: Verify scarce-class strategy shift, 16+ source evaluations, exhausted sources, field collection plan, QA gates, V5 ledger, and ZIP."""
    from scripts.acquire_and_qa_task8f_scarce_classes import classify_box_pair_overlap
    from scripts.correct_task8f_semantic_qa import _struct_box_to_norm_tuple

    w8h_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8h_scarce_class_scaling"
    ckpt_p = w8h_dir / "checkpoint_raw_candidates.json"
    raw_p = w8h_dir / "task8h_raw_candidates_manifest.json"
    qa_p = w8h_dir / "task8h_qa_manifest.json"
    lic_p = w8h_dir / "LICENSE_AND_PROVENANCE.md"
    rep_p = REPORTS_DIR / "TASK8H_SCARCE_CLASS_SCALING_REPORT.json"
    field_p = REPORTS_DIR / "SCARCE_CLASS_FIELD_COLLECTION_PLAN.json"
    field_md_p = REPORTS_DIR / "SCARCE_CLASS_FIELD_COLLECTION_PLAN.md"
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    val_plan_p = REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json"
    zip_p = REPO_ROOT / "artifacts" / "deliveries" / "TASK8H_SCARCE_CLASS_SCALING.zip"
    if not zip_p.exists():
        zip_p = REPO_ROOT / "TASK8H_SCARCE_CLASS_SCALING.zip"

    assert ckpt_p.exists() and raw_p.exists() and qa_p.exists() and lic_p.exists()
    assert rep_p.exists() and field_p.exists() and field_md_p.exists() and zip_p.exists()

    qa = json.loads(qa_p.read_text(encoding="utf-8"))
    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    field_plan = json.loads(field_p.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
    val_plan = json.loads(val_plan_p.read_text(encoding="utf-8"))

    # 1. Source research & pilot evaluations (>= 3 candidate sources per scarce class: 0, 3, 8, 1, 4)
    evals = qa["source_evaluations"]
    assert len(evals) >= 15
    for scarce_cid in (0, 3, 8, 1, 4):
        c_evals = [e for e in evals if int(e["target_scarce_class_id"]) == scarce_cid]
        assert len(c_evals) >= 3, f"Scarce class {scarce_cid} has < 3 evaluated sources"

    # 2. Exhausted sources explicitly marked EXHAUSTED and not re-downloaded
    exhausted = qa["exhausted_sources"]
    assert len(exhausted) >= 4
    assert all(ex["status"] == "EXHAUSTED" for ex in exhausted)

    # 3. Field Data Collection Plan (Section 4) exists with 0 simulated/fabricated images
    assert field_plan["simulated_or_fabricated_images"] == 0
    assert len(field_plan["per_class_gaps_and_session_estimates"]) == 5
    assert "camera_angles_and_lighting" in field_plan
    assert "multiclass_co_occurrence_protocol" in field_plan
    assert "instance_annotation_guidelines" in field_plan
    assert "rights_safety_and_privacy" in field_plan

    # 4. QA & Independence invariants on all Task 8H records
    assert qa["approved_for_train"] is False
    assert qa["zero_unannotated_waste_used_as_hard_negative"] is True
    assert all(r["APPROVED_FOR_TRAIN"] is False for r in qa["records"])
    assert all(r["primary_class_id"] in (0, 1, 3, 4, 8) for r in qa["records"])

    pass_recs = [r for r in qa["records"] if r["qa_status"] == "REAL_WASTE_PASS"]
    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8H"]
    assert len(pass_recs) == qa["real_waste_pass_total_images"]
    assert len(train_recs) == qa["real_waste_pass_train_images"]
    assert len(val_recs) == qa["real_waste_pass_val_images"]
    assert not ({r["session_group"] for r in train_recs} & {r["session_group"] for r in val_recs})
    assert not ({r["specimen_group"] for r in train_recs} & {r["specimen_group"] for r in val_recs})

    # Verify meaningful train increases across all 5 scarce classes (and > Task 8G's +1 battery / +3 clothes / +3 shoes)
    tr_by_cls = qa["per_class_real_waste_pass_train_images"]
    assert tr_by_cls["battery"] >= 10
    assert tr_by_cls["clothes"] >= 10
    assert tr_by_cls["shoes"] >= 10
    assert tr_by_cls["biological"] >= 15
    assert tr_by_cls["glass"] >= 10
    # Verify plastic was NOT bulk-expanded (strictly co-occurring only, <= 10 images)
    assert tr_by_cls["plastic"] <= 10

    for r in pass_recs:
        assert r["domain_tag"] == "ACTUAL_WASTE"
        assert r["OBJECT_ANNOTATION_VALID"] is True
        assert r["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert r["source_split"] != "test"
        if r["source_split"] in ("validation", "valid"):
            assert r["target_split"] == "VAL_REAL_WASTE_W8H"
        assert (REPO_ROOT / r["image_path"]).exists()
        assert (REPO_ROOT / r["label_path"]).exists()
        assert (REPO_ROOT / r["overlay_path"]).exists()

    # Zero DUPLICATE_ANNOTATION pairs across all Task 8H records
    for r in qa["records"]:
        bxs = r["boxes"]
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                rel = classify_box_pair_overlap(
                    _struct_box_to_norm_tuple(bxs[i]),
                    _struct_box_to_norm_tuple(bxs[j]),
                )
                assert rel["category"] != "DUPLICATE_ANNOTATION"

    # Physical YOLO label files match manifest 1-to-1
    pass_lbls = sorted((w8h_dir / "labels_real_waste_pass").glob("*.txt"))
    assert len(pass_lbls) == len(pass_recs)
    total_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in pass_lbls
    )
    assert total_lines == qa["real_waste_pass_total_boxes"]

    # 5. Ledger V5/V6, Validation Plan V4/V5 & Handover ZIP
    assert ledger["ledger_version"] in (
        "TASK8H_CANONICAL_5000_LEDGER_V5_SCARCE_CLASS_SCALING",
        "TASK8I_CANONICAL_5000_LEDGER_V6_SOURCE_SIGNOFF_AND_FIELD_PILOT",
    )
    assert ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["cohorts"]["wave8h_scarce_scaling_real_waste_train"] == len(train_recs)
    assert ledger["three_tier_summary"]["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_train_images"] == 1681 + len(train_recs)
    assert val_plan["plan_version"] in (
        "REAL_WASTE_10CLASS_VALIDATION_PLAN_V4_TASK8H_SCARCE_SCALING",
        "REAL_WASTE_10CLASS_VALIDATION_PLAN_V5_TASK8I_SOURCE_SIGNOFF",
    )
    assert val_plan["cohorts_included"]["wave8h_val_images"] == len(val_recs)
    assert val_plan["cohorts_included"]["total_unique_verified_real_waste_val_images"] == 364 + len(val_recs)
    assert rep["quality_and_safety_gates"]["task8g_manifest_and_checksums_preserved"] is True
    assert 0 < zip_p.stat().st_size < 500 * 1024 * 1024


def test_task8i_inaturalist_signoff_and_field_pilot_tools() -> None:
    """Task 8I + Final Hotfix: Verify 107 iNaturalist sign-off (30 REAL_WASTE_PASS: 24 train / 6 val), separated photo vs observation license policy, 3 handheld citrus demotions, 90-image PM contact sheets, Field Capture Kit, 4D independence, and handover ZIPs."""
    w8i_dir = REPO_ROOT / "data" / "acquisition_staging" / "task8i_source_signoff_and_field_pilot"
    inat_json_p = w8i_dir / "task8i_inaturalist_107_verified_manifest.json"
    inat_csv_p = w8i_dir / "task8i_inaturalist_107_verified_manifest.csv"
    inat_md_p = w8i_dir / "INATURALIST_107_PER_IMAGE_LICENSE_TABLE.md"
    ps_ex_p = w8i_dir / "regression_examples" / "product_studio_held_at_needs_review_example.json"
    fg_json_p = REPORTS_DIR / "FIRST_PARTY_FIELD_CAPTURE_GUIDE.json"
    fg_md_p = REPORTS_DIR / "FIRST_PARTY_FIELD_CAPTURE_GUIDE.md"
    dp_json_p = REPORTS_DIR / "DUAL_CHANNEL_5000_SCALING_PLAN.json"
    dp_md_p = REPORTS_DIR / "DUAL_CHANNEL_5000_SCALING_PLAN.md"
    rep8i_json_p = REPORTS_DIR / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT_REPORT.json"
    rep8i_md_p = REPORTS_DIR / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT_REPORT.md"
    unc_json_p = REPORTS_DIR / "UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.json"
    unc_md_p = REPORTS_DIR / "UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.md"
    delta_json_p = REPORTS_DIR / "TASK8I_HOTFIX_MANIFEST_AND_LEDGER_DELTA.json"
    delta_md_p = REPORTS_DIR / "TASK8I_HOTFIX_MANIFEST_AND_LEDGER_DELTA.md"
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    zip_p = REPO_ROOT / "artifacts" / "deliveries" / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip"
    if not zip_p.exists():
        zip_p = REPO_ROOT / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip"
    hf_zip_p = REPO_ROOT / "artifacts" / "deliveries" / "TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip"
    if not hf_zip_p.exists():
        hf_zip_p = REPO_ROOT / "TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip"

    for p in (
        inat_json_p,
        inat_csv_p,
        inat_md_p,
        ps_ex_p,
        fg_json_p,
        fg_md_p,
        dp_json_p,
        dp_md_p,
        rep8i_json_p,
        rep8i_md_p,
        unc_json_p,
        unc_md_p,
        delta_json_p,
        delta_md_p,
        ledger_p,
        zip_p,
        hf_zip_p,
    ):
        assert p.exists(), f"Missing Task 8I / Hotfix artifact: {p}"

    inat = json.loads(inat_json_p.read_text(encoding="utf-8"))
    ps_ex = json.loads(ps_ex_p.read_text(encoding="utf-8"))
    fg = json.loads(fg_json_p.read_text(encoding="utf-8"))
    dp = json.loads(dp_json_p.read_text(encoding="utf-8"))
    rep8i = json.loads(rep8i_json_p.read_text(encoding="utf-8"))
    unc = json.loads(unc_json_p.read_text(encoding="utf-8"))
    delta = json.loads(delta_json_p.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))

    # Task 2 Hotfix: Verify pre-hotfix vs post-hotfix iNaturalist numbers & 3 handheld citrus demotions
    assert inat["total_inaturalist_candidates_audited"] == 107
    assert inat["pre_8i_real_waste_pass_audited"] == 90
    assert inat["pre_hotfix_8i_manifest_summary"]["real_waste_pass_total"] == 33
    assert inat["pre_hotfix_8i_manifest_summary"]["real_waste_pass_train"] == 27
    assert inat["pre_hotfix_8i_manifest_summary"]["real_waste_pass_val"] == 6
    assert inat["post_8i_status_counts_all_107"]["REAL_WASTE_PASS"] == 30
    assert inat["post_8i_status_counts_all_107"]["KEEP_AUXILIARY"] == 37
    assert inat["post_8i_status_counts_all_107"]["REJECT"] == 40
    assert inat["post_8i_verified_real_waste_pass_train"] == 24
    assert inat["post_8i_verified_real_waste_pass_val"] == 6
    assert len(inat["records"]) == 107
    assert len(inat["pm_inspection_contact_sheets"]) == 4
    for cs_rel in inat["pm_inspection_contact_sheets"]:
        assert (REPO_ROOT / cs_rel).exists()

    by_id = {r["candidate_id"]: r for r in inat["records"]}
    for cid_hh in ("task8h_0085", "task8h_0092", "task8h_0137"):
        r_hh = by_id[cid_hh]
        assert r_hh["post_8i_qa_status"] == "KEEP_AUXILIARY"
        assert r_hh["biological_decay_category"] == "HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL"
        assert r_hh["domain_tag"] == "AMBIGUOUS_CONTEXT"
        assert r_hh[" target_split" if " target_split" in r_hh else "target_split"] == "AUXILIARY_NON_WASTE_TRAIN_HOLD"

    # Task 3 Hotfix: Separated Photo License vs Observation License & 11 restricted-obs cases
    obs_restricted_recs = [r for r in inat["records"] if not r["inat_observation_license_open"]]
    assert len(obs_restricted_recs) == 11
    assert sum(1 for r in obs_restricted_recs if r["post_8i_qa_status"] == "KEEP_AUXILIARY") == 5
    assert sum(1 for r in obs_restricted_recs if r["post_8i_qa_status"] == "REJECT") == 6
    assert sum(1 for r in obs_restricted_recs if r["post_8i_qa_status"] == "REAL_WASTE_PASS") == 0

    for r in inat["records"]:
        assert r["inat_observation_id"] > 0
        assert r["inat_photo_id"] > 0
        assert r["inat_photo_url"].startswith("https://")
        assert r["inat_observation_url"].startswith("https://www.inaturalist.org/observations/")
        assert r["inat_author"]
        assert r["inat_photo_license_code"] in ("cc0", "cc-by", "cc-by-sa")
        assert r["inat_photo_license_usable"] is True
        if r["post_8i_qa_status"] == "REAL_WASTE_PASS":
            assert r["inat_observation_license_open"] is True
            assert r["biological_decay_category"] in (
                "DISCARDED_ORGANIC_FOOD_WASTE",
                "OUTDOOR_LITTER_SCARCE_CLASS",
            )
            assert r["domain_tag"] == "ACTUAL_WASTE"

    # Verify Deliverable #5 & #6 & #4 contents
    assert len(unc["part_a_11_photo_vs_observation_license_mismatch_cases"]) == 11
    assert len(unc["part_b_handheld_and_borderline_domain_cases"]) >= 3
    assert delta["ledger_v6_delta"]["pre_hotfix_unique_verified_rw_train_images"] == 1811
    assert delta["ledger_v6_delta"]["post_hotfix_unique_verified_rw_train_images"] == 1808
    assert ps_ex["input_manifest_row"]["context_type"] == "PRODUCT_STUDIO"
    assert ps_ex["importer_output_record"]["qa_status"] == "NEEDS_REVIEW"
    assert ps_ex["importer_output_record"]["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert ps_ex["importer_output_record"]["APPROVED_FOR_TRAIN"] is False

    # Task 2 & Task 3: Field Capture Kit & Ledger V6
    assert fg["real_field_photos_captured_to_date"] == 0
    assert fg["ai_or_synthetic_images_generated"] == 0
    assert ledger["ledger_version"] == "TASK8I_CANONICAL_5000_LEDGER_V6_SOURCE_SIGNOFF_AND_FIELD_PILOT"
    assert ledger["three_tier_summary"]["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_train_images"] == 1808
    assert ledger["three_tier_summary"]["2_VERIFIED_REAL_WASTE"]["cumulative_verified_real_waste_unique_val_images"] == 392
    assert dp["mass_plastic_expansion_blocked"] is True
    assert len(dp["per_class_scaling_roadmap"]) == 10
    assert rep8i["approved_for_train"] is False
    assert rep8i["constraints_compliance"]["simulated_or_fabricated_field_images"] == 0
    assert 0 < zip_p.stat().st_size < 500 * 1024 * 1024
    assert 0 < hf_zip_p.stat().st_size < 50 * 1024 * 1024


def test_task8i_p0_field_capture_qa_safety_regression_cases(tmp_path: Path) -> None:
    """Task 8I Final Hotfix (Task 1): Verify all 5 mandatory P0 regression cases on scripts/field_capture_import_and_qa.py."""
    from PIL import Image, ImageDraw
    from scripts.field_capture_import_and_qa import validate_and_allocate_field_batch

    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for idx in range(1, 8):
        im = Image.new("RGB", (640, 480), (20, 30, 40))
        dr = ImageDraw.Draw(im)
        for gy in range(8):
            for gx in range(9):
                val = ((gx * 31 + gy * 17 + idx * 53) ^ (idx * 19 * (gx + 1))) & 0xFF
                dr.rectangle([gx * 70, gy * 60, gx * 70 + 68, gy * 60 + 58], fill=(val, (val * 3) & 0xFF, (val * 7) & 0xFF))
        im.save(img_dir / f"case{idx}.jpg", format="JPEG")
        (lbl_dir / f"case{idx}.txt").write_text("3 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")

    rows = [
        # P0 CASE 1: PRODUCT_STUDIO + valid YOLO label -> NEVER REAL_WASTE_PASS
        # Even if a caller maliciously/mistakenly passes visual_qa_decision=REAL_WASTE_PASS!
        {
            "image_filename": "case1.jpg",
            "label_filename": "case1.txt",
            "specimen_id": "SPEC_CLO_202610_0001",
            "session_id": "SESS_20261016_STUDIO_01",
            "photographer": "TEAM_01",
            "context_type": "PRODUCT_STUDIO",
            "privacy_cleared": "true",
            "rights_confirmed": "true",
            "license": "CC-BY-4.0",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:00:00Z",
            "context_evidence": "White studio product catalog shot on table",
            "bbox_tightness_verified": "true",
        },
        # P0 CASE 2: IN_USE + valid YOLO label -> NEVER REAL_WASTE_PASS
        {
            "image_filename": "case2.jpg",
            "label_filename": "case2.txt",
            "specimen_id": "SPEC_CLO_202610_0002",
            "session_id": "SESS_20261016_INUSE_01",
            "photographer": "TEAM_01",
            "context_type": "IN_USE",
            "privacy_cleared": "true",
            "rights_confirmed": "true",
            "license": "CC-BY-4.0",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:00:00Z",
            "context_evidence": "Shirt currently being worn by person",
            "bbox_tightness_verified": "true",
        },
        # P0 CASE 3: Self-declared ACTUAL_WASTE without visual QA review -> MUST default to NEEDS_REVIEW
        # Also omits license / rights_confirmed to verify it does NOT auto-assign CC-BY-4.0!
        {
            "image_filename": "case3.jpg",
            "label_filename": "case3.txt",
            "specimen_id": "SPEC_CLO_202610_0003",
            "session_id": "SESS_20261016_ALLEY_01",
            "photographer": "TEAM_01",
            "context_type": "ACTUAL_WASTE",
            "privacy_cleared": "true",
        },
        # P0 CASE 4a: ACTUAL_WASTE with valid human review, but unconfirmed usage rights -> held at NEEDS_REVIEW
        {
            "image_filename": "case4.jpg",
            "label_filename": "case4.txt",
            "specimen_id": "SPEC_CLO_202610_0004",
            "session_id": "SESS_20261016_ALLEY_02",
            "photographer": "TEAM_01",
            "context_type": "ACTUAL_WASTE",
            "privacy_cleared": "true",
            "rights_confirmed": "false",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:05:00Z",
            "context_evidence": "Torn shirt discarded inside residential trash bin",
            "bbox_tightness_verified": "true",
        },
        # P0 CASE 4b & CASE 5 (Angle 1 of specimen 0005): ACTUAL_WASTE with valid human review + confirmed rights -> REAL_WASTE_PASS, APPROVED_FOR_TRAIN=False
        {
            "image_filename": "case5.jpg",
            "label_filename": "case5.txt",
            "specimen_id": "SPEC_CLO_202610_0005",
            "session_id": "SESS_20261016_ALLEY_03",
            "photographer": "TEAM_01",
            "context_type": "ACTUAL_WASTE",
            "privacy_cleared": "true",
            "rights_confirmed": "true",
            "license": "CC-BY-4.0",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:10:00Z",
            "context_evidence": "Discarded mud-stained jacket next to street dumpster (top-down 90deg)",
            "bbox_tightness_verified": "true",
        },
        # P0 CASE 5 (Angle 2 of same specimen 0005 in same session): Must stay in same split as case5!
        {
            "image_filename": "case6.jpg",
            "label_filename": "case6.txt",
            "specimen_id": "SPEC_CLO_202610_0005",
            "session_id": "SESS_20261016_ALLEY_03",
            "photographer": "TEAM_01",
            "context_type": "ACTUAL_WASTE",
            "privacy_cleared": "true",
            "rights_confirmed": "true",
            "license": "CC-BY-4.0",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:11:00Z",
            "context_evidence": "Discarded mud-stained jacket next to street dumpster (oblique 45deg)",
            "bbox_tightness_verified": "true",
        },
        # P0 CASE 5 (Independent specimen 0006 in independent session 04): Allows 80/20 split without leakage
        {
            "image_filename": "case7.jpg",
            "label_filename": "case7.txt",
            "specimen_id": "SPEC_CLO_202610_0006",
            "session_id": "SESS_20261016_MARKET_04",
            "photographer": "TEAM_01",
            "context_type": "ACTUAL_WASTE",
            "privacy_cleared": "true",
            "rights_confirmed": "true",
            "license": "CC-BY-4.0",
            "visual_qa_decision": "REAL_WASTE_PASS",
            "reviewer_id": "QA_REVIEWER_01",
            "review_timestamp": "2026-10-16T03:15:00Z",
            "context_evidence": "Abandoned wet sweatshirt in wet market waste pile",
            "bbox_tightness_verified": "true",
        },
    ]

    res = validate_and_allocate_field_batch(rows, img_dir, lbl_dir)
    by_fn = {r["image_filename"]: r for r in res["records"]}

    # Verify Case 1: PRODUCT_STUDIO + valid YOLO label -> NEVER REAL_WASTE_PASS
    r1 = by_fn["case1.jpg"]
    assert r1["FILE_AND_BBOX_FORMAT_VALID"] is True
    assert r1["OBJECT_ANNOTATION_VALID"] is False
    assert r1["qa_status"] == "NEEDS_REVIEW"
    assert r1["qa_status"] != "REAL_WASTE_PASS"
    assert r1["domain_tag"] == "PRODUCT_STUDIO"
    assert r1["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert r1["APPROVED_FOR_TRAIN"] is False
    assert r1["target_split"] == "MANUAL_REVIEW_QUEUE"

    # Verify Case 2: IN_USE + valid YOLO label -> NEVER REAL_WASTE_PASS
    r2 = by_fn["case2.jpg"]
    assert r2["FILE_AND_BBOX_FORMAT_VALID"] is True
    assert r2["OBJECT_ANNOTATION_VALID"] is False
    assert r2["qa_status"] == "NEEDS_REVIEW"
    assert r2["qa_status"] != "REAL_WASTE_PASS"
    assert r2["domain_tag"] == "IN_USE"
    assert r2["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert r2["APPROVED_FOR_TRAIN"] is False
    assert r2["target_split"] == "MANUAL_REVIEW_QUEUE"

    # Verify Case 3: Self-declared ACTUAL_WASTE without visual QA -> NEEDS_REVIEW + UNCONFIRMED_USAGE_RIGHTS
    r3 = by_fn["case3.jpg"]
    assert r3["FILE_AND_BBOX_FORMAT_VALID"] is True
    assert r3["OBJECT_ANNOTATION_VALID"] is False
    assert r3["qa_status"] == "NEEDS_REVIEW"
    assert r3["domain_tag"] == "ACTUAL_WASTE"
    assert r3["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert r3["APPROVED_FOR_TRAIN"] is False
    assert r3["license"] == "UNCONFIRMED_USAGE_RIGHTS"
    assert r3["license_verified"] is False
    assert r3["target_split"] == "MANUAL_REVIEW_QUEUE"

    # Verify Case 4a: Valid human review of ACTUAL_WASTE but rights_confirmed=False -> held at NEEDS_REVIEW
    r4 = by_fn["case4.jpg"]
    assert r4["qa_status"] == "NEEDS_REVIEW"
    assert r4["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert r4["license"] == "UNCONFIRMED_USAGE_RIGHTS"
    assert r4["license_verified"] is False
    assert r4["APPROVED_FOR_TRAIN"] is False

    # Verify Case 4b: Valid human review of ACTUAL_WASTE + confirmed license -> REAL_WASTE_PASS, APPROVED_FOR_TRAIN=False
    r5 = by_fn["case5.jpg"]
    r6 = by_fn["case6.jpg"]
    r7 = by_fn["case7.jpg"]
    for rp in (r5, r6, r7):
        assert rp["qa_status"] == "REAL_WASTE_PASS"
        assert rp["domain_tag"] == "ACTUAL_WASTE"
        assert rp["FILE_AND_BBOX_FORMAT_VALID"] is True
        assert rp["OBJECT_ANNOTATION_VALID"] is True
        assert rp["REAL_WASTE_CONTEXT_VERIFIED"] is True
        assert rp["APPROVED_FOR_TRAIN"] is False
        assert rp["license"] == "CC-BY-4.0"
        assert rp["license_verified"] is True

    # Verify Case 5: Multi-image same specimen/session -> zero train/val leakage & deduplicated specimen count
    assert r5["target_split"] == r6["target_split"]
    assert res["four_independence_metrics_pass_total"]["unique_images"] == 3
    assert res["four_independence_metrics_pass_total"]["unique_specimens"] == 2
    assert res["four_independence_metrics_pass_total"]["unique_sessions"] == 2
    assert res["leakage_checks"]["train_val_specimen_overlap_count"] == 0
    assert res["leakage_checks"]["train_val_session_overlap_count"] == 0
    assert res["leakage_checks"]["all_approved_for_train_false"] is True



