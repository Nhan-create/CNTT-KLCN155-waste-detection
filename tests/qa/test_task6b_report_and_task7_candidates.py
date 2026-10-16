"""
Automated QA & Consistency Tests for AI-UPGRADE-01:
- Part A: Task 6B E3 Evidence Correction (Holdout metrics, V3 per-class table, training_log reconciliation, Recall by threshold)
- Part B: Task 7 Diverse Battery Data Acquisition, Full-Object Annotations, Zero-Leakage Audit & V4/E4 Proposal (DO NOT RUN)
"""
import hashlib
import json
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_DIR = REPO_ROOT / "artifacts" / "detection_v3_pilot"
TASK7_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_part_a_task6b_corrected_report_consistency():
    report_path = REPO_ROOT / "TASK6_E3_CORRECTED_EVALUATION_REPORT.md"
    audit_path = V3_DIR / "task6b_report_consistency_audit.json"
    errata_path = V3_DIR / "E3_yolov8n_pilot" / "training_log_errata_and_reconciliation.json"
    eval_json_path = V3_DIR / "evaluation_e1_vs_e3" / "e1_vs_e3_comprehensive_evaluation.json"
    preflight_json_path = V3_DIR / "dataset_v3_materialization_and_preflight_report.json"

    assert report_path.exists(), "Missing TASK6_E3_CORRECTED_EVALUATION_REPORT.md"
    assert audit_path.exists(), "Missing task6b_report_consistency_audit.json"
    assert errata_path.exists(), "Missing training_log_errata_and_reconciliation.json"

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["all_passed"] is True
    assert audit["total_checks"] == 22
    assert all(c["passed"] for c in audit["checks"])

    report_text = report_path.read_text(encoding="utf-8")
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_json_path.read_text(encoding="utf-8"))

    # A1: Holdout metrics exact match
    evalD = eval_data["evaluation_D_holdout_27"]
    h512_e1 = evalD["E1_imgsz512"]
    h640_e1 = evalD["E1_imgsz640"]
    h512_e3 = evalD["E3_imgsz512"]

    assert f"{h512_e1['ap']['mAP50']:.8f}" in report_text  # 0.05913945
    assert f"{h512_e1['ap']['mAP50_95']:.8f}" in report_text  # 0.03790507
    assert f"{h640_e1['ap']['mAP50']:.8f}" in report_text  # 0.05710398
    assert f"{h640_e1['ap']['mAP50_95']:.8f}" in report_text  # 0.02890579
    assert f"{h512_e3['ap']['mAP50']:.8f}" in report_text  # 0.00822891
    assert f"{h512_e3['ap']['mAP50_95']:.8f}" in report_text  # 0.00288265

    # A2: Per-class dataset v3 counts match preflight JSON
    splits = preflight["measured_on_disk_splits"]
    assert sum(splits["train"]["per_class_boxes"].values()) == 4074
    assert sum(splits["val"]["per_class_boxes"].values()) == 930
    assert sum(splits["test"]["per_class_boxes"].values()) == 720
    assert splits["train"]["per_class_boxes"]["battery"] == 112
    assert splits["val"]["per_class_boxes"]["battery"] == 29
    assert splits["test"]["per_class_boxes"]["battery"] == 1
    assert "| **`0: battery`** | `1` | **`112`** (`+111`) | `112` | `0` | **`29`** (`+29`) | `29` | `1` | `1` |" in report_text
    assert "| **`7: plastic`** | `1560` | `1560` | `1560` | `382` | `382` | `382` | `274` | `274` |" in report_text

    # A3: Training log untouched & both SHA-256s documented
    raw_log = (V3_DIR / "E3_yolov8n_pilot" / "training_log.txt").read_text(encoding="utf-8")
    assert "Epoch 23/25" in raw_log
    assert "91efc77df60344a648a4090f384d588e8d27ecf1edc01c11ebdef1dba3ec03e5" in report_text
    assert "e451cb8d0a0faab9374f5dac90811bba681b2827d51aba07f6e6a120b5364169" in report_text

    # A4: Recall by threshold explicitly separated
    assert "0.96139" in report_text
    assert "0.86207" in report_text
    assert "25 / 29" in report_text or "25/29" in report_text


def test_frozen_checkpoints_and_production_default_untouched():
    e1_best = REPO_ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
    e3_best = V3_DIR / "E3_yolov8n_pilot" / "weights" / "best.pt"
    settings_py = REPO_ROOT / "src" / "web" / "detection_settings.py"

    assert _sha256(e1_best) == "7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8"
    assert _sha256(e3_best) == "e451cb8d0a0faab9374f5dac90811bba681b2827d51aba07f6e6a120b5364169"

    settings_text = settings_py.read_text(encoding="utf-8")
    assert "E3_yolov8n_pilot" not in settings_text
    assert not (REPO_ROOT / "artifacts" / "detection_v4").exists()


def test_part_b_task7_candidates_labels_and_zero_leakage():
    qa_manifest_p = TASK7_DIR / "task7_qa_manifest.json"
    leakage_p = TASK7_DIR / "task7_leakage_report.json"
    proposal_p = TASK7_DIR / "task7_dataset_v4_and_e4_proposal.json"
    e4_yaml_p = REPO_ROOT / "configs" / "training_yolov8n_v4_proposed_DO_NOT_RUN.yaml"
    report_p = REPO_ROOT / "TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md"

    for p in (qa_manifest_p, leakage_p, proposal_p, e4_yaml_p, report_p):
        assert p.exists(), f"Missing required Task 7 deliverable: {p}"

    qa = json.loads(qa_manifest_p.read_text(encoding="utf-8"))
    summary = qa["summary_counts"]
    assert qa["total_downloaded_candidates"] == 56
    assert summary["qa_pass_total_images"] == 30
    assert summary["qa_pass_positive_battery_images"] == 26
    assert summary["qa_pass_hard_negative_images"] == 4
    assert summary["qa_pass_total_battery_boxes"] == 61
    assert summary["qa_pass_multi_battery_images_ge2_boxes"] == 14
    assert summary["proposed_train_v4_images"] == 19
    assert summary["proposed_train_v4_battery_boxes"] == 40
    assert summary["proposed_val_real_battery_images"] == 11
    assert summary["proposed_val_real_battery_boxes"] == 21
    assert summary["needs_manual_review_images"] == 10
    assert summary["reject_images"] == 16
    assert summary["approved_for_train"] == 0

    # Verify all 56 images, overlays, and QA_PASS YOLO labels on disk
    for item in qa["items"]:
        img_p = REPO_ROOT / item["local_path"]
        ov_p = REPO_ROOT / item["overlay_path"]
        assert img_p.exists(), f"Missing image {img_p}"
        assert ov_p.exists(), f"Missing overlay {ov_p}"
        assert _sha256(img_p) == item["sha256"]
        with Image.open(img_p) as im:
            w, h = im.size
            assert (w, h) == (item["actual_width"], item["actual_height"])
        assert item["approved_for_train"] is False

        if item["qa_verdict"] == "QA_PASS":
            lbl_p = REPO_ROOT / item["label_path"]
            assert lbl_p.exists(), f"Missing QA_PASS label {lbl_p}"
            assert lbl_p.parent.name == "labels"
            lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(lines) == item["boxes_count"]
            for ln in lines:
                parts = ln.split()
                assert len(parts) == 5
                assert parts[0] == "0"
                xc, yc, bw, bh = map(float, parts[1:])
                assert 0.0 < xc < 1.0 and 0.0 < yc < 1.0
                assert 0.04 <= bw <= 1.0 and 0.04 <= bh <= 1.0
        elif item["qa_verdict"] == "NEEDS_MANUAL_REVIEW":
            lbl_p = REPO_ROOT / item["label_path"]
            assert lbl_p.exists()
            assert lbl_p.parent.name == "labels_needs_manual_review"
        else:
            assert item["label_path"] is None

    # Verify contact sheets
    for cs_rel in qa["contact_sheets"]:
        cs_p = REPO_ROOT / cs_rel
        assert cs_p.exists() and cs_p.stat().st_size > 50_000

    # Verify zero leakage
    leak = json.loads(leakage_p.read_text(encoding="utf-8"))
    assert leak["zero_external_leakage"] is True
    assert leak["zero_cross_split_leakage"] is True
    assert len(leak["exact_sha256_collisions_with_v2_v3_holdout"]) == 0
    assert len(leak["near_phash_le8_collisions_with_v2_v3_holdout"]) == 0
    assert len(leak["cross_split_train_vs_val_real_battery_collisions"]) == 0


def test_task7_final_qa_patch_task7_30_and_task7_53_regression():
    qa_manifest_p = TASK7_DIR / "task7_qa_manifest.json"
    qa = json.loads(qa_manifest_p.read_text(encoding="utf-8"))

    assert "qa_reviewer" in qa and qa["qa_reviewer"]
    assert "qa_patch_history" in qa and len(qa["qa_patch_history"]) == 2

    by_key = {item["short_key"]: item for item in qa["items"]}
    assert len(by_key) == 56  # No duplicate candidate keys

    t30 = by_key["task7_30"]
    t53 = by_key["task7_53"]

    # 1. Verify task7_30 has 6 full-object boxes including protruding positive terminals
    assert t30["qa_verdict"] == "QA_PASS"
    assert t30["boxes_count"] == 6
    assert len(t30["boxes_xyxy_norm"]) == 6
    assert t30["boxes_xyxy_norm"] == [
        [0.073, 0.075, 0.232, 0.710],
        [0.242, 0.158, 0.343, 0.706],
        [0.360, 0.162, 0.453, 0.704],
        [0.383, 0.492, 0.747, 0.912],
        [0.546, 0.563, 0.838, 0.918],
        [0.675, 0.570, 0.948, 0.918],
    ]
    assert t30["qa_patch_note"] and t30["qa_reviewer"] and t30["qa_reviewed_at_utc"]
    assert t30["source_page_url"] and t30["author"] and t30["license"]
    assert t30["approved_for_train"] is False

    # 2. Verify task7_53 has 4 full-object boxes enclosing upper-right slanted ends of Boxes #3 and #4
    assert t53["qa_verdict"] == "QA_PASS"
    assert t53["boxes_count"] == 4
    assert len(t53["boxes_xyxy_norm"]) == 4
    assert t53["boxes_xyxy_norm"] == [
        [0.353, 0.122, 0.478, 0.638],
        [0.476, 0.095, 0.592, 0.592],
        [0.258, 0.530, 0.678, 0.842],
        [0.450, 0.560, 0.868, 0.882],
    ]
    assert t53["qa_patch_note"] and t53["qa_reviewer"] and t53["qa_reviewed_at_utc"]
    assert t53["source_page_url"] and t53["author"] and t53["license"]
    assert t53["approved_for_train"] is False

    # 3. Verify BEFORE and AFTER overlays exist in qa_patch_before_after/
    patch_dir = TASK7_DIR / "qa_patch_before_after"
    for sk in ("task7_30", "task7_53"):
        assert (patch_dir / f"{sk}_BEFORE_overlay.jpg").exists()
        assert (patch_dir / f"{sk}_AFTER_overlay.jpg").exists()
        assert (patch_dir / f"{sk}_BEFORE_vs_AFTER_comparison.jpg").exists()
        assert _sha256(patch_dir / f"{sk}_BEFORE_overlay.jpg") != _sha256(patch_dir / f"{sk}_AFTER_overlay.jpg")

