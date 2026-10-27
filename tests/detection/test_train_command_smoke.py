# Mô tả: Kiểm thử khói (smoke test) giao diện dòng lệnh huấn luyện phát hiện đa đối tượng (python -m src.detection.train).
"""
tests/detection/test_train_command_smoke.py
-------------------------------------------
Smoke tests for official training commands and CLI parsing in scripts/train_detection_v2.py:
1. --run argument is strictly required (missing --run causes exit code 2).
2. Unsupported or spoofed arguments cause exit code 2.
3. Valid CLI arguments are parsed correctly.
4. Preflight and dataset safety checks cannot be bypassed by raw CLI one-liners.
5. Zero multi-epoch training execution during tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

from scripts.train_detection_v2 import verify_training_dataset_safety

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "train_detection_v2.py"


def test_missing_run_argument_fails_smoke():
    """Verify that calling train_detection_v2 without --run fails with exit code 2."""
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "yolo", "--data", "configs/detection_dataset_real_v2.yaml"],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    assert proc.returncode == 2, f"Expected exit code 2 for missing --run, got {proc.returncode}"
    assert "required" in proc.stderr.lower() or "--run" in proc.stderr.lower()


def test_unsupported_flag_fails_smoke():
    """Verify that calling train_detection_v2 with unsupported flag fails with exit code 2."""
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "yolo", "--run", "E1_test", "--unsupported-flag-xyz"],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    assert proc.returncode == 2, f"Expected exit code 2 for unrecognized flag, got {proc.returncode}"
    assert "unrecognized arguments" in proc.stderr.lower()


def test_train_script_help_smoke():
    """Verify that --help exits with 0 and shows correct CLI interface."""
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    assert proc.returncode == 0
    assert "yolo" in proc.stdout
    assert "ssdlite" in proc.stdout
    assert "post" in proc.stdout
    assert "--run" in proc.stdout
    assert "--data" in proc.stdout


def test_verify_training_dataset_safety_blocks_test_leakage():
    """Verify verify_training_dataset_safety aborts if train split contains test image."""
    fake_ds = {
        "splits": {
            "train": {
                "images": [Path("images/real/taco_0068.jpg"), Path("images/real/normal_train.jpg")]
            }
        }
    }
    with pytest.raises(SystemExit) as exc_info:
        verify_training_dataset_safety(Path("configs/detection_dataset_real_v2.yaml"), fake_ds)
    assert "Split contamination" in str(exc_info.value)
    assert "taco_0068.jpg" in str(exc_info.value)


def test_verify_training_dataset_safety_blocks_unapproved_candidate():
    """Verify verify_training_dataset_safety aborts when candidate dataset quality gate is not TRAINING_ELIGIBLE."""
    fake_ds = {
        "splits": {
            "train": {
                "images": [Path("images/real/safe_image.jpg")]
            }
        }
    }
    # Targeting candidate YAML while pilot_quality_gates_report is TRAINING_BLOCKED
    with pytest.raises(SystemExit) as exc_info:
        verify_training_dataset_safety(Path("configs/detection_dataset_real_v3_candidate.yaml"), fake_ds)
    assert "TRAINING BLOCKED" in str(exc_info.value)


def test_resume_mode_checks_dataset_safety_smoke():
    """Verify that train_yolo and train_ssd enforce dataset safety even when resume=True."""
    from scripts.train_detection_v2 import train_yolo, train_ssd
    from unittest.mock import patch

    # Mock preflight guard so hardware check passes
    with patch("src.detection.preflight_guard.enforce_preflight_guard"):
        # Targeting a contaminated run directory
        fake_run_dir = ROOT / "artifacts" / "detection_v2" / "fake_test_run"
        fake_run_dir.mkdir(parents=True, exist_ok=True)
        (fake_run_dir / "weights").mkdir(exist_ok=True)
        (fake_run_dir / "weights" / "last.pt").write_text("fake_weights")
        (fake_run_dir / "run_metadata.json").write_text('{"class_names": ["waste"], "agnostic": true, "dataset_yaml": "configs/detection_dataset_real_v3_candidate.yaml"}')

        try:
            # resume=True on candidate dataset with BLOCKED quality gate MUST abort
            with pytest.raises(SystemExit) as exc_info:
                train_yolo("fake_test_run", None, False, {"device": "cpu"}, resume=True)
            assert "TRAINING BLOCKED" in str(exc_info.value)

            with pytest.raises(SystemExit) as exc_info_ssd:
                train_ssd("fake_test_run", None, False, {"device": "cpu"}, resume=True)
            assert "TRAINING BLOCKED" in str(exc_info_ssd.value)
        finally:
            import shutil
            shutil.rmtree(fake_run_dir, ignore_errors=True)


def test_user_controlled_approval_record_scope_and_image_hash(tmp_path):
    """
    P1 Regression Test:
    Verifies user-controlled approval record checks:
    1. Scope must match or authorize train export.
    2. Image SHA-256 is verified when images_dir is provided.
    """
    from src.detection.export_train_manifest import validate_image_eligibility

    test_img = tmp_path / "test_sample_01.jpg"
    test_img.write_bytes(b"sample_image_bytes_12345")
    test_lbl = tmp_path / "test_sample_01.txt"
    test_lbl.write_text("0 0.5 0.5 0.2 0.2\n")

    import hashlib
    actual_img_sha = hashlib.sha256(test_img.read_bytes()).hexdigest()
    actual_lbl_sha = hashlib.sha256(test_lbl.read_bytes()).hexdigest()

    fn = "test_sample_01.jpg"
    img_meta = {
        "filename": fn,
        "category": "NEW_CANDIDATE",
        "qa_approval_status": "PENDING_USER_APPROVAL"
    }

    # 1. Invalid scope -> REJECTED
    unauthorized_scope_ledger = {
        fn: {
            "decision": "APPROVED",
            "scope": "INSPECTION_ONLY",   # unauthorized for train export
            "approved_by": "authorized_user_lead",
            "label_sha256": actual_lbl_sha
        }
    }
    is_ok, reason, err = validate_image_eligibility(
        img_meta,
        approval_ledger=unauthorized_scope_ledger,
        labels_dir=tmp_path,
        required_scope="TRAIN_EXPANSION"
    )
    assert not is_ok
    assert err == "INVALID_APPROVAL_RECORD"
    assert "scope" in reason

    # 2. Mismatched image SHA -> REJECTED
    bad_img_sha_ledger = {
        fn: {
            "decision": "APPROVED",
            "scope": "TRAIN_EXPANSION",
            "approved_by": "authorized_user_lead",
            "label_sha256": actual_lbl_sha,
            "image_sha256": "0" * 64   # bad hash
        }
    }
    is_ok_img, reason_img, err_img = validate_image_eligibility(
        img_meta,
        approval_ledger=bad_img_sha_ledger,
        labels_dir=tmp_path,
        images_dir=tmp_path
    )
    assert not is_ok_img
    assert err_img == "INVALID_APPROVAL_RECORD"
    assert "image SHA-256 mismatch" in reason_img

    # 3. Valid user-controlled record with matching hashes and scope -> SUCCEEDS
    valid_ledger = {
        fn: {
            "decision": "APPROVED",
            "scope": "TRAIN_EXPANSION",
            "approved_by": "authorized_user_lead",
            "label_sha256": actual_lbl_sha,
            "image_sha256": actual_img_sha
        }
    }
    is_ok_valid, reason_valid, err_valid = validate_image_eligibility(
        img_meta,
        approval_ledger=valid_ledger,
        labels_dir=tmp_path,
        images_dir=tmp_path,
        required_scope="TRAIN_EXPANSION"
    )
    assert is_ok_valid
    assert err_valid is None

    # 4. Self-declared approver ("agent" / "ai" / "system") -> REJECTED
    for forbidden in ("agent", "ai", "system", "anonymous", "bot"):
        self_declared_ledger = {
            fn: {
                "decision": "APPROVED",
                "scope": "TRAIN_EXPANSION",
                "approved_by": forbidden,
                "label_sha256": actual_lbl_sha,
                "image_sha256": actual_img_sha
            }
        }
        is_ok_self, reason_self, err_self = validate_image_eligibility(
            img_meta,
            approval_ledger=self_declared_ledger,
            labels_dir=tmp_path,
            images_dir=tmp_path,
            required_scope="TRAIN_EXPANSION"
        )
        assert not is_ok_self
        assert err_self == "INVALID_APPROVAL_RECORD"
        assert "approver identity" in reason_self

    # 5. Missing scope on new candidate -> REJECTED
    missing_scope_ledger = {
        fn: {
            "decision": "APPROVED",
            "approved_by": "authorized_user_lead",
            "label_sha256": actual_lbl_sha,
            "image_sha256": actual_img_sha
        }
    }
    is_ok_noscope, reason_noscope, err_noscope = validate_image_eligibility(
        img_meta,
        approval_ledger=missing_scope_ledger,
        labels_dir=tmp_path,
        images_dir=tmp_path,
        required_scope="TRAIN_EXPANSION"
    )
    assert not is_ok_noscope
    assert err_noscope == "INVALID_APPROVAL_RECORD"
    assert "scope" in reason_noscope

    # 6. Missing image_sha256 on new candidate -> REJECTED
    missing_img_sha_ledger = {
        fn: {
            "decision": "APPROVED",
            "scope": "TRAIN_EXPANSION",
            "approved_by": "authorized_user_lead",
            "label_sha256": actual_lbl_sha
        }
    }
    is_ok_noimgsha, reason_noimgsha, err_noimgsha = validate_image_eligibility(
        img_meta,
        approval_ledger=missing_img_sha_ledger,
        labels_dir=tmp_path,
        images_dir=tmp_path,
        required_scope="TRAIN_EXPANSION"
    )
    assert not is_ok_noimgsha
    assert err_noimgsha == "INVALID_APPROVAL_RECORD"
    assert "image_sha256" in reason_noimgsha

    # 7. Reviewer Bug Repro: images_dir is None for new candidate -> REJECTED
    is_ok_noimgdir, reason_noimgdir, err_noimgdir = validate_image_eligibility(
        img_meta,
        approval_ledger=valid_ledger,
        labels_dir=tmp_path,
        images_dir=None,  # images_dir omitted
        required_scope="TRAIN_EXPANSION"
    )
    assert not is_ok_noimgdir
    assert err_noimgdir == "MISSING_IMAGE_VERIFICATION"
    assert "images_dir was not provided" in reason_noimgdir

    from src.detection.export_train_manifest import filter_manifest_for_train, MissingImageVerificationError
    with pytest.raises(MissingImageVerificationError):
        filter_manifest_for_train(
            {"images": [img_meta]},
            approval_ledger=valid_ledger,
            labels_dir=tmp_path,
            images_dir=None,
            required_scope="TRAIN_EXPANSION",
            raise_on_violation=True
        )

    # 8. Reviewer Bug Repro: Altering image bytes after approval -> REJECTED
    # Modify image bytes on disk
    test_img.write_bytes(b"tampered_modified_image_bytes_xyz999")
    is_ok_tampered, reason_tampered, err_tampered = validate_image_eligibility(
        img_meta,
        approval_ledger=valid_ledger,
        labels_dir=tmp_path,
        images_dir=tmp_path,
        required_scope="TRAIN_EXPANSION"
    )
    assert not is_ok_tampered
    assert err_tampered == "INVALID_APPROVAL_RECORD"
    assert "image SHA-256 mismatch" in reason_tampered


def test_candidate_manifest_unverified_demotion():
    """Verify that candidate acquisition manifest v3 isolates active shortlist from 35 quarantined items."""
    import json
    manifest_path = ROOT / "data" / "acquisition_staging" / "manifests" / "candidate_acquisition_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_metadata"]["current_stage"] == "ACQUISITION_TARGET"
    assert data["manifest_metadata"]["total_source_identified_images"] == 3
    assert data["manifest_metadata"]["total_blocked_access_requests"] == 2
    assert data["manifest_metadata"]["total_downloaded_images"] == 0
    assert len(data["rejected_unverified_history"]) == 35

    # Quarantined history must not be used for downloading
    for cand in data["rejected_unverified_history"]:
        assert cand["lifecycle_status"] == "REJECTED_UNVERIFIED_HISTORY"
        assert cand["original_url"] == "REJECTED_DO_NOT_DOWNLOAD"
        assert cand["downloaded_bytes_on_disk"] is None

    # Active shortlist must have exactly 5 items (3 OpenImages, 2 Roboflow blocked)
    assert len(data["active_shortlist"]) == 5
    for cand in data["active_shortlist"]:
        assert cand["downloaded_bytes_on_disk"] is None


def test_verified_source_proof_5_samples():
    """Verify that 5 authentic source records are established with real provenance and zero synthetic IDs."""
    import json
    proof_path = ROOT / "data" / "acquisition_staging" / "manifests" / "verified_source_proof_5_samples.json"
    assert proof_path.is_file()
    data = json.loads(proof_path.read_text(encoding="utf-8"))
    assert len(data["samples"]) == 5

    # 2 Roboflow blocked
    rf_samples = [s for s in data["samples"] if "Roboflow" in s["source_dataset"]]
    assert len(rf_samples) == 2
    for s in rf_samples:
        assert s["access_status"] == "BLOCKED_REQUIRES_ACCESS"
        assert s["source_image_id"] == "BLOCKED_REQUIRES_ACCESS"
        assert s["substitute_id_created"] is False
        assert "403" in s["url_verification_status"]

    # 2 OpenImages biological
    bio_samples = [s for s in data["samples"] if s["target_category"] == "biological"]
    assert len(bio_samples) == 2
    for s in bio_samples:
        assert len(s["source_image_id"]) == 16
        assert len(s["raw_csv_rows"]) > 0
        assert s["human_annotations_count"] == 0
        assert "licenses/by" in s["openimages_metadata"]["License"].lower()
        assert s["url_verification_status"].startswith("VERIFIED_HTTP_200")
        assert s["image_disk_status"].startswith("NOT_DOWNLOADED")

    # 1 OpenImages hard-negative
    neg_sample = next(s for s in data["samples"] if s["target_category"] == "hard_negatives")
    assert len(neg_sample["source_image_id"]) == 16
    assert neg_sample["waste_annotations_count"] == 0
    assert neg_sample["human_annotations_count"] == 0
    assert "Building" in neg_sample["classes_and_boxes"][0]["class_name"]
    assert neg_sample["url_verification_status"].startswith("VERIFIED_HTTP_200")

