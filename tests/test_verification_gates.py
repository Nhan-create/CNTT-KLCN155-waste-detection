r"""Automated Test Suite for Verification Gates (Part 1 Acceptance Hardening).

Tests:
1. Missing image folder must cause check_split_leakage to FAIL (exit code 1, FAILED_DISK_IMAGE_VERIFICATION).
2. Unresolved candidate in decision table must cause check_split_leakage to FAIL (exit code 1, UNRESOLVED_CANDIDATES_PRESENT, unresolved_count > 0).
3. Legitimate dataset and decisions must PASS check_split_leakage (exit code 0, VERIFIED_NO_LEAKAGE_WITHIN_HASH_SCOPE).
4. Direct disk image hashing in reproduce_validation must match manifest 100% and reproduce baseline metrics (exit code 0).

Portable:
- Dynamically resolves PROJECT_ROOT relative to this test file.
- Accepts configurable external image directory via --processed-dir and --val-dir.
- Works seamlessly in extracted standalone package folders.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Dynamic path resolution based on file location
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    res = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.returncode, res.stdout, res.stderr


def test_missing_directory_fails():
    print("\n[TEST 1] Testing missing image folder handling...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent_dir = Path(tmp_dir) / "non_existent_image_folder_test"
        tmp_out = Path(tmp_dir) / "out"
        cmd = [
            PYTHON_EXE,
            "scripts/check_split_leakage.py",
            "--processed-dir", str(non_existent_dir),
            "--output-dir", str(tmp_out),
        ]
        ret, stdout, stderr = run_cmd(cmd)
        print(f"  Exit code: {ret}")
        assert ret == 1, f"Expected exit code 1 for missing dir, got {ret}\nStdout: {stdout}\nStderr: {stderr}"

        report_file = tmp_out / "leakage_audit_report.json"
        assert report_file.exists(), "Report JSON not created"
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["audit_status"] == "FAILED_DISK_IMAGE_VERIFICATION", (
            f"Expected status FAILED_DISK_IMAGE_VERIFICATION, got {report['audit_status']}"
        )
        assert report["disk_verification"]["passed"] is False, "disk_verification.passed should be False"
        assert report["disk_verification"]["directory_exists"] is False, "directory_exists should be False"
        print("  -> TEST 1 PASSED: Missing folder correctly flagged as FAILED_DISK_IMAGE_VERIFICATION with exit code 1!")


def test_unresolved_entry_in_decision_table_fails():
    print("\n[TEST 2] Testing unresolved candidate detection in decision table...")
    orig_dec_table = PROJECT_ROOT / "data" / "audit" / "visual_audit_decision_table.csv"
    assert orig_dec_table.exists(), f"Original decision table not found at: {orig_dec_table}"

    df = pd.read_csv(orig_dec_table)
    # Modify the first row to have status UNRESOLVED
    df.loc[0, "status"] = "UNRESOLVED"
    df.loc[0, "relationship"] = "UNRESOLVED_REQUIRES_INSPECTION"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dec_csv = Path(tmp_dir) / "test_decision_table.csv"
        df.to_csv(tmp_dec_csv, index=False)
        tmp_out = Path(tmp_dir) / "out"

        cmd = [
            PYTHON_EXE,
            "scripts/check_split_leakage.py",
            "--decision-table", str(tmp_dec_csv),
            "--output-dir", str(tmp_out),
            "--skip-disk-hash",
        ]
        ret, stdout, stderr = run_cmd(cmd)
        print(f"  Exit code: {ret}")
        assert ret == 1, f"Expected exit code 1 for unresolved candidate, got {ret}\nStdout: {stdout}\nStderr: {stderr}"

        report_file = tmp_out / "leakage_audit_report.json"
        assert report_file.exists(), "Report JSON not created"
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)

        unresolved = report["phash_cross_split_candidates"]["unresolved_candidates"]
        print(f"  Unresolved count detected: {unresolved}")
        assert unresolved >= 1, f"Expected unresolved >= 1, got {unresolved}"
        assert report["audit_status"] == "UNRESOLVED_CANDIDATES_PRESENT", (
            f"Expected status UNRESOLVED_CANDIDATES_PRESENT, got {report['audit_status']}"
        )
        print("  -> TEST 2 PASSED: Unresolved candidate correctly flagged with exit code 1 and unresolved_count > 0!")


def test_real_audit_passes(processed_dir: Path, skip_real: bool):
    print("\n[TEST 3] Running full leakage audit on official dataset...")
    if not processed_dir.exists():
        if skip_real:
            print(f"  SKIPPED: Image directory not found at {processed_dir} and --skip-real-images flag set.")
            return
        raise FileNotFoundError(
            f"Processed image directory not found at: {processed_dir}.\n"
            "Pass --processed-dir <path> pointing to external image folder, or --skip-real-images if running metadata-only checks."
        )

    cmd = [
        PYTHON_EXE,
        "scripts/check_split_leakage.py",
        "--processed-dir", str(processed_dir),
    ]
    ret, stdout, stderr = run_cmd(cmd)
    print(f"  Exit code: {ret}")
    assert ret == 0, f"Expected exit code 0 for official audit, got {ret}\nStderr: {stderr}\nStdout: {stdout}"

    report_file = PROJECT_ROOT / "data" / "audit" / "leakage_audit_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["audit_status"] == "VERIFIED_NO_LEAKAGE_WITHIN_HASH_SCOPE", (
        f"Unexpected status: {report['audit_status']}"
    )
    assert report["disk_verification"]["passed"] is True, "disk_verification.passed must be True"
    assert report["disk_verification"]["missing_files"] == 0, "missing files must be 0"
    assert report["disk_verification"]["sha256_mismatches"] == 0, "hash mismatches must be 0"
    assert report["exact_sha256_overlap"]["total_collisions"] == 0, "collisions must be 0"
    assert report["phash_cross_split_candidates"]["unresolved_candidates"] == 0, "unresolved must be 0"
    assert report["phash_cross_split_candidates"]["confirmed_same_object_leakages"] == 0, "leakages must be 0"
    print("  -> TEST 3 PASSED: Full official audit passed 100% with exit code 0 and all checks verified!")


def test_reproduce_validation_direct_disk_hash(val_dir: Path, skip_real: bool):
    print("\n[TEST 4] Running validation reproduction with direct disk byte hashing...")
    if not val_dir.exists():
        if skip_real:
            print(f"  SKIPPED: Validation image directory not found at {val_dir} and --skip-real-images flag set.")
            return
        raise FileNotFoundError(
            f"Validation image directory not found at: {val_dir}.\n"
            "Pass --val-dir <path> pointing to external validation images, or --skip-real-images if running metadata-only checks."
        )

    cmd = [
        PYTHON_EXE,
        "scripts/reproduce_validation.py",
        "--val-dir", str(val_dir),
    ]
    ret, stdout, stderr = run_cmd(cmd)
    print(f"  Exit code: {ret}")
    assert ret == 0, f"Expected exit code 0 for validation reproduction, got {ret}\nStderr: {stderr}\nStdout: {stdout}"

    summary_file = PROJECT_ROOT / "artifacts" / "official_run" / "reproduced_validation_summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "REPRODUCED_MATCH_VERIFIED"
    assert summary["disk_hash_verification"]["passed"] is True
    assert summary["disk_hash_verification"]["total_samples_hashed_from_disk"] == 2223
    assert summary["disk_hash_verification"]["mismatches_against_manifest"] == 0
    assert summary["disk_hash_verification"]["missing_from_manifest"] == 0
    assert summary["metrics"]["correct_predictions"] == 2138
    assert summary["metrics"]["total_errors"] == 85
    assert abs(summary["metrics"]["accuracy"] - 0.9618) < 1e-3

    # Check predictions CSV
    pred_csv = PROJECT_ROOT / "artifacts" / "official_run" / "val_predictions.csv"
    df_pred = pd.read_csv(pred_csv)
    assert len(df_pred) == 2223
    assert df_pred["sha256"].str.len().eq(64).all(), "All SHA-256 strings must be valid 64-character hex"
    print("  -> TEST 4 PASSED: Validation reproduced with direct disk byte hashing, 0 mismatches, 96.18% accuracy!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Verification Gates Test Suite.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to processed image directory (default: auto-detected from local or D:/CNTT-KLCN155-waste-detection)",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Path to validation image directory (default: processed_dir/val)",
    )
    parser.add_argument(
        "--skip-real-images",
        action="store_true",
        help="Skip Tests 3 & 4 if external real image dataset is not present on this machine",
    )
    args = parser.parse_args()

    # Resolve processed directory
    if args.processed_dir is not None:
        processed_dir = args.processed_dir
    elif (PROJECT_ROOT / "data" / "processed_v2").exists():
        processed_dir = PROJECT_ROOT / "data" / "processed_v2"
    elif Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2").exists():
        processed_dir = Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2")
    else:
        processed_dir = PROJECT_ROOT / "data" / "processed_v2"

    # Resolve validation directory
    if args.val_dir is not None:
        val_dir = args.val_dir
    else:
        val_dir = processed_dir / "val"

    print("=" * 80)
    print("RUNNING VERIFICATION GATES TEST SUITE")
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Python Runtime:  {PYTHON_EXE}")
    print(f"Processed Dir:   {processed_dir} (exists: {processed_dir.exists()})")
    print(f"Validation Dir:  {val_dir} (exists: {val_dir.exists()})")
    print(f"Skip Real Imgs:  {args.skip_real_images}")
    print("=" * 80)

    test_missing_directory_fails()
    test_unresolved_entry_in_decision_table_fails()
    test_real_audit_passes(processed_dir, args.skip_real_images)
    test_reproduce_validation_direct_disk_hash(val_dir, args.skip_real_images)

    print("\n" + "=" * 80)
    print("ALL VERIFICATION GATE TESTS PASSED CONVINCINGLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
