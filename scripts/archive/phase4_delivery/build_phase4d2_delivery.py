# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Đóng gói hồ sơ minh chứng, manifest và tệp nén bàn giao cho `build_phase4d2_delivery`.
"""
scripts/build_phase4d2_delivery.py
----------------------------------
Builds the concise Phase 4D.2 deliverable archive:
~/Downloads/PHASE4D2_REAL_DATA_SHORTLIST_EVIDENCE.zip

Contents:
1. Master report detailing 35 real image shortlist, source catalog, and answers to 3 questions.
2. Verified expansion catalog with Roboflow waste-battery-only/1 URL and analysis.
3. Candidate acquisition manifest v2.0.0 with 35 real source-identified entries.
4. Source code with resume-safety, cryptographic hash binding, and scope-enforced ledger.
5. Downloader script with fail-closed user confirmation flag.
6. Comprehensive test suite and raw pytest_targeted.log (247 passed, 1 skipped).
7. Cryptographic SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DEST = DOWNLOADS_DIR / "PHASE4D2_REAL_DATA_SHORTLIST_EVIDENCE.zip"
MANIFEST_JSON = ROOT / "reports" / "data_v2" / "phase4d2_delivery_sha256_manifest.json"

FILES_TO_INCLUDE = [
    # Master Reports
    ROOT / "reports" / "PHASE4D2_REAL_IMAGE_SHORTLIST.md",
    ROOT / "reports" / "PHASE4D1_FINAL_SAFETY_PATCH_AND_EXECUTION_READINESS.md",

    # Source code with Phase 4D.2 engineering patches
    ROOT / "src" / "detection" / "export_train_manifest.py",
    ROOT / "scripts" / "train_detection_v2.py",
    ROOT / "scripts" / "prepare_acquisition_pilot.py",
    ROOT / "src" / "detection" / "preflight_guard.py",
    ROOT / "src" / "detection" / "ssdlite_train.py",
    ROOT / "scripts" / "validate_pilot_quality_gates.py",

    # Regression and QA tests
    ROOT / "tests" / "detection" / "test_train_command_smoke.py",
    ROOT / "tests" / "qa" / "test_split_contamination_safety.py",
    ROOT / "tests" / "qa" / "test_pilot_dataset_v3.py",
    ROOT / "tests" / "detection" / "test_preflight_guard_integration.py",

    # Raw test logs
    ROOT / "reports" / "data_v2" / "pytest_targeted.log",

    # Catalogs, Plans and Manifests
    ROOT / "reports" / "data_v2" / "real_data_acquisition_plan.md",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.json",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.md",
    ROOT / "reports" / "data_v2" / "pilot_quality_gates_report.json",
    ROOT / "reports" / "data_v2" / "pilot_train_eligible_manifest.json",
    ROOT / "data" / "acquisition_staging" / "manifests" / "candidate_acquisition_manifest.json",
    ROOT / "data" / "acquisition_staging" / "README.md",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Building Phase 4D.2 delivery package...")
    missing = [p for p in FILES_TO_INCLUDE if not p.is_file()]
    if missing:
        print("[ERROR] Missing required files for delivery:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)

    manifest_records = []

    # First compute individual hashes
    for p in FILES_TO_INCLUDE:
        rel = p.relative_to(ROOT).as_posix()
        file_hash = sha256_file(p)
        file_size = p.stat().st_size
        manifest_records.append({
            "relative_path": rel,
            "sha256": file_hash,
            "size_bytes": file_size
        })

    # Write JSON manifest
    manifest_payload = {
        "phase": "4D.2",
        "title": "Phase 4D.2 Real Data Shortlist & Pre-Acquisition Safety Evidence",
        "total_files": len(manifest_records),
        "manifest_version": "2.0.0-real-source-identified",
        "staged_downloaded_images": 0,
        "staged_downloaded_bytes": 0,
        "files": manifest_records
    }
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"Created manifest at {MANIFEST_JSON}")

    # Build ZIP archive
    if ZIP_DEST.exists():
        ZIP_DEST.unlink()

    with zipfile.ZipFile(ZIP_DEST, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in FILES_TO_INCLUDE:
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname=arcname)
        zf.write(MANIFEST_JSON, arcname=MANIFEST_JSON.relative_to(ROOT).as_posix())

    zip_hash = sha256_file(ZIP_DEST)
    zip_size_mb = ZIP_DEST.stat().st_size / (1024 * 1024)

    print("=" * 70)
    print(f"PHASE 4D.2 DELIVERABLE PACKAGE READY:")
    print(f"Destination: {ZIP_DEST}")
    print(f"Size:        {zip_size_mb:.2f} MB ({ZIP_DEST.stat().st_size:,} bytes)")
    print(f"Files Count: {len(FILES_TO_INCLUDE) + 1}")
    print(f"SHA-256:     {zip_hash}")
    print("=" * 70)


if __name__ == "__main__":
    main()
