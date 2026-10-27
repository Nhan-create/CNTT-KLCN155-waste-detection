"""
scripts/build_phase4d2_correction_delivery.py
---------------------------------------------
Builds the small Phase 4D.2 Correction deliverable package:
~/Downloads/PHASE4D2_CORRECTION_REAL_SOURCE_PROOF_EVIDENCE.zip

Contents:
1. reports/REAL_IMAGE_SOURCE_PROOF_5_SAMPLES.md
2. data/acquisition_staging/manifests/verified_source_proof_5_samples.json
3. data/acquisition_staging/manifests/candidate_acquisition_manifest.json
4. src/detection/export_train_manifest.py
5. tests/detection/test_train_command_smoke.py
6. tests/qa/test_split_contamination_safety.py
7. tests/qa/test_pilot_dataset_v3.py
8. reports/data_v2/phase4d2_correction_raw_test.log
9. reports/data_v2/phase4d2_correction_delivery_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DEST = DOWNLOADS_DIR / "PHASE4D2_CORRECTION_REAL_SOURCE_PROOF_EVIDENCE.zip"
MANIFEST_JSON = ROOT / "reports" / "data_v2" / "phase4d2_correction_delivery_manifest.json"

FILES_TO_INCLUDE = [
    ROOT / "reports" / "FINAL_IMAGE_PROOF_AND_ACQUISITION_STATUS.md",
    ROOT / "reports" / "REAL_IMAGE_SOURCE_PROOF_5_SAMPLES.md",
    ROOT / "data" / "acquisition_staging" / "manifests" / "verified_source_proof_5_samples.json",
    ROOT / "data" / "acquisition_staging" / "manifests" / "candidate_acquisition_manifest.json",
    ROOT / "src" / "detection" / "export_train_manifest.py",
    ROOT / "scripts" / "stage_whitelisted_pilot_images.py",
    ROOT / "scripts" / "prepare_acquisition_pilot.py",
    ROOT / "tests" / "detection" / "test_train_command_smoke.py",
    ROOT / "tests" / "qa" / "test_split_contamination_safety.py",
    ROOT / "tests" / "qa" / "test_pilot_dataset_v3.py",
    ROOT / "reports" / "data_v2" / "phase4d2_correction_raw_test.log",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("Building Phase 4D.2 Correction deliverable package...")
    missing = [p for p in FILES_TO_INCLUDE if not p.is_file()]
    if missing:
        print("[ERROR] Missing required files for delivery:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)

    manifest_records = []
    for p in FILES_TO_INCLUDE:
        rel = p.relative_to(ROOT).as_posix()
        file_hash = sha256_file(p)
        file_size = p.stat().st_size
        manifest_records.append({
            "relative_path": rel,
            "sha256": file_hash,
            "size_bytes": file_size
        })

    manifest_payload = {
        "phase": "4D.2_CORRECTION",
        "title": "Phase 4D.2 Correction: 5 Proven Real Image Sources Evidence",
        "total_files": len(manifest_records),
        "manifest_version": "1.0.0-proven-5-samples",
        "staged_downloaded_images": 0,
        "staged_downloaded_bytes": 0,
        "files": manifest_records
    }
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created manifest at {MANIFEST_JSON}")

    if ZIP_DEST.exists():
        ZIP_DEST.unlink()

    with zipfile.ZipFile(ZIP_DEST, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in FILES_TO_INCLUDE:
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname=arcname)
        zf.write(MANIFEST_JSON, arcname=MANIFEST_JSON.relative_to(ROOT).as_posix())

    zip_hash = sha256_file(ZIP_DEST)
    zip_size_kb = ZIP_DEST.stat().st_size / 1024

    print("=" * 70)
    print("PHASE 4D.2 CORRECTION DELIVERABLE PACKAGE READY:")
    print(f"Destination: {ZIP_DEST}")
    print(f"Size:        {zip_size_kb:.2f} KB ({ZIP_DEST.stat().st_size:,} bytes)")
    print(f"Files Count: {len(FILES_TO_INCLUDE) + 1}")
    print(f"SHA-256:     {zip_hash}")
    print("=" * 70)


if __name__ == "__main__":
    main()
