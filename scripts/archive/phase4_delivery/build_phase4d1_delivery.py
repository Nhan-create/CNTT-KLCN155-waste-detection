# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Đóng gói hồ sơ minh chứng, manifest và tệp nén bàn giao cho `build_phase4d1_delivery`.
"""
scripts/build_phase4d1_delivery.py
----------------------------------
Builds the concise Phase 4D.1 deliverable archive:
~/Downloads/PHASE4D1_FINAL_SAFETY_PATCH_EVIDENCE.zip

Contents:
1. Modified source files (export_train_manifest, preflight_guard, ssdlite_train, train_detection_v2).
2. Regression tests (test_split_contamination_safety, test_preflight_guard_integration, test_train_command_smoke).
3. Raw test logs (pytest_targeted.log).
4. Evidence of three bypasses eliminated.
5. Evidence of preflight non-bypassability.
6. Verified dataset sources and catalogs.
7. Acquisition manifest for 35 pilot images in data/acquisition_staging/.
8. Phase 4D.1 master report with user approval actions.
9. Cryptographic SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DEST = DOWNLOADS_DIR / "PHASE4D1_FINAL_SAFETY_PATCH_EVIDENCE.zip"
MANIFEST_JSON = ROOT / "reports" / "data_v2" / "phase4d1_delivery_sha256_manifest.json"

FILES_TO_INCLUDE = [
    # Master Report
    ROOT / "reports" / "PHASE4D1_FINAL_SAFETY_PATCH_AND_EXECUTION_READINESS.md",
    ROOT / "reports" / "PHASE4D_SAFETY_RELEASE_AND_VERIFIED_SOURCES.md",

    # Source code
    ROOT / "src" / "detection" / "export_train_manifest.py",
    ROOT / "src" / "detection" / "preflight_guard.py",
    ROOT / "src" / "detection" / "ssdlite_train.py",
    ROOT / "scripts" / "train_detection_v2.py",
    ROOT / "scripts" / "prepare_acquisition_pilot.py",
    ROOT / "scripts" / "validate_pilot_quality_gates.py",

    # Regression tests
    ROOT / "tests" / "qa" / "test_split_contamination_safety.py",
    ROOT / "tests" / "qa" / "test_pilot_dataset_v3.py",
    ROOT / "tests" / "detection" / "test_preflight_guard_integration.py",
    ROOT / "tests" / "detection" / "test_train_command_smoke.py",

    # Raw test logs
    ROOT / "reports" / "data_v2" / "pytest_targeted.log",

    # Verified Catalogs & Manifests
    ROOT / "reports" / "data_v2" / "real_data_acquisition_plan.md",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.json",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.md",
    ROOT / "reports" / "data_v2" / "pilot_quality_gates_report.json",
    ROOT / "reports" / "data_v2" / "pilot_train_eligible_manifest.json",
    ROOT / "data" / "acquisition_staging" / "manifests" / "candidate_acquisition_manifest.json",
    ROOT / "data" / "acquisition_staging" / "README.md",
]

DIRS_TO_INCLUDE = [
    (ROOT / "data" / "acquisition_staging", "data/acquisition_staging"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== PACKAGING PHASE 4D.1 DELIVERABLE ARCHIVE ===")
    print(f"Destination: {ZIP_DEST}")

    manifest_entries = []

    for f in FILES_TO_INCLUDE:
        if f.exists():
            arcname = f.relative_to(ROOT).as_posix()
            manifest_entries.append({
                "path": arcname,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f)
            })
        else:
            print(f"Warning: Missing file {f}")

    for dpath, darcname in DIRS_TO_INCLUDE:
        if dpath.exists():
            for p in sorted(dpath.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(dpath).as_posix()
                    arcname = f"{darcname}/{rel}"
                    # Avoid duplicates
                    if not any(m["path"] == arcname for m in manifest_entries):
                        manifest_entries.append({
                            "path": arcname,
                            "size_bytes": p.stat().st_size,
                            "sha256": sha256_file(p)
                        })

    with open(MANIFEST_JSON, "w", encoding="utf-8") as mf:
        json.dump({
            "delivery_package": "PHASE4D1_FINAL_SAFETY_PATCH_EVIDENCE.zip",
            "phase": "4D.1",
            "total_items": len(manifest_entries),
            "manifest": manifest_entries
        }, mf, indent=2, ensure_ascii=False)
    print(f"Saved SHA-256 manifest to: {MANIFEST_JSON}")

    count = 0
    with zipfile.ZipFile(ZIP_DEST, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MANIFEST_JSON, "reports/data_v2/phase4d1_delivery_sha256_manifest.json")
        count += 1

        for item in manifest_entries:
            disk_path = ROOT / Path(item["path"])
            zf.write(disk_path, item["path"])
            count += 1

    size_kb = ZIP_DEST.stat().st_size / 1024
    print(f"\nSuccessfully created deliverable ZIP with {count} items.")
    print(f"Archive Size: {size_kb:.2f} KB")
    print(f"Archive Location: {ZIP_DEST}")


if __name__ == "__main__":
    main()
