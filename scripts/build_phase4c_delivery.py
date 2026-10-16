"""
scripts/build_phase4c_delivery.py
---------------------------------
Packages the definitive delivery archive for Phase 4C Acceptance:
~/Downloads/PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip

Includes:
1. 15 pilot images and labels
2. Original source metadata & candidate expansion catalog
3. Pilot manifests & specification
4. 8 negative review images + composite contact sheet + JSON audit
5. 2 pairs of biological before/after overlays + labels + audit evidence
6. TACO COCO category/annotation audit JSON
7. Forensic Battery reconciliation report
8. Split contamination & Data novelty reports
9. Code changes & tests
10. Raw pytest execution logs
11. Complete SHA-256 manifest of all items
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = (Path.home() / "Downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DEST = DOWNLOADS_DIR / "PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip"
MANIFEST_JSON = ROOT / "reports" / "data_v2" / "phase4c_delivery_sha256_manifest.json"

FILES_TO_INCLUDE = [
    # Master reports
    ROOT / "reports" / "PHASE4C_REAL_DATA_PILOT.md",
    ROOT / "reports" / "PHASE4C_CLOSEOUT_AND_DATA_ACQUISITION.md",
    ROOT / "reports" / "PHASE4D_SAFETY_RELEASE_AND_VERIFIED_SOURCES.md",

    # Source, Acquisition & Audit Catalogs
    ROOT / "reports" / "data_v2" / "real_data_acquisition_plan.md",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.json",
    ROOT / "reports" / "data_v2" / "candidate_expansion_sources.md",
    ROOT / "reports" / "data_v2" / "taco_exact_categories_audit.json",
    ROOT / "reports" / "data_v2" / "pilot_quality_gates_report.json",
    ROOT / "reports" / "data_v2" / "pilot_train_eligible_manifest.json",
    ROOT / "reports" / "data_v2" / "phash_leakage_audit_report.json",

    # Forensic & Acceptance Reports
    ROOT / "reports" / "data_v2" / "split_contamination_report.json",
    ROOT / "reports" / "data_v2" / "split_contamination_report.md",
    ROOT / "reports" / "data_v2" / "data_novelty_report.json",
    ROOT / "reports" / "data_v2" / "data_novelty_report.md",
    ROOT / "reports" / "data_v2" / "battery_label_forensic_report.json",
    ROOT / "reports" / "data_v2" / "battery_label_forensic_report.md",

    # Code
    ROOT / "src" / "detection" / "export_train_manifest.py",
    ROOT / "src" / "detection" / "preflight_guard.py",
    ROOT / "src" / "detection" / "ssdlite_train.py",
    ROOT / "scripts" / "train_detection_v2.py",
    ROOT / "scripts" / "generate_phash_contact_sheets.py",
    ROOT / "scripts" / "generate_biological_reannotation_overlays.py",
    ROOT / "scripts" / "audit_hard_negatives_deep.py",
    ROOT / "scripts" / "generate_split_contamination_report.py",
    ROOT / "scripts" / "generate_data_novelty_report.py",
    ROOT / "scripts" / "generate_battery_forensic_report.py",
    ROOT / "scripts" / "build_pilot_dataset_v3.py",
    ROOT / "scripts" / "validate_pilot_quality_gates.py",

    # Tests
    ROOT / "tests" / "detection" / "test_preflight_guard_integration.py",
    ROOT / "tests" / "qa" / "test_pilot_dataset_v3.py",
    ROOT / "tests" / "qa" / "test_split_contamination_safety.py",

    # Pytest execution log
    ROOT / "reports" / "data_v2" / "pytest_targeted.log",
]

DIRS_TO_INCLUDE = [
    # Contact sheets (9 Hamming pairs JPGs + json)
    (ROOT / "reports" / "data_v2" / "contact_sheets", "reports/data_v2/contact_sheets"),

    # Biological overlays (Before/After comparison JPGs + json + md)
    (ROOT / "reports" / "data_v2" / "biological_overlays", "reports/data_v2/biological_overlays"),

    # Negative review (8 individual audit JPGs + composite contact sheet + json)
    (ROOT / "reports" / "data_v2" / "negative_review", "reports/data_v2/negative_review"),

    # Complete Pilot dataset (15 images, 15 labels, manifests)
    (ROOT / "data" / "detection_pilot_v3", "data/detection_pilot_v3"),

    # Acquisition staging (structure ready for approved downloads)
    (ROOT / "data" / "acquisition_staging", "data/acquisition_staging"),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== PACKAGING PHASE 4C ACCEPTANCE DELIVERY ARCHIVE ===")
    print(f"Destination: {ZIP_DEST}")

    # Build manifest
    manifest_entries = []

    # Discrete files
    for f in FILES_TO_INCLUDE:
        if f.exists():
            arcname = f.relative_to(ROOT).as_posix()
            manifest_entries.append({
                "path": arcname,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f)
            })
        else:
            print(f"Warning: Missing discrete file {f}")

    # Directory files
    for dpath, darcname in DIRS_TO_INCLUDE:
        if dpath.exists():
            for p in sorted(dpath.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(dpath).as_posix()
                    arcname = f"{darcname}/{rel}"
                    manifest_entries.append({
                        "path": arcname,
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p)
                    })

    # Save manifest JSON
    with open(MANIFEST_JSON, "w", encoding="utf-8") as mf:
        json.dump({
            "delivery_package": "PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip",
            "generated_at_utc": "2026-10-12T02:00:00Z",
            "total_items": len(manifest_entries),
            "manifest": manifest_entries
        }, mf, indent=2, ensure_ascii=False)
    print(f"Saved SHA-256 manifest to: {MANIFEST_JSON}")

    # Write ZIP
    count = 0
    with zipfile.ZipFile(ZIP_DEST, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include manifest inside zip as well
        zf.write(MANIFEST_JSON, "reports/data_v2/phase4c_delivery_sha256_manifest.json")
        count += 1

        for item in manifest_entries:
            disk_path = ROOT / Path(item["path"])
            zf.write(disk_path, item["path"])
            count += 1

    size_mb = ZIP_DEST.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully created deliverable ZIP with {count} items.")
    print(f"Archive Size: {size_mb:.2f} MB")
    print(f"Archive Location: {ZIP_DEST}")

if __name__ == "__main__":
    main()
