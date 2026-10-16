"""
scripts/build_phase4e_final_patch_delivery.py
---------------------------------------------
Packages all Phase 4E Final Patch artifacts, regression tests, raw logs,
manifests, downloaded images, contact sheet, overlays, and reports into:
~/Downloads/PHASE4E_FINAL_PATCH_AND_MINI_PILOT_EVIDENCE.zip
"""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = (Path.home() / "Downloads" / "PHASE4E_FINAL_PATCH_AND_MINI_PILOT_EVIDENCE.zip")

FILES_TO_PACK = [
    # Catalog & QA Manifests
    ROOT / "data" / "acquisition_staging" / "manifests" / "phase4e_real_waste_catalog.json",
    ROOT / "data" / "acquisition_staging" / "manifests" / "mini_pilot_qa_manifest.json",
    ROOT / "data" / "acquisition_staging" / "manifests" / "train_expansion_candidate_manifest.json",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "mini_pilot_staging_summary.json",
    # Downloaded Real Images
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_street_food_remains.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_neg_junction_boxes.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_neg_utility_box_leeds.jpg",
    # Contact Sheet
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "contact_sheet" / "mini_pilot_contact_sheet.jpg",
    # Visual QA Overlays
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "overlays" / "overlay_wiki_bio_banana_peel_ground.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "overlays" / "overlay_wiki_bio_street_food_remains.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "overlays" / "overlay_wiki_neg_junction_boxes.jpg",
    ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "overlays" / "overlay_wiki_neg_utility_box_leeds.jpg",
    # Ingestion, Staging & Overlay Scripts
    ROOT / "scripts" / "ingest_roboflow_battery_export.py",
    ROOT / "scripts" / "stage_phase4e_whitelist_candidates.py",
    ROOT / "scripts" / "generate_mini_pilot_qa_overlays.py",
    ROOT / "scripts" / "build_phase4e_final_patch_delivery.py",
    # Safety Regression Tests
    ROOT / "tests" / "qa" / "test_phase4e_intake_safety.py",
    # Raw Test Logs
    ROOT / "reports" / "data_v2" / "phase4e4_raw_test.log",
    # Comprehensive Reports
    ROOT / "reports" / "PHASE4E_FINAL_PATCH_AND_MINI_PILOT.md",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    print("=" * 70)
    print("BUILDING DELIVERABLE: PHASE4E_FINAL_PATCH_AND_MINI_PILOT_EVIDENCE.zip")
    print("=" * 70)

    for p in FILES_TO_PACK:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        h = sha256_file(p)
        print(f"Verified [{h[:10]}...] {p.relative_to(ROOT)}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in FILES_TO_PACK:
            arcname = str(p.relative_to(ROOT)).replace("\\", "/")
            zf.write(p, arcname=arcname)

    zip_size = OUTPUT_ZIP.stat().st_size
    zip_sha = sha256_file(OUTPUT_ZIP)

    print("=" * 70)
    print(f"Package created: {OUTPUT_ZIP}")
    print(f"Total files:     {len(FILES_TO_PACK)}")
    print(f"ZIP Size:        {zip_size:,} bytes ({zip_size / (1024 * 1024):.2f} MB)")
    print(f"ZIP SHA-256:     {zip_sha}")
    print("=" * 70)


if __name__ == "__main__":
    main()
