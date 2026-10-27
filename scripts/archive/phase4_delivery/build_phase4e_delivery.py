"""
scripts/build_phase4e_delivery.py
---------------------------------
Packages all Phase 4E manifests, ingestion scripts, staging scripts,
evidence logs, and reports into a compact delivery ZIP archive:
~/Downloads/PHASE4E_REAL_WASTE_DATA_EVIDENCE.zip
"""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ZIP = (Path.home() / "Downloads" / "PHASE4E_REAL_WASTE_DATA_EVIDENCE.zip")

FILES_TO_PACK = [
    # Catalog Manifests
    ROOT / "data" / "acquisition_staging" / "manifests" / "phase4e_real_waste_catalog.json",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "preview_3_manifest.json",
    # Preview 3 Images & Labels
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_078821d86db99fc9.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_74a44994e3d6b962.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_e29db1c1963628e7.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_078821d86db99fc9.txt",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_74a44994e3d6b962.txt",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_e29db1c1963628e7.txt",
    # Overlays & Contact Sheet
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_078821d86db99fc9.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_74a44994e3d6b962.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_e29db1c1963628e7.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "contact_sheet" / "contact_sheet_preview_3.jpg",
    # Ingestion & Staging Scripts
    ROOT / "scripts" / "stage_whitelisted_pilot_images.py",
    ROOT / "scripts" / "build_phase4e_candidate_manifest.py",
    ROOT / "scripts" / "ingest_roboflow_battery_export.py",
    ROOT / "scripts" / "stage_phase4e_whitelist_candidates.py",
    # Comprehensive Reports
    ROOT / "reports" / "PHASE4E_REAL_WASTE_DATA.md",
    ROOT / "reports" / "PILOT_PREVIEW_VISUAL_QA_REPORT.md",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    print("=" * 70)
    print("BUILDING DELIVERABLE PACKAGE: PHASE4E_REAL_WASTE_DATA_EVIDENCE.zip")
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
    print(f"ZIP Size:        {zip_size:,} bytes ({zip_size / 1024:.2f} KB)")
    print(f"ZIP SHA-256:     {zip_sha}")
    print("=" * 70)


if __name__ == "__main__":
    main()
