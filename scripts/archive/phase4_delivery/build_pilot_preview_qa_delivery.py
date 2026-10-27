"""
scripts/build_pilot_preview_qa_delivery.py
-------------------------------------------
Builds a compact deliverable ZIP package containing the 3 preview images,
their bounding box overlays, contact sheet, manifest, raw log, and QA report.
"""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ZIP = (Path.home() / "Downloads" / "PILOT_PREVIEW_3_VISUAL_QA_EVIDENCE.zip")

FILES_TO_PACK = [
    # 3 Real Downloaded Images
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_078821d86db99fc9.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_74a44994e3d6b962.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "images" / "oi_e29db1c1963628e7.jpg",
    # 3 Proposed Labels
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_078821d86db99fc9.txt",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_74a44994e3d6b962.txt",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "labels" / "oi_e29db1c1963628e7.txt",
    # 3 Visual Bounding Box Overlays
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_078821d86db99fc9.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_74a44994e3d6b962.jpg",
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "overlays" / "overlay_oi_e29db1c1963628e7.jpg",
    # Contact Sheet
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "contact_sheet" / "contact_sheet_preview_3.jpg",
    # Manifest
    ROOT / "data" / "acquisition_staging" / "pilot_preview_3" / "preview_3_manifest.json",
    # Execution Scripts
    ROOT / "scripts" / "stage_whitelisted_pilot_images.py",
    # QA Report & Logs
    ROOT / "reports" / "PILOT_PREVIEW_VISUAL_QA_REPORT.md",
    ROOT / "reports" / "data_v2" / "pilot_preview_download_and_qa.log",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    print("=" * 70)
    print("BUILDING DELIVERABLE PACKAGE: PILOT_PREVIEW_3_VISUAL_QA_EVIDENCE.zip")
    print("=" * 70)

    for p in FILES_TO_PACK:
        if not p.exists():
            raise FileNotFoundError(f"Missing required deliverable file: {p}")
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
