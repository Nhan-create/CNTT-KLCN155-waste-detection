"""
scripts/build_phase4e2_delivery.py
-----------------------------------
Packages all Phase 4E.2 scripts, intake safety tests, raw execution logs,
catalogs, and reports into a delivery ZIP archive:
C:\\Users\\ad\\Downloads\\PHASE4E2_FINAL_INGEST_AND_MINI_PILOT_EVIDENCE.zip
"""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = (Path.home() / "Downloads" / "PHASE4E2_FINAL_INGEST_AND_MINI_PILOT_EVIDENCE.zip")

FILES_TO_PACK = [
    # Catalog Manifests
    ROOT / "data" / "acquisition_staging" / "manifests" / "phase4e_real_waste_catalog.json",
    # Fixed Ingestion & Staging Scripts
    ROOT / "scripts" / "ingest_roboflow_battery_export.py",
    ROOT / "scripts" / "stage_phase4e_whitelist_candidates.py",
    # Safety Tests
    ROOT / "tests" / "qa" / "test_phase4e_intake_safety.py",
    # Raw Test Logs
    ROOT / "reports" / "data_v2" / "phase4e2_raw_test.log",
    # Comprehensive Reports
    ROOT / "reports" / "PHASE4E2_FINAL_INGEST_FIX_AND_MINI_PILOT.md",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    print("=" * 70)
    print("BUILDING DELIVERABLE PACKAGE: PHASE4E2_FINAL_INGEST_AND_MINI_PILOT_EVIDENCE.zip")
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
