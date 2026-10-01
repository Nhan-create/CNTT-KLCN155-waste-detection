r"""Package Part 1 Deliverables into ZIP file.

Creates CNTT-KLCN155_PART01_DELIVERY.zip in C:\Users\ad\Downloads\
and copies to D:\CNTT-KLCN155-waste-detection\.
Verifies integrity and computes SHA-256.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("D:/CNTT-KLCN155-waste-detection")
DOWNLOADS_DIR = Path("C:/Users/ad/Downloads")
OUTPUT_ZIP_DOWNLOADS = DOWNLOADS_DIR / "CNTT-KLCN155_PART01_DELIVERY.zip"
OUTPUT_ZIP_PROJECT = PROJECT_ROOT / "CNTT-KLCN155_PART01_DELIVERY.zip"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PACKAGING PART 1 DELIVERABLES (CNTT-KLCN155)")
    print("=" * 80)

    # Directories and files to include
    include_dirs = [
        "docs/plan",
        "scripts",
        "data/audit",
        "data/metadata",
        "artifacts/official_run",
        "artifacts/smoke_test_v2",
        "configs",
    ]
    include_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
    ]

    # Temporary staging or direct write to zip
    if OUTPUT_ZIP_DOWNLOADS.exists():
        OUTPUT_ZIP_DOWNLOADS.unlink()

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP_DOWNLOADS, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add individual files
        for rel_file in include_files:
            p = PROJECT_ROOT / rel_file
            if p.exists():
                arcname = f"CNTT-KLCN155_PART01_DELIVERY/{rel_file}"
                zf.write(p, arcname)
                total_files += 1
                total_uncompressed_bytes += p.stat().st_size
                print(f"Added file: {rel_file} ({p.stat().st_size:,} bytes)")

        # Add directories
        for rel_dir in include_dirs:
            p_dir = PROJECT_ROOT / rel_dir
            if not p_dir.exists():
                print(f"Warning: Directory not found: {rel_dir}")
                continue
            for item in sorted(p_dir.rglob("*")):
                if item.is_file():
                    # Skip __pycache__ or temporary files
                    if "__pycache__" in item.parts or item.suffix in [".pyc", ".pyo"]:
                        continue
                    arcname = f"CNTT-KLCN155_PART01_DELIVERY/{item.relative_to(PROJECT_ROOT).as_posix()}"
                    zf.write(item, arcname)
                    total_files += 1
                    total_uncompressed_bytes += item.stat().st_size
            print(f"Added directory: {rel_dir}/")

    print("\nVerifying ZIP integrity...")
    with zipfile.ZipFile(OUTPUT_ZIP_DOWNLOADS, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupt file in zip: {bad_file}")
            return 1
        namelist = zf.namelist()
        print(f"Integrity check PASSED! Total files in archive: {len(namelist)}")

    # Copy to project root as well
    shutil.copy2(OUTPUT_ZIP_DOWNLOADS, OUTPUT_ZIP_PROJECT)

    # Compute SHA-256
    sha256_dl = compute_sha256(OUTPUT_ZIP_DOWNLOADS)
    sha256_prj = compute_sha256(OUTPUT_ZIP_PROJECT)
    assert sha256_dl == sha256_prj, "Checksum mismatch between copies!"

    size_mb = OUTPUT_ZIP_DOWNLOADS.stat().st_size / (1024 ** 2)
    uncomp_mb = total_uncompressed_bytes / (1024 ** 2)

    print("\nDelivery Package Summary:")
    print(f"  - Primary Path: {OUTPUT_ZIP_DOWNLOADS}")
    print(f"  - Backup Path:  {OUTPUT_ZIP_PROJECT}")
    print(f"  - Total Files:  {total_files}")
    print(f"  - Uncompressed: {uncomp_mb:.2f} MB")
    print(f"  - Archive Size: {size_mb:.2f} MB")
    print(f"  - SHA-256:      {sha256_dl}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
