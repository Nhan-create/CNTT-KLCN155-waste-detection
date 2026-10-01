r"""Package Part 1 Verified Deliverables (P1-R1).

Creates:
1. CNTT-KLCN155_PART01_VERIFIED_R1.zip in C:\Users\ad\Downloads\ and project root.
2. CNTT-KLCN155_PART01_VERIFIED_R1_manifest.csv (file-by-file SHA-256 and size).
Verifies zip integrity and computes top-level archive SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("D:/CNTT-KLCN155-waste-detection")
DOWNLOADS_DIR = Path("C:/Users/ad/Downloads")

ZIP_NAME = "CNTT-KLCN155_PART01_VERIFIED_R1.zip"
MANIFEST_NAME = "CNTT-KLCN155_PART01_VERIFIED_R1_manifest.csv"

OUTPUT_ZIP_DOWNLOADS = DOWNLOADS_DIR / ZIP_NAME
OUTPUT_ZIP_PROJECT = PROJECT_ROOT / ZIP_NAME
OUTPUT_MANIFEST_DOWNLOADS = DOWNLOADS_DIR / MANIFEST_NAME
OUTPUT_MANIFEST_PROJECT = PROJECT_ROOT / MANIFEST_NAME


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("PACKAGING VERIFIED PART 1 DELIVERABLES (P1-R1)")
    print("=" * 80)

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

    if OUTPUT_ZIP_DOWNLOADS.exists():
        OUTPUT_ZIP_DOWNLOADS.unlink()

    files_manifest = []
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP_DOWNLOADS, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Add individual files
        for rel_file in include_files:
            p = PROJECT_ROOT / rel_file
            if p.exists():
                arcname = f"CNTT-KLCN155_PART01_VERIFIED_R1/{rel_file}"
                zf.write(p, arcname)
                f_size = p.stat().st_size
                f_sha = compute_sha256(p)
                total_files += 1
                total_uncompressed_bytes += f_size
                files_manifest.append({
                    "relative_path": arcname,
                    "size_bytes": f_size,
                    "sha256": f_sha,
                })
                print(f"Added file: {rel_file} ({f_size:,} bytes)")

        # 2. Add directories
        for rel_dir in include_dirs:
            p_dir = PROJECT_ROOT / rel_dir
            if not p_dir.exists():
                print(f"Warning: Directory not found: {rel_dir}")
                continue
            for item in sorted(p_dir.rglob("*")):
                if item.is_file():
                    if "__pycache__" in item.parts or item.suffix in [".pyc", ".pyo"]:
                        continue
                    arcname = f"CNTT-KLCN155_PART01_VERIFIED_R1/{item.relative_to(PROJECT_ROOT).as_posix()}"
                    zf.write(item, arcname)
                    f_size = item.stat().st_size
                    f_sha = compute_sha256(item)
                    total_files += 1
                    total_uncompressed_bytes += f_size
                    files_manifest.append({
                        "relative_path": arcname,
                        "size_bytes": f_size,
                        "sha256": f_sha,
                    })
            print(f"Added directory: {rel_dir}/")

    print("\nVerifying ZIP archive integrity (CRC check)...")
    with zipfile.ZipFile(OUTPUT_ZIP_DOWNLOADS, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupt file in zip: {bad_file}")
            return 1
        namelist = zf.namelist()
        print(f"Integrity check PASSED! Total files in archive: {len(namelist)}")

    # Copy zip to project root
    shutil.copy2(OUTPUT_ZIP_DOWNLOADS, OUTPUT_ZIP_PROJECT)

    # Export file-by-file manifest CSV
    with open(OUTPUT_MANIFEST_DOWNLOADS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(files_manifest)
    shutil.copy2(OUTPUT_MANIFEST_DOWNLOADS, OUTPUT_MANIFEST_PROJECT)
    print(f"Saved file-by-file manifest to: {OUTPUT_MANIFEST_DOWNLOADS}")

    # Top-level zip hash
    sha256_dl = compute_sha256(OUTPUT_ZIP_DOWNLOADS)
    sha256_prj = compute_sha256(OUTPUT_ZIP_PROJECT)
    assert sha256_dl == sha256_prj, "Checksum mismatch between copies!"

    size_mb = OUTPUT_ZIP_DOWNLOADS.stat().st_size / (1024 ** 2)
    uncomp_mb = total_uncompressed_bytes / (1024 ** 2)

    print("\n" + "=" * 80)
    print("DELIVERY PACKAGE P1-R1 SUMMARY:")
    print(f"  - Primary ZIP Path:    {OUTPUT_ZIP_DOWNLOADS}")
    print(f"  - Backup ZIP Path:     {OUTPUT_ZIP_PROJECT}")
    print(f"  - Manifest CSV Path:   {OUTPUT_MANIFEST_DOWNLOADS}")
    print(f"  - Total Archived Files: {total_files}")
    print(f"  - Uncompressed Size:   {uncomp_mb:.2f} MB")
    print(f"  - Compressed Size:     {size_mb:.2f} MB")
    print(f"  - ZIP Archive SHA-256: {sha256_dl}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
