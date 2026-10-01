r"""Package Part 1 Verified Deliverables (P1-R2 Hardened Gates).

Creates:
1. CNTT-KLCN155_PART01_VERIFIED_R2.zip in C:\Users\ad\Downloads\ and project root.
2. CNTT-KLCN155_PART01_VERIFIED_R2_manifest.csv (file-by-file SHA-256 and size).
3. Updates CNTT-KLCN155_PART01_VERIFIED_R1.zip for backward compatibility.
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

ZIP_NAME_R2 = "CNTT-KLCN155_PART01_VERIFIED_R2.zip"
MANIFEST_NAME_R2 = "CNTT-KLCN155_PART01_VERIFIED_R2_manifest.csv"
ZIP_NAME_R1 = "CNTT-KLCN155_PART01_VERIFIED_R1.zip"
MANIFEST_NAME_R1 = "CNTT-KLCN155_PART01_VERIFIED_R1_manifest.csv"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_package(zip_name: str, manifest_name: str, root_arcname: str) -> dict:
    out_zip_dl = DOWNLOADS_DIR / zip_name
    out_zip_prj = PROJECT_ROOT / zip_name
    out_man_dl = DOWNLOADS_DIR / manifest_name
    out_man_prj = PROJECT_ROOT / manifest_name

    include_dirs = [
        "docs/plan",
        "scripts",
        "tests",
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

    if out_zip_dl.exists():
        out_zip_dl.unlink()

    files_manifest = []
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(out_zip_dl, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Add individual files
        for rel_file in include_files:
            p = PROJECT_ROOT / rel_file
            if p.exists():
                arcname = f"{root_arcname}/{rel_file}"
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
                    arcname = f"{root_arcname}/{item.relative_to(PROJECT_ROOT).as_posix()}"
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

    # Test integrity
    with zipfile.ZipFile(out_zip_dl, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt file in zip {zip_name}: {bad_file}")

    # Copy zip to project root
    shutil.copy2(out_zip_dl, out_zip_prj)

    # Export file-by-file manifest CSV
    with open(out_man_dl, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(files_manifest)
    shutil.copy2(out_man_dl, out_man_prj)

    sha256_dl = compute_sha256(out_zip_dl)
    size_mb = out_zip_dl.stat().st_size / (1024 ** 2)
    uncomp_mb = total_uncompressed_bytes / (1024 ** 2)

    return {
        "zip_name": zip_name,
        "zip_path_dl": str(out_zip_dl),
        "zip_path_prj": str(out_zip_prj),
        "manifest_path": str(out_man_dl),
        "total_files": total_files,
        "uncompressed_mb": round(uncomp_mb, 2),
        "size_mb": round(size_mb, 2),
        "sha256": sha256_dl,
    }


def main():
    print("=" * 80)
    print("PACKAGING HARDENED PART 1 DELIVERABLES (P1-R2 & P1-R1 UPDATED)")
    print("=" * 80)

    # Build R2 package
    r2_info = build_package(ZIP_NAME_R2, MANIFEST_NAME_R2, "CNTT-KLCN155_PART01_VERIFIED_R2")
    print(f"\n[R2 Package Created]:")
    print(f"  - File: {r2_info['zip_name']}")
    print(f"  - Files in archive: {r2_info['total_files']}")
    print(f"  - Compressed Size:  {r2_info['size_mb']} MB")
    print(f"  - SHA-256:          {r2_info['sha256']}")

    # Build R1 package (updated)
    r1_info = build_package(ZIP_NAME_R1, MANIFEST_NAME_R1, "CNTT-KLCN155_PART01_VERIFIED_R1")
    print(f"\n[R1 Package Updated]:")
    print(f"  - File: {r1_info['zip_name']}")
    print(f"  - Files in archive: {r1_info['total_files']}")
    print(f"  - Compressed Size:  {r1_info['size_mb']} MB")
    print(f"  - SHA-256:          {r1_info['sha256']}")

    print("\n" + "=" * 80)
    print("ALL DELIVERY PACKAGES CREATED AND VERIFIED SUCCESSFULLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
