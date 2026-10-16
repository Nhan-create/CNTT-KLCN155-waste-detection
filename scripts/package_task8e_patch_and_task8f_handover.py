"""Package TASK 8E FINAL QA PATCH + TASK 8F SCARCE-CLASS EXPANSION handover into 2 balanced ZIP files (each ~335 MB < 500 MB) in C:\\Users\\ad\\Downloads."""
from __future__ import annotations

import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNLOADS = (pathlib.Path.home() / "Downloads")

OLD_SINGLE_ZIP = DOWNLOADS / "CNTT-KLCN155_TASK8E_QA_PATCH_AND_TASK8F_HANDOVER.zip"
OUT_ZIP_PART1 = DOWNLOADS / "CNTT-KLCN155_TASK8E_QA_PATCH_AND_TASK8F_HANDOVER_Part1.zip"
OUT_ZIP_PART2 = DOWNLOADS / "CNTT-KLCN155_TASK8E_QA_PATCH_AND_TASK8F_HANDOVER_Part2.zip"

W8E_DIR = ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste"
W8F_DIR = ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes"

# Included in BOTH Part 1 and Part 2 so either archive has all reports, manifests, contact sheets, labels, scripts, and tests
SHARED_DIRS = [
    ROOT / "reports" / "task8_dataset_program",
    W8E_DIR / "contact_sheets",
    W8E_DIR / "qa_patch_before_after",
    W8E_DIR / "labels_real_waste_pass",
    W8E_DIR / "labels_auxiliary",
    W8E_DIR / "labels_needs_review",
    W8F_DIR / "contact_sheets",
    W8F_DIR / "labels_real_waste_pass",
    W8F_DIR / "labels_auxiliary",
    W8F_DIR / "labels_needs_review",
    ROOT / "data" / "acquisition_staging" / "task8d_independent_qa_mini_pack",
]

SHARED_FILES = [
    W8E_DIR / "task8e_qa_manifest.json",
    W8E_DIR / "task8e_qa_manifest.csv",
    W8E_DIR / "task8e_raw_candidates_manifest.json",
    W8F_DIR / "task8f_qa_manifest.json",
    W8F_DIR / "task8f_qa_manifest.csv",
    W8F_DIR / "task8f_raw_candidates_manifest.json",
    ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json",
    ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.csv",
    ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json",
    ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json",
    ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json",
    ROOT / "scripts" / "qa_and_annotate_task8e_batch.py",
    ROOT / "scripts" / "build_canonical_5000_ledger.py",
    ROOT / "scripts" / "acquire_and_qa_task8f_scarce_classes.py",
    ROOT / "scripts" / "generate_task8f_and_validation_reports.py",
    ROOT / "scripts" / "package_task8e_patch_and_task8f_handover.py",
    ROOT / "tests" / "detection" / "test_task8_manifest_report_consistency.py",
]

BULK_DIRS = [
    W8E_DIR / "images",
    W8E_DIR / "overlays",
    W8F_DIR / "images",
    W8F_DIR / "overlays",
]


def collect_shared_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for d in SHARED_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                arc = str(p.relative_to(ROOT)).replace("\\", "/")
                if arc not in seen:
                    seen.add(arc)
                    out.append(p)
    for f in SHARED_FILES:
        if f.exists() and f.is_file():
            arc = str(f.relative_to(ROOT)).replace("\\", "/")
            if arc not in seen:
                seen.add(arc)
                out.append(f)
    return out


def build_zip(out_zip: pathlib.Path, files: list[pathlib.Path]) -> None:
    if out_zip.exists():
        out_zip.unlink()
    seen_arcs: set[str] = set()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            if f.exists() and f.is_file():
                arc = str(f.relative_to(ROOT)).replace("\\", "/")
                if arc not in seen_arcs:
                    zf.write(f, arcname=arc)
                    seen_arcs.add(arc)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Created {out_zip} ({len(seen_arcs)} files, {size_mb:.2f} MB)")
    assert size_mb < 500.0, f"ZIP exceeds 500 MB limit: {out_zip} ({size_mb:.2f} MB)"


def main() -> None:
    if OLD_SINGLE_ZIP.exists():
        OLD_SINGLE_ZIP.unlink()
    shared = collect_shared_files()
    bulk: list[pathlib.Path] = []
    for d in BULK_DIRS:
        if d.exists():
            bulk.extend(sorted(p for p in d.rglob("*") if p.is_file()))

    total_bulk_bytes = sum(p.stat().st_size for p in bulk)
    half_bytes = total_bulk_bytes / 2.0
    part1_bulk: list[pathlib.Path] = []
    part2_bulk: list[pathlib.Path] = []
    running = 0
    for p in bulk:
        sz = p.stat().st_size
        if running + sz <= half_bytes:
            part1_bulk.append(p)
            running += sz
        else:
            part2_bulk.append(p)

    build_zip(OUT_ZIP_PART1, shared + part1_bulk)
    build_zip(OUT_ZIP_PART2, shared + part2_bulk)


if __name__ == "__main__":
    main()

