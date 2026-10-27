"""Build Task 6 Handover Package (`TASK8F_SEMANTIC_QA_CORRECTION.zip`).

Strictly packages ONLY the delta artifacts requested by the PM:
1. Before & After manifests (`task8f_qa_manifest_pre_correction.json`, `task8f_qa_manifest.json`, `task8f_qa_manifest.csv`).
2. Images, YOLO labels, and overlays ONLY for the modified records (`229` modified IDs, including before/after overlays for all modified pre-correction `REAL_WASTE_PASS` records) — does NOT repack unchanged candidate images.
3. Final `REAL_WASTE_PASS` contact sheets (`p1.jpg..p3.jpg`, `task8f_val_real_waste.jpg`, `task8f_keep_auxiliary.jpg`, `task8f_semantic_qa_before_after.jpg`).
4. Class, domain, bbox, and high-IoU correction tables (`TASK8F_SEMANTIC_QA_CORRECTION_REPORT.{md,json}`).
5. Synchronized Three-Tier Ledger & Validation Plan (`CANONICAL_5000_DATASET_LEDGER.{md,json,csv}`, `TASK8F_SCARCE_CLASS_EXPANSION_REPORT.{md,json}`, `REAL_WASTE_10CLASS_VALIDATION_PLAN.{md,json}`).
6. Scripts & regression test suite (`scripts/correct_task8f_semantic_qa.py`, `scripts/acquire_and_qa_task8f_scarce_classes.py`, `scripts/build_canonical_5000_ledger.py`, `scripts/generate_task8f_and_validation_reports.py`, `tests/detection/test_task8_manifest_report_consistency.py`).
"""
from __future__ import annotations

import json
import pathlib
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
DOWNLOADS = (pathlib.Path.home() / "Downloads")
W8F_DIR = ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes"
REPORTS_DIR = ROOT / "reports" / "task8_dataset_program"


def add_file(zf: zipfile.ZipFile, p: pathlib.Path, seen: set[str]) -> None:
    if not p.exists() or not p.is_file():
        return
    arc = str(p.relative_to(ROOT)).replace("\\", "/")
    if arc not in seen:
        zf.write(p, arcname=arc)
        seen.add(arc)


def main() -> None:
    corr = json.loads((REPORTS_DIR / "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.json").read_text(encoding="utf-8"))
    post = json.loads((W8F_DIR / "task8f_qa_manifest.json").read_text(encoding="utf-8"))
    by_id = {r["candidate_id"]: r for r in post["records"]}
    modified_ids = set(corr["modified_candidate_ids"])

    zip_p = ROOT / "TASK8F_SEMANTIC_QA_CORRECTION.zip"
    if zip_p.exists():
        zip_p.unlink()

    seen: set[str] = set()
    with zipfile.ZipFile(zip_p, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Before & After Manifests
        add_file(zf, W8F_DIR / "qa_correction_before_after" / "task8f_qa_manifest_pre_correction.json", seen)
        add_file(zf, W8F_DIR / "task8f_qa_manifest.json", seen)
        add_file(zf, W8F_DIR / "task8f_qa_manifest.csv", seen)

        # 2. Images, labels, and overlays ONLY for modified records + all 86 post-correction REAL_WASTE_PASS labels/overlays
        for cid in sorted(modified_ids):
            rec = by_id[cid]
            add_file(zf, ROOT / rec["image_path"], seen)
            if rec.get("label_path"):
                add_file(zf, ROOT / rec["label_path"], seen)
            if rec.get("overlay_path"):
                add_file(zf, ROOT / rec["overlay_path"], seen)
            add_file(zf, W8F_DIR / "qa_correction_before_after" / f"{cid}_before_overlay.jpg", seen)
            add_file(zf, W8F_DIR / "qa_correction_before_after" / f"{cid}_after_overlay.jpg", seen)

        for r in post["records"]:
            if r["qa_status"] == "REAL_WASTE_PASS":
                add_file(zf, ROOT / r["image_path"], seen)
                if r.get("label_path"):
                    add_file(zf, ROOT / r["label_path"], seen)
                if r.get("overlay_path"):
                    add_file(zf, ROOT / r["overlay_path"], seen)

        # 3. Final Contact Sheets & Before/After Spotlight Sheet
        for p in sorted((W8F_DIR / "contact_sheets").glob("*.jpg")):
            add_file(zf, p, seen)

        # 4. Reports, Class/BBox/High-IoU Tables, Synchronized Ledger & Validation Plan
        for fn in [
            "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md",
            "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.json",
            "CANONICAL_5000_DATASET_LEDGER.md",
            "CANONICAL_5000_DATASET_LEDGER.json",
            "CANONICAL_5000_DATASET_LEDGER.csv",
            "TASK8F_SCARCE_CLASS_EXPANSION_REPORT.md",
            "TASK8F_SCARCE_CLASS_EXPANSION_REPORT.json",
            "REAL_WASTE_10CLASS_VALIDATION_PLAN.md",
            "REAL_WASTE_10CLASS_VALIDATION_PLAN.json",
        ]:
            add_file(zf, REPORTS_DIR / fn, seen)

        # 5. Scripts & Regression Test Suite
        for rel in [
            "scripts/correct_task8f_semantic_qa.py",
            "scripts/acquire_and_qa_task8f_scarce_classes.py",
            "scripts/build_canonical_5000_ledger.py",
            "scripts/generate_task8f_and_validation_reports.py",
            "scripts/package_task8f_semantic_qa_correction.py",
            "tests/detection/test_task8_manifest_report_consistency.py",
        ]:
            add_file(zf, ROOT / rel, seen)

    size_mb = zip_p.stat().st_size / (1024 * 1024)
    assert size_mb < 500.0, f"Archive {zip_p.name} exceeds 500 MB: {size_mb:.2f} MB"
    if DOWNLOADS.exists():
        shutil.copy2(zip_p, DOWNLOADS / zip_p.name)
    print(f"Created {zip_p.name}: {size_mb:.2f} MB ({len(seen)} files) -> copied to {DOWNLOADS / zip_p.name}")


if __name__ == "__main__":
    main()
