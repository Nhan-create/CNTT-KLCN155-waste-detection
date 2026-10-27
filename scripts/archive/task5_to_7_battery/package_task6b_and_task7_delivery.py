"""
Package Task 6B (E3 Evidence Correction) and Task 7 (Diverse Battery Data Acquisition & Visual QA)
into a single self-contained ZIP archive in ~/Downloads/.
"""
import hashlib
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ZIP_PATH = (Path.home() / "Downloads" / "CNTT_KLCN155_Task6B_And_Task7_Battery_Expansion_Delivery.zip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    explicit_files = [
        "TASK6_E3_CORRECTED_EVALUATION_REPORT.md",
        "TASK6_YOLOV8N_V3_PILOT_TRAINING_AND_EVAL_REPORT.md",
        "TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md",
        "configs/training_yolov8n_v4_proposed_DO_NOT_RUN.yaml",
        "configs/detection_dataset_real_v3_pilot.yaml",
        "artifacts/detection_v3_pilot/dataset_v3_materialization_and_preflight_report.json",
        "artifacts/detection_v3_pilot/task6b_report_consistency_audit.json",
        "artifacts/detection_v3_pilot/evaluation_e1_vs_e3/e1_vs_e3_comprehensive_evaluation.json",
        "artifacts/detection_v3_pilot/E3_yolov8n_pilot/epoch_telemetry.json",
        "artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv",
        "artifacts/detection_v3_pilot/E3_yolov8n_pilot/training_log.txt",
        "artifacts/detection_v3_pilot/E3_yolov8n_pilot/training_log_errata_and_reconciliation.json",
        "scripts/verify_and_generate_task6b_corrected_report.py",
        "scripts/search_and_download_task7_battery_candidates.py",
        "scripts/fetch_additional_valid_task7_batteries.py",
        "scripts/annotate_and_qa_task7_battery_batch.py",
        "scripts/generate_task7_report.py",
        "tests/qa/test_task6b_report_and_task7_candidates.py",
    ]

    task7_dir = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
    all_rel_paths = list(explicit_files)

    for p in sorted(task7_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel not in all_rel_paths:
                all_rel_paths.append(rel)

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in all_rel_paths:
            abs_p = REPO_ROOT / rel
            if not abs_p.exists():
                raise FileNotFoundError(f"Missing file for ZIP: {abs_p}")
            zf.write(abs_p, arcname=rel)

    repo_zip_copy = REPO_ROOT / "artifacts" / "detection_v3_pilot" / ZIP_PATH.name
    import shutil
    shutil.copy2(ZIP_PATH, repo_zip_copy)

    zip_sha = sha256_file(ZIP_PATH)
    zip_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    summary = {
        "zip_path": str(ZIP_PATH),
        "repo_zip_copy": str(repo_zip_copy),
        "total_files_packaged": len(all_rel_paths),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_size_mb": round(zip_mb, 2),
        "zip_sha256": zip_sha,
    }
    (REPO_ROOT / "artifacts" / "detection_v3_pilot" / "task6b_and_task7_delivery_zip_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
