"""Package all Task 6 deliverables into C:\\Users\\ad\\Downloads\\CNTT_KLCN155_Task6_E3_Training_Delivery.zip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = (Path.home() / "Downloads" / "CNTT_KLCN155_Task6_E3_Training_Delivery.zip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    include_files: list[Path] = [
        REPO_ROOT / "TASK6_YOLOV8N_V3_PILOT_TRAINING_AND_EVAL_REPORT.md",
        REPO_ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml",
        REPO_ROOT / "configs" / "training_yolov8n_v3_pilot.yaml",
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "gpu_preflight_log.txt",
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "dataset_v3_materialization_and_preflight_report.json",
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "pytest_task6_verification_log.txt",
        REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "candidate_pilot_v3_manifest.json",
        REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "splits" / "train.txt",
        REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "splits" / "val.txt",
        REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "splits" / "test.txt",
        REPO_ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "reannotated_v2_full140" / "full140_reannotation_manifest.json",
        REPO_ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "base_images_qa_manifest.json",
        REPO_ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "training_approval_candidate_list.json",
        REPO_ROOT / "scripts" / "materialize_and_verify_pilot_v3_task6.py",
        REPO_ROOT / "scripts" / "run_task6_e3_pilot_training.py",
        REPO_ROOT / "scripts" / "evaluate_and_compare_e1_vs_e3_task6.py",
        REPO_ROOT / "scripts" / "generate_task6_visual_comparisons.py",
        REPO_ROOT / "scripts" / "package_task6_delivery.py",
        REPO_ROOT / "tests" / "qa" / "test_task6_e3_training_and_eval.py",
    ]

    include_dirs: list[Path] = [
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot",
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "evaluation_e1_vs_e3",
        REPO_ROOT / "artifacts" / "detection_v3_pilot" / "visual_comparison_e1_vs_e3",
    ]

    all_paths: list[Path] = []
    seen: set[Path] = set()

    for f in include_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required deliverable file: {f}")
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            all_paths.append(f)

    for d in include_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing required deliverable directory: {d}")
        for f in sorted(d.rglob("*")):
            if f.is_file():
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    all_paths.append(f)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in all_paths:
            arcname = f.relative_to(REPO_ROOT).as_posix()
            zf.write(f, arcname=arcname)

    zip_sha = sha256_file(ZIP_PATH)
    zip_size = ZIP_PATH.stat().st_size

    summary = {
        "zip_path": str(ZIP_PATH),
        "zip_size_bytes": zip_size,
        "zip_size_mb": round(zip_size / (1024 * 1024), 2),
        "zip_sha256": zip_sha,
        "total_files_packaged": len(all_paths),
    }
    manifest_out = REPO_ROOT / "artifacts" / "detection_v3_pilot" / "task6_delivery_zip_manifest.json"
    manifest_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
