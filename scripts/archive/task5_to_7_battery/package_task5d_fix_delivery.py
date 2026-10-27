# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Thực thi quy trình kiểm định chất lượng và đóng gói bàn giao cho `package_task5d_fix_delivery`.
"""
scripts/package_task5d_fix_delivery.py
--------------------------------------
Packages all Task 5D Fix deliverables into:
  ~/Downloads/CNTT_KLCN155_Task5D_Fix_Delivery.zip
  (and updates ~/Downloads/CNTT_KLCN155_Task5D_Delivery.zip)

Included artifacts:
1. 20 representative battery images (`reannotated_v1_minipilot20/images/*.jpg`)
2. 20 new full-battery YOLO label files (`reannotated_v1_minipilot20/labels/*.txt`)
3. 20 BEFORE/AFTER side-by-side overlays + 2 summary sheets (`reannotated_v1_minipilot20/overlays_before_after/*.jpg`)
4. Updated QA manifests (`minipilot20_reannotation_manifest.json`, `base_images_qa_manifest.json`, `roboflow_battery_staging_manifest.json`, `training_approval_candidate_list.json`) + 7 updated base contact sheets
5. Augmentation & capture-session leakage analysis (`augmentation_and_session_leakage_analysis.json`)
6. Updated code, configs, regression tests, technical report, and raw pytest execution log (`reports/task5d_fix_pytest_log.txt`).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS = (Path.home() / "Downloads")
ZIP_FIX_PATH = DOWNLOADS / "CNTT_KLCN155_Task5D_Fix_Delivery.zip"
ZIP_MAIN_PATH = DOWNLOADS / "CNTT_KLCN155_Task5D_Delivery.zip"


def main() -> int:
    files_to_pack = [
        # Reports & Logs
        ROOT / "TASK5D_FIX_BATTERY_ANNOTATION_REPORT.md",
        ROOT / "TASK5D_FINAL_BATTERY_QA_AND_APPROVAL_REPORT.md",
        ROOT / "reports" / "task5d_fix_pytest_log.txt",
        # Configs
        ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml",
        ROOT / "configs" / "training_yolov8n_v3_pilot.yaml",
        # Updated Source & Scripts
        ROOT / "src" / "detection" / "safe_dataset_builder.py",
        ROOT / "scripts" / "import_and_qa_roboflow_battery.py",
        ROOT / "scripts" / "generate_all_140_base_contact_sheets.py",
        ROOT / "scripts" / "reannotate_battery_minipilot_20.py",
        ROOT / "scripts" / "analyze_battery_variants_and_sessions.py",
        # Regression Tests
        ROOT / "tests" / "qa" / "test_task5d_fix_reannotation_and_qa.py",
        ROOT / "tests" / "qa" / "test_task5d_approval_package.py",
        ROOT / "tests" / "qa" / "test_task5c_battery_qa.py",
        ROOT / "tests" / "qa" / "test_task5b_hotfixes_and_intake.py",
        # Manifests & Session Analysis
        ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "roboflow_battery_staging_manifest.json",
        ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "base_images_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "training_approval_candidate_list.json",
        ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "augmentation_and_session_leakage_analysis.json",
        ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "reannotated_v1_minipilot20" / "minipilot20_reannotation_manifest.json",
    ]

    # Add 20 re-annotated images, 20 new labels, 20 BEFORE/AFTER overlays + 2 summary sheets
    reann_root = ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "reannotated_v1_minipilot20"
    files_to_pack.extend(sorted((reann_root / "images").glob("*.jpg")))
    files_to_pack.extend(sorted((reann_root / "labels").glob("*.txt")))
    files_to_pack.extend(sorted((reann_root / "overlays_before_after").glob("*.jpg")))

    # Add 7 updated base contact sheets
    cs_base_root = ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "contact_sheets_base"
    files_to_pack.extend(sorted(cs_base_root.glob("*.jpg")))

    with zipfile.ZipFile(ZIP_FIX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_pack:
            if not p.exists():
                raise FileNotFoundError(f"Missing required delivery file: {p}")
            arcname = str(p.relative_to(ROOT)).replace("\\", "/")
            zf.write(p, arcname=arcname)

    shutil.copy2(ZIP_FIX_PATH, ZIP_MAIN_PATH)

    print(f"Created delivery archive: {ZIP_FIX_PATH} ({ZIP_FIX_PATH.stat().st_size:,} bytes, {len(files_to_pack)} files)")
    print(f"Updated main archive:     {ZIP_MAIN_PATH} ({ZIP_MAIN_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
