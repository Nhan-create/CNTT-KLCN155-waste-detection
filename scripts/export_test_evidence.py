"""
scripts/export_test_evidence.py
--------------------------------
Executes pytest collection and targeted test execution using .venv Python,
generating complete, raw logs and structured machine-readable JSON artifacts:
- reports/test_evidence/pytest_collection.log
- reports/test_evidence/pytest_targeted.log
- reports/test_evidence/python_environment.json
- reports/test_evidence/tests_summary.json
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "reports" / "test_evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable

# 1. Capture Python Environment
print("[1/4] Capturing Python environment...")
import torch
import torchvision
import ultralytics
import imagehash
from PyQt5 import QtCore

env_info = {
    "sys_executable": str(PYTHON_EXE),
    "python_version": sys.version,
    "platform": platform.platform(),
    "os_name": os.name,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "device_count": torch.cuda.device_count(),
    "pytorch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "ultralytics_version": ultralytics.__version__,
    "imagehash_version": getattr(imagehash, "__version__", "unknown"),
    "pyqt5_version": QtCore.PYQT_VERSION_STR,
    "working_directory": str(ROOT),
}

env_file = EVIDENCE_DIR / "python_environment.json"
env_file.write_text(json.dumps(env_info, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  -> Saved environment info to: {env_file}")

# 2. Pytest Collection Run
print("[2/4] Running pytest collection...")
cmd_collect = [PYTHON_EXE, "-m", "pytest", "--collect-only"]
t0 = time.time()
res_collect = subprocess.run(cmd_collect, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
t_collect = time.time() - t0

collect_log = EVIDENCE_DIR / "pytest_collection.log"
collect_log.write_text(res_collect.stdout + "\n" + res_collect.stderr, encoding="utf-8")
print(f"  -> Exit code: {res_collect.returncode} in {t_collect:.2f}s")
print(f"  -> Saved collection log to: {collect_log}")

# 3. Pytest Targeted Test Run
print("[3/4] Running targeted test suites...")
targeted_test_files = [
    "tests/test_environment_integrity.py",
    "tests/test_detection_annotations.py",
    "tests/detection/test_taco_clipping_geometry.py",
    "tests/qa/test_compare_eval_v1_vs_v2.py",
    "tests/qa/test_ground_truth_v2_integrity.py",
    "tests/detection/test_coordinates.py",
    "tests/detection/test_tiling.py",
    "tests/detection/test_two_stage.py",
    "tests/detection/test_mapping_regression.py",
    "tests/detection/test_real_mapping_integration.py",
    "tests/detection/test_split_real_v2.py",
    "tests/web/test_detection_pipeline.py",
    "tests/web/test_detection_image_outputs.py",
    "tests/web/test_streamlit_contract.py",
    "tests/detection/test_visualization.py",
    "tests/test_verification_gates.py",
    "tests/qa/test_dataset_v3_quality_gates.py",
    "tests/qa/test_yolo_v3_compatibility.py",
]

cmd_targeted = [PYTHON_EXE, "-m", "pytest", "-v"] + targeted_test_files
t0 = time.time()
res_targeted = subprocess.run(cmd_targeted, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
t_targeted = time.time() - t0

targeted_log = EVIDENCE_DIR / "pytest_targeted.log"
targeted_log.write_text(res_targeted.stdout + "\n" + res_targeted.stderr, encoding="utf-8")
print(f"  -> Exit code: {res_targeted.returncode} in {t_targeted:.2f}s")
print(f"  -> Saved targeted run log to: {targeted_log}")

# 4. Parse Results Summary
print("[4/4] Generating structured summary...")
# Parse collection count
collected_count = 0
for line in res_collect.stdout.splitlines():
    if "collected" in line and "items" in line:
        parts = line.split()
        for idx, p in enumerate(parts):
            if p == "collected" and idx + 1 < len(parts):
                try:
                    collected_count = int(parts[idx + 1])
                except ValueError:
                    pass

# Parse targeted counts
passed_count = 0
failed_count = 0
skipped_count = 0
for line in res_targeted.stdout.splitlines()[::-1]:
    if "passed" in line or "failed" in line or "skipped" in line:
        tokens = line.replace("=", "").split(",")
        for tok in tokens:
            t_clean = tok.strip()
            if "passed" in t_clean:
                try:
                    passed_count = int(t_clean.split()[0])
                except ValueError:
                    pass
            elif "failed" in t_clean:
                try:
                    failed_count = int(t_clean.split()[0])
                except ValueError:
                    pass
            elif "skipped" in t_clean:
                try:
                    skipped_count = int(t_clean.split()[0])
                except ValueError:
                    pass
        break

summary = {
    "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "interpreter_path": str(PYTHON_EXE),
    "collection": {
        "command": " ".join(cmd_collect),
        "exit_code": res_collect.returncode,
        "duration_seconds": round(t_collect, 2),
        "total_collected_in_workspace": collected_count,
        "log_file": str(collect_log.relative_to(ROOT)),
    },
    "targeted_suite": {
        "command": " ".join(cmd_targeted),
        "exit_code": res_targeted.returncode,
        "duration_seconds": round(t_targeted, 2),
        "test_files_count": len(targeted_test_files),
        "test_files": targeted_test_files,
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "status": "PASSED" if res_targeted.returncode == 0 and failed_count == 0 else "FAILED",
        "log_file": str(targeted_log.relative_to(ROOT)),
    },
    "scope_clarification": f"Anti reports {passed_count} passed and {skipped_count} skipped out of {passed_count + skipped_count + failed_count} targeted tests executed across {len(targeted_test_files)} test files. Full workspace suite contains {collected_count} collected tests (includes extended GPU/training scripts not executed in this run)."
}

summary_file = EVIDENCE_DIR / "tests_summary.json"
summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  -> Saved structured tests summary to: {summary_file}")
print(f"\nTargeted suite result: {passed_count} passed, {skipped_count} skipped, {failed_count} failed")
