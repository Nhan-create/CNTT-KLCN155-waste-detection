# Mô tả: Chạy kiểm thử đột biến cô lập (mutation test) để chứng minh độ nhạy của các cổng kiểm định chất lượng dữ liệu.
"""
Runs an isolated mutation test:
1. Copies minimal needed source and test files into an isolated directory.
2. Intentionally mutates the product code (removes SSD label shift: class_index = int(label) instead of int(label)-1).
3. Executes pytest on the isolated copy -> confirms FAILURE (RED) with exit code != 0.
4. Restores the unmutated code in the isolated copy.
5. Executes pytest on the restored copy -> confirms PASS (GREEN) with exit code 0.
6. Saves raw logs of both runs for the delivery report.
Does NOT modify or revert the user's working tree.
"""

import shutil
import subprocess
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
scratch_root = (Path(__file__).resolve().parents[2] / "artifacts" / "scratch")
if scratch_root.exists():
    shutil.rmtree(scratch_root)
scratch_root.mkdir(parents=True, exist_ok=True)

# 1. Set up isolated environment
isolated_src = scratch_root / "src"
isolated_tests = scratch_root / "tests"
isolated_configs = scratch_root / "configs"
shutil.copytree(repo_root / "src", isolated_src)
shutil.copytree(repo_root / "tests", isolated_tests)
shutil.copytree(repo_root / "configs", isolated_configs)

# Copy pytest.ini if exists
if (repo_root / "pytest.ini").exists():
    shutil.copy2(repo_root / "pytest.ini", scratch_root / "pytest.ini")

python_exe = repo_root / ".venv" / "Scripts" / "python.exe"

# 2. Introduce deliberate mutations in isolated copy:
# A. Mutate SSDLite loader shift in isolated_src/detection/ssdlite_train.py:
#    labels.append(int(c) + 1) --> labels.append(int(c))
ssdlite_train_file = isolated_src / "detection" / "ssdlite_train.py"
train_content = ssdlite_train_file.read_text(encoding="utf-8")
assert "labels.append(int(c) + 1)" in train_content
mutated_train_content = train_content.replace("labels.append(int(c) + 1)", "labels.append(int(c))")
ssdlite_train_file.write_text(mutated_train_content, encoding="utf-8")

# B. Mutate SSDLite decoder shift in isolated_src/detection/ssdlite.py:
#    class_index = int(label) - 1 --> class_index = int(label)
ssdlite_file = isolated_src / "detection" / "ssdlite.py"
content = ssdlite_file.read_text(encoding="utf-8")
assert "class_index = int(label) - 1" in content
mutated_content = content.replace("class_index = int(label) - 1", "class_index = int(label)")
ssdlite_file.write_text(mutated_content, encoding="utf-8")

print("[1] Mutated isolated copy: altered both loader (+1 shift) and decoder (-1 shift).")

# 3. Run pytest on mutated isolated copy -> expect FAILURE (RED)
cmd_red = [
    str(python_exe),
    "-m", "pytest",
    str(isolated_tests / "detection" / "test_real_mapping_integration.py"),
    "-v"
]
env_red = dict(subprocess.os.environ)
env_red["PYTHONPATH"] = str(scratch_root)

proc_red = subprocess.run(cmd_red, cwd=str(scratch_root), capture_output=True, text=True, env=env_red)
print(f"[2] Ran RED test: exit_code = {proc_red.returncode} (Expected non-zero)")

# 4. Restore original unmutated content in isolated copy
ssdlite_train_file.write_text(train_content, encoding="utf-8")
ssdlite_file.write_text(content, encoding="utf-8")
print("[3] Restored isolated copy to correct logic.")

# 5. Run pytest on restored isolated copy -> expect SUCCESS (GREEN)
proc_green = subprocess.run(cmd_red, cwd=str(scratch_root), capture_output=True, text=True, env=env_red)
print(f"[4] Ran GREEN test: exit_code = {proc_green.returncode} (Expected 0)")

results = {
    "mutation_description": "Deliberately removed the -1 foreground shift in SSDLite tensor_detections: 'class_index = int(label)' instead of 'class_index = int(label) - 1'.",
    "red_run": {
        "exit_code": proc_red.returncode,
        "stdout": proc_red.stdout,
        "stderr": proc_red.stderr,
        "passed": proc_red.returncode != 0
    },
    "green_run": {
        "exit_code": proc_green.returncode,
        "stdout": proc_green.stdout,
        "stderr": proc_green.stderr,
        "passed": proc_green.returncode == 0
    }
}

output_json = repo_root / "reports" / "mutation_test_red_green_log.json"
output_json.parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Mutation RED/GREEN test complete. Results written to: {output_json}")
print("\n--- RED Output Sample ---")
print(proc_red.stdout[:800])
print("\n--- GREEN Output Sample ---")
print(proc_green.stdout[:500])
