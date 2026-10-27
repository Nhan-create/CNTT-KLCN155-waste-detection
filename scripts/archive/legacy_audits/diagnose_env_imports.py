# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `diagnose_env_imports`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from pathlib import Path

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Working Dir: {os.getcwd()}")
print(f"PYTHONPATH env: {os.environ.get('PYTHONPATH')}")

print("\nsys.path:")
for idx, p in enumerate(sys.path):
    print(f"  [{idx}] {p}")

site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
print(f"\nSite-packages path: {site_packages}")
if site_packages.exists():
    pth_files = list(site_packages.glob("*.pth"))
    print(f"Found {len(pth_files)} .pth files:")
    for pth in pth_files:
        print(f"  - {pth.name}:")
        with open(pth, "r", encoding="utf-8", errors="replace") as f:
            for l in f:
                print(f"      {l.strip()}")
                
    dist_info_direct_url = list(site_packages.glob("*.dist-info/direct_url.json"))
    print(f"\nFound {len(dist_info_direct_url)} direct_url.json files:")
    for du in dist_info_direct_url:
        print(f"  - {du.parent.name}:")
        with open(du, "r", encoding="utf-8", errors="replace") as f:
            print(f"      {f.read().strip()}")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import src
print(f"\nsrc module:")
print(f"  src.__file__: {src.__file__}")
print(f"  src.__path__: {src.__path__}")

# Check pycache files in tests/ and src/
print("\nScanning for copied .pyc files in tests/ and src/ with old checkout paths...")
found_foreign_pyc = 0
for target_dir in ["tests", "src"]:
    for root, dirs, files in os.walk(target_dir):
        if "__pycache__" in root:
            for fn in files:
                if fn.endswith(".pyc"):
                    pyc_path = Path(root) / fn
                    try:
                        with open(pyc_path, "rb") as f:
                            data = f.read()
                            if b"waste-classifier-mobilenetv3" in data:
                                found_foreign_pyc += 1
                                if found_foreign_pyc <= 10:
                                    print(f"  Foreign pyc: {pyc_path}")
                    except Exception:
                        pass
print(f"Total foreign .pyc files found in tests/ and src/: {found_foreign_pyc}")
