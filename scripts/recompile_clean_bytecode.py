import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
from pathlib import Path
import compileall

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=== RECOMPILING AND SANITIZING BYTECODE CACHES ===")

# Step 1: Remove all __pycache__ in tests, src, scripts
pycache_count = 0
pyc_count = 0
for target_dir in ["tests", "src", "scripts"]:
    d_path = PROJECT_ROOT / target_dir
    if not d_path.exists():
        continue
    for root, dirs, files in os.walk(d_path):
        if "__pycache__" in root:
            for f in files:
                if f.endswith(".pyc"):
                    pyc_count += 1
                    try:
                        (Path(root) / f).unlink()
                    except Exception as e:
                        print(f"Error removing {f}: {e}")
            try:
                shutil.rmtree(root, ignore_errors=True)
                pycache_count += 1
            except Exception as e:
                pass

print(f"Removed {pyc_count} stale .pyc files across {pycache_count} __pycache__ directories.")

# Also remove .pytest_temp and .pytest_cache if needed
shutil.rmtree(PROJECT_ROOT / ".pytest_temp", ignore_errors=True)
shutil.rmtree(PROJECT_ROOT / ".pytest_cache", ignore_errors=True)

# Step 2: Compile fresh bytecode
print("\nRecompiling clean bytecode with local workspace paths...")
for target_dir in ["tests", "src", "scripts"]:
    compileall.compile_dir(str(PROJECT_ROOT / target_dir), quiet=1, force=True)

# Step 3: Verify no foreign path exists in any .pyc
foreign_count = 0
for target_dir in ["tests", "src", "scripts"]:
    for root, dirs, files in os.walk(PROJECT_ROOT / target_dir):
        if "__pycache__" in root:
            for fn in files:
                if fn.endswith(".pyc"):
                    p = Path(root) / fn
                    try:
                        with open(p, "rb") as f:
                            if b"waste-classifier-mobilenetv3" in f.read():
                                foreign_count += 1
                                print(f"  WARNING: Still foreign: {p}")
                    except Exception:
                        pass

print(f"\nVerification result: Foreign .pyc files remaining: {foreign_count}")
if foreign_count == 0:
    print("SUCCESS: Bytecode sanitized 100%. All tracebacks will now point strictly to D:\\CNTT-KLCN155-waste-detection.")
