import sys
import os
import inspect
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_python_interpreter_is_local_venv():
    """Ensure the test runner is executing from the project-local .venv."""
    exe_path = Path(sys.executable).resolve()
    assert "CNTT-KLCN155-waste-detection" in str(exe_path), f"Python executable {exe_path} is outside project workspace"
    assert ".venv" in str(exe_path), f"Python executable {exe_path} is not in .venv"

def test_src_module_is_workspace_local():
    """Ensure src resolves strictly within this repository."""
    import src
    src_file = Path(src.__file__).resolve()
    assert src_file.is_relative_to(PROJECT_ROOT), f"src resolved outside repository: {src_file}"

def test_no_foreign_bytecode_in_workspace():
    """Scan all .pyc files in tests and src to ensure none reference external repos."""
    foreign_pyc = []
    for d in ["tests", "src"]:
        target = PROJECT_ROOT / d
        if not target.exists():
            continue
        for root, dirs, files in os.walk(target):
            if "__pycache__" in root:
                for fn in files:
                    if fn.endswith(".pyc"):
                        p = Path(root) / fn
                        try:
                            with open(p, "rb") as f:
                                content = f.read()
                                if b"waste-classifier-mobilenetv3" in content:
                                    foreign_pyc.append(str(p))
                        except Exception:
                            pass
    assert len(foreign_pyc) == 0, f"Found foreign bytecode references: {foreign_pyc}"

def test_sys_path_does_not_contain_foreign_repo():
    """Ensure sys.path does not contain any foreign repository path."""
    for p in sys.path:
        assert "waste-classifier-mobilenetv3" not in p, f"Foreign repo in sys.path: {p}"
