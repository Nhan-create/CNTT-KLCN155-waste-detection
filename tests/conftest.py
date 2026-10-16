"""
tests/conftest.py
-----------------
Pytest configuration and fixtures for CNTT-KLCN155 verification test suite.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_BYTECODE_ROOT = PROJECT_ROOT / ".pytest_cache" / "pyc"

# The repository was copied with valid-looking ``.pyc`` files compiled in an
# older checkout. Use a workspace-local cache namespace so pytest never loads
# those files and tracebacks always identify this checkout.
sys.pycache_prefix = str(TEST_BYTECODE_ROOT)

# Keep the complete suite headless on Windows. Callers can still override both.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
# The verification suite does not test CUDA kernels. Keeping CUDA hidden avoids
# unrelated 4 GiB GPU pressure from subprocess imports and validation fixtures.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail early if the top-level package resolves outside this workspace."""
    import src

    source_path = Path(src.__file__).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT):
        raise pytest.UsageError(
            f"src resolved outside the repository: {source_path} (expected under {PROJECT_ROOT})"
        )

@pytest.fixture(scope="session")
def processed_dir() -> Path:
    p = PROJECT_ROOT / "data" / "processed_v2"
    return p

@pytest.fixture(scope="session")
def val_dir(processed_dir) -> Path:
    return processed_dir / "val"

@pytest.fixture(scope="session")
def skip_real() -> bool:
    return False
