"""
tests/conftest.py
-----------------
Pytest configuration and fixtures for CNTT-KLCN155 verification test suite.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
