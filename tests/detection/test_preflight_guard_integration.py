"""
tests/detection/test_preflight_guard_integration.py
---------------------------------------------------
Tests for Preflight Guard Integration into Training Pipelines:
1. Mocking host RAM to test both passing and blocking paths.
2. Mocking GPU VRAM to test VRAM exhaustion detection.
3. Verification that train_ssdlite raises PreflightSafetyError immediately when resources are critical.
4. Verification that train_yolo and train_ssd enforce the preflight guard.
"""

from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.detection.preflight_guard import (
    check_training_safety,
    enforce_preflight_guard,
    PreflightSafetyError,
    PreflightReport
)

# Mock namedtuple matching psutil.virtual_memory return structure
VMem = namedtuple("VMem", ["total", "available", "percent", "used", "free"])

def test_preflight_guard_passes_when_resources_sufficient():
    """Verify preflight guard passes when system has sufficient RAM."""
    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,      # 16 GB
        available=4 * 1024 * 1024 * 1024,   # 4 GB available (> 1.2 GB min)
        percent=75.0,
        used=12 * 1024 * 1024 * 1024,
        free=4 * 1024 * 1024 * 1024
    )
    with patch("psutil.virtual_memory", return_value=mock_vm):
        with patch("torch.cuda.is_available", return_value=False):
            report = check_training_safety(min_free_ram_mb=1200.0)
            assert report.safe_to_train is True
            assert report.status_code == "PREFLIGHT_PASSED"
            assert report.host_ram_available_mb > 1200.0

            # enforce_preflight_guard should return without raising
            ret = enforce_preflight_guard(min_free_ram_mb=1200.0)
            assert isinstance(ret, PreflightReport)
            assert ret.safe_to_train is True

def test_preflight_guard_blocks_when_ram_is_critically_low():
    """Verify preflight guard blocks when available system RAM is critically low."""
    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=400 * 1024 * 1024,        # Only 400 MB available (< 1200 MB min)
        percent=97.5,
        used=15.6 * 1024 * 1024 * 1024,
        free=400 * 1024 * 1024
    )
    with patch("psutil.virtual_memory", return_value=mock_vm):
        report = check_training_safety(min_free_ram_mb=1200.0)
        assert report.safe_to_train is False
        assert report.status_code == "SYSTEM_RAM_CRITICAL"
        assert "critically low" in report.reason

        with pytest.raises(PreflightSafetyError) as exc_info:
            enforce_preflight_guard(min_free_ram_mb=1200.0)
        assert "SYSTEM_RAM_CRITICAL" in str(exc_info.value)

def test_preflight_guard_blocks_when_gpu_vram_is_insufficient():
    """Verify preflight guard blocks when GPU VRAM is below threshold."""
    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=4 * 1024 * 1024 * 1024,   # RAM is OK
        percent=75.0,
        used=12 * 1024 * 1024 * 1024,
        free=4 * 1024 * 1024 * 1024
    )
    # GPU has only 200 MB free (< 500 MB min)
    mock_mem_info = (200 * 1024 * 1024, 4096 * 1024 * 1024)

    with patch("psutil.virtual_memory", return_value=mock_vm):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.current_device", return_value=0):
                with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 2050"):
                    with patch("torch.cuda.mem_get_info", return_value=mock_mem_info):
                        report = check_training_safety(min_free_ram_mb=1200.0, min_free_vram_mb=500.0, device="cuda")
                        assert report.safe_to_train is False
                        assert report.status_code == "GPU_VRAM_INSUFFICIENT"

                        with pytest.raises(PreflightSafetyError) as exc_info:
                            enforce_preflight_guard(min_free_ram_mb=1200.0, min_free_vram_mb=500.0, device="cuda")
                        assert "GPU_VRAM_INSUFFICIENT" in str(exc_info.value)

def test_train_ssdlite_entrypoint_triggers_preflight_guard():
    """Verify that train_ssdlite entrypoint calls preflight guard and aborts immediately when unsafe."""
    from src.detection.ssdlite_train import train_ssdlite

    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=300 * 1024 * 1024,        # 300 MB available
        percent=98.0,
        used=15.7 * 1024 * 1024 * 1024,
        free=300 * 1024 * 1024
    )
    config = {
        "train": {
            "device": "cpu",
            "min_free_ram_mb": 1200.0
        }
    }
    with patch("psutil.virtual_memory", return_value=mock_vm):
        with pytest.raises(PreflightSafetyError) as exc_info:
            train_ssdlite(
                data_path=Path("dummy_path.yaml"),
                config=config,
                run_directory=Path("dummy_run"),
                metadata={},
                resume=False
            )
        assert "SYSTEM_RAM_CRITICAL" in str(exc_info.value)

def test_train_detection_v2_yolo_and_ssd_triggers_preflight_guard():
    """Verify that train_yolo and train_ssd in scripts/train_detection_v2.py enforce preflight guard."""
    from scripts.train_detection_v2 import train_yolo, train_ssd

    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=500 * 1024 * 1024,        # 500 MB available
        percent=96.0,
        used=15.5 * 1024 * 1024 * 1024,
        free=500 * 1024 * 1024
    )
    with patch("psutil.virtual_memory", return_value=mock_vm):
        with pytest.raises(PreflightSafetyError):
            train_yolo("dummy_run", Path("dummy.yaml"), False, {"device": "cpu", "min_free_ram_mb": 1200.0})

        with pytest.raises(PreflightSafetyError):
            train_ssd("dummy_run", Path("dummy.yaml"), False, {"device": "cpu", "min_free_ram_mb": 1200.0})


def test_preflight_guard_cli_exit_codes():
    """Verify CLI python -m src.detection.preflight_guard exits with 0 on pass and 1 on block."""
    import subprocess
    import sys

    # 1. Very low threshold -> PASS -> exit code 0 (unless host physical RAM is critically low < 1000 MB)
    import psutil
    curr_avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    proc_pass = subprocess.run(
        [sys.executable, "-m", "src.detection.preflight_guard", "--min-ram-mb", "1.0", "--min-vram-mb", "1.0"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2])
    )
    if curr_avail_mb < 1000.0:
        # System RAM is legitimately below the 1000MB hard floor; guard correctly triggers SYSTEM_RAM_CRITICAL
        assert proc_pass.returncode == 1
        assert "SYSTEM_RAM_CRITICAL" in proc_pass.stdout
    else:
        assert proc_pass.returncode == 0, f"Expected 0 on pass, got {proc_pass.returncode}: {proc_pass.stderr}"
        assert "TRAINING PREFLIGHT GUARD VERDICT: PASS" in proc_pass.stdout

    # 2. Impossibly high threshold -> BLOCKED -> exit code 1
    proc_blocked = subprocess.run(
        [sys.executable, "-m", "src.detection.preflight_guard", "--min-ram-mb", "999999999.0"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2])
    )
    assert proc_blocked.returncode == 1, f"Expected 1 on block, got {proc_blocked.returncode}: {proc_blocked.stderr}"
    assert "TRAINING PREFLIGHT GUARD VERDICT: BLOCKED" in proc_blocked.stdout


def test_hard_floor_prevents_zero_bypass():
    """
    P0.2 Regression Test:
    Passing min_free_ram_mb=0 cannot bypass the preflight guard if available RAM < 1000 MB.
    """
    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=400 * 1024 * 1024,   # 400 MB available (< 1000 MB floor)
        percent=97.5,
        used=15.6 * 1024 * 1024 * 1024,
        free=400 * 1024 * 1024
    )
    with patch("psutil.virtual_memory", return_value=mock_vm):
        # Passing 0.0 MB must NOT pass
        report = check_training_safety(min_free_ram_mb=0.0)
        assert report.safe_to_train is False
        assert report.status_code == "SYSTEM_RAM_CRITICAL"

        with pytest.raises(PreflightSafetyError):
            enforce_preflight_guard(min_free_ram_mb=0.0)


def test_skip_preflight_guard_ignored_in_training_entrypoints():
    """
    P0.2 Regression Test:
    Attempting to pass skip_preflight_guard=True is ignored and preflight guard still raises.
    """
    from scripts.train_detection_v2 import train_yolo, train_ssd
    from src.detection.ssdlite_train import train_ssdlite

    mock_vm = VMem(
        total=16 * 1024 * 1024 * 1024,
        available=400 * 1024 * 1024,   # 400 MB available
        percent=97.5,
        used=15.6 * 1024 * 1024 * 1024,
        free=400 * 1024 * 1024
    )
    with patch("psutil.virtual_memory", return_value=mock_vm):
        # 1. train_yolo ignores skip_preflight_guard
        with pytest.raises(PreflightSafetyError):
            train_yolo("dummy", Path("dummy.yaml"), False, {"skip_preflight_guard": True, "device": "cpu"})

        # 2. train_ssd ignores skip_preflight_guard
        with pytest.raises(PreflightSafetyError):
            train_ssd("dummy", Path("dummy.yaml"), False, {"skip_preflight_guard": True, "device": "cpu"})

        # 3. train_ssdlite ignores skip_preflight_guard
        with pytest.raises(PreflightSafetyError):
            train_ssdlite(Path("dummy.yaml"), {"train": {"skip_preflight_guard": True, "device": "cpu"}}, Path("dummy"), {}, False)


