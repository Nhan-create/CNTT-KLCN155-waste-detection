"""
src/detection/preflight_guard.py
--------------------------------
Shared system hardware and environment preflight safety guard for training entrypoints.
Enforces system stability by verifying host RAM and GPU VRAM thresholds before
launching memory-heavy training or worker processes.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import psutil
import torch

@dataclass
class PreflightReport:
    safe_to_train: bool
    status_code: str
    reason: str
    host_ram_total_mb: float
    host_ram_available_mb: float
    host_ram_used_percent: float
    gpu_available: bool
    gpu_device_name: str
    gpu_vram_total_mb: float
    gpu_vram_free_mb: float
    recommendation: str

class PreflightSafetyError(RuntimeError):
    """Raised when hardware resources are insufficient to safely train without risk of OS thrashing."""
    pass

HARD_MIN_RAM_FLOOR_MB = 1000.0   # Hard floor: min_free_ram_mb=0 cannot bypass preflight guard
HARD_MIN_VRAM_FLOOR_MB = 300.0   # Hard floor for GPU VRAM


def check_training_safety(
    min_free_ram_mb: float = 1200.0,
    min_free_vram_mb: float = 500.0,
    device: Optional[str] = None
) -> PreflightReport:
    """
    Evaluates hardware safety before training.
    Default thresholds:
      - min_free_ram_mb: 1200 MB (1.2 GB) minimum free system memory to prevent OS lockup
      - min_free_vram_mb: 500 MB minimum free GPU VRAM
    Enforces hard floors: min_free_ram_mb cannot be lowered below 1000 MB,
    and min_free_vram_mb cannot be lowered below 300 MB, preventing zero-threshold bypass.
    """
    eff_min_ram = max(float(min_free_ram_mb), HARD_MIN_RAM_FLOOR_MB)
    eff_min_vram = max(float(min_free_vram_mb), HARD_MIN_VRAM_FLOOR_MB)

    vm = psutil.virtual_memory()
    total_ram_mb = vm.total / (1024 * 1024)
    avail_ram_mb = vm.available / (1024 * 1024)
    used_ram_pct = vm.percent

    # GPU evaluation
    gpu_available = torch.cuda.is_available()
    gpu_name = "None (CPU only)"
    total_vram_mb = 0.0
    free_vram_mb = 0.0

    if gpu_available:
        try:
            dev_idx = torch.cuda.current_device() if device is None or device.startswith("cuda") else 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            free_b, tot_b = torch.cuda.mem_get_info(dev_idx)
            free_vram_mb = free_b / (1024 * 1024)
            total_vram_mb = tot_b / (1024 * 1024)
        except Exception:
            pass

    # Safety checks
    if avail_ram_mb < eff_min_ram:
        return PreflightReport(
            safe_to_train=False,
            status_code="SYSTEM_RAM_CRITICAL",
            reason=f"Available system RAM is critically low: {avail_ram_mb:.1f} MB (< {eff_min_ram:.1f} MB required). RAM is {used_ram_pct:.1f}% utilized.",
            host_ram_total_mb=round(total_ram_mb, 1),
            host_ram_available_mb=round(avail_ram_mb, 1),
            host_ram_used_percent=round(used_ram_pct, 1),
            gpu_available=gpu_available,
            gpu_device_name=gpu_name,
            gpu_vram_total_mb=round(total_vram_mb, 1),
            gpu_vram_free_mb=round(free_vram_mb, 1),
            recommendation="Local multi-epoch training is BLOCKED to prevent OS thrashing. Free system RAM by closing background apps, or run training in Google Colab GPU."
        )

    if gpu_available and free_vram_mb < eff_min_vram:
        return PreflightReport(
            safe_to_train=False,
            status_code="GPU_VRAM_INSUFFICIENT",
            reason=f"Free GPU VRAM is insufficient: {free_vram_mb:.1f} MB (< {eff_min_vram:.1f} MB required).",
            host_ram_total_mb=round(total_ram_mb, 1),
            host_ram_available_mb=round(avail_ram_mb, 1),
            host_ram_used_percent=round(used_ram_pct, 1),
            gpu_available=gpu_available,
            gpu_device_name=gpu_name,
            gpu_vram_total_mb=round(total_vram_mb, 1),
            gpu_vram_free_mb=round(free_vram_mb, 1),
            recommendation="Restart GPU runtime or clear CUDA cache before training."
        )

    return PreflightReport(
        safe_to_train=True,
        status_code="PREFLIGHT_PASSED",
        reason="System RAM and GPU VRAM meet all safety criteria.",
        host_ram_total_mb=round(total_ram_mb, 1),
        host_ram_available_mb=round(avail_ram_mb, 1),
        host_ram_used_percent=round(used_ram_pct, 1),
        gpu_available=gpu_available,
        gpu_device_name=gpu_name,
        gpu_vram_total_mb=round(total_vram_mb, 1),
        gpu_vram_free_mb=round(free_vram_mb, 1),
        recommendation="Hardware resources are safe for training execution."
    )

def enforce_preflight_guard(
    min_free_ram_mb: float = 1200.0,
    min_free_vram_mb: float = 500.0,
    device: Optional[str] = None
) -> PreflightReport:
    """Enforces preflight guard, raising PreflightSafetyError if unsafe."""
    report = check_training_safety(min_free_ram_mb, min_free_vram_mb, device)
    if not report.safe_to_train:
        raise PreflightSafetyError(f"[{report.status_code}] {report.reason} -> {report.recommendation}")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardware and Environment Preflight Safety Guard.")
    parser.add_argument("--min-ram-mb", type=float, default=1200.0, help="Minimum required available RAM in MB (default: 1200)")
    parser.add_argument("--min-vram-mb", type=float, default=500.0, help="Minimum required free VRAM in MB (default: 500)")
    parser.add_argument("--device", type=str, default=None, help="Device to inspect ('cuda' or 'cpu')")
    parser.add_argument("--json", action="store_true", help="Print report as JSON")

    args = parser.parse_args()

    report = check_training_safety(
        min_free_ram_mb=args.min_ram_mb,
        min_free_vram_mb=args.min_vram_mb,
        device=args.device
    )

    if args.json:
        import json
        print(json.dumps(asdict(report), indent=2))
    else:
        verdict_str = "PASS" if report.safe_to_train else "BLOCKED"
        print("=" * 60)
        print(f"TRAINING PREFLIGHT GUARD VERDICT: {verdict_str}")
        print("=" * 60)
        print(f"Status Code:     {report.status_code}")
        print(f"Reason:          {report.reason}")
        print(f"Host RAM:        {report.host_ram_available_mb:.1f} MB available / {report.host_ram_total_mb:.1f} MB total ({report.host_ram_used_percent:.1f}% used)")
        print(f"RAM Threshold:   {args.min_ram_mb:.1f} MB minimum required")
        print(f"GPU Available:   {report.gpu_available} ({report.gpu_device_name})")
        if report.gpu_available:
            print(f"GPU VRAM:        {report.gpu_vram_free_mb:.1f} MB free / {report.gpu_vram_total_mb:.1f} MB total (Threshold: {args.min_vram_mb:.1f} MB)")
        print(f"Recommendation:  {report.recommendation}")
        print("=" * 60)

    if not report.safe_to_train:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

