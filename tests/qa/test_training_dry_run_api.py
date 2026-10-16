"""
tests/qa/test_training_dry_run_api.py
-------------------------------------
P0 Tests for SSDLite Training Dry-Run & Preflight Hardware Safety Guard:
1. Mixed positive + negative batch execution without NaN/Inf.
2. Preflight guard checks host RAM and detects critical low memory conditions.
3. Preflight guard raises PreflightSafetyError when enforced under insufficient resources.
"""

import pytest
import torch
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large

from src.detection.preflight_guard import (
    check_training_safety,
    enforce_preflight_guard,
    PreflightSafetyError
)

def test_ssdlite_training_api_mixed_positive_negative_batch():
    """
    P0 Test: Verifies that SSDLite's training API natively accepts a mixed batch
    containing both positive images (with bounding boxes) and negative images (0 boxes),
    computes loss tensors, and successfully executes a backward pass.
    """
    import psutil
    if psutil.virtual_memory().available < 1200 * 1024 * 1024:
        pytest.skip("Available host RAM is critically low (< 1200 MB); skipping backward pass dry-run per preflight safety policy.")

    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=11)
    model.train()

    # 1. Positive image with 2 boxes
    img_pos = torch.rand((3, 320, 320), dtype=torch.float32)
    target_pos = {
        "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0], [150.0, 120.0, 250.0, 220.0]], dtype=torch.float32),
        "labels": torch.tensor([1, 7], dtype=torch.int64)  # 1-indexed for SSDLite
    }

    # 2. Negative image with 0 boxes
    img_neg = torch.rand((3, 320, 320), dtype=torch.float32)
    target_neg = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64)
    }

    # Forward pass
    losses = model([img_pos, img_neg], [target_pos, target_neg])

    assert "bbox_regression" in losses, "Missing bbox_regression loss"
    assert "classification" in losses, "Missing classification loss"

    total_loss = losses["bbox_regression"] + losses["classification"]
    assert total_loss.item() > 0, "Loss must be positive"
    assert not torch.isnan(total_loss), "Loss must not be NaN"
    assert not torch.isinf(total_loss), "Loss must not be Inf"

    # Backward pass
    total_loss.backward()

    # Check that gradients were computed and finite
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad is True, "Backward pass must compute gradients for model parameters"

    finite_grads = all(
        not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        for p in model.parameters()
        if p.grad is not None
    )
    assert finite_grads is True, "Gradients must all be finite"

    del model, total_loss, losses, img_pos, img_neg, target_pos, target_neg
    import gc
    gc.collect()

def test_preflight_guard_telemetry_and_detection():
    """Verifies that the Preflight Guard accurately reports RAM and GPU status."""
    report = check_training_safety(min_free_ram_mb=100.0)  # low threshold should pass or report accurately
    assert report.host_ram_total_mb > 1000.0
    assert report.host_ram_available_mb > 0.0
    assert 0.0 <= report.host_ram_used_percent <= 100.0

def test_preflight_guard_blocks_critical_ram():
    """Verifies that an excessively high RAM threshold correctly triggers a block."""
    # Requesting 100 TB of free RAM must be blocked
    report = check_training_safety(min_free_ram_mb=100_000_000.0)
    assert report.safe_to_train is False
    assert report.status_code == "SYSTEM_RAM_CRITICAL"

    with pytest.raises(PreflightSafetyError):
        enforce_preflight_guard(min_free_ram_mb=100_000_000.0)
