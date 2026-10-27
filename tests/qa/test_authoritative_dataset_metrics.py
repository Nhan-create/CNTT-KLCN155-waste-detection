# Mô tả: Kiểm thử đối chiếu toán học bảng thống kê số lượng ảnh và số lượng khung bao chuẩn xác của bộ dữ liệu.
import sys
"""
tests/qa/test_authoritative_dataset_metrics.py
----------------------------------------------
Regression tests asserting that authoritative dataset metrics:
1. Reconcile exactly: v2 (5,584) - excluded (1,253) = v3 candidate (4,331).
2. Sum of per-class boxes equals total boxes for every split and overall.
3. Critically starved classes are preserved 100%: battery = 2, biological = 6.
4. Per-split sum matches overall sum for all 10 classes.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "data_v2" / "authoritative_dataset_metrics.json"

@pytest.fixture(scope="module")
def metrics():
    if not METRICS_PATH.exists():
        import subprocess
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "generate_authoritative_dataset_metrics.py")
        ], check=True)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_mathematical_reconciliation_identity(metrics):
    """Verify exact identity: v2 - excluded = v3 candidate."""
    assert metrics["metadata"]["reconciliation_valid"] is True

    v2 = metrics["datasets"]["real_v2"]
    v3 = metrics["datasets"]["v3_candidate"]
    ex = metrics["datasets"]["excluded_subset"]

    assert v2["total_images"] == 1996
    assert v3["total_images"] == 1820
    assert ex["total_images"] == 176
    assert v2["total_images"] == v3["total_images"] + ex["total_images"]

    assert v2["total_boxes"] == 5584
    assert v3["total_boxes"] == 4331
    assert ex["total_boxes"] == 1253
    assert v2["total_boxes"] == v3["total_boxes"] + ex["total_boxes"]

def test_sum_of_classes_equals_total_boxes(metrics):
    """Verify sum of per-class boxes equals total boxes across all datasets."""
    for ds_name in ["real_v2", "v3_candidate", "excluded_subset"]:
        ds = metrics["datasets"][ds_name]
        dist = ds["class_distribution_overall"]
        sum_classes = sum(dist.values())
        assert sum_classes == ds["total_boxes"], f"{ds_name} class sum {sum_classes} != total {ds['total_boxes']}"

def test_split_box_sums_match_overall_boxes(metrics):
    """Verify sum of split boxes matches total boxes for v2 and v3."""
    for ds_name in ["real_v2", "v3_candidate"]:
        ds = metrics["datasets"][ds_name]
        split_sum = sum(ds["split_boxes"].values())
        assert split_sum == ds["total_boxes"], f"{ds_name} split sum {split_sum} != total {ds['total_boxes']}"

def test_critical_classes_preserved_100_percent(metrics):
    """Verify battery (2 boxes) and biological (6 boxes) have zero losses."""
    v2_dist = metrics["datasets"]["real_v2"]["class_distribution_overall"]
    v3_dist = metrics["datasets"]["v3_candidate"]["class_distribution_overall"]
    ex_dist = metrics["datasets"]["excluded_subset"]["class_distribution_overall"]

    assert v2_dist["battery"] == 2
    assert v3_dist["battery"] == 2
    assert ex_dist["battery"] == 0

    assert v2_dist["biological"] == 6
    assert v3_dist["biological"] == 6
    assert ex_dist["biological"] == 0

def test_authoritative_numbers_match_expected_constants(metrics):
    """Assert against the ground-truth constants to prevent regressions."""
    v3_dist = metrics["datasets"]["v3_candidate"]["class_distribution_overall"]
    assert v3_dist["plastic"] == 1890
    assert v3_dist["trash"] == 631
    assert v3_dist["metal"] == 470
    assert v3_dist["clothes"] == 437
    assert v3_dist["shoes"] == 287
    assert v3_dist["cardboard"] == 208
    assert v3_dist["glass"] == 205
    assert v3_dist["paper"] == 195
    assert v3_dist["biological"] == 6
    assert v3_dist["battery"] == 2
