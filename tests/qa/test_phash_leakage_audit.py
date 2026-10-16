import sys
"""
tests/qa/test_phash_leakage_audit.py
------------------------------------
QA test asserting that exhaustive pairwise pHash comparisons verify zero leakage:
- Zero cross-split pairs with Hamming distance <= 8.
- Zero Holdout benchmark pairs with Hamming distance <= 8.
- Final verdict is VERIFIED_ZERO_LEAKAGE.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "data_v2" / "phash_leakage_audit_report.json"

@pytest.fixture(scope="module")
def phash_report():
    if not REPORT_PATH.exists():
        import subprocess
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "verify_phash_cross_split_and_holdout.py")
        ], check=True)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_phash_audit_verdict(phash_report):
    """Verify pHash exhaustive pairwise audit passes with VERIFIED_ZERO_LEAKAGE."""
    assert phash_report["verdict"] == "VERIFIED_ZERO_LEAKAGE"

def test_zero_cross_split_near_duplicates(phash_report):
    """Verify zero cross-split image pairs have Hamming distance <= 8."""
    cs = phash_report["cross_split_audit"]
    assert cs["pairs_hamming_lte_8"] == 0
    assert cs["exact_perceptual_duplicates_count"] == 0

def test_zero_holdout_leakage(phash_report):
    """Verify zero candidate images match Holdout benchmark with Hamming distance <= 8."""
    hl = phash_report["holdout_leakage_audit"]
    assert hl["official_27_benchmark_leaks_lte_8"] == 0
    assert hl["all_holdout_leaks_lte_8"] == 0
    assert hl["exact_holdout_duplicates_count"] == 0

def test_audit_coverage(phash_report):
    """Verify all 1,820 candidate images and all Holdout images were audited."""
    meta = phash_report["metadata"]
    assert meta["v3_candidate_images_tested"] == 1820
    assert meta["official_27_holdout_tested"] == 27
