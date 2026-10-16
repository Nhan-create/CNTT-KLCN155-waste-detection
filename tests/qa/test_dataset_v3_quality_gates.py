import pytest
import sys
import math
from pathlib import Path

# Add scripts to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from validate_dataset_v3_quality_gates import (
    calculate_iou,
    check_box_geometry,
    check_duplicate_boxes,
    check_taxonomy,
    check_hard_negative_label,
    check_leakage_hashes,
    check_leakage_phash,
    validate_dataset_entry,
    STATUS_VALID,
    STATUS_FLAG_FOR_REVIEW,
    STATUS_INVALID
)

# ==========================================
# 1. IoU Tests
# ==========================================

def test_calculate_iou_exact():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [0.0, 0.0, 10.0, 10.0]
    assert calculate_iou(boxA, boxB) == 1.0

def test_calculate_iou_disjoint():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [20.0, 20.0, 30.0, 30.0]
    assert calculate_iou(boxA, boxB) == 0.0

def test_calculate_iou_partial():
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [5.0, 0.0, 15.0, 10.0]
    assert abs(calculate_iou(boxA, boxB) - (1.0 / 3.0)) < 1e-5

# ==========================================
# 2. Geometry Gate Tests (NaN, Inf, Bounds, Inverted)
# ==========================================

def test_geometry_gate_valid_box():
    status, msg = check_box_geometry([50, 50, 200, 200], 640, 480)
    assert status == STATUS_VALID
    assert msg == "OK"

def test_geometry_gate_rejects_nan_in_coords():
    """Regression test: NaN in coordinates must be strictly rejected as INVALID."""
    status, msg = check_box_geometry([float('nan'), 0, 100, 100], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

    status, msg = check_box_geometry([0, float('nan'), 100, 100], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

    status, msg = check_box_geometry([0, 0, float('nan'), 100], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

    status, msg = check_box_geometry([0, 0, 100, float('nan')], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

def test_geometry_gate_rejects_inf_in_coords():
    """Regression test: Inf in coordinates must be strictly rejected as INVALID."""
    status, msg = check_box_geometry([0, 0, float('inf'), 100], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

    status, msg = check_box_geometry([float('-inf'), 0, 100, 100], 640, 480)
    assert status == STATUS_INVALID
    assert "NaN or Inf" in msg

def test_geometry_gate_rejects_invalid_image_dims():
    """Regression test: invalid or non-positive image dimensions must be INVALID."""
    status, msg = check_box_geometry([10, 10, 50, 50], 0, 480)
    assert status == STATUS_INVALID
    assert "Invalid image width" in msg

    status, msg = check_box_geometry([10, 10, 50, 50], 640, -10)
    assert status == STATUS_INVALID
    assert "Invalid image height" in msg

    status, msg = check_box_geometry([10, 10, 50, 50], float('nan'), 480)
    assert status == STATUS_INVALID
    assert "Invalid image width" in msg

def test_geometry_gate_rejects_inverted_coordinates():
    """Regression test: x1 >= x2 or y1 >= y2 must be INVALID."""
    # x1 > x2
    status, msg = check_box_geometry([150, 50, 100, 100], 640, 480)
    assert status == STATUS_INVALID
    assert "Invalid horizontal ordering" in msg

    # y1 > y2
    status, msg = check_box_geometry([50, 150, 100, 100], 640, 480)
    assert status == STATUS_INVALID
    assert "Invalid vertical ordering" in msg

    # x1 == x2 (degenerate)
    status, msg = check_box_geometry([100, 50, 100, 150], 640, 480)
    assert status == STATUS_INVALID
    assert "Invalid horizontal ordering" in msg

def test_geometry_gate_rejects_out_of_bounds_and_negative():
    status, msg = check_box_geometry([-5, 0, 640, 480], 640, 480)
    assert status == STATUS_INVALID
    assert "negative" in msg.lower()

    status, msg = check_box_geometry([0, 0, 700, 480], 640, 480)
    assert status == STATUS_INVALID
    assert "exceeds image boundaries" in msg

# ==========================================
# 3. Large Bounding Box: 3 States (VALID, FLAG_FOR_REVIEW, INVALID)
# ==========================================

def test_geometry_gate_flags_large_box_for_review_not_invalid():
    """Crucial requirement: Boxes > 95% are NOT rejected outright; they are FLAG_FOR_REVIEW."""
    status, msg = check_box_geometry([0, 0, 640, 480], 640, 480, max_area_norm=0.95)
    assert status == STATUS_FLAG_FOR_REVIEW
    assert "requires human/QA verification" in msg

def test_geometry_gate_flags_subresolution_box_for_review():
    """Subresolution boxes (< min_area_norm) are flagged for review, not invalid."""
    status, msg = check_box_geometry([10, 10, 11, 11], 1000, 1000, min_area_norm=0.0001)
    # Area = 1 / 1,000,000 = 0.000001 < 0.0001
    assert status == STATUS_FLAG_FOR_REVIEW
    assert "Sub-resolution" in msg

# ==========================================
# 4. Taxonomy Gate
# ==========================================

def test_taxonomy_gate_valid():
    for cid in range(10):
        status, msg = check_taxonomy(cid)
        assert status == STATUS_VALID

def test_taxonomy_gate_invalid():
    status, msg = check_taxonomy(10)
    assert status == STATUS_INVALID
    assert "Invalid class ID" in msg

    status, msg = check_taxonomy(-1)
    assert status == STATUS_INVALID

# ==========================================
# 5. Duplicate Box Gate
# ==========================================

def test_duplicate_gate_detects_duplicates():
    boxes = [
        (3, [50, 50, 150, 150]),
        (3, [50, 50, 150, 150]), # duplicate
        (7, [50, 50, 150, 150])  # different class
    ]
    dups = check_duplicate_boxes(boxes, iou_threshold=0.95)
    assert len(dups) == 1
    assert dups[0][0] == 0
    assert dups[0][1] == 1
    assert dups[0][2] == 1.0

# ==========================================
# 6. Hard-Negative Label Convention
# ==========================================

def test_hard_negative_empty_0_bytes():
    """Official dataset standard: exactly 0 bytes file."""
    status, code, msg = check_hard_negative_label(b"")
    assert status == STATUS_VALID
    assert code == "EMPTY_0_BYTES"

def test_hard_negative_whitespace_only():
    """Whitespace-only file is flagged for normalization."""
    status, code, msg = check_hard_negative_label(b"   \n  \r\n\t  ")
    assert status == STATUS_FLAG_FOR_REVIEW
    assert code == "WHITESPACE_ONLY"

def test_hard_negative_has_boxes_is_invalid():
    """If file contains boxes, it is INVALID as hard negative."""
    status, code, msg = check_hard_negative_label(b"0 0.5 0.5 0.2 0.2\n")
    assert status == STATUS_INVALID
    assert code == "HAS_BOXES"

def test_hard_negative_missing_file_is_invalid():
    status, code, msg = check_hard_negative_label(None)
    assert status == STATUS_INVALID
    assert code == "MISSING_FILE"

# ==========================================
# 7. Leakage Gate: SHA-256 and pHash
# ==========================================

def test_leakage_sha256_collision():
    train_hashes = {"img1.jpg": "hash_abc", "img2.jpg": "hash_def"}
    holdout_hashes = {"holdout1.jpg": "hash_xyz", "holdout2.jpg": "hash_abc"}

    collisions = check_leakage_hashes(train_hashes, holdout_hashes)
    assert len(collisions) == 1
    assert collisions[0] == ("img1.jpg", "holdout2.jpg", "hash_abc")

def test_leakage_phash_near_duplicate():
    """pHash distance <= 8 indicates near-duplicate leakage."""
    # identical hex hash has distance 0
    h1 = "8f8f8f8f8f8f8f8f"
    h2 = "8f8f8f8f8f8f8f8f" # dist = 0
    h3 = "8f8f8f8f8f8f8f8e" # dist = 1
    h4 = "0000000000000000" # dist = 32

    dataset_phashes = {"train_01.jpg": h1, "train_02.jpg": h4}
    holdout_phashes = {"holdout_01.jpg": h2, "holdout_02.jpg": h3}

    collisions = check_leakage_phash(dataset_phashes, holdout_phashes, max_hamming=8)
    assert len(collisions) >= 1
    # Check that train_01 collides with both holdout_01 (dist 0) and holdout_02 (dist 1)
    colliding_pairs = [(c[0], c[1]) for c in collisions]
    assert ("train_01.jpg", "holdout_01.jpg") in colliding_pairs
    assert ("train_01.jpg", "holdout_02.jpg") in colliding_pairs
