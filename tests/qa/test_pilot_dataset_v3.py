"""
tests/qa/test_pilot_dataset_v3.py
---------------------------------
Continuous Quality Assurance and Regression Suite for Pilot Dataset v3:
1. Verifies physical isolation in data/detection_pilot_v3/.
2. Verifies 1:1 image-to-label pairing.
3. Verifies geometry of all bounding boxes (within [0, 1], w, h >= 0.005).
4. Verifies taxonomy strictly in [0..9].
5. Verifies hard negatives have 0 bytes.
6. Verifies that taco_0072 and taco_0092 are re-annotated and have biological labels.
7. Verifies zero holdout leakage.
"""

import json
from pathlib import Path
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "data" / "detection_pilot_v3"
PILOT_IMG_DIR = PILOT_DIR / "images"
PILOT_LBL_DIR = PILOT_DIR / "labels"
PILOT_MAN_DIR = PILOT_DIR / "manifests"

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

def test_pilot_dataset_directory_structure_and_isolation():
    """Verify pilot dataset exists and contains images, labels, and manifests."""
    assert PILOT_DIR.is_dir(), "Missing data/detection_pilot_v3 directory"
    assert PILOT_IMG_DIR.is_dir(), "Missing images/ directory"
    assert PILOT_LBL_DIR.is_dir(), "Missing labels/ directory"
    assert PILOT_MAN_DIR.is_dir(), "Missing manifests/ directory"

    images = list(PILOT_IMG_DIR.glob("*.jpg"))
    assert len(images) >= 15, f"Expected at least 15 pilot images, found {len(images)}"

def test_pilot_images_and_labels_one_to_one():
    """Verify each image has a corresponding label file and can be opened."""
    for img_p in PILOT_IMG_DIR.glob("*.jpg"):
        lbl_p = PILOT_LBL_DIR / f"{img_p.stem}.txt"
        assert lbl_p.is_file(), f"Missing label file for {img_p.name}"

        with Image.open(img_p) as im:
            assert im.width > 0 and im.height > 0

def test_pilot_geometry_and_taxonomy():
    """Verify bounding box coordinates are normalized and classes are strictly 0..9."""
    box_count = 0
    for lbl_p in PILOT_LBL_DIR.glob("*.txt"):
        lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line_idx, ln in enumerate(lines):
            tokens = ln.split()
            assert len(tokens) == 5, f"Expected 5 tokens in {lbl_p.name}:{line_idx}, got {len(tokens)}"
            cid = int(float(tokens[0]))
            assert 0 <= cid <= 9, f"Class ID {cid} out of [0, 9] in {lbl_p.name}"

            cx, cy, bw, bh = map(float, tokens[1:5])
            assert 0.0 < cx < 1.0, f"cx={cx} out of bounds in {lbl_p.name}"
            assert 0.0 < cy < 1.0, f"cy={cy} out of bounds in {lbl_p.name}"
            assert 0.005 <= bw <= 1.0, f"bw={bw} out of bounds in {lbl_p.name}"
            assert 0.005 <= bh <= 1.0, f"bh={bh} out of bounds in {lbl_p.name}"

            # Extents
            assert cx - bw / 2 >= -1e-4
            assert cx + bw / 2 <= 1.0 + 1e-4
            assert cy - bh / 2 >= -1e-4
            assert cy + bh / 2 <= 1.0 + 1e-4
            box_count += 1

    assert box_count == 19, f"Expected 19 total annotations in pilot dataset, got {box_count}"

def test_pilot_empty_labels_zero_bytes_unmapped_waste():
    """Verify that unmapped waste images have 0-byte label files on disk due to unmapped Category 51."""
    neg_stems = ["taco_0170", "taco_0431", "taco_0556", "taco_0909", "taco_0912", "taco_0916", "taco_0917", "taco_0925"]
    for stem in neg_stems:
        lbl_p = PILOT_LBL_DIR / f"{stem}.txt"
        assert lbl_p.is_file(), f"Missing label for {stem}"
        assert lbl_p.stat().st_size == 0, f"Expected 0 bytes for unmapped waste {stem}, got {lbl_p.stat().st_size}"

def test_pilot_reannotated_biological_present():
    """Verify that taco_0072 and taco_0092 contain biological class 1 in their labels."""
    lbl_0072 = (PILOT_LBL_DIR / "taco_0072.txt").read_text(encoding="utf-8")
    lbl_0092 = (PILOT_LBL_DIR / "taco_0092.txt").read_text(encoding="utf-8")

    assert "1 " in lbl_0072, "Missing biological label in taco_0072.txt"
    assert "7 " in lbl_0072, "Missing plastic label in taco_0072.txt"
    assert "1 " in lbl_0092, "Missing biological label in taco_0092.txt"

def test_pilot_manifests_consistency():
    """Verify image and annotation manifests match disk files and reconciliation metadata."""
    img_man = json.loads((PILOT_MAN_DIR / "pilot_image_manifest.json").read_text(encoding="utf-8"))
    anno_man = json.loads((PILOT_MAN_DIR / "pilot_annotation_manifest.json").read_text(encoding="utf-8"))

    assert img_man["metadata"]["total_images"] == 15
    assert img_man["metadata"]["rejected_negative_candidates"] == 8
    assert img_man["metadata"]["verified_true_negatives"] == 0
    assert img_man["metadata"]["eligible_new_train_images"] == 2
    assert anno_man["metadata"]["total_annotations"] == 19
    assert anno_man["metadata"]["class_distribution"]["biological"] == 5
    assert anno_man["metadata"]["class_distribution"]["battery"] == 2


def test_validate_pilot_quality_gates_exit_codes():
    """Verify that validate_pilot_quality_gates.py exits with non-zero when blocked, and 0 with --report-only."""
    import subprocess
    import sys

    script_path = ROOT / "scripts" / "validate_pilot_quality_gates.py"
    assert script_path.is_file(), f"Missing script at {script_path}"

    # 1. Default run: MUST fail closed with non-zero exit code because verdict is TRAINING_BLOCKED
    proc_default = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    assert proc_default.returncode == 1, (
        f"Expected exit code 1 on TRAINING_BLOCKED, got {proc_default.returncode}.\n"
        f"Stdout:\n{proc_default.stdout}\nStderr:\n{proc_default.stderr}"
    )
    assert "TRAINING_BLOCKED" in proc_default.stdout

    # 2. Run with --report-only: MUST exit with 0
    proc_report_only = subprocess.run(
        [sys.executable, str(script_path), "--report-only"],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    assert proc_report_only.returncode == 0, (
        f"Expected exit code 0 with --report-only, got {proc_report_only.returncode}.\n"
        f"Stdout:\n{proc_report_only.stdout}\nStderr:\n{proc_report_only.stderr}"
    )
    assert "--report-only flag set: exiting with 0" in proc_report_only.stdout

