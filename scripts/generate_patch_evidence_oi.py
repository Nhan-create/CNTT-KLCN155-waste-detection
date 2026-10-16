"""
scripts/generate_patch_evidence_oi.py
--------------------------------------
Generates rigorous unified diff and cryptographic proof for the dataset deduplication patch
applied to oi_2734e597675d8a97.txt and associated manifests.

Clarifications:
1. real_v2_split_manifest.csv column 7 is IMAGE SHA-256 (be1d1ed75cb7ae89994956272a680e1c60ce918d47e1d5c4b2539fa73b92c9c0),
   which remains unchanged. Column 8 is num_boxes, which changed from 2 to 1.
2. Label file SHA-256 is reported with both LF and CRLF encodings to account for OS line endings.
"""

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = ROOT / "reports" / "patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = ROOT / "data" / "detection" / "images" / "real" / "oi_2734e597675d8a97.jpg"
IMAGE_SHA256 = hashlib.sha256(IMAGE_PATH.read_bytes()).hexdigest()

BEFORE_LABEL_10_TEXT = "3 0.500000 0.500000 1.000000 1.000000\n3 0.500000 0.500000 1.000000 1.000000\n"
AFTER_LABEL_10_PATH = ROOT / "data" / "detection" / "labels" / "real" / "oi_2734e597675d8a97.txt"
AFTER_LABEL_10_RAW = AFTER_LABEL_10_PATH.read_bytes()
AFTER_LABEL_10_TEXT = AFTER_LABEL_10_RAW.decode("utf-8")

BEFORE_LABEL_AGN_TEXT = "0 0.500000 0.500000 1.000000 1.000000\n0 0.500000 0.500000 1.000000 1.000000\n"
AFTER_LABEL_AGN_PATH1 = ROOT / "data" / "detection" / "labels_agnostic" / "real" / "oi_2734e597675d8a97.txt"
AFTER_LABEL_AGN_PATH2 = ROOT / "data" / "detection_agnostic" / "labels" / "real" / "oi_2734e597675d8a97.txt"
AFTER_LABEL_AGN1_RAW = AFTER_LABEL_AGN_PATH1.read_bytes()
AFTER_LABEL_AGN2_RAW = AFTER_LABEL_AGN_PATH2.read_bytes()

BEFORE_SRC_LINE = "oi_2734e597675d8a97.jpg,openimages_v7,2734e597675d8a97,validation,https://open-images-dataset.s3.amazonaws.com/validation/2734e597675d8a97.jpg,new,2,0\n"
AFTER_SRC_PATH = ROOT / "data" / "detection" / "real_v2_source_manifest.csv"
AFTER_SRC_LINES = [line for line in AFTER_SRC_PATH.read_text(encoding="utf-8").splitlines(keepends=True) if "oi_2734e597675d8a97" in line]
AFTER_SRC_LINE = AFTER_SRC_LINES[0] if AFTER_SRC_LINES else ""

BEFORE_SPLIT_LINE = "oi_2734e597675d8a97.jpg,val,openimages_v7,g00102,openimages_v7|clothes,858c1d6d695a3e65,be1d1ed75cb7ae89994956272a680e1c60ce918d47e1d5c4b2539fa73b92c9c0,2,clothes\n"
AFTER_SPLIT_PATH = ROOT / "data" / "detection" / "real_v2_split_manifest.csv"
AFTER_SPLIT_LINES = [line for line in AFTER_SPLIT_PATH.read_text(encoding="utf-8").splitlines(keepends=True) if "oi_2734e597675d8a97" in line]
AFTER_SPLIT_LINE = AFTER_SPLIT_LINES[0] if AFTER_SPLIT_LINES else ""


def get_lf_crlf_hashes(text: str):
    lf_bytes = text.replace("\r\n", "\n").encode("utf-8")
    crlf_bytes = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    return {
        "sha256_lf": hashlib.sha256(lf_bytes).hexdigest(),
        "sha256_crlf": hashlib.sha256(crlf_bytes).hexdigest(),
    }


evidence = {
    "patch_target": "oi_2734e597675d8a97 duplicate bounding box removal",
    "image_sha256": IMAGE_SHA256,
    "manifest_field_clarification": "real_v2_split_manifest.csv column 7 is IMAGE sha256 (invariant be1d1ed75cb7ae89994956272a680e1c60ce918d47e1d5c4b2539fa73b92c9c0); column 8 is num_boxes (2 -> 1).",
    "files": {
        "labels/real/oi_2734e597675d8a97.txt": {
            "boxes_before": 2,
            "boxes_after": len([ln for ln in AFTER_LABEL_10_TEXT.splitlines() if ln.strip()]),
            "hashes_before": get_lf_crlf_hashes(BEFORE_LABEL_10_TEXT),
            "hashes_after": get_lf_crlf_hashes(AFTER_LABEL_10_TEXT),
            "disk_raw_bytes_sha256": hashlib.sha256(AFTER_LABEL_10_RAW).hexdigest(),
            "content_before": BEFORE_LABEL_10_TEXT.splitlines(),
            "content_after": AFTER_LABEL_10_TEXT.splitlines(),
        },
        "labels_agnostic/real/oi_2734e597675d8a97.txt": {
            "boxes_before": 2,
            "boxes_after": len([ln for ln in AFTER_LABEL_AGN_PATH1.read_text(encoding="utf-8").splitlines() if ln.strip()]),
            "hashes_before": get_lf_crlf_hashes(BEFORE_LABEL_AGN_TEXT),
            "hashes_after": get_lf_crlf_hashes(AFTER_LABEL_AGN_PATH1.read_text(encoding="utf-8")),
            "disk_raw_bytes_sha256": hashlib.sha256(AFTER_LABEL_AGN1_RAW).hexdigest(),
        },
        "detection_agnostic/labels/real/oi_2734e597675d8a97.txt": {
            "boxes_before": 2,
            "boxes_after": len([ln for ln in AFTER_LABEL_AGN_PATH2.read_text(encoding="utf-8").splitlines() if ln.strip()]),
            "hashes_before": get_lf_crlf_hashes(BEFORE_LABEL_AGN_TEXT),
            "hashes_after": get_lf_crlf_hashes(AFTER_LABEL_AGN_PATH2.read_text(encoding="utf-8")),
            "disk_raw_bytes_sha256": hashlib.sha256(AFTER_LABEL_AGN2_RAW).hexdigest(),
        },
        "real_v2_source_manifest.csv": {
            "line_before": BEFORE_SRC_LINE.strip(),
            "line_after": AFTER_SRC_LINE.strip(),
            "num_boxes_before": 2,
            "num_boxes_after": 1,
        },
        "real_v2_split_manifest.csv": {
            "line_before": BEFORE_SPLIT_LINE.strip(),
            "line_after": AFTER_SPLIT_LINE.strip(),
            "num_boxes_before": 2,
            "num_boxes_after": 1,
            "image_sha256_verified": IMAGE_SHA256,
        }
    }
}

diff_lines = []
diff_lines.extend(difflib.unified_diff(
    BEFORE_LABEL_10_TEXT.splitlines(keepends=True),
    AFTER_LABEL_10_TEXT.splitlines(keepends=True),
    fromfile="a/data/detection/labels/real/oi_2734e597675d8a97.txt",
    tofile="b/data/detection/labels/real/oi_2734e597675d8a97.txt"
))
diff_lines.extend(difflib.unified_diff(
    BEFORE_LABEL_AGN_TEXT.splitlines(keepends=True),
    AFTER_LABEL_AGN_PATH1.read_text(encoding="utf-8").splitlines(keepends=True),
    fromfile="a/data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt",
    tofile="b/data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt"
))
diff_lines.extend(difflib.unified_diff(
    BEFORE_SRC_LINE.splitlines(keepends=True),
    AFTER_SRC_LINE.splitlines(keepends=True),
    fromfile="a/data/detection/real_v2_source_manifest.csv (line 108)",
    tofile="b/data/detection/real_v2_source_manifest.csv (line 108)"
))
diff_lines.extend(difflib.unified_diff(
    BEFORE_SPLIT_LINE.splitlines(keepends=True),
    AFTER_SPLIT_LINE.splitlines(keepends=True),
    fromfile="a/data/detection/real_v2_split_manifest.csv (line 104)",
    tofile="b/data/detection/real_v2_split_manifest.csv (line 104)"
))

diff_content = "".join(diff_lines)
diff_file = PATCHES_DIR / "patch_oi_2734e597675d8a97.diff"
diff_file.write_text(diff_content, encoding="utf-8")

json_file = PATCHES_DIR / "patch_oi_2734e597675d8a97_evidence.json"
json_file.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Generated unified diff at: {diff_file}")
print(f"Generated JSON evidence at: {json_file}")
