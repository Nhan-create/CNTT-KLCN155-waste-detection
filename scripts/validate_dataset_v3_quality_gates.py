# Mô tả: Thư viện kiểm tra các tiêu chuẩn Quality Gates đối với hình học khung bao, taxonomy 10 lớp và trùng lặp bbox.
"""
Dataset Quality Gates Validation Library (CNTT-KLCN155 Phase 4B)
Enforces mathematical, perceptual, and structural integrity rules for Object Detection datasets:
1. Geometry Gate: valid coordinates, NaN/Inf rejection, non-negative, ordered (x1<x2, y1<y2),
   3-state categorization (VALID, FLAG_FOR_REVIEW, INVALID).
2. Taxonomy Gate: class ID within [0..9].
3. Duplicate Gate: detect identical or near-identical bounding boxes (IoU >= 0.95).
4. Hard-Negative Gate: official standard is 0-byte file, distinguishes MISSING, WHITESPACE, HAS_BOXES.
5. Leakage Gate: exact SHA-256 collision and perceptual pHash Hamming distance (<= 8) vs Holdout benchmark.
6. Integration Runner: audits real disk images and labels, outputting structured JSON reports.
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from PIL import Image
import imagehash

# Three-state validation constants
STATUS_VALID = "VALID"
STATUS_FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
STATUS_INVALID = "INVALID"

SEVERITY_NONE = "NONE"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def check_box_geometry(
    box_xyxy: List[float],
    img_width: float,
    img_height: float,
    min_area_norm: float = 0.0001,
    max_area_norm: float = 0.95
) -> Tuple[str, str]:
    """
    Check if bounding box satisfies geometry constraints.
    Returns (status, message) where status is:
      - STATUS_VALID: Normal valid bounding box.
      - STATUS_FLAG_FOR_REVIEW: Geometrically valid, but unusually large (>95%),
        sub-resolution (<0.0001), or extreme aspect ratio.
      - STATUS_INVALID: Mathematical violation (NaN, Inf, negative, out-of-bounds, inverted).
    """
    # 1. Validate image dimensions
    if not (isinstance(img_width, (int, float)) and isinstance(img_height, (int, float))):
        return STATUS_INVALID, "Image dimensions must be numeric"
    if math.isnan(img_width) or math.isinf(img_width) or img_width <= 0:
        return STATUS_INVALID, f"Invalid image width: {img_width}"
    if math.isnan(img_height) or math.isinf(img_height) or img_height <= 0:
        return STATUS_INVALID, f"Invalid image height: {img_height}"

    # 2. Validate box list length and numeric types
    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        return STATUS_INVALID, f"Box must contain exactly 4 coordinates, got {box_xyxy}"

    for idx, c in enumerate(box_xyxy):
        if not isinstance(c, (int, float)):
            return STATUS_INVALID, f"Coordinate {idx} must be numeric, got {type(c)}"
        if math.isnan(c) or math.isinf(c):
            return STATUS_INVALID, f"Coordinate {idx} is NaN or Inf: {c}"

    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])

    # 3. Allow epsilon tolerance (0.01 px) for floating-point rounding from normalized formats
    eps = 0.01
    if -eps <= x1 < 0.0:
        x1 = 0.0
    if -eps <= y1 < 0.0:
        y1 = 0.0
    if img_width < x2 <= (img_width + eps):
        x2 = float(img_width)
    if img_height < y2 <= (img_height + eps):
        y2 = float(img_height)

    # Check non-negative
    if x1 < 0.0 or y1 < 0.0:
        return STATUS_INVALID, f"Coordinates cannot be negative: [{x1}, {y1}, {x2}, {y2}]"

    # 4. Check upper boundary against image dimensions
    if x2 > img_width or y2 > img_height:
        return STATUS_INVALID, f"Box [{x1}, {y1}, {x2}, {y2}] exceeds image boundaries ({img_width}x{img_height})"

    # 5. Check ordering (x1 < x2, y1 < y2)
    if x1 >= x2:
        return STATUS_INVALID, f"Invalid horizontal ordering: x1 ({x1}) >= x2 ({x2})"
    if y1 >= y2:
        return STATUS_INVALID, f"Invalid vertical ordering: y1 ({y1}) >= y2 ({y2})"

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    img_area = float(img_width * img_height)
    norm_area = area / img_area

    # 6. Flag for human/QA review if unusually large or sub-resolution (Do NOT auto-delete)
    if norm_area > max_area_norm:
        return STATUS_FLAG_FOR_REVIEW, (
            f"Unusually large bounding box ({norm_area*100:.2f}% of canvas); "
            f"requires human/QA verification (potential full-canvas or pile)"
        )

    if norm_area < min_area_norm:
        return STATUS_FLAG_FOR_REVIEW, (
            f"Sub-resolution bounding box ({norm_area*100:.4f}% of canvas); check if noise"
        )

    aspect_ratio = bw / bh
    if aspect_ratio < 0.02 or aspect_ratio > 50.0:
        return STATUS_FLAG_FOR_REVIEW, f"Extreme aspect ratio ({aspect_ratio:.2f}); check if sliver or edge artifact"

    return STATUS_VALID, "OK"


def check_taxonomy(class_id: int, num_classes: int = 10) -> Tuple[str, str]:
    """Check if class ID is valid for the 10 fixed classes."""
    if not isinstance(class_id, int):
        return STATUS_INVALID, f"Class ID must be an integer, got {type(class_id)}"
    if not (0 <= class_id < num_classes):
        return STATUS_INVALID, f"Invalid class ID {class_id}; must be in [0..{num_classes-1}]"
    return STATUS_VALID, "OK"


def check_duplicate_boxes(
    boxes_with_class: List[Tuple[int, List[float]]],
    iou_threshold: float = 0.95
) -> List[Tuple[int, int, float, str]]:
    """Detect duplicate bounding boxes belonging to the same class."""
    duplicates = []
    n = len(boxes_with_class)
    for i in range(n):
        cid_i, box_i = boxes_with_class[i]
        for j in range(i + 1, n):
            cid_j, box_j = boxes_with_class[j]
            if cid_i == cid_j:
                iou = calculate_iou(box_i, box_j)
                if iou >= iou_threshold:
                    duplicates.append((
                        i, j, float(iou),
                        f"Duplicate boxes {i} and {j} for class {cid_i} (IoU={iou:.4f})"
                    ))
    return duplicates


def check_hard_negative_label(
    label_source: Union[bytes, str, Path, None]
) -> Tuple[str, str, str]:
    """
    Validate hard-negative label convention.
    Official dataset standard: EXACTLY 0 bytes file.
    
    Returns:
      (status, code, message)
      - (STATUS_VALID, "EMPTY_0_BYTES", "Standard 0-byte negative label")
      - (STATUS_FLAG_FOR_REVIEW, "WHITESPACE_ONLY", "Label contains only whitespace; recommend normalize to 0 bytes")
      - (STATUS_INVALID, "HAS_BOXES", "Hard negative file contains bounding box annotations")
      - (STATUS_INVALID, "MISSING_FILE", "Label file does not exist")
    """
    if label_source is None:
        return STATUS_INVALID, "MISSING_FILE", "Label file does not exist"

    raw_bytes: bytes
    if isinstance(label_source, Path):
        if not label_source.exists():
            return STATUS_INVALID, "MISSING_FILE", f"Label file not found on disk: {label_source}"
        raw_bytes = label_source.read_bytes()
    elif isinstance(label_source, str):
        p = Path(label_source)
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
        else:
            raw_bytes = label_source.encode("utf-8")
    elif isinstance(label_source, bytes):
        raw_bytes = label_source
    else:
        return STATUS_INVALID, "INVALID_INPUT", f"Unsupported label source type: {type(label_source)}"

    if len(raw_bytes) == 0:
        return STATUS_VALID, "EMPTY_0_BYTES", "Standard 0-byte negative label"

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return STATUS_INVALID, "CORRUPT_BYTES", "Label file cannot be decoded as utf-8"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 0:
        return STATUS_FLAG_FOR_REVIEW, "WHITESPACE_ONLY", (
            "Label file contains only whitespace; recommended action: normalize to 0-byte file"
        )

    return STATUS_INVALID, "HAS_BOXES", (
        f"Hard negative image must not have bounding boxes, but found {len(lines)} annotations"
    )


def check_leakage_hashes(
    dataset_hashes: Dict[str, str],
    holdout_hashes: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """Detect exact SHA-256 collisions between dataset images and holdout images."""
    collisions = []
    holdout_lookup = {h.lower(): fn for fn, h in holdout_hashes.items()}
    for fn, h in dataset_hashes.items():
        hl = h.lower()
        if hl in holdout_lookup:
            collisions.append((fn, holdout_lookup[hl], h))
    return collisions


def check_leakage_phash(
    dataset_phashes: Dict[str, Union[str, imagehash.ImageHash]],
    holdout_phashes: Dict[str, Union[str, imagehash.ImageHash]],
    max_hamming: int = 8
) -> List[Tuple[str, str, int, str]]:
    """
    Detect perceptual near-duplicate leakage using pHash Hamming distance.
    Hamming distance <= max_hamming (default 8) indicates near-duplicate image.
    """
    collisions = []
    # Convert strings to ImageHash objects
    h_objs = {}
    for fn, val in holdout_phashes.items():
        if isinstance(val, imagehash.ImageHash):
            h_objs[fn] = val
        else:
            h_objs[fn] = imagehash.hex_to_hash(str(val))

    for fn, val in dataset_phashes.items():
        d_obj = val if isinstance(val, imagehash.ImageHash) else imagehash.hex_to_hash(str(val))
        for h_fn, h_obj in h_objs.items():
            dist = int(d_obj - h_obj)
            if dist <= max_hamming:
                collisions.append((
                    fn, h_fn, dist,
                    f"Perceptual near-duplicate detected (Hamming distance={dist} <= {max_hamming})"
                ))
    return collisions


def validate_dataset_entry(
    image_path: Path,
    label_path: Optional[Path],
    is_negative: bool = False,
    num_classes: int = 10,
    holdout_sha256_set: Optional[Dict[str, str]] = None,
    holdout_phash_set: Optional[Dict[str, Union[str, imagehash.ImageHash]]] = None
) -> Dict[str, Any]:
    """
    Validates a real image and label file against all quality gates.
    Returns structured audit result dictionary.
    """
    fn = image_path.name
    result = {
        "filename": fn,
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path else None,
        "validation_status": STATUS_VALID,
        "error_category": "NONE",
        "severity": SEVERITY_NONE,
        "evidence": {},
        "recommended_action": "KEEP"
    }

    # 1. Image Existence & Integrity
    if not image_path.exists():
        result["validation_status"] = STATUS_INVALID
        result["error_category"] = "MISSING_IMAGE"
        result["severity"] = SEVERITY_CRITICAL
        result["evidence"] = {"message": f"Image file not found: {image_path}"}
        result["recommended_action"] = "INVESTIGATE_MISSING_IMAGE"
        return result

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            img_format = img.format
            # Compute pHash for leakage check
            p_hash_obj = imagehash.phash(img)
            p_hash_str = str(p_hash_obj)
    except Exception as e:
        result["validation_status"] = STATUS_INVALID
        result["error_category"] = "CORRUPT_IMAGE"
        result["severity"] = SEVERITY_CRITICAL
        result["evidence"] = {"error": str(e)}
        result["recommended_action"] = "QUARANTINE_CORRUPT_IMAGE"
        return result

    result["evidence"]["image_size"] = [w, h]
    result["evidence"]["image_format"] = img_format
    result["evidence"]["phash"] = p_hash_str

    # 2. Leakage check against holdout
    if holdout_phash_set:
        phash_collisions = check_leakage_phash({fn: p_hash_obj}, holdout_phash_set, max_hamming=8)
        if phash_collisions:
            result["validation_status"] = STATUS_INVALID
            result["error_category"] = "HOLDOUT_LEAKAGE"
            result["severity"] = SEVERITY_CRITICAL
            result["evidence"]["phash_collision"] = phash_collisions[0]
            result["recommended_action"] = "QUARANTINE_LEAKAGE"
            return result

    # 3. Label Validation
    if is_negative:
        status, code, msg = check_hard_negative_label(label_path)
        if status != STATUS_VALID:
            result["validation_status"] = status
            result["error_category"] = f"NEGATIVE_LABEL_{code}"
            result["severity"] = SEVERITY_CRITICAL if status == STATUS_INVALID else SEVERITY_WARNING
            result["evidence"]["negative_check"] = {"code": code, "message": msg}
            result["recommended_action"] = "NORMALIZE_TO_EMPTY_FILE" if code == "WHITESPACE_ONLY" else "QUARANTINE"
        return result

    # Positive detection image: label must exist
    if label_path is None or not label_path.exists():
        result["validation_status"] = STATUS_INVALID
        result["error_category"] = "MISSING_LABEL"
        result["severity"] = SEVERITY_CRITICAL
        result["evidence"] = {"message": f"Label file not found: {label_path}"}
        result["recommended_action"] = "LOCATE_OR_ANNOTATE_LABEL"
        return result

    with open(label_path, "r", encoding="utf-8") as lf:
        lines = [line.strip() for line in lf if line.strip()]

    if len(lines) == 0:
        result["validation_status"] = STATUS_FLAG_FOR_REVIEW
        result["error_category"] = "EMPTY_LABEL_FOR_POSITIVE_SAMPLE"
        result["severity"] = SEVERITY_WARNING
        result["evidence"] = {"message": "Positive sample has 0 bounding boxes"}
        result["recommended_action"] = "VERIFY_IF_INTENDED_NEGATIVE"
        return result

    parsed_boxes = []
    box_statuses = []
    for line_idx, line in enumerate(lines):
        toks = line.split()
        if len(toks) < 5:
            box_statuses.append((STATUS_INVALID, f"Malformed line {line_idx}: '{line}'"))
            continue

        try:
            cid = int(toks[0])
            xc, yc, bw_norm, bh_norm = float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
        except ValueError as ve:
            box_statuses.append((STATUS_INVALID, f"Non-numeric values in line {line_idx}: {ve}"))
            continue

        # Check taxonomy
        tax_status, tax_msg = check_taxonomy(cid, num_classes)
        if tax_status != STATUS_VALID:
            box_statuses.append((tax_status, f"Line {line_idx}: {tax_msg}"))
            continue

        # Convert YOLO (xc, yc, w, h) normalized to absolute [x1, y1, x2, y2]
        x1 = (xc - bw_norm / 2.0) * w
        y1 = (yc - bh_norm / 2.0) * h
        x2 = (xc + bw_norm / 2.0) * w
        y2 = (yc + bh_norm / 2.0) * h

        geo_status, geo_msg = check_box_geometry([x1, y1, x2, y2], w, h)
        box_statuses.append((geo_status, f"Line {line_idx}: {geo_msg}"))
        parsed_boxes.append((cid, [x1, y1, x2, y2]))

    # Check duplicates
    dups = check_duplicate_boxes(parsed_boxes, iou_threshold=0.95)
    if dups:
        box_statuses.append((STATUS_FLAG_FOR_REVIEW, f"Found {len(dups)} duplicate boxes: {dups[0][3]}"))

    # Determine overall status
    has_invalid = any(st == STATUS_INVALID for st, _ in box_statuses)
    has_flag = any(st == STATUS_FLAG_FOR_REVIEW for st, _ in box_statuses)

    if has_invalid:
        result["validation_status"] = STATUS_INVALID
        result["error_category"] = "GEOMETRY_OR_TAXONOMY_ERROR"
        result["severity"] = SEVERITY_CRITICAL
        result["recommended_action"] = "FIX_ANNOTATION_COORDINATES"
    elif has_flag:
        result["validation_status"] = STATUS_FLAG_FOR_REVIEW
        result["error_category"] = "ANOMALY_FLAGGED"
        result["severity"] = SEVERITY_WARNING
        result["recommended_action"] = "MANUAL_QA_REVIEW"
    else:
        result["validation_status"] = STATUS_VALID
        result["recommended_action"] = "KEEP"

    result["evidence"]["num_boxes"] = len(lines)
    result["evidence"]["box_details"] = [msg for _, msg in box_statuses if _ != STATUS_VALID]
    return result
