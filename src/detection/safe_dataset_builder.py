"""
src/detection/safe_dataset_builder.py
-------------------------------------
Fail-closed, allowlist-driven dataset intake and builder for waste detection.

Core Invariants (Sprint AI-UPGRADE-01 Task 5B Hotfix):
1. STRICT ALLOWLIST GATEKEEPER:
   - Positive candidates are ONLY accepted if visual_qa is in VALID_POSITIVE_QA_STATUSES.
   - Clean negatives are ONLY accepted if visual_qa is in VALID_CLEAN_NEGATIVE_QA_STATUSES.
   - QUARANTINED, PENDING, REJECTED, SOURCE_UNVERIFIED, QA_FAIL, empty/unknown visual_qa
     are ALWAYS rejected unconditionally.
   - OBJECT_ONLY candidates are segregated and rejected by default unless explicit slice flag
     and approval are provided.
2. DUAL-FLAG APPROVAL VERIFICATION:
   - When require_approval=True, requires BOTH approved_for_train is True
     AND user_train_approval == "APPROVED_FOR_TRAIN".
3. PHYSICAL STAGING & VERIFICATION:
   - Physical output is created ONLY for approved images passing all allowlists.
   - Images and labels are verified on disk (PIL decode, valid YOLO format).
   - Empty labels are NEVER accepted unless certified QA_PASS_CLEAN_NEGATIVE.
   - Protects baseline datasets from overwrite (fail-closed guard).
   - When 0 approved images exist, returns clear EMPTY_STAGING status.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

# Strict Allowlist Definitions
VALID_POSITIVE_QA_STATUSES: Set[str] = {
    "VISUAL_QA_PASS",
    "QA_PASS",
    "QA_PASS_SINGLE_OBJECT",
    "QA_PASS_MULTI_OBJECT",
}

VALID_ISOLATED_OBJECT_QA_STATUSES: Set[str] = {
    "QA_PASS_OBJECT_ONLY_WHITE_BG",
    "QA_PASS_ISOLATED_OBJECT",
}

VALID_CLEAN_NEGATIVE_QA_STATUSES: Set[str] = {
    "QA_PASS_CLEAN_NEGATIVE",
}

BLOCKED_QA_SUBSTRINGS: Tuple[str, ...] = (
    "QUARANTINE",
    "PENDING",
    "REJECT",
    "UNVERIFIED",
    "FAIL",
    "ERROR",
    "BLOCKED",
    "CORRUPT",
    "NEEDS_REANNOTATION",
    "NEEDS_FIX",
    "TECHNICAL_VALID",
)


def verify_visual_qa_pass_evidence(item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verifies that a candidate claiming VISUAL_QA_PASS has an explicit, non-empty
    human/visual review record (`review_evidence` with `reviewed_by`, `review_timestamp`,
    and `review_note`). Prevents automated scripts from assigning VISUAL_QA_PASS by default.
    """
    cid = str(item.get("candidate_id") or item.get("base_id") or "UNKNOWN")
    ev = item.get("review_evidence")
    if not isinstance(ev, dict):
        return False, f"Candidate {cid} claims VISUAL_QA_PASS without a 'review_evidence' dictionary."

    for req_key in ("reviewed_by", "review_timestamp", "review_note"):
        val = str(ev.get(req_key) or "").strip()
        if not val:
            return False, f"Candidate {cid} claims VISUAL_QA_PASS but review_evidence['{req_key}'] is missing or empty."

    return True, None


class SafeDatasetBuilderError(Exception):
    """Base exception for safe dataset builder failures."""
    pass


class UnmanifestedScanError(SafeDatasetBuilderError):
    """Raised when an attempt is made to glob labels/images without manifest validation."""
    pass


class OverwriteBaselineError(SafeDatasetBuilderError):
    """Raised when output_dir targets a protected/frozen dataset directory."""
    pass


class InvalidAnnotationError(SafeDatasetBuilderError):
    """Raised when on-disk annotation is corrupt or invalid."""
    pass


class InvalidImageError(SafeDatasetBuilderError):
    """Raised when on-disk image is missing or cannot be decoded."""
    pass


def validate_candidate_for_intake(
    item: Dict[str, Any],
    require_approval: bool = True,
    allow_isolated_objects: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single candidate dictionary from a synchronized manifest.

    Enforces strict FAIL-CLOSED allowlist:
    - Rejects missing, empty, or blocked visual_qa (including NEEDS_REANNOTATION and TECHNICAL_VALID).
    - Requires explicit review_evidence when visual_qa == 'VISUAL_QA_PASS'.
    - Rejects SOURCE_UNVERIFIED or unverified sources.
    - Rejects positive candidates not in VALID_POSITIVE_QA_STATUSES.
    - Rejects negative candidates (num_boxes == 0) not in VALID_CLEAN_NEGATIVE_QA_STATUSES.
    - Rejects isolated objects unless allow_isolated_objects=True.
    - Enforces dual approval flags (approved_for_train=True and user_train_approval="APPROVED_FOR_TRAIN").

    Returns:
        (is_valid, rejection_reason)
    """
    cid = str(item.get("candidate_id") or "UNKNOWN")
    v_qa = str(item.get("visual_qa") or "").strip()
    user_app = str(item.get("user_train_approval") or "").strip()
    approved = item.get("approved_for_train", False)
    num_boxes = item.get("num_boxes")
    scene_context = str(item.get("scene_context") or "").strip()
    verification_status = str(item.get("verification_status") or "").strip()

    # 1. Missing or empty visual_qa
    if not v_qa:
        return False, f"Candidate {cid} has empty or missing visual_qa. Rejected."

    # 2. Blocked keywords in visual_qa
    v_qa_upper = v_qa.upper()
    for blocked_kw in BLOCKED_QA_SUBSTRINGS:
        if blocked_kw in v_qa_upper:
            return False, f"Candidate {cid} has blocked visual_qa='{v_qa}' (matched '{blocked_kw}')."

    # 2b. If VISUAL_QA_PASS is claimed, require explicit review_evidence
    if v_qa == "VISUAL_QA_PASS":
        ev_ok, ev_err = verify_visual_qa_pass_evidence(item)
        if not ev_ok:
            return False, ev_err

    # 3. Source verification check
    if verification_status and verification_status != "VERIFIED":
        return False, f"Candidate {cid} has unverified source status: '{verification_status}'."

    # 4. Box count presence
    if num_boxes is None:
        return False, f"Candidate {cid} has missing 'num_boxes' field."

    try:
        num_boxes = int(num_boxes)
    except (ValueError, TypeError):
        return False, f"Candidate {cid} has non-integer num_boxes: {num_boxes}."

    if num_boxes < 0:
        return False, f"Candidate {cid} has negative num_boxes: {num_boxes}."

    # 5. Clean Negative validation (0 boxes)
    if num_boxes == 0:
        if v_qa not in VALID_CLEAN_NEGATIVE_QA_STATUSES:
            return False, (
                f"Candidate {cid} has 0 boxes but visual_qa='{v_qa}' (not QA_PASS_CLEAN_NEGATIVE). "
                f"Negative candidates must have visual_qa in {sorted(VALID_CLEAN_NEGATIVE_QA_STATUSES)}."
            )
        boxes = item.get("boxes", [])
        if len(boxes) != 0:
            return False, f"Candidate {cid} claims 0 boxes but 'boxes' list has {len(boxes)} items."

    # 6. Positive candidates (> 0 boxes)
    else:
        is_isolated = (v_qa in VALID_ISOLATED_OBJECT_QA_STATUSES) or ("OBJECT_ONLY" in scene_context)
        if is_isolated:
            if not allow_isolated_objects:
                return False, (
                    f"Candidate {cid} is marked OBJECT_ONLY ({v_qa}, {scene_context}). "
                    "Segregated from in-the-wild discarded waste dataset."
                )
            if v_qa not in VALID_ISOLATED_OBJECT_QA_STATUSES:
                return False, (
                    f"Candidate {cid} is marked OBJECT_ONLY but visual_qa='{v_qa}' is not in "
                    f"{sorted(VALID_ISOLATED_OBJECT_QA_STATUSES)}."
                )
        else:
            if v_qa not in VALID_POSITIVE_QA_STATUSES:
                return False, (
                    f"Candidate {cid} has visual_qa='{v_qa}' which is not in positive allowlist: "
                    f"{sorted(VALID_POSITIVE_QA_STATUSES)}."
                )

        # Validate bounding box objects if present in manifest
        boxes = item.get("boxes", [])
        if boxes and len(boxes) != num_boxes:
            return False, f"Candidate {cid} num_boxes ({num_boxes}) does not match len(boxes) ({len(boxes)})."

    # 7. Explicit dual approval check
    if require_approval:
        if approved is not True or user_app != "APPROVED_FOR_TRAIN":
            return False, (
                f"Candidate {cid} lacks explicit human/PM approval "
                f"(approved_for_train={approved} must be True, user_train_approval='{user_app}' must be 'APPROVED_FOR_TRAIN')."
            )

    return True, None


def scan_directory_blindly_guard(directory: Path) -> None:
    """
    Prevents blind directory ingestion.
    Raises UnmanifestedScanError unconditionally to enforce fail-closed architecture.
    """
    raise UnmanifestedScanError(
        f"Blind filesystem scanning of '{directory}' is forbidden. "
        "All dataset samples must be loaded through an authorized manifest with verified QA records."
    )


def load_and_filter_manifest(
    manifest_path: Path,
    require_approval: bool = True,
    allow_isolated_objects: bool = False,
) -> Dict[str, Any]:
    """
    Loads a manifest file and partitions candidates into approved and rejected lists.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    accepted = []
    rejected = []

    for item in items:
        ok, reason = validate_candidate_for_intake(
            item,
            require_approval=require_approval,
            allow_isolated_objects=allow_isolated_objects,
        )
        if ok:
            accepted.append(item)
        else:
            rejected.append({"candidate_id": item.get("candidate_id"), "reason": reason})

    return {
        "manifest_name": data.get("manifest_name", manifest_path.name),
        "total_items": len(items),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
    }


import math

def validate_on_disk_sample(
    img_path: Path,
    lbl_path: Path,
    expected_boxes: int,
    is_clean_negative: bool = False,
) -> None:
    """
    Validates on-disk image and YOLO label file pair before staging.
    Enforces:
    - Finite float coordinates (no NaN, no Inf).
    - Positive width and height (w > 0, h > 0).
    - Entire bounding box strictly within [0.0, 1.0] (0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1).
    - Class ID strictly in 0..9.
    - Rejects invalid annotations outright rather than silently modifying them.
    """
    # 1. Image check
    if not img_path.exists():
        raise InvalidImageError(f"Image file does not exist on disk: {img_path}")
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                raise InvalidImageError(f"Image {img_path.name} has non-positive dimensions ({w}x{h}).")
            im.verify()
    except Exception as exc:
        raise InvalidImageError(f"Failed to decode image {img_path.name}: {exc}")

    # 2. Label check
    if not lbl_path.exists():
        raise InvalidAnnotationError(f"Label file does not exist on disk: {lbl_path}")

    content = lbl_path.read_text(encoding="utf-8").strip()

    if is_clean_negative or expected_boxes == 0:
        if content:
            raise InvalidAnnotationError(
                f"Candidate {img_path.stem} is certified clean negative but label file is not empty ({len(content)} chars)."
            )
        return

    # Positive sample: validate format
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if len(lines) != expected_boxes:
        raise InvalidAnnotationError(
            f"Candidate {img_path.stem} label has {len(lines)} boxes, expected {expected_boxes}."
        )

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            raise InvalidAnnotationError(
                f"Line {i+1} in {lbl_path.name} has {len(parts)} tokens (expected 5)."
            )
        try:
            cid = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError as exc:
            raise InvalidAnnotationError(f"Line {i+1} in {lbl_path.name} has non-numeric values: {exc}")

        if not (0 <= cid <= 9):
            raise InvalidAnnotationError(f"Line {i+1} in {lbl_path.name} has invalid class ID {cid} (must be 0..9).")

        xc, yc, bw, bh = coords
        # Check finite values (NaN / Inf)
        for val, name in [(xc, "xc"), (yc, "yc"), (bw, "w"), (bh, "h")]:
            if math.isnan(val) or math.isinf(val):
                raise InvalidAnnotationError(f"Line {i+1} in {lbl_path.name} has non-finite value in {name}: {val}.")

        # Dimensions must be strictly positive
        if bw <= 0.0 or bh <= 0.0:
            raise InvalidAnnotationError(f"Line {i+1} in {lbl_path.name} has non-positive dimensions (w={bw}, h={bh}).")

        # Centers must be in [0.0, 1.0]
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise InvalidAnnotationError(f"Line {i+1} in {lbl_path.name} center out of bounds (xc={xc}, yc={yc}).")

        # Full bounding box boundaries check: [x1, y1, x2, y2] strictly within [0.0, 1.0]
        x1 = xc - bw / 2.0
        x2 = xc + bw / 2.0
        y1 = yc - bh / 2.0
        y2 = yc + bh / 2.0

        TOLERANCE = 1e-6
        if (x1 < -TOLERANCE) or (y1 < -TOLERANCE) or (x2 > 1.0 + TOLERANCE) or (y2 > 1.0 + TOLERANCE):
            raise InvalidAnnotationError(
                f"Line {i+1} in {lbl_path.name} bounding box exceeds image bounds: "
                f"[x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f}]. Coordinates must be in [0.0, 1.0]."
            )


def build_safe_pilot_dataset_staging(
    manifest_path: Path,
    output_dir: Path,
    source_images_dir: Optional[Path] = None,
    source_labels_dir: Optional[Path] = None,
    allow_isolated_objects: bool = False,
    require_approval: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Safely builds physical staging dataset output strictly from approved, validated candidates.

    Guarantees:
    - Never allows bypassing approval: require_approval=False is strictly forbidden and raises SafeDatasetBuilderError.
    - Never overwrites v2 baseline, holdout, or test sets.
    - Pre-validates on-disk image and label integrity before copy.
    - If 0 approved candidates exist, outputs a clear EMPTY_STAGING status with 0 files staged.
    - Quarantined or pending candidates are NEVER turned into clean negatives.
    """
    if require_approval is False:
        raise SafeDatasetBuilderError(
            "Security violation: require_approval cannot be False in build_safe_pilot_dataset_staging. "
            "Training dataset staging unconditionally requires verified human approval."
        )

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Protected paths guard
    output_str = str(output_dir).replace("\\", "/").lower()
    protected_patterns = [
        "data/detection/labels/real",
        "data/detection/images/real",
        "data/detection_real_v2",
        "splits_real_v2",
        "hinh_anh_rac",
        "holdout",
    ]
    for pattern in protected_patterns:
        if pattern in output_str:
            raise OverwriteBaselineError(
                f"Security violation: output_dir '{output_dir}' matches protected baseline path '{pattern}'."
            )

    # Resolve source directories
    if source_images_dir is None:
        source_images_dir = manifest_path.parent / "images"
    else:
        source_images_dir = Path(source_images_dir)

    if source_labels_dir is None:
        source_labels_dir = manifest_path.parent / "labels"
    else:
        source_labels_dir = Path(source_labels_dir)

    # Filter manifest - ALWAYS enforces require_approval=True
    filter_res = load_and_filter_manifest(
        manifest_path,
        require_approval=True,
        allow_isolated_objects=allow_isolated_objects,
    )

    out_images_dir = output_dir / "images"
    out_labels_dir = output_dir / "labels"
    out_manifests_dir = output_dir / "manifests"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_manifests_dir.mkdir(parents=True, exist_ok=True)

    accepted = filter_res["accepted_candidates"]

    # When no candidates are approved, return empty staging cleanly
    if not accepted:
        empty_manifest = {
            "status": "EMPTY_STAGING_NO_APPROVED_CANDIDATES",
            "manifest_name": filter_res["manifest_name"],
            "staged_images_count": 0,
            "staged_labels_count": 0,
            "rejected_count": filter_res["rejected_count"],
            "staged_items": [],
        }
        with open(out_manifests_dir / "staging_manifest.json", "w", encoding="utf-8") as f:
            json.dump(empty_manifest, f, indent=2)

        return {
            "status": "EMPTY_STAGING_NO_APPROVED_CANDIDATES",
            "output_dir": str(output_dir),
            "staged_images_count": 0,
            "staged_labels_count": 0,
            "staged_items": [],
            "rejected_candidates": filter_res["rejected_candidates"],
        }

    # Physical staging of approved candidates
    staged_items = []
    for cand in accepted:
        # Defense-in-depth: double-check approval flags
        if cand.get("approved_for_train") is not True or cand.get("user_train_approval") != "APPROVED_FOR_TRAIN":
            raise SafeDatasetBuilderError(
                f"Candidate {cand.get('candidate_id')} lacks verified approval flags. Staging aborted."
            )

        fname = cand.get("filename") or f"{cand['candidate_id']}.jpg"
        img_src = source_images_dir / fname
        lbl_src = source_labels_dir / f"{Path(fname).stem}.txt"

        num_b = cand.get("num_boxes", 0)
        is_neg = (cand.get("visual_qa") == "QA_PASS_CLEAN_NEGATIVE")

        # Validate on disk before staging
        validate_on_disk_sample(img_src, lbl_src, expected_boxes=num_b, is_clean_negative=is_neg)

        # Copy to staging
        img_dest = out_images_dir / fname
        lbl_dest = out_labels_dir / lbl_src.name

        shutil.copy2(img_src, img_dest)
        shutil.copy2(lbl_src, lbl_dest)

        staged_items.append({
            "candidate_id": cand.get("candidate_id"),
            "image": str(img_dest.relative_to(output_dir)),
            "label": str(lbl_dest.relative_to(output_dir)),
            "num_boxes": num_b,
            "visual_qa": cand.get("visual_qa"),
        })

    staging_manifest = {
        "status": "STAGING_SUCCESS",
        "manifest_name": filter_res["manifest_name"],
        "staged_images_count": len(staged_items),
        "staged_labels_count": len(staged_items),
        "rejected_count": filter_res["rejected_count"],
        "staged_items": staged_items,
    }
    with open(out_manifests_dir / "staging_manifest.json", "w", encoding="utf-8") as f:
        json.dump(staging_manifest, f, indent=2)

    return {
        "status": "STAGING_SUCCESS",
        "output_dir": str(output_dir),
        "staged_images_count": len(staged_items),
        "staged_labels_count": len(staged_items),
        "staged_items": staged_items,
        "rejected_candidates": filter_res["rejected_candidates"],
    }


FIXED_10_CLASS_NAMES: Dict[int, str] = {
    0: "battery",
    1: "biological",
    2: "cardboard",
    3: "clothes",
    4: "glass",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "shoes",
    9: "trash",
}


def resolve_yolo_split_image_and_label(line: str, split_txt_path: Path, dataset_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Resolves a single entry from a YOLO split .txt file into (image_path, label_path)
    following Ultralytics YOLODataset resolution rules:
    - Lines starting with './' are resolved relative to split_txt_path.parent.
    - Relative lines not starting with './' are resolved relative to dataset_root (or split_txt_path.parent.parent).
    - Label path replaces the '/images/' path segment with '/labels/' and suffix with '.txt'.
    """
    raw = line.strip()
    if raw.startswith("./") or raw.startswith(".\\"):
        img_path = (split_txt_path.parent / raw[2:]).resolve()
    else:
        p = Path(raw)
        if p.is_absolute():
            img_path = p.resolve()
        else:
            base = dataset_root.resolve() if dataset_root is not None else split_txt_path.parent.parent.resolve()
            img_path = (base / p).resolve()

    parts = list(img_path.parts)
    # Replace last occurrence of 'images' with 'labels' (Ultralytics img2label_paths rule)
    rev_idx = None
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            rev_idx = idx
            break
    if rev_idx is None:
        raise InvalidAnnotationError(f"Image path '{img_path}' does not contain an 'images' directory segment for YOLO label resolution.")
    parts[rev_idx] = "labels"
    lbl_path = Path(*parts).with_suffix(".txt")
    return img_path, lbl_path


def count_yolo_split_labels_directly(
    split_txt_path: Path,
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Reads a YOLO split .txt file and counts images and bounding boxes per class (0..9)
    directly from the resolved on-disk label files. Zero hardcoded formulas.
    """
    split_txt_path = Path(split_txt_path).resolve()
    if not split_txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_txt_path}")

    lines = [l.strip() for l in split_txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_class_boxes = {FIXED_10_CLASS_NAMES[i]: 0 for i in range(10)}
    images_with_class = {FIXED_10_CLASS_NAMES[i]: 0 for i in range(10)}
    missing_images: List[str] = []
    missing_labels: List[str] = []
    empty_label_images: List[str] = []
    invalid_lines: List[str] = []

    for line in lines:
        img_p, lbl_p = resolve_yolo_split_image_and_label(line, split_txt_path, dataset_root=dataset_root)
        if not img_p.exists():
            missing_images.append(str(img_p))
        if not lbl_p.exists():
            missing_labels.append(str(lbl_p))
            continue

        lbl_lines = [x.strip() for x in lbl_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lbl_lines:
            empty_label_images.append(img_p.name)
            continue

        present_classes: Set[int] = set()
        for idx, ll in enumerate(lbl_lines):
            toks = ll.split()
            if len(toks) != 5:
                invalid_lines.append(f"{lbl_p.name}:{idx+1}")
                continue
            cid = int(toks[0])
            if cid not in FIXED_10_CLASS_NAMES:
                invalid_lines.append(f"{lbl_p.name}:{idx+1}:cid={cid}")
                continue
            cname = FIXED_10_CLASS_NAMES[cid]
            per_class_boxes[cname] += 1
            present_classes.add(cid)

        for cid in present_classes:
            images_with_class[FIXED_10_CLASS_NAMES[cid]] += 1

    return {
        "split_file": str(split_txt_path),
        "images_count": len(lines),
        "total_boxes": sum(per_class_boxes.values()),
        "per_class_boxes": per_class_boxes,
        "images_with_class": images_with_class,
        "missing_images_count": len(missing_images),
        "missing_labels_count": len(missing_labels),
        "empty_labels_count": len(empty_label_images),
        "invalid_label_lines_count": len(invalid_lines),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "empty_label_images": empty_label_images,
        "invalid_label_lines": invalid_lines,
    }


def build_pilot_v3_battery_extension_dataset(
    battery_manifest_path: Path,
    v2_root_dir: Path,
    output_pilot_dir: Path,
    source_battery_root_dir: Optional[Path] = None,
    require_approval: bool = True,
) -> Dict[str, Any]:
    """
    Executable Dataset Builder for Pilot V3 (Battery-Only Extension on top of frozen v2).

    Invariants enforced:
    1. require_approval=False is strictly forbidden (raises SafeDatasetBuilderError).
    2. Protected baseline paths (data/detection, splits_real_v2, holdout, Hinh_Anh_Rac) cannot be overwritten.
    3. Reads frozen v2 splits (train.txt=1398, val.txt=299, test.txt=299) and preserves all v2 samples.
    4. Filters battery_manifest_path through validate_candidate_for_intake(..., require_approval=True).
    5. Verifies strict physical battery specimen separation between train and valid splits.
    6. Copies only approved battery images/labels into output_pilot_dir/{images,labels}/roboflow_battery/.
    7. Writes splits/{train,val,test}.txt and dataset_pilot_v3.yaml, then counts all images and boxes
       directly from the resolved on-disk files (zero hardcoded arithmetic).
    """
    if require_approval is False:
        raise SafeDatasetBuilderError(
            "Security violation: require_approval cannot be False in build_pilot_v3_battery_extension_dataset."
        )

    battery_manifest_path = Path(battery_manifest_path).resolve()
    v2_root_dir = Path(v2_root_dir).resolve()
    output_pilot_dir = Path(output_pilot_dir).resolve()
    if source_battery_root_dir is None:
        source_battery_root_dir = battery_manifest_path.parent.parent.resolve()
    else:
        source_battery_root_dir = Path(source_battery_root_dir).resolve()

    out_str = str(output_pilot_dir).replace("\\", "/").lower()
    for pattern in ["data/detection/labels/real", "data/detection/images/real", "splits_real_v2", "hinh_anh_rac", "holdout"]:
        if pattern in out_str:
            raise OverwriteBaselineError(
                f"Security violation: output_pilot_dir '{output_pilot_dir}' matches protected path '{pattern}'."
            )

    with open(battery_manifest_path, "r", encoding="utf-8") as f:
        bat_manifest = json.load(f)

    items = bat_manifest.get("items", [])
    approved_train: List[Dict[str, Any]] = []
    approved_val: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for it in items:
        cand = dict(it)
        cand.setdefault("candidate_id", cand.get("base_id") or cand.get("filename"))
        cand.setdefault("num_boxes", cand.get("new_num_boxes", 1))
        ok, reason = validate_candidate_for_intake(cand, require_approval=True, allow_isolated_objects=False)
        if not ok:
            rejected.append({"base_id": cand.get("base_id"), "filename": cand.get("filename"), "reason": reason})
            continue

        split_target = cand.get("recommended_leak_free_split")
        if split_target == "train":
            approved_train.append(cand)
        elif split_target in ("valid", "val"):
            approved_val.append(cand)
        else:
            raise SafeDatasetBuilderError(f"Invalid recommended_leak_free_split '{split_target}' for {cand.get('base_id')}")

    # Verify zero specimen overlap between approved train and valid battery items
    train_specimens = {x.get("specimen_cluster_id") for x in approved_train if x.get("specimen_cluster_id")}
    val_specimens = {x.get("specimen_cluster_id") for x in approved_val if x.get("specimen_cluster_id")}
    overlap_specimens = sorted(train_specimens & val_specimens)
    if overlap_specimens:
        raise SafeDatasetBuilderError(f"Specimen leakage detected between train and val: {overlap_specimens}")

    splits_dir = output_pilot_dir / "splits"
    bat_img_out_dir = output_pilot_dir / "images" / "roboflow_battery"
    bat_lbl_out_dir = output_pilot_dir / "labels" / "roboflow_battery"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Read v2 split lines and convert to relative or absolute paths resolvable from splits_dir
    v2_split_lines: Dict[str, List[str]] = {}
    for sp in ("train", "val", "test"):
        v2_sp_file = v2_root_dir / "splits_real_v2" / f"{sp}.txt"
        raw_lines = [l.strip() for l in v2_sp_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        converted: List[str] = []
        for rl in raw_lines:
            abs_img = (v2_root_dir / rl).resolve()
            try:
                rel_from_splits = os.path.relpath(abs_img, splits_dir).replace("\\", "/")
                if not rel_from_splits.startswith("./"):
                    rel_from_splits = "./" + rel_from_splits
                converted.append(rel_from_splits)
            except ValueError:
                # Different drive letter on Windows during temp dry-run
                converted.append(str(abs_img).replace("\\", "/"))
        v2_split_lines[sp] = converted

    staged_train_lines: List[str] = []
    staged_val_lines: List[str] = []

    if approved_train or approved_val:
        bat_img_out_dir.mkdir(parents=True, exist_ok=True)
        bat_lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for split_list, out_lines in [(approved_train, staged_train_lines), (approved_val, staged_val_lines)]:
        for cand in split_list:
            img_src = (source_battery_root_dir / Path(cand["image_path"])).resolve()
            lbl_src = (source_battery_root_dir / Path(cand["label_path"])).resolve()
            validate_on_disk_sample(img_src, lbl_src, expected_boxes=int(cand.get("num_boxes", 1)), is_clean_negative=False)

            # Verify class 0 battery in label
            lbl_text = lbl_src.read_text(encoding="utf-8").strip()
            for ll in lbl_text.splitlines():
                if int(ll.split()[0]) != 0:
                    raise InvalidAnnotationError(f"Battery candidate {cand['filename']} has non-zero class ID in {lbl_src}")

            img_dst = bat_img_out_dir / img_src.name
            lbl_dst = bat_lbl_out_dir / lbl_src.name
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

            rel_line = f"./../images/roboflow_battery/{img_dst.name}"
            out_lines.append(rel_line)

    final_train_lines = v2_split_lines["train"] + staged_train_lines
    final_val_lines = v2_split_lines["val"] + staged_val_lines
    final_test_lines = v2_split_lines["test"]

    (splits_dir / "train.txt").write_text("\n".join(final_train_lines) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(final_val_lines) + "\n", encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(final_test_lines) + "\n", encoding="utf-8")

    # Directly count from generated split files
    train_stats = count_yolo_split_labels_directly(splits_dir / "train.txt", dataset_root=output_pilot_dir)
    val_stats = count_yolo_split_labels_directly(splits_dir / "val.txt", dataset_root=output_pilot_dir)
    test_stats = count_yolo_split_labels_directly(splits_dir / "test.txt", dataset_root=output_pilot_dir)

    status = "BUILT_WITH_APPROVED_BATTERY_EXTENSION" if (approved_train or approved_val) else "PROPOSAL_ONLY_NO_APPROVED_CANDIDATES"
    return {
        "status": status,
        "output_pilot_dir": str(output_pilot_dir),
        "approved_battery_train_count": len(approved_train),
        "approved_battery_val_count": len(approved_val),
        "rejected_battery_count": len(rejected),
        "train_specimens": sorted(train_specimens),
        "val_specimens": sorted(val_specimens),
        "overlapping_specimens": overlap_specimens,
        "measured_splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
    }

