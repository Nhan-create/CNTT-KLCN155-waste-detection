"""
src/detection/export_train_manifest.py
--------------------------------------
Production-grade Manifest Export and Training Split Safety Gate:
1. Authoritative Split Cross-Referencing: Checks disk split files (splits_real_v2/test.txt, val.txt, Holdout),
   ignoring any untrusted or spoofed 'category' metadata in manifest JSON.
2. Cryptographic Approval Verification: Eliminates boolean bypasses. Quarantined/pending annotations
   are ONLY accepted if backed by a valid Approval Record linking filename, label SHA-256, and APPROVED decision.
3. Unmapped Waste Gating: Strictly blocks 0-byte labels containing unmapped waste (Category 51 rope & strings).
4. Fail-Closed Security: Any missing split reference or unverified record blocks training export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = ROOT / "data" / "detection" / "splits_real_v2"
DEFAULT_HOLDOUT_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"


class TrainExportError(Exception):
    """Base exception for train export validation errors."""
    pass


class SplitContaminationError(TrainExportError):
    """Raised when an image belonging to TEST, VAL, or HOLDOUT is targeted for train export."""
    pass


class UnmappedWasteLeakageError(TrainExportError):
    """Raised when unmapped waste with empty labels is targeted as a training sample."""
    pass


class PendingApprovalError(TrainExportError):
    """Raised when an unapproved/quarantined annotation is targeted without valid approval record."""
    pass


class ApprovalRecordVerificationError(TrainExportError):
    """Raised when an approval record has decision != APPROVED or hash mismatch."""
    pass


class UnverifiedSplitMembershipError(TrainExportError):
    """Raised when an image claims EXISTING_TRAIN but is not in authoritative train split on disk."""
    pass


class UnverifiedTrueNegativeError(TrainExportError):
    """Raised when an image claims VERIFIED_TRUE_NEGATIVE without QA approval evidence."""
    pass


class MissingLabelVerificationError(TrainExportError):
    """Raised when disk label SHA-256 cannot be verified because labels_dir was omitted."""
    pass


class MissingImageVerificationError(TrainExportError):
    """Raised when disk image SHA-256 cannot be verified because images_dir was omitted."""
    pass


def load_authoritative_splits(
    splits_dir: Optional[Path] = None,
    holdout_manifest_path: Optional[Path] = None
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Load authoritative image basenames directly from disk split files.
    Returns: (test_set, val_set, train_set, holdout_set)

    FAIL-CLOSED: If splits_dir or holdout_manifest_path does not exist,
    or if any of train.txt, test.txt, val.txt is missing, raises TrainExportError.
    """
    sdir = splits_dir or DEFAULT_SPLITS_DIR
    hpath = holdout_manifest_path or DEFAULT_HOLDOUT_MANIFEST

    if not sdir.is_dir():
        raise TrainExportError(f"Authoritative splits directory missing: {sdir}")
    if not hpath.is_file():
        raise TrainExportError(f"Authoritative holdout manifest missing: {hpath}")

    train_file = sdir / "train.txt"
    test_file = sdir / "test.txt"
    val_file = sdir / "val.txt"

    for req_f in (train_file, test_file, val_file):
        if not req_f.is_file():
            raise TrainExportError(f"Authoritative split file missing: {req_f}")

    test_set: Set[str] = set()
    val_set: Set[str] = set()
    train_set: Set[str] = set()
    holdout_set: Set[str] = set()

    for line in test_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            test_set.add(Path(clean).name)
    test_ultra = sdir / "test_ultra.txt"
    if test_ultra.is_file():
        for line in test_ultra.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean:
                test_set.add(Path(clean).name)

    for line in val_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            val_set.add(Path(clean).name)
    val_ultra = sdir / "val_ultra.txt"
    if val_ultra.is_file():
        for line in val_ultra.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean:
                val_set.add(Path(clean).name)

    for line in train_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            train_set.add(Path(clean).name)
    train_ultra = sdir / "train_ultra.txt"
    if train_ultra.is_file():
        for line in train_ultra.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean:
                train_set.add(Path(clean).name)

    try:
        with open(hpath, "r", encoding="utf-8") as f:
            h_data = json.load(f)
            for item in h_data.get("files", []):
                holdout_set.add(item["filename"])
    except Exception as e:
        raise TrainExportError(f"Failed to parse holdout manifest at {hpath}: {e}")

    if not train_set:
        raise TrainExportError(f"Authoritative train split {train_file} is empty!")
    if not test_set:
        raise TrainExportError(f"Authoritative test split {test_file} is empty!")

    return test_set, val_set, train_set, holdout_set


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image_eligibility(
    image_meta: Dict[str, Any],
    approval_ledger: Optional[Dict[str, Dict[str, Any]]] = None,
    authoritative_splits: Optional[Tuple[Set[str], Set[str], Set[str], Set[str]]] = None,
    labels_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    required_scope: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate whether a single image entry is eligible for train export.

    Returns:
        (is_eligible, reason, error_type)
        where error_type is one of:
          - 'SPLIT_CONTAMINATION'
          - 'UNMAPPED_WASTE_LEAKAGE'
          - 'PENDING_APPROVAL'
          - 'INVALID_APPROVAL_RECORD'
          - None (if eligible)
    """
    fn = image_meta.get("filename", "unknown")
    category = image_meta.get("category", "")
    qa_status = image_meta.get("qa_approval_status", "")
    meta_split = image_meta.get("official_split", "").lower()

    # Rule 1: Authoritative Split Cross-Check (HIGHEST PRIORITY)
    # Check physical disk split lists first, ignoring any claimed 'category' in manifest JSON.
    if authoritative_splits is None:
        authoritative_splits = load_authoritative_splits()

    test_set, val_set, train_set, holdout_set = authoritative_splits

    if fn in test_set:
        return (
            False,
            f"{fn} is listed in authoritative TEST split (splits_real_v2/test.txt): train export forbidden",
            "SPLIT_CONTAMINATION"
        )

    if fn in holdout_set:
        return (
            False,
            f"{fn} is listed in official HOLDOUT acceptance manifest: train export forbidden",
            "SPLIT_CONTAMINATION"
        )

    if fn in val_set:
        return (
            False,
            f"{fn} is listed in authoritative VAL split (splits_real_v2/val.txt): train export forbidden",
            "SPLIT_CONTAMINATION"
        )

    # Check manifest metadata for test/val indicators as secondary defense
    if category == "EXISTING_TEST" or "TEST" in qa_status.split("_") or meta_split in ("test", "testing"):
        return False, f"{fn} metadata indicates TEST split: train export forbidden", "SPLIT_CONTAMINATION"

    if category == "EXISTING_VAL" or "VAL" in qa_status.split("_") or "VALIDATION" in qa_status or meta_split in ("val", "valid", "validation"):
        return False, f"{fn} metadata indicates VAL split: train export forbidden", "SPLIT_CONTAMINATION"

    # Rule 2: Unmapped Waste / False Negative Leakage Check
    if (
        category == "EMPTY_LABEL_UNMAPPED_WASTE"
        or "REJECTED" in qa_status
        or "ROPE" in qa_status
        or category == "UNMAPPED_OBJECT"
    ):
        return (
            False,
            f"{fn} contains unmapped waste (Category 51 rope/strings) with empty label; not a verified negative",
            "UNMAPPED_WASTE_LEAKAGE"
        )

    # Rule 3: Authoritative Train Split Membership for Existing Train Images
    if category == "EXISTING_TRAIN" or qa_status in ("VERIFIED_EXISTING_TRAIN", "APPROVED_TRAIN"):
        if fn not in train_set:
            return (
                False,
                f"{fn} claims EXISTING_TRAIN but is not present in authoritative train split (splits_real_v2/train.txt)",
                "UNVERIFIED_SPLIT_MEMBERSHIP"
            )
        return True, f"{fn} is verified member of authoritative train split", None

    # Rule 4: True Negatives Must Have Verified Approval Record (No Metadata-Only Acceptance)
    if category == "VERIFIED_TRUE_NEGATIVE" or qa_status == "VERIFIED_TRUE_NEGATIVE" or image_meta.get("is_hard_negative"):
        if not approval_ledger or fn not in approval_ledger:
            return (
                False,
                f"{fn} claims VERIFIED_TRUE_NEGATIVE in metadata but lacks QA approval record in ledger: train export forbidden",
                "UNVERIFIED_TRUE_NEGATIVE"
            )
        # Fall through to approval record validation below

    # Rule 5: Approval Record Verification (Pending, Quarantined, New Candidates, True Negatives)
    needs_approval_record = (
        category in ("QUARANTINED", "NEW_CANDIDATE", "PILOT_IMAGE", "VERIFIED_TRUE_NEGATIVE")
        or "PENDING" in qa_status
        or (approval_ledger is not None and fn in approval_ledger)
    )

    if needs_approval_record:
        if not approval_ledger or fn not in approval_ledger:
            return (
                False,
                f"{fn} is in {category} status ({qa_status}): requires explicit verified approval record in ledger",
                "PENDING_APPROVAL"
            )

        record = approval_ledger[fn]
        decision = record.get("decision", "").upper()
        if decision != "APPROVED":
            return (
                False,
                f"{fn} approval record rejected or pending: decision='{decision}'",
                "INVALID_APPROVAL_RECORD"
            )

        is_new_candidate = (
            category in ("NEW_CANDIDATE", "ACQUISITION_TARGET", "PILOT_IMAGE_NEW")
            or image_meta.get("is_new_candidate", False)
        )

        # Mandatory Scope for new candidates or when required_scope is requested
        rec_scope = record.get("scope")
        if required_scope or is_new_candidate:
            if not rec_scope:
                return (
                    False,
                    f"{fn} approval record missing mandatory scope for train export",
                    "INVALID_APPROVAL_RECORD"
                )
            allowed_scopes = (
                (required_scope, "ALL", "TRAIN_EXPANSION", "TRAIN_EXPORT")
                if required_scope
                else ("ALL", "TRAIN_EXPANSION", "TRAIN_EXPORT")
            )
            if rec_scope not in allowed_scopes:
                return (
                    False,
                    f"{fn} approval record scope '{rec_scope}' does not authorize train export (required: '{required_scope or 'TRAIN_EXPANSION'}')",
                    "INVALID_APPROVAL_RECORD"
                )

        approver = str(record.get("approved_by", "")).strip()
        FORBIDDEN_APPROVERS = {
            "unspecified", "anonymous", "none", "unknown",
            "system", "agent", "ai", "bot", "auto", "autogenerated",
            "assistant", "self", "script"
        }
        if not approver or approver.lower() in FORBIDDEN_APPROVERS:
            return (
                False,
                f"{fn} approval record lacks valid authoritative approver identity (got '{approver}')",
                "INVALID_APPROVAL_RECORD"
            )

        # Image SHA-256 is mandatory for new candidates
        expected_img_hash = record.get("image_sha256", "").lower().strip()
        if is_new_candidate:
            if not expected_img_hash or len(expected_img_hash) != 64 or not all(c in "0123456789abcdef" for c in expected_img_hash):
                return (
                    False,
                    f"{fn} approval record for new candidate missing mandatory 64-char hexadecimal image_sha256",
                    "INVALID_APPROVAL_RECORD"
                )

        expected_hash = record.get("label_sha256", "").lower().strip()
        if len(expected_hash) != 64 or not all(c in "0123456789abcdef" for c in expected_hash):
            return (
                False,
                f"{fn} approval record missing valid 64-char hexadecimal label_sha256",
                "INVALID_APPROVAL_RECORD"
            )

        # Mandatory disk verification of label SHA-256 (Fail-closed if labels_dir is missing)
        if labels_dir is None:
            return (
                False,
                f"{fn} requires disk label SHA-256 verification, but labels_dir was not provided: train export forbidden",
                "MISSING_LABEL_VERIFICATION"
            )

        lbl_path = labels_dir / f"{Path(fn).stem}.txt"
        if not lbl_path.is_file():
            return (
                False,
                f"{fn} label file missing on disk at {lbl_path}",
                "INVALID_APPROVAL_RECORD"
            )

        actual_hash = compute_file_sha256(lbl_path).lower()
        if actual_hash != expected_hash:
            return (
                False,
                f"{fn} label SHA-256 mismatch (actual: {actual_hash[:8]}... vs expected: {expected_hash[:8]}...)",
                "INVALID_APPROVAL_RECORD"
            )

        # Mandatory disk verification of image SHA-256 for new candidates
        if is_new_candidate:
            if images_dir is None:
                return (
                    False,
                    f"{fn} is a new candidate requiring disk image SHA-256 verification, but images_dir was not provided: train export forbidden",
                    "MISSING_IMAGE_VERIFICATION"
                )
            img_path = images_dir / fn
            if not img_path.is_file():
                return (
                    False,
                    f"{fn} image file missing on disk at {img_path}",
                    "INVALID_APPROVAL_RECORD"
                )
            actual_img_hash = compute_file_sha256(img_path).lower()
            if actual_img_hash != expected_img_hash:
                return (
                    False,
                    f"{fn} image SHA-256 mismatch (actual: {actual_img_hash[:8]}... vs expected: {expected_img_hash[:8]}...)",
                    "INVALID_APPROVAL_RECORD"
                )
        elif images_dir and expected_img_hash:
            img_path = images_dir / fn
            if not img_path.is_file():
                return (
                    False,
                    f"{fn} image file missing on disk at {img_path}",
                    "INVALID_APPROVAL_RECORD"
                )
            actual_img_hash = compute_file_sha256(img_path).lower()
            if actual_img_hash != expected_img_hash:
                return (
                    False,
                    f"{fn} image SHA-256 mismatch (actual: {actual_img_hash[:8]}... vs expected: {expected_img_hash[:8]}...)",
                    "INVALID_APPROVAL_RECORD"
                )

        return True, f"{fn} approved for train export by user '{approver}' (image & label hashes verified against physical disk files)", None

    # Fallback default: fail closed
    return False, f"{fn} has unverified status: category='{category}', qa='{qa_status}'", "UNAPPROVED"


def filter_manifest_for_train(
    manifest_data: Dict[str, Any],
    approval_ledger: Optional[Dict[str, Dict[str, Any]]] = None,
    authoritative_splits: Optional[Tuple[Set[str], Set[str], Set[str], Set[str]]] = None,
    labels_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    required_scope: Optional[str] = None,
    raise_on_violation: bool = False
) -> Dict[str, Any]:
    """
    Filter a manifest dictionary to retain only train-eligible images.

    If raise_on_violation is True, encountering a forbidden image (e.g. test split or unmapped waste)
    will immediately raise the corresponding TrainExportError subclass.
    """
    if authoritative_splits is None:
        authoritative_splits = load_authoritative_splits()

    images = manifest_data.get("images", [])
    eligible_images = []
    blocked_images = []

    for img in images:
        is_ok, reason, err_type = validate_image_eligibility(
            img,
            approval_ledger=approval_ledger,
            authoritative_splits=authoritative_splits,
            labels_dir=labels_dir,
            images_dir=images_dir,
            required_scope=required_scope
        )
        if is_ok:
            eligible_images.append(img)
        else:
            blocked_images.append({"image": img, "reason": reason, "error_type": err_type})
            if raise_on_violation:
                if err_type == "SPLIT_CONTAMINATION":
                    raise SplitContaminationError(reason)
                elif err_type == "UNMAPPED_WASTE_LEAKAGE":
                    raise UnmappedWasteLeakageError(reason)
                elif err_type == "PENDING_APPROVAL":
                    raise PendingApprovalError(reason)
                elif err_type == "INVALID_APPROVAL_RECORD":
                    raise ApprovalRecordVerificationError(reason)
                elif err_type == "UNVERIFIED_SPLIT_MEMBERSHIP":
                    raise UnverifiedSplitMembershipError(reason)
                elif err_type == "UNVERIFIED_TRUE_NEGATIVE":
                    raise UnverifiedTrueNegativeError(reason)
                elif err_type == "MISSING_LABEL_VERIFICATION":
                    raise MissingLabelVerificationError(reason)
                elif err_type == "MISSING_IMAGE_VERIFICATION":
                    raise MissingImageVerificationError(reason)
                else:
                    raise TrainExportError(reason)

    return {
        "metadata": {
            "source_total_images": len(images),
            "eligible_train_images_count": len(eligible_images),
            "blocked_images_count": len(blocked_images),
            "approval_ledger_provided": approval_ledger is not None,
            "blocked_breakdown": {
                "split_contamination": sum(1 for b in blocked_images if b["error_type"] == "SPLIT_CONTAMINATION"),
                "unmapped_waste_leakage": sum(1 for b in blocked_images if b["error_type"] == "UNMAPPED_WASTE_LEAKAGE"),
                "pending_approval": sum(1 for b in blocked_images if b["error_type"] == "PENDING_APPROVAL"),
                "invalid_approval_record": sum(1 for b in blocked_images if b["error_type"] == "INVALID_APPROVAL_RECORD"),
                "unverified_split_membership": sum(1 for b in blocked_images if b["error_type"] == "UNVERIFIED_SPLIT_MEMBERSHIP"),
                "unverified_true_negative": sum(1 for b in blocked_images if b["error_type"] == "UNVERIFIED_TRUE_NEGATIVE"),
                "missing_label_verification": sum(1 for b in blocked_images if b["error_type"] == "MISSING_LABEL_VERIFICATION"),
                "missing_image_verification": sum(1 for b in blocked_images if b["error_type"] == "MISSING_IMAGE_VERIFICATION"),
                "unapproved_other": sum(1 for b in blocked_images if b["error_type"] == "UNAPPROVED")
            }
        },
        "eligible_images": eligible_images,
        "blocked_images": blocked_images
    }


def export_train_manifest(
    manifest_path: Path,
    output_path: Path,
    approval_ledger: Optional[Dict[str, Dict[str, Any]]] = None,
    authoritative_splits: Optional[Tuple[Set[str], Set[str], Set[str], Set[str]]] = None,
    labels_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    required_scope: Optional[str] = None
) -> Dict[str, Any]:
    """Read a manifest file, filter for training eligibility, and write safe output."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = filter_manifest_for_train(
        data,
        approval_ledger=approval_ledger,
        authoritative_splits=authoritative_splits,
        labels_dir=labels_dir,
        images_dir=images_dir,
        required_scope=required_scope,
        raise_on_violation=False
    )

    out_data = {
        "metadata": {
            "generated_by": "src.detection.export_train_manifest",
            "source_manifest": str(manifest_path),
            **result["metadata"]
        },
        "images": result["eligible_images"]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Export clean train manifest preventing contamination.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Output filtered manifest JSON path")
    parser.add_argument("--approval-ledger", type=Path, default=None, help="JSON path to approval records")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Directory containing label files for hash verification")
    parser.add_argument("--strict", action="store_true", help="Raise error if any blocked images found")

    args = parser.parse_args()

    ledger = None
    if args.approval_ledger and args.approval_ledger.is_file():
        with open(args.approval_ledger, "r", encoding="utf-8") as lf:
            ledger = json.load(lf)

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        result = filter_manifest_for_train(
            data,
            approval_ledger=ledger,
            labels_dir=args.labels_dir,
            raise_on_violation=args.strict
        )
        print(f"Total images evaluated: {result['metadata']['source_total_images']}")
        print(f"Eligible for train:    {result['metadata']['eligible_train_images_count']}")
        print(f"Blocked from train:     {result['metadata']['blocked_images_count']}")
        for k, v in result["metadata"]["blocked_breakdown"].items():
            print(f"  - {k}: {v}")

        export_train_manifest(
            args.manifest,
            args.output,
            approval_ledger=ledger,
            labels_dir=args.labels_dir
        )
        print(f"Clean train manifest written to: {args.output}")
        sys.exit(0)
    except TrainExportError as e:
        print(f"[BLOCKED] Train Export Violation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
