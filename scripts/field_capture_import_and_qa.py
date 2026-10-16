"""Task 8I (Final Hotfix) — Safe First-Party Field Capture Import, Split-Isolation & QA Tool.

Enforces strict separation between technical file/YOLO validation and human real-waste context verification:
1. 10-class canonical schema (`[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`).
2. Technical validation (`FILE_AND_BBOX_FORMAT_VALID`) is strictly separated from semantic context verification (`REAL_WASTE_CONTEXT_VERIFIED`).
3. Newly imported records default to `qa_status = "NEEDS_REVIEW"` (`target_split = "MANUAL_REVIEW_QUEUE"`).
4. NEVER auto-promotes `REAL_WASTE_PASS` from `context_type`, filename, or valid YOLO coordinates:
   - `PRODUCT_STUDIO` + valid YOLO label -> NEVER `REAL_WASTE_PASS` (`NEEDS_REVIEW` by default, or `KEEP_AUXILIARY` if human-reviewed as auxiliary; `REAL_WASTE_CONTEXT_VERIFIED = False`).
   - `IN_USE` + valid YOLO label -> NEVER `REAL_WASTE_PASS` (`NEEDS_REVIEW` by default, or `KEEP_AUXILIARY` if human-reviewed as auxiliary; `REAL_WASTE_CONTEXT_VERIFIED = False`).
   - `ACTUAL_WASTE` self-declared without human visual QA sign-off -> held at `NEEDS_REVIEW` (`REAL_WASTE_CONTEXT_VERIFIED = False`).
5. `REAL_WASTE_CONTEXT_VERIFIED = True` and `qa_status = "REAL_WASTE_PASS"` require an explicit human visual QA sign-off record (`visual_qa_review` or `visual_qa_*` fields) with a human `reviewer_id`, `review_timestamp`, `context_evidence` (>= 10 chars), `bbox_tightness_verified = True`, and confirmed usage rights/license.
6. NEVER auto-assigns `CC-BY-4.0` when the photographer/owner has not explicitly provided a license and confirmed usage rights (`rights_confirmed = True`).
7. `APPROVED_FOR_TRAIN = False` is unconditionally enforced on 100% of records, even when human visual QA passes `REAL_WASTE_PASS`.
8. Zero `specimen_id` and zero `session_id` leakage across `TRAIN` and `VAL` splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from PIL import Image, ImageDraw, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquire_and_qa_task8f_scarce_classes import (
    CLASS_COLORS,
    CLASS_NAMES,
    classify_box_pair_overlap,
)


def compute_dhash_int(im: Image.Image) -> int:
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(g.get_flattened_data() if hasattr(g, "get_flattened_data") else g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            idx = row * 9 + col
            bits = (bits << 1) | (1 if px[idx] > px[idx + 1] else 0)
    return bits


def hamming_distance_64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


SPECIMEN_ID_RE = re.compile(r"^SPEC_(BAT|BIO|CAR|CLO|GLA|MET|PAP|PLA|SHO|TRA)_\d{6,8}_\d{3,4}$")
SESSION_ID_RE = re.compile(r"^SESS_\d{8}_[A-Z0-9_]{3,32}$")
DISALLOWED_AUTO_REVIEWERS = {"", "AUTO", "SCRIPT", "IMPORTER", "SYSTEM", "NONE", "NULL", "BOT", "AI"}


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def parse_yolo_label_file(label_path: pathlib.Path, img_w: int, img_h: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a YOLO `.txt` file into structured boxes and return technical format/geometry errors only.

    Note: Valid YOLO coordinates only establish `FILE_AND_BBOX_FORMAT_VALID = True`.
    They NEVER establish `REAL_WASTE_CONTEXT_VERIFIED` or `OBJECT_ANNOTATION_VALID`.
    """
    errors: list[str] = []
    boxes: list[dict[str, Any]] = []
    if not label_path.exists():
        return [], [f"Missing YOLO label file: {label_path.name}"]

    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return [], [f"Empty YOLO label file (zero annotated instances): {label_path.name}"]

    for idx, ln in enumerate(lines, start=1):
        parts = ln.split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: expected 5 tokens '<cid> <xc> <yc> <w> <h>', got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"Line {idx}: non-numeric token in '{ln}'")
            continue

        if not (0 <= cid < 10):
            errors.append(f"Line {idx}: class_id {cid} outside canonical 0..9 range")
            continue
        if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            errors.append(f"Line {idx}: normalized coordinates out of bounds ({xc}, {yc}, {bw}, {bh})")
            continue
        area = bw * bh
        if area < 0.002:
            errors.append(f"Line {idx}: bounding box area too tiny ({area:.5f} < 0.002)")
            continue
        if area > 0.96:
            errors.append(f"Line {idx}: full-image coarse box ({area:.4f} > 0.96) not allowed")
            continue

        x1n = max(0.0, xc - bw / 2.0)
        y1n = max(0.0, yc - bh / 2.0)
        x2n = min(1.0, xc + bw / 2.0)
        y2n = min(1.0, yc + bh / 2.0)
        boxes.append(
            {
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "yolo_xywhn": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                "bbox_xyxy": [
                    round(x1n * img_w, 2),
                    round(y1n * img_h, 2),
                    round(x2n * img_w, 2),
                    round(y2n * img_h, 2),
                ],
                "norm_tuple": (cid, round(x1n, 4), round(y1n, 4), round(x2n, 4), round(y2n, 4)),
            }
        )

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            rel = classify_box_pair_overlap(boxes[i]["norm_tuple"], boxes[j]["norm_tuple"])
            if rel["category"] == "DUPLICATE_ANNOTATION":
                errors.append(
                    f"Boxes {i + 1} and {j + 1} are DUPLICATE_ANNOTATION of same instance "
                    f"(class={boxes[i]['class_name']}, IoU={rel['iou']}, max_edge_delta={rel['max_edge_delta']})"
                )

    for b in boxes:
        b.pop("norm_tuple", None)
    return boxes, errors


def _extract_visual_qa_review(row: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate an optional human visual QA review record from a manifest row."""
    sub = row.get("visual_qa_review")
    src = sub if isinstance(sub, dict) else row
    decision = str(src.get("decision") or src.get("visual_qa_decision") or "").strip().upper()
    reviewer_id = str(src.get("reviewer_id") or src.get("visual_qa_reviewer") or "").strip()
    review_timestamp = str(
        src.get("review_timestamp") or src.get("visual_qa_timestamp") or src.get("reviewed_at") or ""
    ).strip()
    context_evidence = str(
        src.get("context_evidence") or src.get("visual_qa_evidence") or src.get("evidence") or ""
    ).strip()
    bbox_verified = _as_bool(
        src.get("bbox_tightness_verified", src.get("visual_qa_bbox_verified", False))
    )
    verified_domain = str(
        src.get("verified_domain_tag") or src.get("visual_qa_verified_domain") or ""
    ).strip().upper()

    is_human_reviewer = bool(reviewer_id) and (reviewer_id.upper() not in DISALLOWED_AUTO_REVIEWERS)
    has_evidence = len(context_evidence) >= 10 and bool(review_timestamp)
    return {
        "decision": decision,
        "reviewer_id": reviewer_id,
        "review_timestamp": review_timestamp,
        "context_evidence": context_evidence,
        "bbox_tightness_verified": bbox_verified,
        "verified_domain_tag": verified_domain,
        "is_valid_human_review": bool(is_human_reviewer and has_evidence and decision in {
            "REAL_WASTE_PASS",
            "KEEP_AUXILIARY",
            "NEEDS_REVIEW",
            "REJECT",
        }),
    }


def _normalize_declared_domain(raw_context: str) -> str:
    val = (raw_context or "").strip().upper()
    if val in ("PRODUCT_STUDIO", "STUDIO", "CATALOG", "WHITE_BACKGROUND"):
        return "PRODUCT_STUDIO"
    if val in ("IN_USE", "ACTIVE_USE", "NATURAL_DECAY", "NATURAL_ORCHARD_PLANT_DECAY"):
        return "IN_USE"
    if val == "ACTUAL_WASTE":
        return "ACTUAL_WASTE"
    return "AMBIGUOUS_CONTEXT"


def validate_and_allocate_field_batch(
    manifest_rows: list[dict[str, Any]],
    images_dir: pathlib.Path,
    labels_dir: pathlib.Path,
    existing_sha256: set[str] | None = None,
    existing_dhashes: list[int] | None = None,
    max_angles_per_specimen: int = 3,
) -> dict[str, Any]:
    """Validate a batch of field-captured images + annotations with strict P0 safety gates.

    Invariants enforced:
    - Newly imported records without human visual QA sign-off ALWAYS default to `NEEDS_REVIEW`.
    - `PRODUCT_STUDIO` and `IN_USE` records can NEVER become `REAL_WASTE_PASS` or `REAL_WASTE_CONTEXT_VERIFIED=True`.
    - Self-declared `context_type="ACTUAL_WASTE"` without human `visual_qa_review` stays `NEEDS_REVIEW`.
    - Missing explicit license or unconfirmed owner usage rights (`rights_confirmed=False`) never auto-defaults to `CC-BY-4.0`.
    - `APPROVED_FOR_TRAIN` is unconditionally `False` on 100% of records.
    """
    seen_sha = set(existing_sha256 or set())
    seen_dhash = list(existing_dhashes or [])
    specimen_angle_counts: Counter[str] = Counter()
    records_out: list[dict[str, Any]] = []

    for idx, row in enumerate(manifest_rows, start=1):
        rec_id = str(row.get("capture_id") or f"field8i_{idx:04d}").strip()
        img_rel = str(row.get("image_filename") or "").strip()
        lbl_rel = str(row.get("label_filename") or pathlib.Path(img_rel).with_suffix(".txt").name).strip()
        specimen_id = str(row.get("specimen_id") or "").strip()
        session_id = str(row.get("session_id") or "").strip()
        photographer = str(row.get("photographer") or row.get("photographer_id") or "").strip()

        # P0 Fix #6: Never auto-assign CC-BY-4.0 if photographer/owner has not explicitly provided license & confirmed rights
        raw_license = str(row.get("license") or row.get("license_code") or "").strip()
        rights_confirmed = _as_bool(
            row.get("rights_confirmed", row.get("consent_and_privacy_verified", False))
        )
        if raw_license and rights_confirmed:
            license_str = raw_license
            license_verified = True
        else:
            license_str = "UNCONFIRMED_USAGE_RIGHTS"
            license_verified = False

        raw_context = str(row.get("context_type") or row.get("domain_tag") or "").strip()
        declared_domain = _normalize_declared_domain(raw_context)
        angle_code = str(row.get("angle_code") or row.get("angle_type") or "oblique_45deg").strip()
        distance_m = float(row.get("distance_m") or 1.0)
        lighting = str(row.get("lighting") or row.get("lighting_condition") or "daylight").strip()
        privacy_cleared = _as_bool(
            row.get("privacy_cleared", row.get("consent_and_privacy_verified", False))
        )

        technical_errors: list[str] = []
        review_notes: list[str] = []

        if not SPECIMEN_ID_RE.match(specimen_id):
            technical_errors.append(f"Invalid specimen_id '{specimen_id}' (expected SPEC_<CLS>_<YYYYMM[DD]>_<NNN>)")
        if not SESSION_ID_RE.match(session_id):
            technical_errors.append(f"Invalid session_id '{session_id}' (expected SESS_<YYYYMMDD>_<LOC>)")
        if not photographer:
            technical_errors.append("Missing photographer attribution")
        if not privacy_cleared:
            technical_errors.append("privacy_cleared is False (faces/license plates/private addresses must be cleared)")
        if not license_verified:
            review_notes.append(
                "Missing explicit license or owner usage-rights confirmation (rights_confirmed=False); cannot auto-assign CC-BY-4.0"
            )
        if not (0.3 <= distance_m <= 5.0):
            technical_errors.append(f"Capture distance {distance_m}m outside recommended [0.3m, 5.0m] range")

        img_p = images_dir / img_rel
        lbl_p = labels_dir / lbl_rel
        sha256_hex = ""
        dhash_hex = ""
        w, h = 0, 0
        boxes: list[dict[str, Any]] = []

        if not img_p.exists() or not img_p.is_file():
            technical_errors.append(f"Missing image file: {img_rel}")
        else:
            raw_bytes = img_p.read_bytes()
            sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
            if sha256_hex in seen_sha:
                technical_errors.append(f"Exact duplicate SHA-256 detected: {sha256_hex[:16]}")
            try:
                with Image.open(img_p) as im:
                    im_exif = ImageOps.exif_transpose(im).convert("RGB")
                    w, h = im_exif.size
                    if min(w, h) < 480:
                        technical_errors.append(f"Image resolution too low: {w}x{h} (min short side >= 480px)")
                    dh_int = compute_dhash_int(im_exif)
                    dhash_hex = f"{dh_int:016x}"
                    if any(hamming_distance_64(dh_int, prev_dh) <= 3 for prev_dh in seen_dhash):
                        technical_errors.append(f"Near-duplicate perceptual dHash detected: {dhash_hex}")
                    else:
                        seen_dhash.append(dh_int)
                seen_sha.add(sha256_hex)
            except Exception as exc:
                technical_errors.append(f"Corrupt image file '{img_rel}': {exc}")

        if w > 0 and h > 0:
            boxes, box_errs = parse_yolo_label_file(lbl_p, w, h)
            technical_errors.extend(box_errs)

        specimen_angle_counts[specimen_id] += 1
        angle_rank = specimen_angle_counts[specimen_id]
        if angle_rank > max_angles_per_specimen:
            technical_errors.append(
                f"Specimen '{specimen_id}' exceeds max_angles_per_specimen={max_angles_per_specimen} (angle #{angle_rank})"
            )

        file_and_bbox_format_valid = (len(technical_errors) == 0) and (len(boxes) > 0)

        cls_ids = sorted({int(b["class_id"]) for b in boxes})
        cls_names = [CLASS_NAMES[c] for c in cls_ids]
        primary_cid = cls_ids[0] if cls_ids else -1

        # P0 Fix #1-#5: Separate technical validation from human visual QA context verification
        vqa = _extract_visual_qa_review(row)
        hard_reject = any(
            ("Missing" in e or "duplicate" in e.lower() or "exceeds" in e or "Corrupt" in e)
            for e in technical_errors
        )

        qa_status = "NEEDS_REVIEW"
        domain_tag = declared_domain
        real_waste_context_verified = False
        object_annotation_valid = False

        if hard_reject or vqa["decision"] == "REJECT":
            qa_status = "REJECT"
            if vqa["decision"] == "REJECT":
                review_notes.append(f"Explicitly rejected by human visual QA ({vqa['reviewer_id']})")
        elif declared_domain in ("PRODUCT_STUDIO", "IN_USE"):
            # P0 Invariant: PRODUCT_STUDIO and IN_USE can NEVER become REAL_WASTE_PASS
            domain_tag = declared_domain
            real_waste_context_verified = False
            if (
                file_and_bbox_format_valid
                and license_verified
                and vqa["is_valid_human_review"]
                and vqa["decision"] == "KEEP_AUXILIARY"
            ):
                qa_status = "KEEP_AUXILIARY"
                object_annotation_valid = bool(vqa["bbox_tightness_verified"])
                review_notes.append(
                    f"Human visual QA ({vqa['reviewer_id']}) routed {declared_domain} item to KEEP_AUXILIARY"
                )
            else:
                qa_status = "NEEDS_REVIEW"
                object_annotation_valid = False
                review_notes.append(
                    f"Held at NEEDS_REVIEW: context_type='{declared_domain}' cannot be promoted to REAL_WASTE_PASS"
                )
        elif not file_and_bbox_format_valid or not license_verified:
            qa_status = "NEEDS_REVIEW"
            real_waste_context_verified = False
            object_annotation_valid = False
        elif len(boxes) > 6:
            qa_status = "NEEDS_REVIEW"
            real_waste_context_verified = False
            object_annotation_valid = False
            review_notes.append(f"Dense scene ({len(boxes)} instances > 6); routed to NEEDS_REVIEW")
        elif not vqa["is_valid_human_review"]:
            # P0 Invariant: Newly imported images (even with context_type='ACTUAL_WASTE' and valid YOLO labels)
            # MUST remain NEEDS_REVIEW until human visual QA sign-off is provided.
            qa_status = "NEEDS_REVIEW"
            real_waste_context_verified = False
            object_annotation_valid = False
            review_notes.append(
                "Held at NEEDS_REVIEW: awaiting human visual QA verification of real-waste context and bbox tightness"
            )
        else:
            # Has a valid human review record
            if vqa["decision"] == "KEEP_AUXILIARY" or vqa["verified_domain_tag"] in ("PRODUCT_STUDIO", "IN_USE", "AMBIGUOUS_CONTEXT"):
                qa_status = "KEEP_AUXILIARY"
                domain_tag = vqa["verified_domain_tag"] or "AMBIGUOUS_CONTEXT"
                real_waste_context_verified = False
                object_annotation_valid = bool(vqa["bbox_tightness_verified"])
            elif (
                vqa["decision"] == "REAL_WASTE_PASS"
                and vqa["bbox_tightness_verified"] is True
                and declared_domain == "ACTUAL_WASTE"
                and vqa["verified_domain_tag"] in ("", "ACTUAL_WASTE")
            ):
                qa_status = "REAL_WASTE_PASS"
                domain_tag = "ACTUAL_WASTE"
                real_waste_context_verified = True
                object_annotation_valid = True
            else:
                qa_status = "NEEDS_REVIEW"
                real_waste_context_verified = False
                object_annotation_valid = False
                review_notes.append("Human visual QA did not confirm both ACTUAL_WASTE context and bbox tightness")

        if qa_status == "REJECT":
            target_split = "REJECTED"
        elif qa_status == "KEEP_AUXILIARY":
            target_split = "AUXILIARY_NON_WASTE_TRAIN_HOLD"
        else:
            target_split = "MANUAL_REVIEW_QUEUE"

        records_out.append(
            {
                "capture_id": rec_id,
                "image_filename": img_rel,
                "label_filename": lbl_rel,
                "specimen_id": specimen_id,
                "session_id": session_id,
                "angle_code": angle_code,
                "angle_index_for_specimen": angle_rank,
                "distance_m": distance_m,
                "lighting": lighting,
                "declared_context_type": raw_context,
                "photographer": photographer,
                "license": license_str,
                "rights_confirmed": rights_confirmed,
                "license_verified": license_verified,
                "privacy_cleared": privacy_cleared,
                "sha256": sha256_hex,
                "dhash_hex": dhash_hex,
                "width": w,
                "height": h,
                "primary_class_id": primary_cid,
                "primary_class_name": CLASS_NAMES[primary_cid] if 0 <= primary_cid < 10 else "none",
                "classes_present_ids": cls_ids,
                "classes_present_names": cls_names,
                "is_multiclass": len(cls_ids) > 1,
                "is_multiobject": len(boxes) >= 2,
                "bbox_count": len(boxes),
                "boxes": boxes,
                "FILE_AND_BBOX_FORMAT_VALID": file_and_bbox_format_valid,
                "qa_status": qa_status,
                "domain_tag": domain_tag,
                "OBJECT_ANNOTATION_VALID": object_annotation_valid,
                "REAL_WASTE_CONTEXT_VERIFIED": real_waste_context_verified,
                "APPROVED_FOR_TRAIN": False,  # Unconditionally False even when REAL_WASTE_PASS
                "visual_qa_review": vqa,
                "target_split": target_split,
                "qa_errors": technical_errors,
                "review_notes": review_notes,
            }
        )

    # Group-aware split allocation by connected components of (session_id, specimen_id)
    # Only records that have passed human visual QA (`REAL_WASTE_PASS`) are allocated to TRAIN/VAL staging splits,
    # and even those keep `APPROVED_FOR_TRAIN = False`.
    pass_recs = [r for r in records_out if r["qa_status"] == "REAL_WASTE_PASS"]

    # Build connected components of pass_recs sharing either session_id or specimen_id
    adj: dict[int, set[int]] = defaultdict(set)
    by_sess: dict[str, list[int]] = defaultdict(list)
    by_spec: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pass_recs):
        by_sess[r["session_id"]].append(i)
        by_spec[r["specimen_id"]].append(i)
    for idxs in list(by_sess.values()) + list(by_spec.values()):
        for a in idxs:
            for b in idxs:
                if a != b:
                    adj[a].add(b)

    visited: set[int] = set()
    components: list[list[dict[str, Any]]] = []
    for i in range(len(pass_recs)):
        if i in visited:
            continue
        comp_idxs: list[int] = []
        stack = [i]
        visited.add(i)
        while stack:
            cur = stack.pop()
            comp_idxs.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        comp_recs = [pass_recs[k] for k in sorted(comp_idxs)]
        components.append(comp_recs)

    components.sort(key=lambda grp: (grp[0]["session_id"], grp[0]["specimen_id"]))
    for c_idx, grp in enumerate(components):
        assigned_split = (
            "VAL_FIELD_PILOT_W8I"
            if (len(components) >= 2 and c_idx % 5 == len(components) - 1 and c_idx > 0)
            or (len(components) >= 5 and c_idx % 5 == 4)
            else "TRAIN_FIELD_PILOT_W8I"
        )
        for r in grp:
            r["target_split"] = assigned_split
            r["APPROVED_FOR_TRAIN"] = False

    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_FIELD_PILOT_W8I"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_FIELD_PILOT_W8I"]

    train_specimens = {r["specimen_id"] for r in train_recs}
    val_specimens = {r["specimen_id"] for r in val_recs}
    train_sessions = {r["session_id"] for r in train_recs}
    val_sessions = {r["session_id"] for r in val_recs}
    assert not (train_specimens & val_specimens), f"Specimen leakage: {train_specimens & val_specimens}"
    assert not (train_sessions & val_sessions), f"Session leakage: {train_sessions & val_sessions}"

    per_class_independence: dict[str, dict[str, int]] = {}
    for cid_num, cname in enumerate(CLASS_NAMES):
        c_tr = [r for r in train_recs if cid_num in r["classes_present_ids"]]
        c_val = [r for r in val_recs if cid_num in r["classes_present_ids"]]
        per_class_independence[cname] = {
            "train_unique_images": len(c_tr),
            "train_unique_specimens": len({r["specimen_id"] for r in c_tr}),
            "train_unique_sessions": len({r["session_id"] for r in c_tr}),
            "train_instance_boxes": sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_tr),
            "val_unique_images": len(c_val),
            "val_unique_specimens": len({r["specimen_id"] for r in c_val}),
        }

    status_counts = Counter(r["qa_status"] for r in records_out)
    return {
        "tool_version": "TASK8I_FIELD_CAPTURE_IMPORT_QA_V2_P0_HOTFIX",
        "approved_for_train": False,
        "total_submitted_records": len(records_out),
        "technical_format_valid_count": sum(1 for r in records_out if r["FILE_AND_BBOX_FORMAT_VALID"]),
        "real_waste_context_verified_count": sum(1 for r in records_out if r["REAL_WASTE_CONTEXT_VERIFIED"]),
        "qa_status_counts": dict(status_counts),
        "qa_pass_rate": round(len(pass_recs) / max(1, len(records_out)), 4),
        "four_independence_metrics_pass_total": {
            "unique_images": len(pass_recs),
            "unique_specimens": len({r["specimen_id"] for r in pass_recs}),
            "unique_sessions": len({r["session_id"] for r in pass_recs}),
            "per_class_image_appearances": sum(len(r["classes_present_ids"]) for r in pass_recs),
        },
        "four_independence_metrics_train": {
            "unique_images": len(train_recs),
            "unique_specimens": len(train_specimens),
            "unique_sessions": len(train_sessions),
            "per_class_image_appearances": sum(len(r["classes_present_ids"]) for r in train_recs),
        },
        "leakage_checks": {
            "train_val_specimen_overlap_count": len(train_specimens & val_specimens),
            "train_val_session_overlap_count": len(train_sessions & val_sessions),
            "all_approved_for_train_false": all(r["APPROVED_FOR_TRAIN"] is False for r in records_out),
        },
        "multiclass_pass_images": sum(1 for r in pass_recs if r["is_multiclass"]),
        "multiobject_pass_images": sum(1 for r in pass_recs if r["is_multiobject"]),
        "per_class_independence": per_class_independence,
        "records": records_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8I First-Party Field Capture Import & QA Tool (V2 P0 Hotfix)")
    parser.add_argument("--manifest", type=pathlib.Path, help="Path to field capture CSV or JSON manifest")
    parser.add_argument("--images-dir", type=pathlib.Path, help="Directory containing field-captured images")
    parser.add_argument("--labels-dir", type=pathlib.Path, help="Directory containing YOLO .txt annotations")
    parser.add_argument("--output-json", type=pathlib.Path, help="Output QA manifest JSON path")
    args = parser.parse_args()

    if not args.manifest or not args.images_dir or not args.labels_dir or not args.output_json:
        parser.print_help()
        return

    if args.manifest.suffix.lower() == ".json":
        rows = json.loads(args.manifest.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("records", [])
    else:
        with args.manifest.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    res = validate_and_allocate_field_batch(rows, args.images_dir, args.labels_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Field Capture QA Complete] Saved {args.output_json}")


if __name__ == "__main__":
    main()
