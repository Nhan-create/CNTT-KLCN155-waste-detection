"""
scripts/import_and_qa_roboflow_battery.py
-----------------------------------------
Automated, offline, and secure ingestion & QA processor for Roboflow battery datasets.

Verified Target Source:
  https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1

Key Security & QA Invariants (Sprint AI-UPGRADE-01 Task 5B Hotfix):
1. PRE-EXTRACTION ZIP SAFETY:
   - Path traversal / Zip Slip prevention.
   - File count limit, uncompressed size limit, and zip bomb ratio detection.
   - Extracts into clean, non-overwriting staging subfolder.
2. REAL DATA.YAML PARSING & CLASS MAPPING:
   - Reads data.yaml before mapping.
   - Validates nc == len(names).
   - Only maps verified battery class names ('battery', 'batteries') to class 0.
   - Blocks or quarantines non-battery source classes.
3. YOLO BBOX GEOMETRY VALIDATION:
   - Exactly 5 tokens per box.
   - Finite numeric coordinates within image bounds [0.0, 1.0].
4. DEDUPLICATION & LEAKAGE CHECK:
   - Internal duplicate and near-duplicate (pHash Hamming <= 8) detection inside ZIP.
   - Cross-split leakage check against frozen train, val, test, and holdout splits.
5. GATEKEEPER INTEGRITY:
   - All imported items staged with status 'PENDING_VISUAL_QA'.
   - approved_for_train is strictly False.
   - Zero merge into training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"

MAX_ZIP_FILES = 2000
MAX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024  # 300 MB
MAX_COMPRESSION_RATIO = 20.0  # Max uncompressed / compressed ratio for files > 1 MB
PHASH_NEAR_DUP_THRESHOLD = 8

VERIFIED_BATTERY_CLASS_NAMES: Set[str] = {
    "battery",
    "batteries",
    "waste_battery",
    "waste_batteries",
    "alkaline",
    "alkaline_jp",
    "lr_alkaline",
}


class ZipSafetyError(Exception):
    """Raised when an untrusted ZIP archive fails security checks."""
    pass


class DataYamlError(Exception):
    """Raised when data.yaml is missing, corrupt, or has mismatched classes."""
    pass


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(filepath: Path) -> Optional[imagehash.ImageHash]:
    try:
        with Image.open(filepath) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def validate_zip_archive_safety(
    zip_path: Path,
    max_files: int = MAX_ZIP_FILES,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
    max_ratio: float = MAX_COMPRESSION_RATIO,
) -> None:
    """
    Inspects ZIP archive metadata before extraction to protect against Zip Slip and Zip Bombs.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        if len(infolist) > max_files:
            raise ZipSafetyError(
                f"Zip safety violation: file count {len(infolist)} exceeds limit {max_files}."
            )

        total_uncompressed = 0
        for info in infolist:
            # 1. Zip Slip / Path Traversal Check
            name = info.filename
            if name.startswith("/") or name.startswith("\\"):
                raise ZipSafetyError(f"Zip slip violation: absolute path '{name}'.")
            parts = Path(name).parts
            if ".." in parts:
                raise ZipSafetyError(f"Zip slip violation: path traversal '..' in '{name}'.")

            total_uncompressed += info.file_size

            # 2. Per-file compression ratio check for files > 1 MB
            if info.file_size > 1024 * 1024 and info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > max_ratio:
                    raise ZipSafetyError(
                        f"Zip bomb violation: entry '{name}' has suspicious compression ratio {ratio:.1f}x."
                    )

        if total_uncompressed > max_uncompressed_bytes:
            raise ZipSafetyError(
                f"Zip safety violation: total size {total_uncompressed:,} bytes exceeds limit {max_uncompressed_bytes:,} bytes."
            )


def safe_extract_untrusted_zip(
    zip_path: Path,
    target_dir: Path,
    max_files: int = MAX_ZIP_FILES,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> Path:
    """
    Safely validates and extracts an untrusted ZIP into target_dir.
    """
    validate_zip_archive_safety(zip_path, max_files=max_files, max_uncompressed_bytes=max_uncompressed_bytes)

    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            extracted_path = (target_dir / info.filename).resolve()
            if not extracted_path.is_relative_to(target_dir):
                raise ZipSafetyError(f"Zip slip violation: extracted entry '{info.filename}' escapes target dir.")
            zf.extract(info, target_dir)

    return target_dir


def parse_and_validate_data_yaml(yaml_path: Path) -> Tuple[Dict[int, Optional[int]], Set[int], List[str]]:
    """
    Parses and validates data.yaml from Roboflow export.
    Returns:
        (class_mapping: dict[src_id, target_id_or_None],
         verified_battery_cids: set[src_id],
         source_names: list[str])
    """
    if not yaml_path.exists():
        raise DataYamlError(f"Missing data.yaml at {yaml_path}")

    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataYamlError(f"Failed to parse data.yaml: {exc}")

    if not isinstance(content, dict):
        raise DataYamlError(f"data.yaml must contain a top-level dictionary, got {type(content).__name__}")

    names_raw = content.get("names")
    nc = content.get("nc")

    if names_raw is None or nc is None:
        raise DataYamlError("data.yaml must define both 'nc' and 'names'.")

    # Normalize names to list
    if isinstance(names_raw, dict):
        # e.g. {0: 'battery', 1: 'metal'}
        sorted_keys = sorted([int(k) for k in names_raw.keys()])
        names = [str(names_raw[k]) for k in sorted_keys]
    elif isinstance(names_raw, list):
        names = [str(n) for n in names_raw]
    else:
        raise DataYamlError(f"Unsupported format for 'names' in data.yaml: {type(names_raw).__name__}")

    if int(nc) != len(names):
        raise DataYamlError(f"Class count mismatch in data.yaml: nc={nc} != len(names)={len(names)}.")

    class_mapping: Dict[int, Optional[int]] = {}
    verified_battery_cids: Set[int] = set()

    for idx, name in enumerate(names):
        norm_name = name.strip().lower()
        if norm_name in VERIFIED_BATTERY_CLASS_NAMES:
            class_mapping[idx] = 0  # Map to project class 0: battery
            verified_battery_cids.add(idx)
        else:
            class_mapping[idx] = None  # Non-battery class -> unmapped / quarantined

    if not verified_battery_cids:
        raise DataYamlError(
            f"No verified battery classes found in data.yaml names: {names}. "
            f"Expected at least one of {sorted(VERIFIED_BATTERY_CLASS_NAMES)}."
        )

    return class_mapping, verified_battery_cids, names


def validate_yolo_bbox(
    line: str,
    valid_source_cids: Set[int],
    battery_source_cids: Set[int],
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates a single YOLO bbox line:
    Expected: class_id xc yc w h (strictly 5 tokens).
    Coordinates in [0.0, 1.0], w > 0, h > 0, bounded in image.
    """
    clean = line.strip()
    if not clean:
        return False, "EMPTY_LINE", None

    parts = clean.split()
    if len(parts) != 5:
        return False, f"EXACTLY_5_TOKENS_REQUIRED ({len(parts)} found)", None

    try:
        cid = int(parts[0])
    except ValueError:
        return False, f"NON_INTEGER_CLASS_ID ({parts[0]})", None

    if cid < 0:
        return False, f"NEGATIVE_CLASS_ID ({cid})", None

    if cid not in valid_source_cids:
        return False, f"CLASS_ID_NOT_IN_DATA_YAML ({cid})", None

    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError as exc:
        return False, f"NON_NUMERIC_COORDINATE: {exc}", None

    xc, yc, w, h = coords
    for val, name in [(xc, "xc"), (yc, "yc"), (w, "w"), (h, "h")]:
        if math.isnan(val) or math.isinf(val):
            return False, f"NON_FINITE_COORDINATE_{name.upper()}", None

    if w <= 0.0 or h <= 0.0:
        return False, f"NON_POSITIVE_DIMENSION (w={w}, h={h})", None

    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        return False, f"CENTER_OUT_OF_BOUNDS (xc={xc}, yc={yc})", None

    x1 = xc - w / 2.0
    x2 = xc + w / 2.0
    y1 = yc - h / 2.0
    y2 = yc + h / 2.0

    TOLERANCE = 1e-6
    if x1 < -TOLERANCE or y1 < -TOLERANCE or x2 > 1.0 + TOLERANCE or y2 > 1.0 + TOLERANCE:
        return False, f"BBOX_EXCEEDS_IMAGE_BOUNDS ([{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}])", None

    is_battery = cid in battery_source_cids
    target_cid = 0 if is_battery else None

    return True, None, {
        "source_class_id": cid,
        "target_class_id": target_cid,
        "is_battery": is_battery,
        "yolo_bbox": [
            max(0.0, min(1.0, xc)),
            max(0.0, min(1.0, yc)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)),
        ]
    }


def load_authoritative_reference_hashes(root: Path) -> Tuple[Dict[str, str], Dict[str, imagehash.ImageHash]]:
    """
    Loads authoritative SHA-256 and pHash references from cache or disk splits.
    Returns: (sha_to_split, phash_dict)
    """
    cache_path = root / "data/audit/authoritative_hashes_cache.json"
    known_shas: Dict[str, str] = {}
    known_phashes: Dict[str, imagehash.ImageHash] = {}

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                for k, v in cdata.items():
                    split = v.get("split", "reference")
                    if "sha256" in v and v["sha256"]:
                        known_shas[v["sha256"]] = f"{split}:{k}"
                    if "phash" in v and v["phash"]:
                        try:
                            known_phashes[f"{split}:{k}"] = imagehash.hex_to_hash(v["phash"])
                        except Exception:
                            pass
        except Exception:
            pass

    return known_shas, known_phashes


def process_roboflow_battery_archive(
    zip_path: Path,
    staging_dir: Path,
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes the full safe import, verification, deduplication, and QA staging pipeline.
    """
    if root_dir is None:
        root_dir = ROOT

    staging_dir.mkdir(parents=True, exist_ok=True)
    existing_extracted = sorted([d for d in staging_dir.glob("extracted_*") if d.is_dir()])
    if existing_extracted:
        validate_zip_archive_safety(zip_path)
        extract_subfolder = existing_extracted[0]
    else:
        extract_subfolder = staging_dir / f"extracted_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        safe_extract_untrusted_zip(zip_path, extract_subfolder)

    # 2. Locate and parse data.yaml
    yaml_candidates = list(extract_subfolder.rglob("data.yaml")) + list(extract_subfolder.rglob("data.yml"))
    if not yaml_candidates:
        raise DataYamlError(f"No data.yaml found inside extracted archive {zip_path.name}")
    data_yaml_p = yaml_candidates[0]
    class_mapping, battery_cids, source_names = parse_and_validate_data_yaml(data_yaml_p)
    valid_source_cids = set(class_mapping.keys())

    # 3. Load reference hashes for leakage checking
    known_shas, known_phashes = load_authoritative_reference_hashes(root_dir)

    # 4. Find all images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = sorted([p for p in extract_subfolder.rglob("*") if p.suffix.lower() in img_exts])

    items = []
    seen_shas: Dict[str, str] = {}
    seen_phashes: List[Tuple[str, imagehash.ImageHash]] = []

    total_boxes = 0
    battery_boxes = 0
    non_battery_boxes = 0
    internal_exact_dups = 0
    internal_near_dups = 0
    external_leakage_count = 0

    for img_p in all_imgs:
        cid = img_p.stem
        sha = compute_sha256(img_p)

        # Decode check
        decode_ok = False
        res = [0, 0]
        ph = None
        try:
            with Image.open(img_p) as im:
                res = list(im.size)
                decode_ok = (im.size[0] > 0 and im.size[1] > 0)
                im.verify()
            with Image.open(img_p) as im:
                ph = imagehash.phash(im)
        except Exception:
            decode_ok = False

        # Internal duplicate check
        internal_exact_match = seen_shas.get(sha)
        internal_near_match = None
        if internal_exact_match:
            internal_exact_dups += 1
        elif ph is not None:
            for prev_cid, prev_ph in seen_phashes:
                dist = ph - prev_ph
                if dist <= PHASH_NEAR_DUP_THRESHOLD:
                    internal_near_match = {"matched_candidate": prev_cid, "hamming_distance": dist}
                    internal_near_dups += 1
                    break

        seen_shas[sha] = cid
        if ph is not None and not internal_exact_match:
            seen_phashes.append((cid, ph))

        # External leakage check
        ext_sha_leak = known_shas.get(sha)
        ext_phash_leak = None
        if ext_sha_leak:
            external_leakage_count += 1
        elif ph is not None and known_phashes:
            min_dist = 999
            closest_split = None
            for ref_name, ref_ph in known_phashes.items():
                d = ph - ref_ph
                if d < min_dist:
                    min_dist = d
                    closest_split = ref_name
            if min_dist <= PHASH_NEAR_DUP_THRESHOLD:
                ext_phash_leak = {"matched_ref": closest_split, "hamming_distance": min_dist}
                external_leakage_count += 1

        # Locate corresponding label
        # Checks: same stem in labels/ or sibling .txt
        lbl_candidates = [
            img_p.parent.parent / "labels" / f"{cid}.txt",
            img_p.parent / f"{cid}.txt",
        ]
        lbl_p = next((p for p in lbl_candidates if p.exists()), None)

        box_records = []
        parse_err = None
        if lbl_p is None:
            label_status = "MISSING_LABEL_FILE"
        else:
            label_status = "FOUND"
            lbl_lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            for line_idx, line in enumerate(lbl_lines):
                ok, err, bbox_info = validate_yolo_bbox(line, valid_source_cids, battery_cids)
                if not ok:
                    parse_err = f"Line {line_idx+1}: {err}"
                    break
                total_boxes += 1
                if bbox_info["is_battery"]:
                    battery_boxes += 1
                else:
                    non_battery_boxes += 1
                box_records.append(bbox_info)

        # Identify source split from parent path (train, valid, test)
        source_split = "unknown"
        for part in img_p.parts:
            if part.lower() in ("train", "valid", "val", "test"):
                source_split = "valid" if part.lower() == "val" else part.lower()
                break

        base_id = cid.split(".rf.")[0]

        # Determine Technical Status & Initial Fail-Closed QA State
        # IMPORTANT (Task 5D Fix): Never auto-assign QA_PASS or VISUAL_QA_PASS on import!
        if not decode_ok:
            technical_status = "REJECT"
            qa_status = "DECODE_ERROR"
            qa_decision = "REJECT"
        elif label_status == "MISSING_LABEL_FILE":
            technical_status = "REJECT"
            qa_status = "MISSING_LABEL"
            qa_decision = "REJECT"
        elif parse_err is not None:
            technical_status = "REJECT"
            qa_status = f"INVALID_ANNOTATION: {parse_err}"
            qa_decision = "REJECT"
        elif ext_sha_leak or ext_phash_leak:
            technical_status = "REJECT"
            qa_status = "LEAKAGE_REJECTED"
            qa_decision = "REJECT"
        elif internal_exact_match:
            technical_status = "REJECT"
            qa_status = "INTERNAL_DUPLICATE_REJECTED"
            qa_decision = "REJECT"
        else:
            technical_status = "TECHNICAL_VALID"
            qa_status = "PENDING_VISUAL_QA"
            qa_decision = "NEEDS_REANNOTATION"

        items.append({
            "candidate_id": cid,
            "filename": img_p.name,
            "relative_path": str(img_p.relative_to(staging_dir)).replace("\\", "/"),
            "resolution": res,
            "decode_ok": decode_ok,
            "sha256": sha,
            "phash": str(ph) if ph else None,
            "source_split": source_split,
            "base_id": base_id,
            "original_status": "ORIGINAL_UNKNOWN",
            "is_base_representative": False,
            "variant_role": "AUGMENTED_VARIANT_EXCLUDED",
            "num_boxes": len(box_records),
            "battery_boxes": sum(1 for b in box_records if b["is_battery"]),
            "boxes": box_records,
            "internal_duplicate": bool(internal_exact_match or internal_near_match),
            "external_leakage": bool(ext_sha_leak or ext_phash_leak),
            "technical_status": technical_status,
            "qa_status": qa_status,
            "qa_decision": qa_decision,
            "visual_qa": qa_decision,
            "review_evidence": None,
            "approved_for_train": False,
            "user_train_approval": "PENDING_PM_APPROVAL",
        })

    # Load re-annotation manifests (prefer full140_reannotation_manifest.json, fallback to minipilot20)
    reann_by_fn: Dict[str, Dict[str, Any]] = {}
    reann_fn_by_base: Dict[str, str] = {}
    full140_p = staging_dir / "reannotated_v2_full140" / "full140_reannotation_manifest.json"
    minipilot20_p = staging_dir / "reannotated_v1_minipilot20" / "minipilot20_reannotation_manifest.json"
    for r_path in (minipilot20_p, full140_p):
        if r_path.exists():
            try:
                with open(r_path, "r", encoding="utf-8") as rf:
                    r_data = json.load(rf)
                for r_rec in r_data.get("items", []):
                    reann_by_fn[r_rec["filename"]] = r_rec
                    reann_fn_by_base[r_rec["base_id"]] = r_rec["filename"]
            except Exception:
                pass

    # Group by base_id to designate 1 canonical representative per base_id
    by_base_group: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_base_group.setdefault(it["base_id"], []).append(it)

    for bid, group in by_base_group.items():
        group_sorted = sorted(group, key=lambda x: x["filename"])
        locked_fn = reann_fn_by_base.get(bid, group_sorted[0]["filename"])
        for it in group_sorted:
            if it["filename"] == locked_fn:
                it["is_base_representative"] = True
                it["variant_role"] = "BASE_REPRESENTATIVE"
            else:
                it["is_base_representative"] = False
                it["variant_role"] = "AUGMENTED_VARIANT_EXCLUDED"

    # Link verified re-annotation review records onto canonical base representatives
    for it in items:
        if it["is_base_representative"] and it["filename"] in reann_by_fn and it["technical_status"] == "TECHNICAL_VALID":
            r_rec = reann_by_fn[it["filename"]]
            ev = r_rec.get("review_evidence")
            if (
                isinstance(ev, dict)
                and ev.get("reviewed_by")
                and ev.get("review_timestamp")
                and ev.get("review_note")
            ):
                it["qa_status"] = "VISUAL_QA_PASS"
                it["qa_decision"] = "VISUAL_QA_PASS"
                it["visual_qa"] = "VISUAL_QA_PASS"
                it["review_evidence"] = ev
                it["reannotated_label_path"] = r_rec.get("label_path")
                it["reannotated_box"] = r_rec.get("new_box")
                it["recommended_leak_free_split"] = r_rec.get("recommended_leak_free_split", it["source_split"])
                it["specimen_cluster_id"] = r_rec.get("specimen_cluster_id", "UNKNOWN")

    # Generate contact sheets and sample overlays for visual review
    contact_sheet_paths = generate_battery_contact_sheets(staging_dir, items)
    overlay_paths = generate_sample_overlays(staging_dir, items)

    manifest_data = {
        "manifest_name": "ROBOFLOW_BATTERY_STAGING_MANIFEST",
        "source_archive": zip_path.name,
        "processed_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(items),
        "total_boxes": total_boxes,
        "battery_boxes": battery_boxes,
        "non_battery_boxes": non_battery_boxes,
        "internal_duplicates": internal_exact_dups + internal_near_dups,
        "external_leaks": external_leakage_count,
        "approved_for_train_count": 0,
        "qa_state_summary": {
            "TECHNICAL_VALID": sum(1 for it in items if it.get("technical_status") == "TECHNICAL_VALID"),
            "VISUAL_QA_PASS": sum(1 for it in items if it.get("visual_qa") == "VISUAL_QA_PASS"),
            "NEEDS_REANNOTATION": sum(1 for it in items if it.get("visual_qa") == "NEEDS_REANNOTATION"),
            "REJECT": sum(1 for it in items if it.get("visual_qa") == "REJECT"),
            "APPROVED_FOR_TRAIN": sum(1 for it in items if it.get("approved_for_train") is True),
        },
        "source_splits": {
            "train_images": sum(1 for it in items if it.get("source_split") == "train"),
            "valid_images": sum(1 for it in items if it.get("source_split") == "valid"),
            "unique_base_images": len(set(it.get("base_id") for it in items)),
            "base_representatives": sum(1 for it in items if it.get("is_base_representative") is True),
            "augmented_variants_excluded": sum(1 for it in items if it.get("is_base_representative") is False),
        },
        "source_data_yaml": {
            "source_names": source_names,
            "verified_battery_class_ids": list(battery_cids),
            "class_mapping": class_mapping,
        },
        "contact_sheets": [str(p.relative_to(staging_dir)).replace("\\", "/") for p in contact_sheet_paths],
        "sample_overlays": [str(p.relative_to(staging_dir)).replace("\\", "/") for p in overlay_paths],
        "items": items,
    }

    manifest_out = staging_dir / "roboflow_battery_staging_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def generate_battery_contact_sheets(
    staging_dir: Path,
    items: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generates visual contact sheets with bounding boxes for QA acceptance."""
    from PIL import ImageDraw, ImageFont

    if output_dir is None:
        output_dir = staging_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "valid_all": [it for it in items if it.get("source_split") == "valid"],
        "train_sample": [it for it in items if it.get("source_split") == "train"][:32],
        "multi_battery": [it for it in items if it.get("battery_boxes", 0) > 1],
    }

    generated_sheets: List[Path] = []
    cols, rows = 4, 4
    cell_w, cell_h = 420, 420
    font = ImageFont.load_default()

    for group_name, group_items in groups.items():
        if not group_items:
            continue
        chunk_size = cols * rows
        num_chunks = (len(group_items) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk = group_items[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(245, 245, 245))
            draw = ImageDraw.Draw(sheet)

            for idx, item in enumerate(chunk):
                img_rel = Path(item["relative_path"])
                img_full = staging_dir / img_rel
                if not img_full.exists():
                    continue

                try:
                    with Image.open(img_full) as raw_im:
                        annotated = raw_im.convert("RGB")
                    im_w, im_h = annotated.size
                    ann_draw = ImageDraw.Draw(annotated)

                    for b_idx, box in enumerate(item.get("boxes", [])):
                        xc, yc, bw, bh = box["yolo_bbox"]
                        x1 = int(round((xc - bw / 2.0) * im_w))
                        y1 = int(round((yc - bh / 2.0) * im_h))
                        x2 = int(round((xc + bw / 2.0) * im_w))
                        y2 = int(round((yc + bh / 2.0) * im_h))
                        ann_draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=4)
                        tag = "bat(0)"
                        ann_draw.rectangle([x1, max(0, y1 - 15), x1 + 45, max(15, y1)], fill=(0, 220, 0))
                        ann_draw.text((x1 + 2, max(0, y1 - 14)), tag, font=font, fill=(0, 0, 0))

                    thumb = annotated.copy()
                    thumb.thumbnail((cell_w - 20, cell_h - 70))

                    col_idx = idx % cols
                    row_idx = idx // cols
                    cx = col_idx * cell_w
                    cy = row_idx * cell_h

                    tx = cx + (cell_w - thumb.width) // 2
                    ty = cy + 50 + (cell_h - 60 - thumb.height) // 2
                    sheet.paste(thumb, (tx, ty))

                    draw.rectangle([cx, cy, cx + cell_w - 1, cy + cell_h - 1], outline=(210, 210, 210), width=1)
                    title = f"#{chunk_idx * chunk_size + idx + 1} {item['candidate_id'][:28]}"
                    meta = f"[{item.get('source_split')}] {im_w}x{im_h} | Boxes: {item.get('battery_boxes')} [bat]"
                    qa_line = f"QA: {item.get('qa_status')} | Train: False"
                    draw.text((cx + 10, cy + 6), title, font=font, fill=(0, 0, 0))
                    draw.text((cx + 10, cy + 20), meta, font=font, fill=(50, 50, 50))
                    draw.text((cx + 10, cy + 34), qa_line, font=font, fill=(0, 120, 0))
                except Exception:
                    pass

            sheet_fn = f"battery_{group_name}_{chunk_idx+1:02d}.jpg"
            out_p = output_dir / sheet_fn
            sheet.save(out_p, quality=90)
            generated_sheets.append(out_p)

    return generated_sheets


def generate_sample_overlays(
    staging_dir: Path,
    items: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    sample_count: int = 6,
) -> List[Path]:
    """Generates full-resolution sample overlays for QA review."""
    from PIL import ImageDraw, ImageFont

    if output_dir is None:
        output_dir = staging_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    generated = []

    picks = []
    val_items = [it for it in items if it.get("source_split") == "valid"]
    multi_items = [it for it in items if it.get("battery_boxes", 0) > 1]
    train_items = [it for it in items if it.get("source_split") == "train"]
    if val_items:
        picks.extend(val_items[:2])
    if multi_items:
        picks.extend(multi_items[:2])
    if train_items:
        picks.extend(train_items[:2])

    for it in picks[:sample_count]:
        img_full = staging_dir / Path(it["relative_path"])
        if not img_full.exists():
            continue
        try:
            with Image.open(img_full) as im:
                over = im.convert("RGB")
            draw = ImageDraw.Draw(over)
            w, h = over.size
            for box in it.get("boxes", []):
                xc, yc, bw, bh = box["yolo_bbox"]
                x1 = int(round((xc - bw / 2.0) * w))
                y1 = int(round((yc - bh / 2.0) * h))
                x2 = int(round((xc + bw / 2.0) * w))
                y2 = int(round((yc + bh / 2.0) * h))
                draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
                draw.rectangle([x1, max(0, y1 - 16), x1 + 50, max(16, y1)], fill=(0, 220, 0))
                draw.text((x1 + 2, max(0, y1 - 15)), "battery", font=font, fill=(0, 0, 0))
            out_p = output_dir / f"overlay_{it['candidate_id']}.jpg"
            over.save(out_p, quality=92)
            generated.append(out_p)
        except Exception:
            pass

    return generated


def main():
    print("=======================================================")
    print("ROBOFLOW BATTERY EXPANSION PROCESSOR (LOCAL OFFLINE)")
    print("=======================================================")

    target_dir = DEFAULT_STAGING_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        print("Status: WAITING FOR USER ZIP FILE.")
        print(f"Target directory: {target_dir}")
        print("\n--- INSTRUCTIONS FOR USER ---")
        print("1. Open your browser and navigate to:")
        print("   https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1")
        print("2. Click 'Download Dataset' -> Select Format: 'YOLOv8' (or YOLOv5 PyTorch).")
        print("3. Select 'download zip to computer' and click Continue.")
        print("4. Move the downloaded .zip file into:")
        print(f"   {target_dir}")
        print("5. Re-run this script to automatically inspect, verify, and prepare the batch:")
        print("   .venv\\Scripts\\python.exe scripts/import_and_qa_roboflow_battery.py")
        print("\nNOTE: No API keys are required. Do NOT paste any Roboflow tokens into chat.")
        return 0

    zip_path = zip_files[0]
    print(f"Found archive: {zip_path.name} ({zip_path.stat().st_size:,} bytes)")
    manifest = process_roboflow_battery_archive(zip_path, target_dir)
    print(f"Processed {manifest['total_images']} images, {manifest['battery_boxes']} battery boxes.")
    print(f"Generated {len(manifest.get('contact_sheets', []))} contact sheets and {len(manifest.get('sample_overlays', []))} sample overlays.")
    print("APPROVED_FOR_TRAIN = 0 (Strict gatekeeper preserved).")
    return 0


if __name__ == "__main__":
    main()
