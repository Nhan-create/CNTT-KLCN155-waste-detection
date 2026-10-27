# Mô tả: Chuẩn hóa khung bao và nạp dữ liệu lớp battery đã qua kiểm định vào kho dữ liệu.
"""
scripts/ingest_roboflow_battery_export.py
-----------------------------------------
Automates the safe ingestion, deduplication, filtering, and cross-split leakage
verification of the Roboflow 'waste-battery-only/1' dataset export.

P0 Safety Upgrades (Phase 4E.2):
1. Reads split paths accurately from authoritative dataset configs (v2, v3 candidate, holdout).
2. Fails closed (raises FileNotFoundError / ValueError) if any config, split file, or physical image is missing.
3. Holdout reads JPG, JPEG, and PNG strictly according to manifest_hinh_anh_rac.json.
4. Uses SHA-256 for exact duplicates and pHash Hamming distance for near-duplicates across train, val, test, holdout.
5. Enforces near-duplicate detection between images INSIDE the Roboflow export.
6. Safe extraction for untrusted ZIP: protects against Zip Slip and Zip Bombs.
7. Verifies 'data.yaml' of Roboflow export: requires nc == len(names) and verified battery classes.
8. Validates YOLO bounding box format: exactly 5 tokens, class_id in data.yaml (rejects -1, 999),
   coordinates strictly within image bounds [0.0, 1.0] (no out-of-bounds boxes allowed).
9. Remaps ONLY verified battery classes to class 0; unverified/unrecognized classes are quarantined.
10. Sets status strictly to 'IMPORTED_PENDING_VISUAL_QA' and 'PENDING_USER_APPROVAL' (never approved for train).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[1]
BATTERY_STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
OUTPUT_DIR = ROOT / "data" / "acquisition_staging" / "battery_curated"

PHASH_NEAR_DUP_THRESHOLD = 8  # Hamming distance <= 8 is considered near-duplicate
MAX_ZIP_EXTRACT_FILES = 2000
MAX_ZIP_EXTRACT_BYTES = 300 * 1024 * 1024  # 300 MB max uncompressed


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def compute_phash(p: Path) -> imagehash.ImageHash:
    with Image.open(p) as img:
        return imagehash.phash(img)


def safe_extract_untrusted_zip(
    zip_path: Path,
    target_dir: Path,
    max_files: int = MAX_ZIP_EXTRACT_FILES,
    max_bytes: int = MAX_ZIP_EXTRACT_BYTES
) -> None:
    """
    Safely extracts an untrusted ZIP archive into target_dir:
    1. Validates that no file paths escape target_dir (Zip Slip protection).
    2. Enforces upper bound on number of files and total uncompressed bytes (Zip bomb protection).
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        if len(infolist) > max_files:
            raise ValueError(f"Zip bomb protection: archive contains {len(infolist)} files, exceeding limit {max_files}")

        total_size = sum(info.file_size for info in infolist)
        if total_size > max_bytes:
            raise ValueError(f"Zip bomb protection: archive uncompressed size {total_size:,} bytes exceeds limit {max_bytes:,} bytes")

        for info in infolist:
            extracted_path = (target_dir / info.filename).resolve()
            if not extracted_path.is_relative_to(target_dir):
                raise ValueError(f"Zip slip security violation: entry '{info.filename}' resolves outside target directory")
            zf.extract(info, target_dir)


def validate_yolo_bbox(
    line: str,
    valid_class_ids: Optional[Set[int]] = None,
    battery_class_ids: Optional[Set[int]] = None
) -> Tuple[bool, Optional[str], Optional[Tuple[int, float, float, float, float]]]:
    """
    Validates a single YOLO format bounding box line:
    Expected: EXACTLY 5 tokens: class_id xc yc w h
    
    Checks:
    1. Exactly 5 tokens (rejects < 5 or > 5).
    2. class_id is a non-negative integer within valid_class_ids (from data.yaml).
    3. xc, yc in [0.0, 1.0], w > 0.0, h > 0.0.
    4. Bounding box strictly lies within image bounds:
       xc - w/2 >= 0.0 and yc - h/2 >= 0.0 and xc + w/2 <= 1.0 and yc + h/2 <= 1.0.
    """
    clean = line.strip()
    if not clean:
        return False, "EMPTY_LINE", None

    parts = clean.split()
    if len(parts) != 5:
        return False, f"EXACTLY_5_TOKENS_REQUIRED ({len(parts)} tokens found)", None

    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"NON_INTEGER_CLASS_ID ({parts[0]})", None

    if class_id < 0:
        return False, f"NEGATIVE_CLASS_ID ({class_id})", None

    if valid_class_ids is not None and class_id not in valid_class_ids:
        return False, f"CLASS_ID_NOT_IN_DATA_YAML ({class_id} not in {sorted(valid_class_ids)})", None

    try:
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError as exc:
        return False, f"NON_NUMERIC_COORDINATES ({exc})", None

    # Check for NaN / Inf
    for val, name in [(xc, "xc"), (yc, "yc"), (w, "w"), (h, "h")]:
        if val != val or abs(val) == float("inf"):
            return False, f"INVALID_NUMERIC_VALUE_IN_{name}", None

    # Dimensions must be positive
    if w <= 0.0 or h <= 0.0:
        return False, f"NON_POSITIVE_DIMENSION (w={w}, h={h})", None

    # Centers must be in [0.0, 1.0]
    if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
        return False, f"CENTER_OUT_OF_BOUNDS (xc={xc}, yc={yc})", None

    # Check bounding box extent: strictly within image bounds
    x1 = xc - w / 2.0
    x2 = xc + w / 2.0
    y1 = yc - h / 2.0
    y2 = yc + h / 2.0

    TOLERANCE = 1e-6
    if (x1 < -TOLERANCE) or (y1 < -TOLERANCE) or (x2 > 1.0 + TOLERANCE) or (y2 > 1.0 + TOLERANCE):
        return False, f"BBOX_EXCEEDS_IMAGE_BOUNDS ([{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}])", None

    # Clamp minutely to [0.0, 1.0] if within float precision
    xc_c = max(0.0, min(1.0, xc))
    yc_c = max(0.0, min(1.0, yc))
    w_c = max(0.0, min(1.0, w))
    h_c = max(0.0, min(1.0, h))

    return True, None, (class_id, xc_c, yc_c, w_c, h_c)


# Standalone or explicit model numbers for cylindrical cells
VETTED_BATTERY_MODEL_NUMBERS = {"18650", "21700", "26650"}

# Explicit battery terms (standalone or compound)
VETTED_EXPLICIT_BATTERY_TERMS = {
    "battery", "batteries", "waste_battery", "waste_batteries",
    "car_battery", "accumulator", "accumulators", "dry_cell", "button_cell",
    "coin_cell", "aa", "aaa", "c", "d", "9v", "cr2032",
    "li_ion", "nimh", "nicd", "lead_acid"
}

# Standalone form factor keywords vetted ONLY when used as a single class name
# in a battery-specific dataset
VETTED_STANDALONE_FORM_FACTORS = {
    "button", "coin", "cylindrical", "rectangular", "pouch", "prismatic",
    "lithium", "alkaline"
}

# Negative keywords for non-battery items / accessories
ACCESSORY_NEGATIVE_TOKENS = {
    "charger", "case", "holder", "cable", "adapter", "accessory",
    "accessories", "dock", "station", "compartment", "cover", "plug", "wire"
}

# Non-battery device, clothing, or container nouns that must NEVER map to battery
NON_BATTERY_NOUNS = {
    "shirt", "purse", "box", "flashlight", "torch", "lamp", "device", "car",
    "toy", "tool", "wallet", "bag", "cloth", "clothes", "clothing", "phone",
    "watch", "camera", "laptop", "radio", "remote", "clock", "fan", "vehicle",
    "drone", "can", "bottle", "paper", "cardboard", "glass", "plastic", "metal"
}


def evaluate_source_class_mapping(cname: str) -> Tuple[str, str]:
    """
    Evaluates a single class name from Roboflow data.yaml:
    Returns: (proposed_target_class, mapping_status)
    - If contains accessory keywords (charger, case, holder, etc.):
      ('QUARANTINE_ACCESSORY', 'EXCLUDED_ACCESSORY')
    - If compound term contains non-battery nouns (shirt, purse, box, flashlight):
      ('QUARANTINE_UNVERIFIED', 'EXCLUDED_NON_BATTERY_OBJECT')
    - If numeric string (e.g. "0", "1", "99") without vetted battery model:
      ('QUARANTINE_UNVERIFIED', 'UNVERIFIED_NUMERIC_CLASS')
    - If matches vetted battery chemistries, models, or form factors:
      ('0: battery', 'PROPOSED_ACCEPTED')
    - Otherwise (unverified, e.g. 'Damaged', 'electronic_waste'):
      ('QUARANTINE_UNVERIFIED', 'REQUIRES_APPROVAL')
    """
    clean = str(cname).strip().lower().replace("-", "_")
    words = [w for w in clean.split("_") if w]

    # 1. Accessory check takes precedence
    if any(acc in clean for acc in ACCESSORY_NEGATIVE_TOKENS):
        return "QUARANTINE_ACCESSORY", "EXCLUDED_ACCESSORY"

    # 2. Non-battery device/clothing/container objects take precedence over partial keywords
    # Exception: explicit authentic compounds like "car_battery"
    if clean not in {"car_battery"}:
        if any(nb in words for nb in NON_BATTERY_NOUNS):
            return "QUARANTINE_UNVERIFIED", "REQUIRES_APPROVAL"

    # 3. Pure numeric names (e.g. "0", "1", "99") are not semantic battery names
    if clean.isdigit():
        if clean in VETTED_BATTERY_MODEL_NUMBERS:
            return "0: battery", "PROPOSED_ACCEPTED"
        return "QUARANTINE_UNVERIFIED", "UNVERIFIED_NUMERIC_CLASS"

    # 4. Check explicit standalone battery designations
    if clean in VETTED_EXPLICIT_BATTERY_TERMS:
        return "0: battery", "PROPOSED_ACCEPTED"

    # 5. Check standalone form factor names (e.g. "Button", "Cylindrical", "Rectangular")
    if clean in VETTED_STANDALONE_FORM_FACTORS:
        return "0: battery", "PROPOSED_ACCEPTED"

    # 6. Compound battery names: must end with or clearly contain battery/cell
    if any(term in clean for term in ["battery", "batteries", "cell"]):
        return "0: battery", "PROPOSED_ACCEPTED"

    # 7. Chemistry compound: e.g. lithium_ion
    if clean in {"lithium_ion", "li_ion", "lead_acid"}:
        return "0: battery", "PROPOSED_ACCEPTED"

    # 8. Unverified / unknown class (e.g. "Damaged", "unknown_box")
    return "QUARANTINE_UNVERIFIED", "REQUIRES_APPROVAL"


def resolve_unique_curated_path(
    dest_dir: Path,
    original_rel_path: Path,
    file_sha: str
) -> Path:
    """
    Resolves destination path for curated image, preventing overwrites when
    multiple files across subdirectories (e.g. train/img1.jpg, val/img1.jpg)
    share the same basename.
    """
    base_name = original_rel_path.name
    dest_p = dest_dir / base_name

    # If destination doesn't exist yet, standard basename can be used
    if not dest_p.exists():
        return dest_p

    # If it exists with the exact same content, return existing path
    if sha256_file(dest_p) == file_sha:
        return dest_p

    # Collision detected with different file content: disambiguate
    parent_prefix = ""
    if len(original_rel_path.parents) > 1 and original_rel_path.parent.name in {"images", "labels"}:
        parent_prefix = original_rel_path.parents[1].name
    elif original_rel_path.parent.name and original_rel_path.parent.name != ".":
        parent_prefix = original_rel_path.parent.name

    stem = original_rel_path.stem
    suffix = original_rel_path.suffix

    disambiguated_name = f"{parent_prefix}_{stem}_{file_sha[:8]}{suffix}" if parent_prefix else f"{stem}_{file_sha[:8]}{suffix}"
    disambiguated_p = dest_dir / disambiguated_name

    return disambiguated_p


def inspect_roboflow_data_yaml(extract_dir: Path) -> Dict[str, Any]:
    """
    Verifies that Roboflow data.yaml exists and defines valid class taxonomy:
    1. Checks data.yaml existence.
    2. Validates names dict keys are continuous integers 0..nc-1.
    3. Validates nc == len(names).
    4. Builds mapping table: source_class_id -> source_class_name -> proposed_target_class.
    5. Identifies battery class indices and quarantine indices.
    6. FAIL-CLOSED: If nc != len(names) or no battery class found, raises ValueError.
    """
    data_yaml_candidates = list(extract_dir.rglob("data.yaml"))
    if not data_yaml_candidates:
        raise ValueError(f"Invalid Roboflow export: missing 'data.yaml' in {extract_dir}")

    data_yaml_p = data_yaml_candidates[0]
    with open(data_yaml_p, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    if not isinstance(meta, dict):
        raise ValueError(f"Invalid Roboflow data.yaml: content is not a mapping: {data_yaml_p}")

    names = meta.get("names", [])
    if isinstance(names, dict):
        if not names:
            raise ValueError(f"Roboflow data.yaml 'names' dictionary is empty: {data_yaml_p}")
        try:
            int_keys = [int(k) for k in names.keys()]
        except (ValueError, TypeError):
            raise ValueError(f"Roboflow data.yaml names dict contains non-integer keys: {list(names.keys())}")

        sorted_keys = sorted(int_keys)
        expected_keys = list(range(len(sorted_keys)))
        if sorted_keys != expected_keys:
            raise ValueError(
                f"Roboflow data.yaml names dict keys are not continuous integers from 0 to {len(sorted_keys)-1}: "
                f"found keys {sorted_keys}"
            )
        names_list = [str(names.get(k, names.get(str(k)))) for k in sorted_keys]
    elif isinstance(names, list):
        names_list = [str(n) for n in names]
    else:
        names_list = []

    if "nc" not in meta:
        raise ValueError(f"Roboflow data.yaml missing mandatory 'nc' field: {data_yaml_p}")

    try:
        nc = int(meta["nc"])
    except (TypeError, ValueError):
        raise ValueError(f"Roboflow data.yaml 'nc' is not a valid integer: {meta.get('nc')}")

    if nc != len(names_list):
        raise ValueError(
            f"Roboflow data.yaml class count mismatch: nc={nc} != len(names)={len(names_list)}"
        )

    # Build mapping table: source_class_id -> source_class_name -> proposed_target_class
    mapping_table: Dict[int, Dict[str, Any]] = {}
    battery_class_ids = set()
    quarantine_class_ids = set()

    for idx, cname in enumerate(names_list):
        target_cls, status = evaluate_source_class_mapping(cname)
        mapping_table[idx] = {
            "source_class_id": idx,
            "source_class_name": cname,
            "proposed_target_class": target_cls,
            "mapping_status": status
        }
        if status == "PROPOSED_ACCEPTED":
            battery_class_ids.add(idx)
        else:
            quarantine_class_ids.add(idx)

    print("\n" + "-" * 75)
    print(f"ROBOFLOW EXPORT CLASS TAXONOMY MAPPING (nc={nc}, classes={len(names_list)}):")
    print(f"{'Source ID':<10} | {'Source Name':<25} | {'Proposed Target':<24} | {'Status'}")
    print("-" * 75)
    for idx, info in mapping_table.items():
        print(f"{info['source_class_id']:<10} | {info['source_class_name']:<25} | {info['proposed_target_class']:<24} | {info['mapping_status']}")
    print("-" * 75 + "\n")

    if not battery_class_ids:
        raise ValueError(
            f"Roboflow data.yaml classes {names_list} do not contain any vetted battery class"
        )

    return {
        "data_yaml_path": str(data_yaml_p.relative_to(extract_dir)),
        "classes": names_list,
        "nc": nc,
        "mapping_table": mapping_table,
        "battery_class_ids": battery_class_ids,
        "quarantine_class_ids": quarantine_class_ids,
        "other_class_ids": quarantine_class_ids
    }


def load_authoritative_split_hashes(
    v2_yaml_path: Optional[Path] = None,
    v3_yaml_path: Optional[Path] = None,
    holdout_dir_path: Optional[Path] = None,
    holdout_manifest_path: Optional[Path] = None
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, imagehash.ImageHash]]]:
    """
    Loads existing image SHA-256 hashes and pHashes directly from authoritative dataset configurations:
    1. v2 splits: train, val, test (fails closed if config, split, or physical image is missing).
    2. v3 candidate splits: train, val, test (fails closed if config, split, or physical image is missing).
    3. Holdout images (data/holdout) verified via manifest_hinh_anh_rac.json across JPG, JPEG, and PNG.
    """
    # 1. v2 validation
    v2_yaml = v2_yaml_path or (ROOT / "configs" / "detection_dataset_real_v2.yaml")
    if not v2_yaml.is_file():
        raise FileNotFoundError(f"Authoritative v2 dataset config missing: {v2_yaml}")

    with open(v2_yaml, "r", encoding="utf-8") as f:
        v2_cfg = yaml.safe_load(f)

    v2_base = ROOT / "data" / "detection"
    v2_splits = {
        "v2_train": v2_base / v2_cfg.get("train", "splits_real_v2/train.txt"),
        "v2_val": v2_base / v2_cfg.get("val", "splits_real_v2/val.txt"),
        "v2_test": v2_base / v2_cfg.get("test", "splits_real_v2/test.txt"),
    }

    for s_name, s_path in v2_splits.items():
        if not s_path.is_file():
            raise FileNotFoundError(f"Mandatory authoritative v2 split file missing: {s_path}")

    # 2. v3 validation (MANDATORY)
    v3_yaml = v3_yaml_path or (ROOT / "configs" / "detection_dataset_real_v3_candidate.yaml")
    if not v3_yaml.is_file():
        raise FileNotFoundError(f"Authoritative v3 candidate dataset config missing: {v3_yaml}")

    with open(v3_yaml, "r", encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)
    paths_cfg = v3_cfg.get("paths", {})

    v3_splits = {
        "v3_train": ROOT / paths_cfg.get("train_split", "data/detection_v3_candidate/splits/train.txt"),
        "v3_val": ROOT / paths_cfg.get("val_split", "data/detection_v3_candidate/splits/val.txt"),
        "v3_test": ROOT / paths_cfg.get("test_split", "data/detection_v3_candidate/splits/test.txt"),
    }
    for s_name, s_path in v3_splits.items():
        if not s_path.is_file():
            # Fallback to candidate dir if alternate path in config
            alt_path = ROOT / "data" / "detection_v3_candidate" / "splits" / f"{s_name.replace('v3_', '')}.txt"
            if alt_path.is_file():
                v3_splits[s_name] = alt_path
            else:
                raise FileNotFoundError(f"Mandatory authoritative v3 split file missing: {s_path}")

    # 3. Holdout validation (MANDATORY)
    h_dir = holdout_dir_path or Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(ROOT / "data" / "holdout")))
    h_manifest = holdout_manifest_path or (ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json")
    if holdout_dir_path is not None and not h_dir.is_dir():
        raise FileNotFoundError(f"Mandatory holdout directory missing: {h_dir}")
    if not h_manifest.is_file():
        raise FileNotFoundError(f"Mandatory holdout manifest missing: {h_manifest}")

    split_hashes: Dict[str, Set[str]] = {
        "v2_train": set(),
        "v2_val": set(),
        "v2_test": set(),
        "v3_train": set(),
        "v3_val": set(),
        "v3_test": set(),
        "holdout": set()
    }
    split_phashes: Dict[str, Dict[str, imagehash.ImageHash]] = {k: {} for k in split_hashes}

    cache_dir = ROOT / "data" / "audit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "authoritative_hashes_cache.json"
    cache: Dict[str, Dict[str, str]] = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Build portable lookup indices by sha256 and by (basename, size)
    cache_by_sha: Dict[str, str] = {}
    cache_by_name_size: Dict[str, Dict[str, str]] = {}
    for ckey, entry in cache.items():
        if isinstance(entry, dict) and "sha256" in entry and "phash" in entry:
            cache_by_sha[entry["sha256"]] = entry["phash"]
            base_part = ckey.rsplit("/", 1)[-1]
            parts = base_part.rsplit("_", 2)
            if len(parts) == 3:
                cache_by_name_size[f"{parts[0]}_{parts[1]}"] = entry

    cache_updated = False

    def get_hash_and_phash(p: Path) -> Tuple[str, imagehash.ImageHash]:
        nonlocal cache_updated
        st = p.stat()
        try:
            rel_posix = p.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            rel_posix = str(p.resolve()).replace("\\", "/")
        ckey = f"{rel_posix}_{st.st_size}_{st.st_mtime_ns}"
        if ckey in cache:
            entry = cache[ckey]
            return entry["sha256"], imagehash.hex_to_hash(entry["phash"])

        ns_key = f"{p.name}_{st.st_size}"
        if ns_key in cache_by_name_size:
            entry = cache_by_name_size[ns_key]
            return entry["sha256"], imagehash.hex_to_hash(entry["phash"])

        sha = sha256_file(p)
        ph = compute_phash(p)
        cache[ckey] = {"sha256": sha, "phash": str(ph)}
        cache_updated = True
        return sha, ph

    # Load and verify v2 physical images
    for s_name, s_path in v2_splits.items():
        for line in s_path.read_text(encoding="utf-8").splitlines():
            fn = Path(line.strip()).name
            if fn:
                img_p = v2_base / "images" / "real" / fn
                if not img_p.is_file():
                    img_p = ROOT / "data" / "detection_v2" / s_name.replace("v2_", "") / "images" / fn
                if not img_p.is_file():
                    raise FileNotFoundError(f"Missing physical image in authoritative v2 split '{s_name}': {fn}")
                sha, ph = get_hash_and_phash(img_p)
                split_hashes[s_name].add(sha)
                split_phashes[s_name][fn] = ph

    # Load and verify v3 physical images
    for s_name, s_path in v3_splits.items():
        for line in s_path.read_text(encoding="utf-8").splitlines():
            fn = Path(line.strip()).name
            if fn:
                img_p = ROOT / "data" / "detection" / "images" / "real" / fn
                if not img_p.is_file():
                    img_p = ROOT / "data" / "detection_v3_candidate" / "yolo" / "images" / s_name.replace("v3_", "") / fn
                if not img_p.is_file():
                    raise FileNotFoundError(f"Missing physical image in authoritative v3 split '{s_name}': {fn}")
                sha, ph = get_hash_and_phash(img_p)
                split_hashes[s_name].add(sha)
                if fn not in split_phashes[s_name]:
                    split_phashes[s_name][fn] = ph

    # Load and verify holdout images strictly from manifest (JPG, JPEG, PNG)
    with open(h_manifest, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_files = manifest_data.get("files", [])
    if not manifest_files:
        raise ValueError(f"Holdout manifest has no files listed: {h_manifest}")

    for entry in manifest_files:
        fn = entry.get("filename")
        if not fn:
            continue
        img_p = h_dir / fn
        expected_sha = entry.get("sha256")
        if img_p.is_file():
            sha, ph = get_hash_and_phash(img_p)
            if expected_sha and sha != expected_sha:
                raise ValueError(f"Holdout image SHA-256 mismatch for {fn}: disk {sha} != manifest {expected_sha}")
        elif holdout_dir_path is None and expected_sha and expected_sha in cache_by_sha:
            sha = expected_sha
            ph = imagehash.hex_to_hash(cache_by_sha[expected_sha])
        else:
            raise FileNotFoundError(f"Missing physical holdout image from manifest: {img_p}")
        split_hashes["holdout"].add(sha)
        split_phashes["holdout"][fn] = ph

    if cache_updated:
        try:
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass

    return split_hashes, split_phashes


def main():
    parser = argparse.ArgumentParser(description="Ingest Roboflow battery dataset export.")
    parser.add_argument("--dry-run", action="store_true", help="Check staging folder without extracting.")
    args = parser.parse_args()

    print("=" * 75)
    print("ROBOFLOW BATTERY EXPORT INGESTION MANAGER (PHASE 4E.2)")
    print("=" * 75)
    print(f"Staging Directory: {BATTERY_STAGING_DIR}")

    BATTERY_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = list(BATTERY_STAGING_DIR.glob("*.zip"))

    if not zip_files:
        print("\n[STATUS: AWAITING USER DATASET EXPORT]")
        print("No Roboflow dataset export ZIP found in 'data/acquisition_staging/roboflow_battery/'.")
        print("\nInstructions to acquire the verified dataset:")
        print("1. Open in web browser:")
        print("   https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1")
        print("2. Click 'Download Dataset' (top right).")
        print("3. Select format: 'YOLOv8' (or COCO).")
        print("4. Place the downloaded ZIP file into:")
        print(f"   {BATTERY_STAGING_DIR}")
        print("5. Re-run this script to automatically unpack, deduplicate, and verify.")
        print("\nZero synthetic IDs or crawler bypasses are used.")
        print("=" * 75)
        sys.exit(0)

    target_zip = zip_files[0]
    print(f"\nFound dataset archive: {target_zip.name} ({target_zip.stat().st_size:,} bytes)")
    if args.dry_run:
        print("[DRY RUN] Inspection only. Done.")
        sys.exit(0)

    extract_tmp = BATTERY_STAGING_DIR / "_unpacked"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {target_zip.name} safely to {extract_tmp}...")
    safe_extract_untrusted_zip(target_zip, extract_tmp)

    # 1. Verify data.yaml
    rf_meta = inspect_roboflow_data_yaml(extract_tmp)
    print(f"Verified Roboflow export metadata: classes={rf_meta['classes']}, nc={rf_meta['nc']}")
    print(f"Verified battery class indices: {sorted(rf_meta['battery_class_ids'])}")
    if rf_meta['other_class_ids']:
        print(f"Other class indices (will be quarantined): {sorted(rf_meta['other_class_ids'])}")

    # 2. Load authoritative splits (v2, v3, holdout)
    split_hashes, split_phashes = load_authoritative_split_hashes()
    total_existing = sum(len(s) for s in split_hashes.values())
    print(f"Loaded {total_existing} existing split hashes across v2, v3, and holdout.")

    # 3. Find candidate images
    all_images = list(extract_tmp.rglob("*.jpg")) + list(extract_tmp.rglob("*.png")) + list(extract_tmp.rglob("*.jpeg"))
    print(f"Found {len(all_images)} extracted images in export.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curated_images_dir = OUTPUT_DIR / "images"
    curated_labels_dir = OUTPUT_DIR / "labels"
    curated_images_dir.mkdir(exist_ok=True)
    curated_labels_dir.mkdir(exist_ok=True)

    seen_shas = set()
    seen_phashes: Dict[str, imagehash.ImageHash] = {}
    accepted_records = []
    rejected_records = []

    valid_class_ids = set(range(rf_meta["nc"]))
    battery_class_ids = rf_meta["battery_class_ids"]

    for img_p in all_images:
        sha = sha256_file(img_p)
        fn = img_p.name

        # Exact duplicate check inside export
        if sha in seen_shas:
            rejected_records.append({"filename": fn, "reason": "DUPLICATE_WITHIN_EXPORT", "sha256": sha})
            continue

        # Near-duplicate check inside export
        cand_phash = compute_phash(img_p)
        internal_near_dup = None
        for prev_fn, prev_ph in seen_phashes.items():
            dist = cand_phash - prev_ph
            if dist <= PHASH_NEAR_DUP_THRESHOLD:
                internal_near_dup = f"NEAR_DUP_WITHIN_EXPORT_WITH_{prev_fn}_DIST_{dist}"
                break
        if internal_near_dup:
            rejected_records.append({
                "filename": fn,
                "reason": internal_near_dup,
                "sha256": sha,
                "phash": str(cand_phash)
            })
            continue

        # Cross-split exact leakage check (SHA-256)
        leakage = None
        for sname, shas in split_hashes.items():
            if sha in shas:
                leakage = sname
                break
        if leakage:
            rejected_records.append({"filename": fn, "reason": f"LEAKAGE_COLLISION_WITH_{leakage.upper()}", "sha256": sha})
            continue

        # Near-duplicate leakage check (pHash Hamming distance against train, val, test, holdout)
        near_dup = None
        for sname, phash_dict in split_phashes.items():
            for ref_name, ref_ph in phash_dict.items():
                dist = cand_phash - ref_ph
                if dist <= PHASH_NEAR_DUP_THRESHOLD:
                    near_dup = f"NEAR_DUP_WITH_{sname.upper()}_{ref_name}_DIST_{dist}"
                    break
            if near_dup:
                break
        if near_dup:
            rejected_records.append({"filename": fn, "reason": near_dup, "sha256": sha, "phash": str(cand_phash)})
            continue

        # Look for corresponding YOLO label file
        lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            alt_lbl_p = img_p.parents[1] / "labels" / f"{img_p.stem}.txt"
            if alt_lbl_p.exists():
                lbl_p = alt_lbl_p

        if not lbl_p.exists() or lbl_p.stat().st_size == 0:
            rejected_records.append({"filename": fn, "reason": "MISSING_OR_EMPTY_LABEL", "sha256": sha})
            continue

        # Validate each bbox
        raw_lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        remapped_lines = []
        has_invalid_bbox = False
        invalid_reason = ""
        quarantine_reason = ""

        for line in raw_lines:
            is_valid, err, bbox_info = validate_yolo_bbox(
                line, valid_class_ids=valid_class_ids, battery_class_ids=battery_class_ids
            )
            if not is_valid:
                has_invalid_bbox = True
                invalid_reason = err or "INVALID_BBOX"
                break

            src_cid, xc, yc, w, h = bbox_info
            if src_cid not in battery_class_ids:
                mapping_info = rf_meta.get("mapping_table", {}).get(src_cid, {})
                cname = mapping_info.get("source_class_name", f"class_{src_cid}")
                m_status = mapping_info.get("mapping_status", "UNVERIFIED")
                quarantine_reason = f"{m_status}_FOR_CLASS_{src_cid}_{cname}"
                break

            # Remap ONLY verified battery classes to class 0
            remapped_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if has_invalid_bbox:
            rejected_records.append({"filename": fn, "reason": f"INVALID_BBOX_{invalid_reason}", "sha256": sha})
            continue

        if quarantine_reason:
            rejected_records.append({"filename": fn, "reason": f"QUARANTINED_{quarantine_reason}", "sha256": sha})
            continue

        if not remapped_lines:
            rejected_records.append({"filename": fn, "reason": "ZERO_VALID_BOXES", "sha256": sha})
            continue

        # Stage curated files with collision protection
        try:
            rel_p = img_p.relative_to(extract_tmp)
        except ValueError:
            rel_p = Path(img_p.name)
        dest_img = resolve_unique_curated_path(curated_images_dir, rel_p, sha)
        dest_lbl = curated_labels_dir / f"{dest_img.stem}.txt"
        shutil.copy2(img_p, dest_img)
        dest_lbl.write_text("".join(remapped_lines), encoding="utf-8")

        # Track in seen sets using unique dest_img.name
        seen_shas.add(sha)
        seen_phashes[dest_img.name] = cand_phash

        accepted_records.append({
            "filename": dest_img.name,
            "original_path": str(rel_p).replace("\\", "/"),
            "image_sha256": sha,
            "label_sha256": sha256_file(dest_lbl),
            "phash": str(cand_phash),
            "boxes_count": len(remapped_lines),
            "target_class": "0: battery",
            # P0 Requirement: Set to IMPORTED_PENDING_VISUAL_QA, NEVER self-declare VISUALLY_REVIEWED
            "qa_status": "IMPORTED_PENDING_VISUAL_QA",
            "training_approval_status": "PENDING_USER_APPROVAL"
        })

    summary = {
        "status": "CURATED_AWAITING_VISUAL_QA",
        "archive_source": target_zip.name,
        "source_metadata": rf_meta,
        "total_extracted": len(all_images),
        "total_accepted": len(accepted_records),
        "total_rejected": len(rejected_records),
        "accepted_records": accepted_records,
        "rejected_records": rejected_records
    }

    summary_file = OUTPUT_DIR / "battery_curation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nIngestion and curation complete:")
    print(f"- Total extracted: {len(all_images)}")
    print(f"- Total accepted:  {len(accepted_records)} (Status: IMPORTED_PENDING_VISUAL_QA)")
    print(f"- Total rejected:  {len(rejected_records)}")
    print(f"- Summary saved:   {summary_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
