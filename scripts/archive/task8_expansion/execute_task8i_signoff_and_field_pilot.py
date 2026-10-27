"""Task 8I (Including Final Hotfix) — Real-Waste Source Sign-Off & Field Data Pilot QA Safety.

Executes all Tasks of Task 8I and Task 8I Final Hotfix:
- Task 1 (Hotfix): Verifies and records a concrete simulation example of `scripts/field_capture_import_and_qa.py`
  holding a `PRODUCT_STUDIO` record (with valid image & valid YOLO label) at `qa_status="NEEDS_REVIEW"`,
  `REAL_WASTE_CONTEXT_VERIFIED=False`, `APPROVED_FOR_TRAIN=False`.
- Task 2 (Hotfix): Synchronizes all report summaries with the exact manifest numbers and re-evaluates
  `task8h_0085`, `task8h_0092`, `task8h_0137` (moldy citrus held in hand with empty observation description
  and no disposal evidence), demoting all 3 from `REAL_WASTE_PASS` to `KEEP_AUXILIARY`
  (`HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL`).
  - Pre-hotfix Task 8I manifest count: `33 REAL_WASTE_PASS` (`27 train`, `6 val`).
  - Post-hotfix Task 8I manifest count: `30 REAL_WASTE_PASS` (`24 train`, `6 val`).
- Task 3 (Hotfix): Separates `inat_photo_license_code` (rights over the photograph) from
  `inat_observation_license_code` (rights over the observation occurrence metadata). Re-evaluates all 11
  cases where observation license is `cc-by-nc` or `all-rights-reserved` while photo license is open
  (`cc0`/`cc-by`/`cc-by-sa`):
  - 5 cases with valid object boxes but unverified disposal context (`0072, 0102, 0114, 0141, 0145`) -> `KEEP_AUXILIARY`.
  - 6 cases with visual/taxonomy defects (`0087, 0118` macro mold; `0151, 0152` plastic bottles in glass query;
    `0149, 0155` wildlife/insect macro on bottle necks) -> remain `REJECT` for visual/taxonomy reasons.
  - 0 cases promoted to `REAL_WASTE_PASS`.
- Task 4 & Handover: Updates Ledger V6, generates `UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.{md,json}`,
  `TASK8I_HOTFIX_MANIFEST_AND_LEDGER_DELTA.{md,json}`, and packages both
  `TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip` and `TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip`.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import zipfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.acquire_and_qa_task8f_scarce_classes import classify_box_pair_overlap
from scripts.correct_task8f_semantic_qa import (
    CLASS_COLORS,
    CLASS_NAMES,
    _struct_box_to_norm_tuple,
)
from scripts.field_capture_import_and_qa import validate_and_allocate_field_batch

REPORTS_DIR = REPO_ROOT / "reports" / "task8_dataset_program"


def write_yolo_labels(lbl_path: Path, boxes: list[dict[str, Any]]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        cid = int(b["class_id"])
        xc, yc, wn, hn = b["yolo_xywhn"]
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_overlay(
    img_path: Path,
    boxes: list[dict[str, Any]],
    out_path: Path,
    header_text: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        font = ImageFont.load_default()
        for b in boxes:
            cid = int(b["class_id"])
            cname = b["class_name"]
            x1, y1, x2, y2 = [float(v) for v in b["bbox_xyxy"]]
            col = CLASS_COLORS.get(cid, (0, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=col, width=4)
            tag = f"{cid}:{cname}"
            draw.rectangle([x1, max(0.0, y1 - 18.0), x1 + len(tag) * 7 + 8, y1], fill=col)
            draw.text((x1 + 4, max(0.0, y1 - 16.0)), tag, fill=(0, 0, 0), font=font)
        draw.rectangle([0, 0, min(rgb.width, 460), 22], fill=(20, 20, 28))
        draw.text((6, 4), header_text[:68], fill=(255, 255, 255), font=font)
        rgb.save(out_path, format="JPEG", quality=88)


W8H_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8h_scarce_class_scaling"
W8I_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8i_source_signoff_and_field_pilot"
W8I_CS_DIR = W8I_DIR / "contact_sheets"
W8I_CHANGED_LBL_PASS_DIR = W8I_DIR / "changed_labels_real_waste_pass"
W8I_CHANGED_LBL_AUX_DIR = W8I_DIR / "changed_labels_auxiliary"
W8I_CHANGED_OVL_DIR = W8I_DIR / "changed_overlays"
W8I_TEMPLATES_DIR = W8I_DIR / "field_pilot_templates"
W8I_REG_EX_DIR = W8I_DIR / "regression_examples"

# 1. Exact classification of the 90 pre-8I REAL_WASTE_PASS iNaturalist images (Post-Hotfix Task 2 & Task 3):
# - 30 Verified REAL_WASTE_PASS (24 train + 6 val): 24 biological, 4 glass, 2 shoes
INAT_VERIFIED_REAL_WASTE_PASS_IDS: set[str] = {
    # 24 Discarded organic household/food waste (1: biological)
    "task8h_0080", "task8h_0083", "task8h_0084",
    "task8h_0090", "task8h_0091", "task8h_0093", "task8h_0094",
    "task8h_0095", "task8h_0096", "task8h_0097", "task8h_0098", "task8h_0099",
    "task8h_0101", "task8h_0103", "task8h_0115", "task8h_0116",
    "task8h_0117", "task8h_0119", "task8h_0121", "task8h_0124",
    "task8h_0138", "task8h_0142", "task8h_0143", "task8h_0144",
    # 4 Outdoor discarded glass bottles/jars (4: glass)
    "task8h_0147", "task8h_0150", "task8h_0153", "task8h_0157",
    # 2 Outdoor discarded footwear (8: shoes)
    "task8h_0158", "task8h_0161",
}

# Task 2 Hotfix: 3 handheld moldy fruit images without disposal evidence -> KEEP_AUXILIARY
INAT_DEMOTE_TO_AUX_HANDHELD_IDS: set[str] = {
    "task8h_0085",
    "task8h_0092",
    "task8h_0137",
}

# 27 Natural Monilinia brown-rot or woodland fruit decay on tree branches / orchard grass / ivy -> KEEP_AUXILIARY
INAT_DEMOTE_TO_AUX_ORCHARD_IDS: set[str] = {
    "task8h_0055", "task8h_0056", "task8h_0057", "task8h_0058", "task8h_0060",
    "task8h_0061", "task8h_0063", "task8h_0066", "task8h_0067", "task8h_0068",
    "task8h_0069", "task8h_0070", "task8h_0071", "task8h_0073", "task8h_0074",
    "task8h_0075", "task8h_0077", "task8h_0078", "task8h_0089", "task8h_0127",
    "task8h_0128", "task8h_0129", "task8h_0131", "task8h_0133", "task8h_0134",
    "task8h_0135", "task8h_0136",
}

# Task 3 Hotfix: Among the 9 pre-8I REAL_WASTE_PASS records with restricted observation license (cc-by-nc / ARR)
# but open photo license (cc0 / cc-by / cc-by-sa):
# - 5 have valid discrete object boundaries but unverified disposal context -> KEEP_AUXILIARY
INAT_OBS_RESTRICTED_TO_AUX_IDS: set[str] = {
    "task8h_0072",  # Photo CC0, Obs ARR: rotting apple in orchard grass (NATURAL_ORCHARD_PLANT_DECAY)
    "task8h_0102",  # Photo CC-BY-SA, Obs CC-BY-NC: shriveled citrus on clean wooden table (unverified disposal)
    "task8h_0114",  # Photo CC-BY, Obs CC-BY-NC: moldy orange on metal beam (unverified disposal)
    "task8h_0141",  # Photo CC-BY, Obs CC-BY-NC: dried orange peels on wooden board (unverified disposal)
    "task8h_0145",  # Photo CC-BY, Obs CC-BY-NC: half moldy lemon on dark studio surface (unverified disposal)
}

# - 4 of the 9 pre-8I REAL_WASTE_PASS records with restricted observation license fail visual/taxonomy QA -> REJECT
INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS: set[str] = {
    "task8h_0087",  # Photo CC-BY, Obs CC-BY-NC: frameless macro mold colonies on agar/peel
    "task8h_0118",  # Photo CC-BY, Obs CC-BY-NC: frameless macro mold colonies
    "task8h_0151",  # Photo CC0, Obs CC-BY-NC: plastic bottle mis-queried under 4:glass
    "task8h_0152",  # Photo CC0, Obs CC-BY-NC: plastic bottle ('discarded plastic bottle') mis-queried under 4:glass
}

# 21 pre-8I REAL_WASTE_PASS records with open photo+obs licenses that are frameless macro mold colonies or duplicates -> REJECT
INAT_REJECT_MACRO_OR_DUP_IDS: set[str] = {
    "task8h_0059", "task8h_0062", "task8h_0065", "task8h_0076", "task8h_0079",
    "task8h_0081", "task8h_0082", "task8h_0088", "task8h_0100", "task8h_0104",
    "task8h_0105", "task8h_0106", "task8h_0107", "task8h_0110", "task8h_0111",
    "task8h_0112", "task8h_0120", "task8h_0122", "task8h_0123", "task8h_0130",
    "task8h_0140",
}

TASK8I_BBOX_OVERRIDES: dict[str, list[tuple[int, float, float, float, float]]] = {
    "task8h_0066": [(1, 0.22, 0.24, 0.88, 0.95)],
    "task8h_0071": [(1, 0.34, 0.14, 0.88, 0.78)],
    "task8h_0080": [(1, 0.14, 0.22, 0.98, 0.98)],
    "task8h_0083": [(1, 0.08, 0.18, 0.96, 0.96)],
    "task8h_0091": [(1, 0.02, 0.04, 0.54, 0.56), (1, 0.30, 0.04, 0.98, 0.92)],
    "task8h_0092": [(1, 0.14, 0.34, 0.88, 0.92)],
    "task8h_0094": [(1, 0.04, 0.04, 0.96, 0.94)],
    "task8h_0095": [(1, 0.24, 0.04, 0.98, 0.96)],
    "task8h_0099": [(1, 0.12, 0.22, 0.88, 0.72)],
    "task8h_0116": [(1, 0.02, 0.02, 0.96, 0.42), (1, 0.04, 0.38, 0.94, 0.96)],
    "task8h_0117": [(1, 0.02, 0.04, 0.96, 0.94)],
    "task8h_0119": [(1, 0.02, 0.02, 0.98, 0.96)],
    "task8h_0121": [(1, 0.02, 0.22, 0.98, 0.92)],
    "task8h_0137": [(1, 0.36, 0.36, 0.86, 0.72)],
    "task8h_0142": [(1, 0.10, 0.32, 0.84, 0.68)],
    "task8h_0167": [(0, 0.38, 0.34, 0.82, 0.92)],
}


def _classify_license_pair(photo_lic: str, obs_lic: str) -> dict[str, Any]:
    """Task 3: Separately evaluate Photo License (image copyright) vs Observation License (occurrence metadata)."""
    p = (photo_lic or "all-rights-reserved").strip().lower()
    o = (obs_lic or "all-rights-reserved").strip().lower()
    allowed = {"cc0", "pdm", "cc-by", "cc-by-sa"}
    photo_usable = p in allowed
    obs_open = o in allowed

    if p == "cc0":
        photo_cat = "CC0"
    elif p == "cc-by":
        photo_cat = "CC-BY"
    elif p == "cc-by-sa":
        photo_cat = "CC-BY-SA"
    elif "nc" in p:
        photo_cat = "CC-BY-NC"
    else:
        photo_cat = "ALL_RIGHTS_RESERVED"

    if obs_open:
        combined_cat = photo_cat
    elif "nc" in o:
        combined_cat = f"PHOTO_{photo_cat}_OBS_CC-BY-NC"
    else:
        combined_cat = f"PHOTO_{photo_cat}_OBS_ARR"

    return {
        "photo_license_category": photo_cat,
        "photo_license_usable_for_image": photo_usable,
        "observation_metadata_license_open": obs_open,
        "photo_obs_license_mismatch": bool(photo_usable and not obs_open),
        "combined_license_status": combined_cat,
        "both_photo_and_obs_open": bool(photo_usable and obs_open),
    }


def _make_box_structs(
    norm_tuples: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
    origin: str,
) -> list[dict[str, Any]]:
    boxes = []
    for cid, x1n, y1n, x2n, y2n in norm_tuples:
        x1 = round(x1n * width, 1)
        y1 = round(y1n * height, 1)
        x2 = round(x2n * width, 1)
        y2 = round(y2n * height, 1)
        xc = round((x1n + x2n) / 2.0, 6)
        yc = round((y1n + y2n) / 2.0, 6)
        wn = round(x2n - x1n, 6)
        hn = round(y2n - y1n, 6)
        boxes.append(
            {
                "class_id": int(cid),
                "class_name": CLASS_NAMES[int(cid)],
                "bbox_xyxy": [x1, y1, x2, y2],
                "yolo_xywhn": [xc, yc, wn, hn],
                "source_box_origin": origin,
            }
        )
    return boxes


def render_pm_audit_contact_sheet(
    records: list[dict[str, Any]],
    out_path: Path,
    title: str,
    cols: int = 5,
    thumb_w: int = 340,
    thumb_h: int = 260,
) -> None:
    """Render high-legibility contact sheet showing candidate_id, post-8I verdict, photo/obs license, and bbox overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(records)
    rows = max(1, math.ceil(n / cols))
    header_h = 54
    canvas = Image.new("RGB", (cols * thumb_w, header_h + rows * thumb_h), (18, 22, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 18), f"{title} ({n} images)", fill=(255, 255, 255), font=font)

    status_colors = {
        "REAL_WASTE_PASS": (28, 168, 76),
        "KEEP_AUXILIARY": (214, 144, 20),
        "REJECT": (204, 44, 44),
        "NEEDS_REVIEW": (148, 82, 212),
    }

    for idx, r in enumerate(records):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = header_h + row * thumb_h

        ovl_p = REPO_ROOT / r["overlay_path"]
        img_p = REPO_ROOT / r["image_path"]
        src_p = ovl_p if ovl_p.exists() else img_p
        try:
            with Image.open(src_p) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail((thumb_w - 10, thumb_h - 56))
                ox = x0 + (thumb_w - rgb.width) // 2
                oy = y0 + 6
                canvas.paste(rgb, (ox, oy))
        except Exception:
            pass

        st = r["qa_status"]
        badge_col = status_colors.get(st, (90, 90, 90))
        draw.rectangle([x0 + 4, y0 + thumb_h - 48, x0 + thumb_w - 4, y0 + thumb_h - 4], fill=(28, 32, 40))
        draw.rectangle([x0 + 4, y0 + thumb_h - 48, x0 + 12, y0 + thumb_h - 4], fill=badge_col)

        p_lic = r.get("inat_photo_license_code", r.get("license", ""))
        o_lic = r.get("inat_observation_license_code", p_lic)
        cat = r.get("biological_decay_category", r.get("domain_tag", ""))
        line1 = f"{r['candidate_id']} | {st} | {r['primary_class_name']} ({r['bbox_count']}b)"
        line2 = f"P:{p_lic} / O:{o_lic} | {cat[:28]}"
        draw.text((x0 + 16, y0 + thumb_h - 44), line1[:48], fill=(245, 245, 245), font=font)
        draw.text((x0 + 16, y0 + thumb_h - 24), line2[:48], fill=(200, 210, 225), font=font)

    canvas.save(out_path, format="JPEG", quality=90)


def generate_product_studio_regression_example() -> Path:
    """Deliverable #6: Concrete simulated test example where PRODUCT_STUDIO + valid image + valid YOLO label is held at NEEDS_REVIEW."""
    W8I_REG_EX_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        img_dir = td_p / "images"
        lbl_dir = td_p / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        im = Image.new("RGB", (640, 480), (250, 250, 250))
        dr = ImageDraw.Draw(im)
        dr.rectangle([160, 100, 480, 380], fill=(80, 120, 200), outline=(20, 40, 80), width=4)
        im.save(img_dir / "studio_catalog_shoe_01.jpg", format="JPEG")
        (lbl_dir / "studio_catalog_shoe_01.txt").write_text("8 0.500000 0.500000 0.500000 0.583333\n", encoding="utf-8")

        manifest_rows = [
            {
                "capture_id": "sim_product_studio_0001",
                "image_filename": "studio_catalog_shoe_01.jpg",
                "label_filename": "studio_catalog_shoe_01.txt",
                "specimen_id": "SPEC_SHO_202610_0001",
                "session_id": "SESS_20261016_STUDIO_TEST01",
                "photographer": "QA_SIMULATION_HARNESS",
                "license": "CC-BY-4.0",
                "rights_confirmed": True,
                "privacy_cleared": True,
                "context_type": "PRODUCT_STUDIO",
                "distance_m": 1.0,
            }
        ]
        res = validate_and_allocate_field_batch(manifest_rows, img_dir, lbl_dir)

    rec = res["records"][0]
    assert rec["FILE_AND_BBOX_FORMAT_VALID"] is True
    assert rec["qa_status"] == "NEEDS_REVIEW"
    assert rec["domain_tag"] == "PRODUCT_STUDIO"
    assert rec["REAL_WASTE_CONTEXT_VERIFIED"] is False
    assert rec["OBJECT_ANNOTATION_VALID"] is False
    assert rec["APPROVED_FOR_TRAIN"] is False
    assert rec["target_split"] == "MANUAL_REVIEW_QUEUE"

    out_p = W8I_REG_EX_DIR / "product_studio_held_at_needs_review_example.json"
    out_p.write_text(
        json.dumps(
            {
                "example_type": "SIMULATED_REGRESSION_TEST_FIXTURE_NOT_DATASET_DATA",
                "description": (
                    "Demonstrates that a PRODUCT_STUDIO image with a valid image file and valid YOLO label "
                    "is NEVER promoted to REAL_WASTE_PASS by scripts/field_capture_import_and_qa.py; "
                    "it is held at qa_status='NEEDS_REVIEW' with REAL_WASTE_CONTEXT_VERIFIED=False and APPROVED_FOR_TRAIN=False."
                ),
                "input_manifest_row": manifest_rows[0],
                "importer_output_summary": {
                    "tool_version": res["tool_version"],
                    "qa_status_counts": res["qa_status_counts"],
                    "technical_format_valid_count": res["technical_format_valid_count"],
                    "real_waste_context_verified_count": res["real_waste_context_verified_count"],
                    "qa_pass_rate": res["qa_pass_rate"],
                },
                "importer_output_record": rec,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_p


def main() -> None:
    for d in (
        W8I_DIR,
        W8I_CS_DIR,
        W8I_CHANGED_LBL_PASS_DIR,
        W8I_CHANGED_LBL_AUX_DIR,
        W8I_CHANGED_OVL_DIR,
        W8I_TEMPLATES_DIR,
        W8I_REG_EX_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Clear changed label/overlay staging dirs before regenerating
    for p in list(W8I_CHANGED_LBL_PASS_DIR.glob("*.txt")) + list(W8I_CHANGED_LBL_AUX_DIR.glob("*.txt")) + list(W8I_CHANGED_OVL_DIR.glob("*.jpg")):
        p.unlink()

    qa_p = W8H_DIR / "task8h_qa_manifest.json"
    inat_cache_p = W8H_DIR / "inat_api_Metadata_cache.json"
    pre_snap_p = W8I_DIR / "pre_8i_task8h_qa_manifest_snapshot.json"

    if not pre_snap_p.exists():
        shutil.copy2(qa_p, pre_snap_p)

    # Read from the immutable pre-8I snapshot so running this script is 100% idempotent
    qa = json.loads(pre_snap_p.read_text(encoding="utf-8"))
    inat_cache = json.loads(inat_cache_p.read_text(encoding="utf-8"))

    # Verify the 90 pre-8I REAL_WASTE_PASS iNaturalist partition is disjoint and complete
    union_90 = (
        INAT_VERIFIED_REAL_WASTE_PASS_IDS
        | INAT_DEMOTE_TO_AUX_HANDHELD_IDS
        | INAT_DEMOTE_TO_AUX_ORCHARD_IDS
        | INAT_OBS_RESTRICTED_TO_AUX_IDS
        | INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS
        | INAT_REJECT_MACRO_OR_DUP_IDS
    )
    assert len(union_90) == 90, f"Expected 90 disjoint IDs, got {len(union_90)}"

    lbl_pass_dir = W8H_DIR / "labels_real_waste_pass"
    lbl_aux_dir = W8H_DIR / "labels_auxiliary"
    lbl_rev_dir = W8H_DIR / "labels_needs_review"
    ovl_dir = W8H_DIR / "overlays"

    changed_records: list[dict[str, Any]] = []
    inat_verified_records: list[dict[str, Any]] = []
    inat_90_pm_audit_records: list[dict[str, Any]] = []
    obs_restricted_11_records: list[dict[str, Any]] = []

    for r in qa["records"]:
        cid = r["candidate_id"]
        pre_status = r["qa_status"]
        r["pre_8i_qa_status"] = pre_status
        r["bbox_changed_in_8i"] = False
        r["status_changed_in_8i"] = False

        if cid in TASK8I_BBOX_OVERRIDES:
            new_norm = TASK8I_BBOX_OVERRIDES[cid]
            r["boxes"] = _make_box_structs(
                new_norm,
                int(r["width"]),
                int(r["height"]),
                "task8i_visual_qa_tightened_instance_box",
            )
            r["bbox_count"] = len(r["boxes"])
            r["label_corrections_count"] = int(r.get("label_corrections_count", 0)) + 1
            r["bbox_changed_in_8i"] = True

        if r["source_id"] == "SRC_8H_04_INATURALIST_BIOLOGICAL_DECAY_FIELD":
            obs_id = r["source_image_id"].replace("inat_", "")
            obs = inat_cache[obs_id]
            photos = obs.get("photos") or []
            ph = photos[0] if photos else {}
            ph_id = int(ph.get("id", 0))
            ph_url = (ph.get("url") or r["original_url"]).replace("square", "large")
            obs_url = f"https://www.inaturalist.org/observations/{obs_id}"
            user_obj = obs.get("user") or {}
            author_login = user_obj.get("login") or r["author"]
            author_name = user_obj.get("name") or author_login
            ph_attr = ph.get("attribution") or ""
            ph_lic = (ph.get("license_code") or "all-rights-reserved").lower()
            obs_lic = (obs.get("license_code") or "all-rights-reserved").lower()
            lic_info = _classify_license_pair(ph_lic, obs_lic)

            r["inat_observation_id"] = int(obs_id)
            r["inat_observation_url"] = obs_url
            r["inat_photo_id"] = ph_id
            r["inat_photo_url"] = ph_url
            r["inat_user_login"] = author_login
            r["inat_author"] = author_name
            r["inat_photo_attribution"] = ph_attr
            r["inat_photo_license_code"] = ph_lic
            r["inat_observation_license_code"] = obs_lic
            r["inat_photo_license_category"] = lic_info["photo_license_category"]
            r["inat_photo_license_usable"] = lic_info["photo_license_usable_for_image"]
            r["inat_observation_license_open"] = lic_info["observation_metadata_license_open"]
            r["inat_photo_obs_license_mismatch"] = lic_info["photo_obs_license_mismatch"]
            r["inat_license_category"] = lic_info["combined_license_status"]
            r["inat_license_commercial_ok"] = lic_info["both_photo_and_obs_open"]
            r["license"] = (
                f"Photo:{ph_lic.upper()} (Image Reuse OK) / Obs:{obs_lic.upper()}"
                if lic_info["photo_license_usable_for_image"]
                else f"Photo:{ph_lic.upper()} / Obs:{obs_lic.upper()}"
            )

            if pre_status == "REAL_WASTE_PASS":
                if cid in INAT_VERIFIED_REAL_WASTE_PASS_IDS:
                    assert lic_info["photo_license_usable_for_image"] and lic_info["observation_metadata_license_open"]
                    r["qa_status"] = "REAL_WASTE_PASS"
                    r["domain_tag"] = "ACTUAL_WASTE"
                    r["OBJECT_ANNOTATION_VALID"] = True
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = True
                    if r["primary_class_id"] == 1:
                        r["biological_decay_category"] = "DISCARDED_ORGANIC_FOOD_WASTE"
                        r["qa_reason"] = (
                            "Task 8I verified: genuine discarded organic food/kitchen waste with tight "
                            f"instance box and verified {lic_info['photo_license_category']} photo+observation license -> REAL_WASTE_PASS."
                        )
                    else:
                        r["biological_decay_category"] = "OUTDOOR_LITTER_SCARCE_CLASS"
                        r["qa_reason"] = (
                            f"Task 8I verified: authentic outdoor discarded {r['primary_class_name']} litter "
                            f"with tight instance box and verified {lic_info['photo_license_category']} license -> REAL_WASTE_PASS."
                        )
                elif cid in INAT_DEMOTE_TO_AUX_HANDHELD_IDS:
                    r["qa_status"] = "KEEP_AUXILIARY"
                    r["domain_tag"] = "AMBIGUOUS_CONTEXT"
                    r["OBJECT_ANNOTATION_VALID"] = True
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = False
                    r["target_split"] = "AUXILIARY_NON_WASTE_TRAIN_HOLD"
                    r["biological_decay_category"] = "HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL"
                    r["status_changed_in_8i"] = True
                    r["qa_reason"] = (
                        "Task 8I Hotfix (Task 2) re-marked REAL_WASTE_PASS -> KEEP_AUXILIARY: moldy citrus fruit "
                        "held in hand with empty observation description and no visual evidence of waste disposal."
                    )
                elif cid in INAT_DEMOTE_TO_AUX_ORCHARD_IDS:
                    r["qa_status"] = "KEEP_AUXILIARY"
                    r["domain_tag"] = "IN_USE"
                    r["OBJECT_ANNOTATION_VALID"] = True
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = False
                    r["target_split"] = "AUXILIARY_NON_WASTE_TRAIN_HOLD"
                    r["biological_decay_category"] = "NATURAL_ORCHARD_PLANT_DECAY"
                    r["status_changed_in_8i"] = True
                    r["qa_reason"] = (
                        "Task 8I re-marked REAL_WASTE_PASS -> KEEP_AUXILIARY: natural orchard/plant decay "
                        "(Monilinia brown rot on fruit on tree branch or orchard grass, not discarded waste)."
                    )
                elif cid in INAT_OBS_RESTRICTED_TO_AUX_IDS:
                    # Task 3 Hotfix: Photo license is CC0/CC-BY/CC-BY-SA (usable), observation is CC-BY-NC/ARR,
                    # and visual context is orchard rot (0072) or isolated specimen without disposal proof -> KEEP_AUXILIARY
                    r["qa_status"] = "KEEP_AUXILIARY"
                    r["domain_tag"] = "IN_USE" if cid == "task8h_0072" else "AMBIGUOUS_CONTEXT"
                    r["OBJECT_ANNOTATION_VALID"] = True
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = False
                    r["target_split"] = "AUXILIARY_NON_WASTE_TRAIN_HOLD"
                    r["biological_decay_category"] = (
                        "NATURAL_ORCHARD_PLANT_DECAY_OBS_RESTRICTED"
                        if cid == "task8h_0072"
                        else "ISOLATED_SPECIMEN_UNVERIFIED_DISPOSAL_OBS_RESTRICTED"
                    )
                    r["status_changed_in_8i"] = True
                    r["qa_reason"] = (
                        f"Task 8I Hotfix (Task 3): photo license '{ph_lic}' permits image reuse (observation metadata '{obs_lic}' not redistributed), "
                        "but visual context lacks disposal evidence -> routed to KEEP_AUXILIARY (not auto-rejected for obs license, not promoted to REAL_WASTE_PASS)."
                    )
                elif cid in INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS:
                    r["qa_status"] = "REJECT"
                    r["domain_tag"] = "IN_USE"
                    r["OBJECT_ANNOTATION_VALID"] = False
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = False
                    r["target_split"] = "REJECTED"
                    if cid in ("task8h_0151", "task8h_0152"):
                        r["biological_decay_category"] = "TAXONOMY_MISMATCH_PLASTIC_IN_GLASS_QUERY"
                        r["qa_reason"] = (
                            f"Task 8I Hotfix (Task 3): photo license '{ph_lic}' is open (obs '{obs_lic}'), "
                            "but image depicts a plastic water bottle mis-queried under 4:glass -> REJECT for taxonomy mismatch."
                        )
                    else:
                        r["biological_decay_category"] = "MACRO_MOLD_COLONY_CLOSEUP"
                        r["qa_reason"] = (
                            f"Task 8I Hotfix (Task 3): photo license '{ph_lic}' is open (obs '{obs_lic}'), "
                            "but image is a frameless macro mold colony close-up without discrete waste boundary -> REJECT for visual QA."
                        )
                    r["status_changed_in_8i"] = True
                elif cid in INAT_REJECT_MACRO_OR_DUP_IDS:
                    r["qa_status"] = "REJECT"
                    r["domain_tag"] = "IN_USE"
                    r["OBJECT_ANNOTATION_VALID"] = False
                    r["REAL_WASTE_CONTEXT_VERIFIED"] = False
                    r["target_split"] = "REJECTED"
                    if cid == "task8h_0079":
                        r["biological_decay_category"] = "SAME_SPECIMEN_CLOSEUP_DUPLICATE"
                        r["qa_reason"] = (
                            "Task 8I re-marked REAL_WASTE_PASS -> REJECT: macro close-up of the same "
                            "moldy tomato specimen as task8h_0097 by the same author."
                        )
                    else:
                        r["biological_decay_category"] = "MACRO_MOLD_COLONY_CLOSEUP"
                        r["qa_reason"] = (
                            "Task 8I re-marked REAL_WASTE_PASS -> REJECT: frameless macro close-up of "
                            "mold mycelium/hyphae or wild bark fungus without discrete waste item boundary."
                        )
                    r["status_changed_in_8i"] = True
                else:
                    raise AssertionError(f"Unhandled pre-8I REAL_WASTE_PASS iNaturalist ID: {cid}")
            elif pre_status == "KEEP_AUXILIARY":
                r["biological_decay_category"] = "AUXILIARY_SORTING_OR_NATURAL"
            else:
                if cid in ("task8h_0149", "task8h_0155"):
                    r["biological_decay_category"] = "NON_WASTE_WILDLIFE_ON_BOTTLE_NECK"
                    r["qa_reason"] = (
                        f"Task 8I Hotfix (Task 3): photo license '{ph_lic}' is open (obs '{obs_lic}'), "
                        "but image is a wildlife/insect macro shot on a bottle neck -> remains REJECT for visual context."
                    )
                else:
                    r["biological_decay_category"] = "NON_WASTE_OR_UNANNOTATABLE"

            r["post_8i_qa_status"] = r["qa_status"]
            inat_verified_records.append(r)
            if pre_status == "REAL_WASTE_PASS":
                inat_90_pm_audit_records.append(r)
            if lic_info["photo_obs_license_mismatch"]:
                obs_restricted_11_records.append(r)
        else:
            r["post_8i_qa_status"] = r["qa_status"]

        # Synchronize physical YOLO label files and overlays
        old_pass_lbl = lbl_pass_dir / f"{cid}.txt"
        old_aux_lbl = lbl_aux_dir / f"{cid}.txt"
        old_rev_lbl = lbl_rev_dir / f"{cid}.txt"
        for p in (old_pass_lbl, old_aux_lbl, old_rev_lbl):
            if p.exists():
                p.unlink()

        if r["qa_status"] == "REAL_WASTE_PASS":
            lbl_p = lbl_pass_dir / f"{cid}.txt"
            write_yolo_labels(lbl_p, r["boxes"])
            r["label_path"] = lbl_p.relative_to(REPO_ROOT).as_posix()
        elif r["qa_status"] == "KEEP_AUXILIARY":
            lbl_p = lbl_aux_dir / f"{cid}.txt"
            write_yolo_labels(lbl_p, r["boxes"])
            r["label_path"] = lbl_p.relative_to(REPO_ROOT).as_posix()
        elif r["qa_status"] == "NEEDS_REVIEW":
            lbl_p = lbl_rev_dir / f"{cid}.txt"
            write_yolo_labels(lbl_p, r["boxes"])
            r["label_path"] = lbl_p.relative_to(REPO_ROOT).as_posix()
        else:
            r["label_path"] = ""

        if r["bbox_changed_in_8i"] or r["status_changed_in_8i"]:
            ovl_p = REPO_ROOT / r["overlay_path"]
            img_p = REPO_ROOT / r["image_path"]
            render_overlay(img_p, r["boxes"], ovl_p, f"{cid} [{r['qa_status']}]")
            shutil.copy2(ovl_p, W8I_CHANGED_OVL_DIR / ovl_p.name)
            if r["qa_status"] == "REAL_WASTE_PASS":
                shutil.copy2(lbl_pass_dir / f"{cid}.txt", W8I_CHANGED_LBL_PASS_DIR / f"{cid}.txt")
            elif r["qa_status"] == "KEEP_AUXILIARY":
                shutil.copy2(lbl_aux_dir / f"{cid}.txt", W8I_CHANGED_LBL_AUX_DIR / f"{cid}.txt")
            changed_records.append(r)

    assert len(obs_restricted_11_records) == 11, f"Expected 11 obs-restricted records, got {len(obs_restricted_11_records)}"

    # Recompute all top-level summary metrics in task8h_qa_manifest.json
    records = qa["records"]
    pass_recs = [r for r in records if r["qa_status"] == "REAL_WASTE_PASS"]
    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8H"]

    assert not ({r["session_group"] for r in train_recs} & {r["session_group"] for r in val_recs})
    assert not ({r["specimen_group"] for r in train_recs} & {r["specimen_group"] for r in val_recs})

    for src_id, s_info in qa["source_stats"].items():
        s_recs = [r for r in records if r["source_id"] == src_id]
        s_pass = [r for r in s_recs if r["qa_status"] == "REAL_WASTE_PASS"]
        s_tr = [r for r in s_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
        s_va = [r for r in s_pass if r["target_split"] == "VAL_REAL_WASTE_W8H"]
        s_aux = [r for r in s_recs if r["qa_status"] == "KEEP_AUXILIARY"]
        s_rev = [r for r in s_recs if r["qa_status"] == "NEEDS_REVIEW"]
        s_rej = [r for r in s_recs if r["qa_status"] == "REJECT"]
        s_info["downloaded"] = len(s_recs)
        s_info["real_waste_pass"] = len(s_pass)
        s_info["train_real_waste"] = len(s_tr)
        s_info["val_real_waste"] = len(s_va)
        s_info["keep_auxiliary"] = len(s_aux)
        s_info["needs_review"] = len(s_rev)
        s_info["reject"] = len(s_rej)
        s_info["pass_yield_rate"] = round(len(s_pass) / max(1, len(s_recs)), 4)

    qa["qa_status_counts"] = dict(Counter(r["qa_status"] for r in records))
    qa["domain_counts"] = dict(Counter(r["domain_tag"] for r in records))
    qa["split_counts"] = dict(Counter(r["target_split"] for r in records))
    qa["real_waste_pass_total_images"] = len(pass_recs)
    qa["real_waste_pass_train_images"] = len(train_recs)
    qa["real_waste_pass_val_images"] = len(val_recs)
    qa["real_waste_pass_total_boxes"] = sum(r["bbox_count"] for r in pass_recs)
    qa["real_waste_pass_multiobject_images"] = sum(1 for r in pass_recs if r["bbox_count"] >= 2)
    qa["real_waste_pass_multiclass_images"] = sum(1 for r in pass_recs if r["is_multiclass"])
    qa["real_waste_pass_unique_specimens"] = len({r["specimen_group"] for r in pass_recs})
    qa["real_waste_pass_unique_sessions"] = len({r["session_group"] for r in pass_recs})
    qa["real_waste_pass_train_unique_specimens"] = len({r["specimen_group"] for r in train_recs})
    qa["real_waste_pass_train_unique_sessions"] = len({r["session_group"] for r in train_recs})
    qa["actual_waste_domain_rate"] = round(len(pass_recs) / len(records), 4)
    qa["real_waste_pass_rate"] = round(len(pass_recs) / len(records), 4)
    qa["records_with_label_corrections"] = sum(1 for r in records if r["label_corrections_count"] > 0)
    qa["total_label_corrections"] = sum(r["label_corrections_count"] for r in records)
    qa["label_correction_rate"] = round(qa["records_with_label_corrections"] / len(records), 4)

    for cid_int, cname in enumerate(CLASS_NAMES):
        qa["per_class_real_waste_pass_images"][cname] = sum(
            1 for r in pass_recs if cid_int in r["classes_present_ids"]
        )
        qa["per_class_real_waste_pass_train_images"][cname] = sum(
            1 for r in train_recs if cid_int in r["classes_present_ids"]
        )
        qa["per_class_real_waste_pass_val_images"][cname] = sum(
            1 for r in val_recs if cid_int in r["classes_present_ids"]
        )
        qa["per_class_real_waste_pass_boxes"][cname] = sum(
            sum(1 for b in r["boxes"] if b["class_id"] == cid_int) for r in pass_recs
        )
        qa["per_class_real_waste_pass_train_boxes"][cname] = sum(
            sum(1 for b in r["boxes"] if b["class_id"] == cid_int) for r in train_recs
        )
        qa["per_class_real_waste_pass_val_boxes"][cname] = sum(
            sum(1 for b in r["boxes"] if b["class_id"] == cid_int) for r in val_recs
        )

    dup_pairs = 0
    same_ovl = 0
    cross_ovl = 0
    for r in records:
        bxs = r["boxes"]
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                rel = classify_box_pair_overlap(
                    _struct_box_to_norm_tuple(bxs[i]),
                    _struct_box_to_norm_tuple(bxs[j]),
                )
                if rel["category"] == "DUPLICATE_ANNOTATION":
                    dup_pairs += 1
                elif rel["category"] == "VALID_MULTI_OBJECT_SAME_CLASS":
                    same_ovl += 1
                elif rel["category"] == "VALID_CO_OCCURRING_CROSS_CLASS":
                    cross_ovl += 1
    assert dup_pairs == 0, f"Found {dup_pairs} duplicate annotation pairs!"
    qa["overlap_audit"] = {
        "duplicate_annotation_pairs_remaining": dup_pairs,
        "preserved_valid_same_class_overlap_pairs": same_ovl,
        "preserved_valid_cross_class_overlap_pairs": cross_ovl,
    }
    qa["task8i_inaturalist_signoff_applied"] = True
    qa["task8i_final_hotfix_applied"] = True

    qa_p.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render dedicated contact sheets for the 90 pre-8I REAL_WASTE_PASS iNaturalist images (3 pages of 30)
    cs_90_paths: list[str] = []
    for p_idx in range(3):
        chunk = inat_90_pm_audit_records[p_idx * 30 : (p_idx + 1) * 30]
        cs_p = W8I_CS_DIR / f"inat90_pm_audit_p{p_idx + 1}.jpg"
        render_pm_audit_contact_sheet(
            chunk,
            cs_p,
            f"Task 8I PM Sign-Off — iNaturalist 90 Pre-8I REAL_WASTE_PASS Audit (Page {p_idx + 1}/3)",
        )
        cs_90_paths.append(cs_p.relative_to(REPO_ROOT).as_posix())

    inat_pass_recs = [r for r in inat_verified_records if r["qa_status"] == "REAL_WASTE_PASS"]
    inat_pass_tr_n = sum(1 for r in inat_pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8H")
    inat_pass_va_n = sum(1 for r in inat_pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8H")
    inat_pass_bio_n = sum(1 for r in inat_pass_recs if r["primary_class_id"] == 1)
    inat_pass_gla_n = sum(1 for r in inat_pass_recs if r["primary_class_id"] == 4)
    inat_pass_sho_n = sum(1 for r in inat_pass_recs if r["primary_class_id"] == 8)
    cs_pass_p = W8I_CS_DIR / "inat35_verified_real_waste_pass.jpg"
    render_pm_audit_contact_sheet(
        inat_pass_recs,
        cs_pass_p,
        f"Task 8I Verified iNaturalist REAL_WASTE_PASS ({len(inat_pass_recs)} Images: {inat_pass_bio_n} Bio, {inat_pass_gla_n} Glass, {inat_pass_sho_n} Shoes)",
    )
    cs_90_paths.append(cs_pass_p.relative_to(REPO_ROOT).as_posix())

    # Write Task 1/2/3 Verified iNaturalist 107 Manifest (JSON + CSV) and Per-Image License Markdown Table
    inat_lic_counts = Counter(r["inat_license_category"] for r in inat_verified_records)
    inat_photo_lic_counts = Counter(r["inat_photo_license_code"] for r in inat_verified_records)
    inat_obs_lic_counts = Counter(r["inat_observation_license_code"] for r in inat_verified_records)
    inat_post_status_counts = Counter(r["qa_status"] for r in inat_verified_records)
    inat_90_post_status_counts = Counter(r["qa_status"] for r in inat_90_pm_audit_records)
    inat_decay_cat_counts = Counter(r["biological_decay_category"] for r in inat_verified_records)

    inat_manifest_obj = {
        "manifest_version": "TASK8I_INATURALIST_107_VERIFIED_MANIFEST_V2_HOTFIX",
        "source_id": "SRC_8H_04_INATURALIST_BIOLOGICAL_DECAY_FIELD",
        "total_inaturalist_candidates_audited": len(inat_verified_records),
        "pre_8i_real_waste_pass_audited": len(inat_90_pm_audit_records),
        "pre_hotfix_8i_manifest_summary": {
            "real_waste_pass_total": 33,
            "real_waste_pass_train": 27,
            "real_waste_pass_val": 6,
            "keep_auxiliary": 29,
            "reject": 45,
            "note": "Exact numbers from pre-hotfix task8i_inaturalist_107_verified_manifest.json before Task 2/3 hotfix adjustments.",
        },
        "post_8i_status_counts_all_107": dict(inat_post_status_counts),
        "post_8i_status_counts_from_pre_8i_90_pass": dict(inat_90_post_status_counts),
        "post_8i_verified_real_waste_pass_train": inat_pass_tr_n,
        "post_8i_verified_real_waste_pass_val": inat_pass_va_n,
        "per_class_verified_real_waste_pass_images": {
            "biological": inat_pass_bio_n,
            "glass": inat_pass_gla_n,
            "shoes": inat_pass_sho_n,
        },
        "per_class_verified_real_waste_pass_train_images": {
            "biological": sum(
                1 for r in inat_pass_recs if r["primary_class_id"] == 1 and r["target_split"] == "TRAIN_REAL_WASTE_W8H"
            ),
            "glass": sum(
                1 for r in inat_pass_recs if r["primary_class_id"] == 4 and r["target_split"] == "TRAIN_REAL_WASTE_W8H"
            ),
            "shoes": sum(
                1 for r in inat_pass_recs if r["primary_class_id"] == 8 and r["target_split"] == "TRAIN_REAL_WASTE_W8H"
            ),
        },
        "license_policy_and_summary_all_107": {
            "policy_distinction": (
                "Photo license (photo_license_code) governs copyright/reuse of the image file itself; "
                "Observation license (observation_license_code) governs reuse of the observation occurrence/GPS record. "
                "Images with open photo licenses (CC0/CC-BY/CC-BY-SA) and restricted observation licenses (CC-BY-NC/ARR) "
                "are NOT automatically banned for license; they are evaluated on visual waste context without redistributing observation metadata."
            ),
            "photo_license_usable_count": sum(1 for r in inat_verified_records if r["inat_photo_license_usable"]),
            "both_photo_and_observation_open_count": sum(
                1 for r in inat_verified_records if r["inat_license_commercial_ok"]
            ),
            "photo_open_but_observation_restricted_count": len(obs_restricted_11_records),
            "obs_restricted_11_outcomes": {
                "moved_to_keep_auxiliary": sorted(list(INAT_OBS_RESTRICTED_TO_AUX_IDS)),
                "kept_at_reject_due_to_visual_or_taxonomy_defect": sorted(
                    list(INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS | {"task8h_0149", "task8h_0155"})
                ),
                "promoted_to_real_waste_pass": [],
            },
            "photo_license_code_counts": dict(inat_photo_lic_counts),
            "observation_license_code_counts": dict(inat_obs_lic_counts),
            "combined_license_category_counts": dict(inat_lic_counts),
        },
        "decay_and_context_category_counts": dict(inat_decay_cat_counts),
        "bbox_tightened_records_count_in_8i": sum(1 for r in changed_records if r["bbox_changed_in_8i"]),
        "pm_inspection_contact_sheets": cs_90_paths,
        "records": [
            {
                "candidate_id": r["candidate_id"],
                "inat_observation_id": r["inat_observation_id"],
                "inat_photo_id": r["inat_photo_id"],
                "inat_photo_url": r["inat_photo_url"],
                "inat_observation_url": r["inat_observation_url"],
                "inat_author": r["inat_author"],
                "inat_user_login": r["inat_user_login"],
                "inat_photo_attribution": r["inat_photo_attribution"],
                "inat_photo_license_code": r["inat_photo_license_code"],
                "inat_observation_license_code": r["inat_observation_license_code"],
                "inat_photo_license_usable": r["inat_photo_license_usable"],
                "inat_observation_license_open": r["inat_observation_license_open"],
                "inat_photo_obs_license_mismatch": r["inat_photo_obs_license_mismatch"],
                "inat_license_category": r["inat_license_category"],
                "inat_license_commercial_ok": r["inat_license_commercial_ok"],
                "primary_class_id": r["primary_class_id"],
                "primary_class_name": r["primary_class_name"],
                "pre_8i_qa_status": r["pre_8i_qa_status"],
                "post_8i_qa_status": r["post_8i_qa_status"],
                "status_changed_in_8i": r["status_changed_in_8i"],
                "bbox_changed_in_8i": r["bbox_changed_in_8i"],
                "biological_decay_category": r["biological_decay_category"],
                "domain_tag": r["domain_tag"],
                "target_split": r["target_split"],
                "specimen_group": r["specimen_group"],
                "session_group": r["session_group"],
                "bbox_count": r["bbox_count"],
                "boxes": r["boxes"],
                "image_path": r["image_path"],
                "label_path": r["label_path"],
                "overlay_path": r["overlay_path"],
                "sha256": r["sha256"],
                "qa_reason": r["qa_reason"],
            }
            for r in inat_verified_records
        ],
    }

    inat_json_p = W8I_DIR / "task8i_inaturalist_107_verified_manifest.json"
    inat_csv_p = W8I_DIR / "task8i_inaturalist_107_verified_manifest.csv"
    inat_md_p = W8I_DIR / "INATURALIST_107_PER_IMAGE_LICENSE_TABLE.md"
    inat_json_p.write_text(json.dumps(inat_manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    with inat_csv_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_id",
                "inat_observation_id",
                "inat_photo_id",
                "inat_author",
                "inat_photo_license_code",
                "inat_observation_license_code",
                "inat_photo_license_usable",
                "inat_observation_license_open",
                "inat_license_category",
                "primary_class_name",
                "pre_8i_qa_status",
                "post_8i_qa_status",
                "biological_decay_category",
                "bbox_count",
                "bbox_changed_in_8i",
                "target_split",
                "inat_photo_url",
                "inat_observation_url",
            ]
        )
        for r in inat_manifest_obj["records"]:
            writer.writerow(
                [
                    r["candidate_id"],
                    r["inat_observation_id"],
                    r["inat_photo_id"],
                    r["inat_author"],
                    r["inat_photo_license_code"],
                    r["inat_observation_license_code"],
                    r["inat_photo_license_usable"],
                    r["inat_observation_license_open"],
                    r["inat_license_category"],
                    r["primary_class_name"],
                    r["pre_8i_qa_status"],
                    r["post_8i_qa_status"],
                    r["biological_decay_category"],
                    r["bbox_count"],
                    r["bbox_changed_in_8i"],
                    r["target_split"],
                    r["inat_photo_url"],
                    r["inat_observation_url"],
                ]
            )

    md_lines = [
        "# TASK 8I (FINAL HOTFIX) — BÁO CÁO & BẢNG XÁC MINH GIẤY PHÉP / NGỮ CẢNH 107 ẢNH INATURALIST",
        "",
        "## 1. Tóm tắt chuẩn xác trước & sau Hotfix (`Task 2 & Task 3`)",
        "",
        "- **Số liệu thực trong `task8i_inaturalist_107_verified_manifest.json` trước Hotfix (đã chỉnh khớp 100%):**",
        "  - **`33` `REAL_WASTE_PASS`** (**`27` train** + **`6` validation**), `29` `KEEP_AUXILIARY`, `45` `REJECT`.",
        "- **Số liệu thực sau khi áp dụng Task 2 & Task 3 Hotfix:**",
        "  - **Task 2 (`task8h_0085`, `task8h_0092`, `task8h_0137`):** Cả 3 ảnh chụp quả cam/chanh mốc cầm trên tay có `description` rỗng và không có bằng chứng vứt bỏ trong thùng/bối cảnh rác đã được chuyển từ `REAL_WASTE_PASS` (`train`) sang **`KEEP_AUXILIARY` (`HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL`)**.",
        "  - **Task 3 (Chính sách tách riêng `Photo License` vs `Observation License` trên 11 ca observation hạn chế):**",
        "    - Cả 11 ca đều có `photo_license_code` thuộc nhóm mở (`3` `cc0`, `7` `cc-by`, `1` `cc-by-sa` $\\rightarrow$ `inat_photo_license_usable = True`), trong khi `observation_license_code` là `cc-by-nc` (`9` ca) hoặc `all-rights-reserved` (`2` ca).",
        "    - Không mặc định cấm sử dụng ảnh chỉ vì `observation_license_code`: đánh giá từng ca theo chất lượng hình ảnh & ngữ cảnh rác thực tế $\\rightarrow$ **`5` ca có vật thể rõ nhưng chưa đủ bằng chứng vứt bỏ (`task8h_0072, 0102, 0114, 0141, 0145`) chuyển sang `KEEP_AUXILIARY`**; **`6` ca lỗi hiển vi/sai lớp (`task8h_0087, 0118, 0149, 0151, 0152, 0155`) giữ ở `REJECT`**; **`0` ca nâng lên `REAL_WASTE_PASS`**.",
        f"  - **Tổng kết 107 ảnh iNaturalist sau Hotfix:** **`{inat_post_status_counts['REAL_WASTE_PASS']}` `REAL_WASTE_PASS`** (**`{inat_pass_tr_n}` train** + **`{inat_pass_va_n}` validation** — gồm `{inat_pass_bio_n}` `biological`, `{inat_pass_gla_n}` `glass`, `{inat_pass_sho_n}` `shoes`), **`{inat_post_status_counts['KEEP_AUXILIARY']}` `KEEP_AUXILIARY`**, **`{inat_post_status_counts['REJECT']}` `REJECT`**.",
        "",
        "## 2. Phân biệt `Photo License` và `Observation License` trên 107 ảnh",
        "",
        "| Photo License | Observation License | Số lượng | Quyền sử dụng bức ảnh (`Photo`) | Quyền tái phân phối dữ liệu `Observation` | Xử lý trong dự án |",
        "|---|---|---:|---|---|---|",
        f"| `cc0 / cc-by / cc-by-sa` | `cc0 / cc-by / cc-by-sa` | `96` | Cho phép (`True`) | Cho phép (`True`) | Đủ điều kiện `REAL_WASTE_PASS` nếu đạt QA ngữ cảnh & bbox (`30` PASS, `32` AUX, `34` REJECT) |",
        f"| `cc0 / cc-by / cc-by-sa` | `cc-by-nc` (`9`) / `ARR` (`2`) | `11` | **Cho phép sử dụng ảnh kèm attribution (`True`)** | Hạn chế (`False` — không xuất metadata GPS/occurrence) | Không auto-reject vì obs license; xét theo ngữ cảnh (`5` `KEEP_AUXILIARY`, `6` `REJECT`, `0` `REAL_WASTE_PASS`) |",
        "",
        "## 3. Bảng tra cứu chi tiết 107 ảnh iNaturalist",
        "",
        "| ID | Obs ID | Photo ID | Tác giả | Photo Lic | Obs Lic | Photo Usable | Lớp | Trước 8I | Sau 8I Hotfix | Phân loại ngữ cảnh | BBox Fix | URL Ảnh |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for r in inat_manifest_obj["records"]:
        md_lines.append(
            f"| `{r['candidate_id']}` | [{r['inat_observation_id']}]({r['inat_observation_url']}) | `{r['inat_photo_id']}` | {r['inat_author']} | `{r['inat_photo_license_code']}` | `{r['inat_observation_license_code']}` | `{r['inat_photo_license_usable']}` | `{r['primary_class_name']}` | `{r['pre_8i_qa_status']}` | **`{r['post_8i_qa_status']}`** | `{r['biological_decay_category']}` | `{'YES' if r['bbox_changed_in_8i'] else 'NO'}` | [Link]({r['inat_photo_url']}) |"
        )
    inat_md_p.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Deliverable #5: Write UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.{json,md}
    uncertain_domain_watchlist = [
        {
            "candidate_id": "task8h_0085",
            "action_taken_in_hotfix": "DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY",
            "reason": "Moldy lemon held in hand indoors; empty observation description; no trash bin or disposal context.",
            "post_hotfix_status": "KEEP_AUXILIARY",
        },
        {
            "candidate_id": "task8h_0092",
            "action_taken_in_hotfix": "DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY",
            "reason": "Moldy mandarin held in hand indoors; None observation description; no visual disposal proof.",
            "post_hotfix_status": "KEEP_AUXILIARY",
        },
        {
            "candidate_id": "task8h_0137",
            "action_taken_in_hotfix": "DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY",
            "reason": "Moldy mandarin held in hand; None observation description; no visual disposal proof.",
            "post_hotfix_status": "KEEP_AUXILIARY",
        },
        {
            "candidate_id": "task8h_0094",
            "action_taken_in_hotfix": "FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE",
            "reason": "Close-up of moldy tangerine held in fingers; unambiguous spoiled food specimen, flagged if PM prefers zero handheld shots in train.",
            "post_hotfix_status": "REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)",
        },
        {
            "candidate_id": "task8h_0101",
            "action_taken_in_hotfix": "FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE",
            "reason": "Cut half of spent moldy lemon held in hand over kitchen box ('This lemon lived a good life...'); flagged for PM visibility.",
            "post_hotfix_status": "REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)",
        },
        {
            "candidate_id": "task8h_0142",
            "action_taken_in_hotfix": "FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE",
            "reason": "Moldy citrus held over outdoor foliage/bin; flagged for PM visibility.",
            "post_hotfix_status": "REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)",
        },
        {
            "candidate_id": "task8h_0150",
            "action_taken_in_hotfix": "FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE",
            "reason": "Clear bottle base buried in forest litter; observer note mentions 'plastic drink bottle' though rim appears rigid.",
            "post_hotfix_status": "REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)",
        },
    ]

    uncertain_obj = {
        "report_version": "TASK8I_UNCERTAIN_LICENSE_AND_DOMAIN_CASES_V1",
        "part_a_11_photo_vs_observation_license_mismatch_cases": [
            {
                "candidate_id": r["candidate_id"],
                "inat_observation_id": r["inat_observation_id"],
                "inat_photo_id": r["inat_photo_id"],
                "primary_class_name": r["primary_class_name"],
                "photo_license_code": r["inat_photo_license_code"],
                "observation_license_code": r["inat_observation_license_code"],
                "photo_license_usable": r["inat_photo_license_usable"],
                "pre_8i_status": r["pre_8i_qa_status"],
                "pre_hotfix_8i_status": "REJECT",
                "post_hotfix_8i_status": r["post_8i_qa_status"],
                "visual_and_domain_assessment": r["biological_decay_category"],
                "decision_rationale": r["qa_reason"],
            }
            for r in obs_restricted_11_records
        ],
        "part_b_handheld_and_borderline_domain_cases": uncertain_domain_watchlist,
    }
    unc_json_p = REPORTS_DIR / "UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.json"
    unc_md_p = REPORTS_DIR / "UNCERTAIN_LICENSE_AND_DOMAIN_CASES_TASK8I.md"
    unc_json_p.write_text(json.dumps(uncertain_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    unc_md_lines = [
        "# TASK 8I FINAL HOTFIX — DANH SÁCH CA LICENSE & DOMAIN CHƯA CHẮC CHẮN / CẦN LƯU Ý",
        "",
        "## Phần A — 11 Ca Chênh lệch Giấy phép Ảnh (`Photo License`) và Giấy phép Quan sát (`Observation License`)",
        "",
        "Theo tài liệu chính thức của iNaturalist, `photo.license_code` quy định quyền bản quyền đối với **bức ảnh**, còn `observation.license_code` quy định quyền đối với **bản ghi dữ liệu quan sát** (tọa độ GPS, ngày giờ, loài). Trong cả 11 ca dưới đây, bức ảnh có giấy phép mở (`cc0`, `cc-by`, `cc-by-sa`), nên không bị cấm sử dụng về mặt bản quyền ảnh nếu không tái phân phối metadata quan sát. Tuy nhiên, khi đánh giá độc lập về **ngữ cảnh rác thật (`ACTUAL_WASTE`)**, không có ca nào đủ điều kiện nâng lên `REAL_WASTE_PASS` (`5` ca chuyển sang `KEEP_AUXILIARY`, `6` ca giữ ở `REJECT` do lỗi hiển vi/sai lớp):",
        "",
        "| ID | Obs ID | Photo ID | Lớp | Photo License | Obs License | Đánh giá ngữ cảnh & hình ảnh | Trạng thái Trước Hotfix | Trạng thái Sau Hotfix |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for item in uncertain_obj["part_a_11_photo_vs_observation_license_mismatch_cases"]:
        unc_md_lines.append(
            f"| `{item['candidate_id']}` | `{item['inat_observation_id']}` | `{item['inat_photo_id']}` | `{item['primary_class_name']}` | `{item['photo_license_code']}` | `{item['observation_license_code']}` | `{item['visual_and_domain_assessment']}` | `{item['pre_hotfix_8i_status']}` | **`{item['post_hotfix_8i_status']}`** |"
        )
    unc_md_lines.extend(
        [
            "",
            "## Phần B — Các Ca Trái Cây Cầm Tay (`Task 2`) & Ca Biên Ngữ Cảnh (`Domain Watchlist`)",
            "",
            "| ID | Hành động trong Hotfix | Lý do / Bằng chứng | Trạng thái Sau Hotfix |",
            "|---|---|---|---|",
        ]
    )
    for w_item in uncertain_domain_watchlist:
        unc_md_lines.append(
            f"| `{w_item['candidate_id']}` | `{w_item['action_taken_in_hotfix']}` | {w_item['reason']} | **`{w_item['post_hotfix_status']}`** |"
        )
    unc_md_p.write_text("\n".join(unc_md_lines) + "\n", encoding="utf-8")

    # Generate Deliverable #6: Concrete simulated test example of PRODUCT_STUDIO held at NEEDS_REVIEW
    reg_example_p = generate_product_studio_regression_example()

    # Update Task 8H report JSON so its quality gates and counts stay synchronized with the post-8I manifest
    rep8h_p = REPORTS_DIR / "TASK8H_SCARCE_CLASS_SCALING_REPORT.json"
    rep8h = json.loads(rep8h_p.read_text(encoding="utf-8"))
    rep8h["direct_deliverable_metrics"]["real_waste_pass_total_images"] = qa["real_waste_pass_total_images"]
    rep8h["direct_deliverable_metrics"]["real_waste_pass_train_images_added"] = qa["real_waste_pass_train_images"]
    rep8h["direct_deliverable_metrics"]["real_waste_pass_val_images_added"] = qa["real_waste_pass_val_images"]
    rep8h["direct_deliverable_metrics"]["real_waste_pass_total_boxes"] = qa["real_waste_pass_total_boxes"]
    rep8h["direct_deliverable_metrics"]["independent_train_images_added_per_class"] = qa[
        "per_class_real_waste_pass_train_images"
    ]
    rep8h["direct_deliverable_metrics"]["independent_val_images_added_per_class"] = qa[
        "per_class_real_waste_pass_val_images"
    ]
    rep8h["direct_deliverable_metrics"]["selection_rate_by_source"] = qa["source_stats"]
    rep8h["direct_deliverable_metrics"]["multi_object_and_multiclass_coverage"] = {
        "real_waste_pass_multiobject_images": qa["real_waste_pass_multiobject_images"],
        "real_waste_pass_multiobject_rate": round(
            qa["real_waste_pass_multiobject_images"] / max(1, qa["real_waste_pass_total_images"]), 4
        ),
        "real_waste_pass_multiclass_images": qa["real_waste_pass_multiclass_images"],
        "real_waste_pass_multiclass_rate": round(
            qa["real_waste_pass_multiclass_images"] / max(1, qa["real_waste_pass_total_images"]), 4
        ),
    }
    rep8h["task8i_inaturalist_signoff_applied"] = True
    rep8h["task8i_final_hotfix_applied"] = True
    rep8h_p.write_text(json.dumps(rep8h, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update Canonical 5000 Dataset Ledger to V6 (reflecting post-hotfix counts)
    ledger_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
    ledger = json.loads(ledger_p.read_text(encoding="utf-8"))

    w8h_train_n = len(train_recs)  # 127 (was 130 pre-hotfix)
    w8h_val_n = len(val_recs)      # 28 (unchanged)
    pending_unique_train = 723 + w8h_train_n  # 850 (was 853 pre-hotfix)
    cum_unique_train = 958 + pending_unique_train  # 1808 (was 1811 pre-hotfix)
    cum_unique_val = 364 + w8h_val_n  # 392 (unchanged)

    pre_8h_train_imgs = {
        0: 52, 1: 90, 2: 185, 3: 34, 4: 90, 5: 301, 6: 201, 7: 985, 8: 55, 9: 361
    }
    pre_8h_train_boxes = {
        0: 120, 1: 114, 2: 253, 3: 36, 4: 143, 5: 466, 6: 250, 7: 1602, 8: 84, 9: 655
    }
    pre_8h_specimens = {
        0: 48, 1: 83, 2: 185, 3: 34, 4: 90, 5: 301, 6: 200, 7: 984, 8: 55, 9: 360
    }
    pre_8h_sessions = {
        0: 46, 1: 64, 2: 58, 3: 13, 4: 31, 5: 89, 6: 71, 7: 159, 8: 40, 9: 96
    }
    pre_8h_val_imgs = {
        0: 23, 1: 20, 2: 44, 3: 1, 4: 27, 5: 56, 6: 47, 7: 177, 8: 11, 9: 72
    }

    for row in ledger["per_class_ledger"]:
        cid_int = int(row["class_id"])
        cname = row["class_name"]
        add_tr_img = qa["per_class_real_waste_pass_train_images"][cname]
        add_tr_box = qa["per_class_real_waste_pass_train_boxes"][cname]
        add_va_img = qa["per_class_real_waste_pass_val_images"][cname]
        add_va_box = qa["per_class_real_waste_pass_val_boxes"][cname]

        cls_tr_recs = [r for r in train_recs if cid_int in r["classes_present_ids"]]
        add_spec = len({r["specimen_group"] for r in cls_tr_recs})
        add_sess = len({r["session_group"] for r in cls_tr_recs})

        taco_tr_img = int(row["tier1a_clean_baseline_taco_verified_rw_train_images"])
        taco_tr_box = int(row["tier1a_clean_baseline_taco_verified_rw_train_boxes"])

        new_cum_tr_img = pre_8h_train_imgs[cid_int] + add_tr_img
        new_cum_tr_box = pre_8h_train_boxes[cid_int] + add_tr_box
        new_pend_tr_img = new_cum_tr_img - taco_tr_img
        new_pend_tr_box = new_cum_tr_box - taco_tr_box

        row["wave8h_scarce_scaling_rw_train"] = add_tr_img
        row["wave8h_scarce_scaling_rw_train_boxes"] = add_tr_box
        row["wave8h_scarce_scaling_rw_val"] = add_va_img
        row["wave8h_scarce_scaling_rw_val_boxes"] = add_va_box
        row["tier3_pending_approval_rw_train_images"] = new_pend_tr_img
        row["tier3_pending_approval_rw_train_boxes"] = new_pend_tr_box
        row["tier2_verified_real_waste_cumulative_train_images"] = new_cum_tr_img
        row["tier2_verified_real_waste_cumulative_train_boxes"] = new_cum_tr_box
        row["tier2_verified_real_waste_unique_specimens"] = pre_8h_specimens[cid_int] + add_spec
        row["tier2_verified_real_waste_unique_sessions"] = pre_8h_sessions[cid_int] + add_sess
        row["verified_real_waste_val_total_images"] = pre_8h_val_imgs[cid_int] + add_va_img
        row["gap_to_m1_500_from_verified_rw"] = max(0, 500 - new_cum_tr_img)
        row["gap_to_m2_1500_from_verified_rw"] = max(0, 1500 - new_cum_tr_img)
        row["gap_to_m3_3000_from_verified_rw"] = max(0, 3000 - new_cum_tr_img)
        row["gap_to_m4_5000_from_verified_rw"] = max(0, 5000 - new_cum_tr_img)

    total_per_class_appearances = sum(
        int(r["tier2_verified_real_waste_cumulative_train_images"]) for r in ledger["per_class_ledger"]
    )
    pending_per_class_appearances = sum(
        int(r["tier3_pending_approval_rw_train_images"]) for r in ledger["per_class_ledger"]
    )
    cum_unique_specimens = 1675 + qa["real_waste_pass_train_unique_specimens"]
    cum_unique_sessions = 457 + qa["real_waste_pass_train_unique_sessions"]

    ledger["ledger_version"] = "TASK8I_CANONICAL_5000_LEDGER_V6_SOURCE_SIGNOFF_AND_FIELD_PILOT"
    ledger["three_tier_summary"]["2_VERIFIED_REAL_WASTE"] = {
        "approved_baseline_taco_train_images": 958,
        "pending_staged_verified_real_waste_train_images": pending_unique_train,
        "cumulative_verified_real_waste_unique_train_images": cum_unique_train,
        "cumulative_verified_real_waste_per_class_appearances": total_per_class_appearances,
        "cumulative_verified_real_waste_unique_specimens": cum_unique_specimens,
        "cumulative_verified_real_waste_unique_sessions": cum_unique_sessions,
        "cumulative_verified_real_waste_unique_val_images": cum_unique_val,
        "train_val_id_overlap_count": 0,
    }
    ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["pending_verified_real_waste_unique_train_images"] = (
        pending_unique_train
    )
    ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["pending_verified_real_waste_per_class_appearances"] = (
        pending_per_class_appearances
    )
    ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["cohorts"]["wave8h_scarce_scaling_real_waste_train"] = (
        w8h_train_n
    )
    ledger["three_tier_summary"]["3_PENDING_APPROVAL"]["cohorts"]["wave8i_field_pilot_captured_train"] = 0

    ledger["historical_audit_trail"]["task8h_scarce_class_scaling"] = {
        "raw_downloaded_candidates": 419,
        "pre_8i_real_waste_pass_images": 215,
        "pre_8i_real_waste_pass_train": 176,
        "pre_8i_real_waste_pass_val": 39,
        "real_waste_pass_images": qa["real_waste_pass_total_images"],
        "real_waste_pass_train": w8h_train_n,
        "real_waste_pass_val": w8h_val_n,
        "real_waste_pass_boxes": qa["real_waste_pass_total_boxes"],
        "real_waste_pass_multiobject_images": qa["real_waste_pass_multiobject_images"],
        "real_waste_pass_multiclass_images": qa["real_waste_pass_multiclass_images"],
        "keep_auxiliary_images": qa["qa_status_counts"].get("KEEP_AUXILIARY", 0),
        "needs_review_images": qa["qa_status_counts"].get("NEEDS_REVIEW", 0),
        "reject_images": qa["qa_status_counts"].get("REJECT", 0),
        "duplicate_annotation_pairs_remaining": 0,
        "preserved_valid_same_class_overlaps": same_ovl,
        "preserved_valid_cross_class_overlaps": cross_ovl,
    }
    ledger["historical_audit_trail"]["task8i_inaturalist_signoff_and_field_pilot"] = {
        "inaturalist_candidates_audited": len(inat_verified_records),
        "pre_8i_inaturalist_real_waste_pass": len(inat_90_pm_audit_records),
        "pre_hotfix_8i_inaturalist_real_waste_pass": 33,
        "pre_hotfix_8i_inaturalist_real_waste_pass_train": 27,
        "pre_hotfix_8i_inaturalist_real_waste_pass_val": 6,
        "post_8i_inaturalist_real_waste_pass": len(inat_pass_recs),
        "post_8i_inaturalist_real_waste_pass_train": inat_pass_tr_n,
        "post_8i_inaturalist_real_waste_pass_val": inat_pass_va_n,
        "demoted_to_keep_auxiliary_natural_orchard_decay": len(INAT_DEMOTE_TO_AUX_ORCHARD_IDS),
        "demoted_to_keep_auxiliary_handheld_unverified_disposal": len(INAT_DEMOTE_TO_AUX_HANDHELD_IDS),
        "obs_license_restricted_routed_to_keep_auxiliary": len(INAT_OBS_RESTRICTED_TO_AUX_IDS),
        "obs_license_restricted_kept_at_reject_visual_or_taxonomy": len(INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS) + 2,
        "rejected_due_to_macro_mold_colony_or_duplicate": len(INAT_REJECT_MACRO_OR_DUP_IDS),
        "bbox_tightened_records_in_8i": len([r for r in changed_records if r["bbox_changed_in_8i"]]),
        "field_pilot_captured_images_to_date": 0,
        "signoff_reason": (
            f"Task 8I + Final Hotfix verification of {len(inat_verified_records)} iNaturalist images: "
            f"pre-hotfix manifest had 33 REAL_WASTE_PASS (27 train, 6 val); Task 2 hotfix demoted 3 handheld "
            f"moldy citrus images without disposal proof (0085, 0092, 0137) to KEEP_AUXILIARY, yielding "
            f"{len(inat_pass_recs)} REAL_WASTE_PASS ({inat_pass_tr_n} train, {inat_pass_va_n} val). "
            f"Task 3 hotfix separated photo license from observation license across all 11 restricted-obs cases "
            f"(5 moved to KEEP_AUXILIARY, 6 kept at REJECT for visual/taxonomy reasons, 0 promoted to REAL_WASTE_PASS)."
        ),
    }
    ledger["four_dimensional_counts"] = {
        "unique_images_verified_real_waste_train": cum_unique_train,
        "per_class_image_appearances_verified_real_waste_train": total_per_class_appearances,
        "unique_specimens_verified_real_waste_train": cum_unique_specimens,
        "unique_sessions_verified_real_waste_train": cum_unique_sessions,
        "unique_images_clean_baseline_train": 1277,
        "per_class_image_appearances_clean_baseline_train": 1746,
    }

    ledger_p.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    ledger_csv_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.csv"
    with ledger_csv_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger["per_class_ledger"][0].keys()))
        writer.writeheader()
        writer.writerows(ledger["per_class_ledger"])

    ledger_md_p = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.md"
    l_md = [
        "# CANONICAL 5,000-IMAGE/CLASS DATASET LEDGER (V6 — POST-TASK 8I FINAL HOTFIX)",
        "",
        "## 1. Bốn chiều độc lập dữ liệu (`Task 3 — Data Independence`)",
        "",
        f"- **Unique Verified Real-Waste Train Images (`unique_images`):** **`{cum_unique_train}`** (`958` TACO Baseline + `{pending_unique_train}` Pending Staged Real-Waste Train)",
        f"- **Unique Physical Specimens (`unique_specimens`):** **`{cum_unique_specimens}`**",
        f"- **Unique Capture Sessions / Locations (`unique_sessions`):** **`{cum_unique_sessions}`**",
        f"- **Per-Class Image Appearances (`per_class_image_appearances`):** **`{total_per_class_appearances}`**",
        f"- **Unique Verified Real-Waste Validation Images:** **`{cum_unique_val}`** (`train_val_id_overlap_count = 0`)",
        "",
        "## 2. Bảng Ledger 10 Lớp Sau Task 8I Final Hotfix",
        "",
        "| Class ID | Lớp | TACO RW Train | Staged Pre-8H RW Train | Wave 8H Verified RW Train (Post-8I Hotfix) | Tổng Verified RW Train (`Tier 2`) | Unique Specimens | Unique Sessions | Verified RW Val | Khoảng cách tới M1 (`500`) | Khoảng cách tới M4 (`5.000`) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ledger["per_class_ledger"]:
        cid_int = int(r["class_id"])
        pre_staged = pre_8h_train_imgs[cid_int] - int(r["tier1a_clean_baseline_taco_verified_rw_train_images"])
        l_md.append(
            f"| {cid_int} | `{r['class_name']}` | {r['tier1a_clean_baseline_taco_verified_rw_train_images']} | "
            f"{pre_staged} | **{r['wave8h_scarce_scaling_rw_train']}** | "
            f"**{r['tier2_verified_real_waste_cumulative_train_images']}** | "
            f"{r['tier2_verified_real_waste_unique_specimens']} | "
            f"{r['tier2_verified_real_waste_unique_sessions']} | "
            f"{r['verified_real_waste_val_total_images']} | "
            f"{r['gap_to_m1_500_from_verified_rw']} | {r['gap_to_m4_5000_from_verified_rw']} |"
        )
    ledger_md_p.write_text("\n".join(l_md) + "\n", encoding="utf-8")

    # Deliverable #4: Write Manifest & Ledger Delta Report (JSON & MD)
    delta_obj = {
        "delta_version": "TASK8I_FINAL_HOTFIX_MANIFEST_AND_LEDGER_DELTA_V1",
        "inaturalist_107_manifest_delta": {
            "pre_hotfix_real_waste_pass_total": 33,
            "pre_hotfix_real_waste_pass_train": 27,
            "pre_hotfix_real_waste_pass_val": 6,
            "pre_hotfix_keep_auxiliary": 29,
            "pre_hotfix_reject": 45,
            "post_hotfix_real_waste_pass_total": len(inat_pass_recs),
            "post_hotfix_real_waste_pass_train": inat_pass_tr_n,
            "post_hotfix_real_waste_pass_val": inat_pass_va_n,
            "post_hotfix_keep_auxiliary": inat_post_status_counts["KEEP_AUXILIARY"],
            "post_hotfix_reject": inat_post_status_counts["REJECT"],
            "delta_real_waste_pass_total": len(inat_pass_recs) - 33,
            "delta_real_waste_pass_train": inat_pass_tr_n - 27,
            "delta_real_waste_pass_val": inat_pass_va_n - 6,
            "delta_keep_auxiliary": inat_post_status_counts["KEEP_AUXILIARY"] - 29,
            "delta_reject": inat_post_status_counts["REJECT"] - 45,
            "records_changed_in_hotfix": {
                "task2_handheld_demoted_from_pass_to_keep_auxiliary": sorted(list(INAT_DEMOTE_TO_AUX_HANDHELD_IDS)),
                "task3_obs_license_restricted_moved_from_reject_to_keep_auxiliary": sorted(
                    list(INAT_OBS_RESTRICTED_TO_AUX_IDS)
                ),
            },
        },
        "wave8h_qa_manifest_delta": {
            "pre_hotfix_real_waste_pass_total": 158,
            "pre_hotfix_real_waste_pass_train": 130,
            "pre_hotfix_real_waste_pass_val": 28,
            "post_hotfix_real_waste_pass_total": qa["real_waste_pass_total_images"],
            "post_hotfix_real_waste_pass_train": w8h_train_n,
            "post_hotfix_real_waste_pass_val": w8h_val_n,
        },
        "ledger_v6_delta": {
            "pre_hotfix_unique_verified_rw_train_images": 1811,
            "post_hotfix_unique_verified_rw_train_images": cum_unique_train,
            "pre_hotfix_biological_verified_rw_train_images": 122,
            "post_hotfix_biological_verified_rw_train_images": 119,
            "other_9_classes_verified_rw_train_delta": 0,
        },
    }
    delta_json_p = REPORTS_DIR / "TASK8I_HOTFIX_MANIFEST_AND_LEDGER_DELTA.json"
    delta_md_p = REPORTS_DIR / "TASK8I_HOTFIX_MANIFEST_AND_LEDGER_DELTA.md"
    delta_json_p.write_text(json.dumps(delta_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    delta_md_lines = [
        "# TASK 8I FINAL HOTFIX — BẢNG ĐỐI CHIẾU THAY ĐỔI MANIFEST & LEDGER DELTA",
        "",
        "## 1. Thay đổi trên `task8i_inaturalist_107_verified_manifest.json`",
        "",
        "| Chỉ số | Trước Hotfix (Số thực trong Manifest) | Sau Hotfix (Task 2 + Task 3) | Chênh lệch (`Delta`) | Giải thích |",
        "|---|---:|---:|---:|---|",
        f"| `REAL_WASTE_PASS` (Tổng) | `33` | **`{len(inat_pass_recs)}`** | `-3` | Hạ `task8h_0085, 0092, 0137` (trái cây mốc cầm tay không có bằng chứng vứt bỏ) sang `KEEP_AUXILIARY` |",
        f"| `REAL_WASTE_PASS` (`Train`) | `27` | **`{inat_pass_tr_n}`** | `-3` | Cả 3 ảnh `0085, 0092, 0137` thuộc tập `TRAIN_REAL_WASTE_W8H` |",
        f"| `REAL_WASTE_PASS` (`Val`) | `6` | **`{inat_pass_va_n}`** | `0` | Không thay đổi |",
        f"| `KEEP_AUXILIARY` | `29` | **`{inat_post_status_counts['KEEP_AUXILIARY']}`** | `+8` | `+3` ca cầm tay (`0085, 0092, 0137`) + `+5` ca giấy phép ảnh mở nhưng ngữ cảnh phụ trợ (`0072, 0102, 0114, 0141, 0145`) |",
        f"| `REJECT` | `45` | **`{inat_post_status_counts['REJECT']}`** | `-5` | `5` ca (`0072, 0102, 0114, 0141, 0145`) có `photo_license` hợp lệ chuyển từ `REJECT` sang `KEEP_AUXILIARY` |",
        "",
        "## 2. Thay đổi trên `task8h_qa_manifest.json` & `CANONICAL_5000_DATASET_LEDGER.json` (V6)",
        "",
        "| Chỉ số | Trước Hotfix | Sau Hotfix | Chênh lệch (`Delta`) |",
        "|---|---:|---:|---:|",
        f"| Wave 8H `REAL_WASTE_PASS` (Tổng) | `158` | **`{qa['real_waste_pass_total_images']}`** | `-3` |",
        f"| Wave 8H `REAL_WASTE_PASS` (`Train`) | `130` | **`{w8h_train_n}`** | `-3` |",
        f"| Wave 8H `REAL_WASTE_PASS` (`Val`) | `28` | **`{w8h_val_n}`** | `0` |",
        f"| Lớp `1: biological` Verified RW Train (Toàn chương trình) | `122` | **`119`** | `-3` |",
        f"| Tổng Unique Verified RW Train Images (`Tier 2`) | `1811` | **`{cum_unique_train}`** | `-3` |",
        f"| 9 lớp còn lại (`0, 2, 3, 4, 5, 6, 7, 8, 9`) | Giữ nguyên | **Giữ nguyên** | `0` |",
    ]
    delta_md_p.write_text("\n".join(delta_md_lines) + "\n", encoding="utf-8")

    # Update REAL_WASTE_10CLASS_VALIDATION_PLAN.json (V5)
    val_plan_p = REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json"
    val_plan = json.loads(val_plan_p.read_text(encoding="utf-8"))
    val_plan["plan_version"] = "REAL_WASTE_10CLASS_VALIDATION_PLAN_V5_TASK8I_SOURCE_SIGNOFF"
    val_plan["cohorts_included"]["wave8h_val_images"] = w8h_val_n
    val_plan["cohorts_included"]["total_unique_verified_real_waste_val_images"] = cum_unique_val
    val_plan_p.write_text(json.dumps(val_plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update Task 8I Master Report (JSON & MD)
    rep8i_obj = {
        "report_version": "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT_REPORT_V2_FINAL_HOTFIX",
        "approved_for_train": False,
        "constraints_compliance": {
            "e4_e5_trained": False,
            "deployed": False,
            "e1_e3_checkpoints_modified": False,
            "frozen_test_299_modified": False,
            "candidate_test_274_modified": False,
            "external_holdout_27_modified": False,
            "historical_audit_reopened": False,
            "simulated_or_fabricated_field_images": 0,
        },
        "task1_p0_field_importer_hotfix": {
            "script_path": "scripts/field_capture_import_and_qa.py",
            "tool_version": "TASK8I_FIELD_CAPTURE_IMPORT_QA_V2_P0_HOTFIX",
            "default_qa_status_for_new_imports": "NEEDS_REVIEW",
            "technical_validation_separated_from_context_verification": True,
            "product_studio_and_in_use_blocked_from_real_waste_pass": True,
            "unreviewed_actual_waste_held_at_needs_review": True,
            "requires_human_reviewer_and_evidence_for_real_waste_context_verified": True,
            "auto_cc_by_4_0_license_assignment_removed": True,
            "approved_for_train_always_false": True,
            "simulated_product_studio_example_json": reg_example_p.relative_to(REPO_ROOT).as_posix(),
        },
        "task1_inaturalist_signoff_summary": {
            "total_inaturalist_audited": len(inat_verified_records),
            "pre_8i_real_waste_pass": len(inat_90_pm_audit_records),
            "pre_hotfix_8i_manifest_real_waste_pass": 33,
            "pre_hotfix_8i_manifest_real_waste_pass_train": 27,
            "pre_hotfix_8i_manifest_real_waste_pass_val": 6,
            "post_8i_verified_real_waste_pass": len(inat_pass_recs),
            "post_8i_verified_real_waste_pass_train": inat_pass_tr_n,
            "post_8i_verified_real_waste_pass_val": inat_pass_va_n,
            "demoted_to_keep_auxiliary_orchard_decay": len(INAT_DEMOTE_TO_AUX_ORCHARD_IDS),
            "demoted_to_keep_auxiliary_handheld_unverified_disposal": len(INAT_DEMOTE_TO_AUX_HANDHELD_IDS),
            "obs_license_restricted_routed_to_keep_auxiliary": len(INAT_OBS_RESTRICTED_TO_AUX_IDS),
            "obs_license_restricted_kept_at_reject_visual_or_taxonomy": len(INAT_OBS_RESTRICTED_VISUAL_REJECT_IDS) + 2,
            "rejected_due_to_macro_mold_or_duplicate": len(INAT_REJECT_MACRO_OR_DUP_IDS),
            "pre_existing_keep_auxiliary": 2,
            "pre_existing_reject": 13,
            "bbox_tightened_records_count": len([r for r in changed_records if r["bbox_changed_in_8i"]]),
            "changed_records_total": len(changed_records),
            "pm_inspection_contact_sheets": cs_90_paths,
        },
        "task3_four_dimensional_independence": ledger["four_dimensional_counts"],
    }
    rep8i_json_p = REPORTS_DIR / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT_REPORT.json"
    rep8i_md_p = REPORTS_DIR / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT_REPORT.md"
    rep8i_json_p.write_text(json.dumps(rep8i_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    rep8i_md_lines = [
        "# BÁO CÁO NGHIỆM THU TASK 8I (FINAL HOTFIX) — FIELD CAPTURE QA SAFETY & INATURALIST SIGN-OFF",
        "",
        "## 1. Task 1 (Hotfix P0) — Tách biệt Kiểm tra Kỹ thuật & Xác nhận Bối cảnh Rác thật trong `scripts/field_capture_import_and_qa.py`",
        "- Đã sửa triệt để lỗi P0 trong [`scripts/field_capture_import_and_qa.py`](./scripts/field_capture_import_and_qa.py):",
        "  1. **Tách riêng `FILE_AND_BBOX_FORMAT_VALID` và `REAL_WASTE_CONTEXT_VERIFIED`:** Việc ảnh hợp lệ và tọa độ YOLO `.txt` đúng chuẩn chỉ thiết lập `FILE_AND_BBOX_FORMAT_VALID = True`, tuyệt đối không tự suy ra `REAL_WASTE_CONTEXT_VERIFIED = True`.",
        "  2. **Mặc định `qa_status = NEEDS_REVIEW`:** Mọi ảnh mới nhập chưa có bản ghi `visual_qa_review` của người duyệt đều giữ ở `NEEDS_REVIEW` (`target_split = MANUAL_REVIEW_QUEUE`).",
        "  3. **Chặn nâng `PRODUCT_STUDIO` và `IN_USE` lên `REAL_WASTE_PASS`:** Bất kể nhãn YOLO hợp lệ hay người nhập khai báo sai, `context_type in (PRODUCT_STUDIO, IN_USE)` luôn có `REAL_WASTE_CONTEXT_VERIFIED = False` và không bao giờ đạt `REAL_WASTE_PASS`.",
        "  4. **Yêu cầu bằng chứng người duyệt thật (`reviewer_id`, `review_timestamp`, `context_evidence >= 10 ký tự`, `bbox_tightness_verified = True`):** Chỉ khi có đủ các trường này trên mẫu `ACTUAL_WASTE` đã xác nhận quyền sử dụng thì mới đạt `REAL_WASTE_CONTEXT_VERIFIED = True`.",
        "  5. **Giữ `APPROVED_FOR_TRAIN = False` trên 100% bản ghi:** Kể cả khi `qa_status == REAL_WASTE_PASS`.",
        "  6. **Không tự gán `CC-BY-4.0`:** Nếu thiếu giấy phép rõ ràng hoặc `rights_confirmed = False`, gán `UNCONFIRMED_USAGE_RIGHTS` và giữ ở `NEEDS_REVIEW`.",
        f"- **Ví dụ kiểm thử giả lập `PRODUCT_STUDIO` bị giữ ở `NEEDS_REVIEW`:** [`{reg_example_p.relative_to(REPO_ROOT).as_posix()}`](../../{reg_example_p.relative_to(REPO_ROOT).as_posix()})",
        "",
        "## 2. Task 2 (Hotfix) — Đồng bộ Báo cáo iNaturalist & Xử lý 3 Ca Trái cây Mốc Cầm tay (`task8h_0085`, `0092`, `0137`)",
        "- **Số liệu thực trong manifest trước Hotfix:** `33` `REAL_WASTE_PASS` (`27` train + `6` validation), `29` `KEEP_AUXILIARY`, `45` `REJECT`.",
        "- **Kiểm tra lại `task8h_0085`, `task8h_0092`, `task8h_0137`:** Cả 3 ảnh đều chụp trái cây có mốc được cầm trên tay trong nhà, `description` trên iNaturalist trống (`''` / `None`), không có thùng rác hay bằng chứng đã thải bỏ $\\rightarrow$ chuyển cả 3 ảnh từ `REAL_WASTE_PASS` (`train`) sang **`KEEP_AUXILIARY` (`HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL`)**.",
        f"- **Số liệu thực sau Hotfix:** **`{len(inat_pass_recs)}` `REAL_WASTE_PASS`** (**`{inat_pass_tr_n}` train** + **`{inat_pass_va_n}` validation**), **`{inat_post_status_counts['KEEP_AUXILIARY']}` `KEEP_AUXILIARY`**, **`{inat_post_status_counts['REJECT']}` `REJECT`**.",
        "",
        "## 3. Task 3 (Hotfix) — Chính sách Giấy phép Kép (`Photo License` vs `Observation License`) trên 11 Ca Hạn chế",
        "- Đã tách riêng trường `inat_photo_license_usable` (quyền bản quyền bức ảnh) và `inat_observation_license_open` (quyền đối với bản ghi dữ liệu quan sát).",
        "- Kiểm tra lại toàn bộ 11 ca có `observation_license_code` là `cc-by-nc` (`9` ca) hoặc `all-rights-reserved` (`2` ca) nhưng `photo_license_code` mở (`cc0`/`cc-by`/`cc-by-sa`):",
        "  - **`5` ca (`task8h_0072, 0102, 0114, 0141, 0145`)** có vật thể rõ và giấy phép ảnh hợp lệ nhưng ngữ cảnh chưa đủ bằng chứng vứt bỏ $\\rightarrow$ chuyển từ `REJECT` sang **`KEEP_AUXILIARY`**.",
        "  - **`6` ca (`task8h_0087, 0118, 0149, 0151, 0152, 0155`)** giữ ở **`REJECT`** do lỗi hình ảnh hiển vi (`0087, 0118`), chụp côn trùng/động vật (`0149, 0155`), hoặc chai nhựa bị lọt vào truy vấn thủy tinh (`0151, 0152`).",
        "  - **`0` ca** được nâng lên `REAL_WASTE_PASS`.",
    ]
    rep8i_md_p.write_text("\n".join(rep8i_md_lines) + "\n", encoding="utf-8")

    # Package compact hotfix ZIP (`TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip`) AND update `TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip`
    hotfix_zip_p = REPO_ROOT / "TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip"
    full_zip_p = REPO_ROOT / "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip"

    hotfix_files: list[Path] = [
        REPO_ROOT / "scripts" / "field_capture_import_and_qa.py",
        REPO_ROOT / "scripts" / "execute_task8i_signoff_and_field_pilot.py",
        REPO_ROOT / "tests" / "detection" / "test_task8_manifest_report_consistency.py",
        reg_example_p,
        inat_json_p,
        inat_csv_p,
        inat_md_p,
        unc_json_p,
        unc_md_p,
        delta_json_p,
        delta_md_p,
        rep8i_json_p,
        rep8i_md_p,
        ledger_p,
        ledger_csv_p,
        ledger_md_p,
        qa_p,
        rep8h_p,
    ]
    # Add pytest log if present
    pytest_log_p = W8I_DIR / "pytest_task8i_hotfix_run.log"
    if pytest_log_p.exists():
        hotfix_files.append(pytest_log_p)

    with zipfile.ZipFile(hotfix_zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in hotfix_files:
            if fp.exists():
                zf.write(fp, arcname=fp.relative_to(REPO_ROOT).as_posix())

    full_files = list(hotfix_files) + [
        pre_snap_p,
        inat_cache_p,
        val_plan_p,
        REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.md",
        REPORTS_DIR / "FIRST_PARTY_FIELD_CAPTURE_GUIDE.json",
        REPORTS_DIR / "FIRST_PARTY_FIELD_CAPTURE_GUIDE.md",
        REPORTS_DIR / "DUAL_CHANNEL_5000_SCALING_PLAN.json",
        REPORTS_DIR / "DUAL_CHANNEL_5000_SCALING_PLAN.md",
        W8I_TEMPLATES_DIR / "field_capture_import_template.csv",
        W8I_TEMPLATES_DIR / "field_capture_import_template.json",
    ]
    for p in sorted(W8I_CS_DIR.glob("*.jpg")):
        full_files.append(p)
    for p in sorted(W8I_CHANGED_LBL_PASS_DIR.glob("*.txt")):
        full_files.append(p)
    for p in sorted(W8I_CHANGED_LBL_AUX_DIR.glob("*.txt")):
        full_files.append(p)
    for p in sorted(W8I_CHANGED_OVL_DIR.glob("*.jpg")):
        full_files.append(p)

    seen_arc: set[str] = set()
    with zipfile.ZipFile(full_zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in full_files:
            if fp.exists():
                arc = fp.relative_to(REPO_ROOT).as_posix()
                if arc not in seen_arc:
                    seen_arc.add(arc)
                    zf.write(fp, arcname=arc)

    for src_z, dst_name in [
        (hotfix_zip_p, "TASK8I_FINAL_HOTFIX_FIELD_QA_SAFETY.zip"),
        (full_zip_p, "TASK8I_SOURCE_SIGNOFF_AND_FIELD_PILOT.zip"),
    ]:
        try:
            shutil.copy2(src_z, (Path.home() / "Downloads") / dst_name)
        except Exception:
            pass

    print(
        json.dumps(
            {
                "status": "TASK8I_FINAL_HOTFIX_COMPLETED",
                "w8h_post_hotfix_real_waste_pass_total": qa["real_waste_pass_total_images"],
                "w8h_post_hotfix_real_waste_pass_train": w8h_train_n,
                "w8h_post_hotfix_real_waste_pass_val": w8h_val_n,
                "inat_post_hotfix_status_counts": dict(inat_post_status_counts),
                "inat_post_hotfix_train": inat_pass_tr_n,
                "inat_post_hotfix_val": inat_pass_va_n,
                "hotfix_zip_size_mb": round(hotfix_zip_p.stat().st_size / (1024 * 1024), 2),
                "full_zip_size_mb": round(full_zip_p.stat().st_size / (1024 * 1024), 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
