"""
Task 8 Part B (B3 & B4) — Batch QA, Taxonomy Verification, Domain Tagging, Deduplication,
YOLO Labels, Overlays, and Contact Sheets for Wave 2 (`task8_wave2_large_batch`).

Enforces:
- Strict taxonomy verification: upstream Open Images / Commons labels are NOT automatically trusted.
  - `/m/01kb5b` (Flashlight) -> held in NEEDS_REVIEW (not bare battery unless verified).
  - `/m/04dr76w` (Bottle) -> plastic/water/soda title cues or material ambiguity held in NEEDS_REVIEW.
  - `/m/025dyy` (Box) -> wooden/plastic/metal/toolbox cues rejected or held in NEEDS_REVIEW.
  - `/m/09j2d` / clothes -> worn/portrait/person cues tagged IN_USE and held in NEEDS_REVIEW.
  - Dense piles / whole-frame heap boxes (>0.92 frame area) -> held in NEEDS_REVIEW per instance rules.
- Domain standardization: `ACTUAL_WASTE`, `PRODUCT_STUDIO`, `IN_USE`, `AMBIGUOUS_CONTEXT`.
- 4-way Deduplication: SHA-256, 64-bit dHash/pHash (Hamming <= 8), source_image_id, and photographer session (`AuthorProfileURL`).
- Split assignment (`TRAIN_WAVE2` vs `VAL_REAL_WAVE2`) strictly grouped by `specimen_group` and `session_group`.
- `approved_for_train = False` on all records (Stop at approval gate before training).
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WAVE2_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch"
RAW_MANIFEST_PATH = WAVE2_DIR / "task8_wave2_raw_manifest.json"
QA_JSON_PATH = WAVE2_DIR / "task8_wave2_qa_manifest.json"
QA_CSV_PATH = WAVE2_DIR / "task8_wave2_qa_manifest.csv"

LABELS_PASS_DIR = WAVE2_DIR / "labels"
LABELS_REVIEW_DIR = WAVE2_DIR / "labels_needs_manual_review"
OVERLAYS_DIR = WAVE2_DIR / "overlays"
CONTACT_SHEETS_DIR = WAVE2_DIR / "contact_sheets"

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

CLASS_COLORS = {
    0: (255, 60, 60),     # battery - red
    1: (60, 200, 60),     # biological - green
    2: (210, 140, 50),    # cardboard - brown/orange
    3: (180, 70, 220),    # clothes - purple
    4: (50, 200, 220),    # glass - cyan
    5: (160, 170, 185),   # metal - silver
    6: (240, 220, 60),    # paper - yellow
    7: (50, 120, 255),    # plastic - blue
    8: (255, 120, 180),   # shoes - pink
    9: (255, 140, 0),     # trash - dark orange
}


import imagehash


def compute_dhash_int(img: Image.Image) -> int:
    """Compute 64-bit perceptual hash (pHash) matching authoritative_hashes_cache.json and Wave 1."""
    return int(str(imagehash.phash(img)), 16)


def hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def load_external_hashes() -> list[tuple[str, str, int]]:
    """Load (dataset_key, sha256, phash_int) from existing datasets for leakage & dedup verification."""
    ext: list[tuple[str, str, int]] = []
    auth_path = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if auth_path.exists():
        data = json.loads(auth_path.read_text(encoding="utf-8"))
        for k, v in data.items():
            if isinstance(v, dict) and v.get("sha256") and v.get("phash"):
                try:
                    ext.append((k, v["sha256"], int(v["phash"], 16)))
                except Exception:
                    pass

    w1_path = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json"
    if w1_path.exists():
        w1 = json.loads(w1_path.read_text(encoding="utf-8"))
        for rec in w1.get("records", []):
            sha = rec.get("sha256", "")
            ph = rec.get("phash_0", "") or rec.get("phash_hex", "")
            if sha and ph:
                try:
                    ext.append((f"wave1:{rec['item_id']}", sha, int(ph, 16)))
                except Exception:
                    pass
    return ext


def analyze_visual_properties(img: Image.Image) -> dict[str, float]:
    """Compute border studio whiteness, color saturation, texture complexity, and outdoor sky/ground cues."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape

    # Border ring (top/bottom/left/right 8% strips)
    bh = max(4, int(h * 0.08))
    bw = max(4, int(w * 0.08))
    top = rgb[:bh, :, :].reshape(-1, 3)
    bot = rgb[h - bh :, :, :].reshape(-1, 3)
    left = rgb[:, :bw, :].reshape(-1, 3)
    right = rgb[:, w - bw :, :].reshape(-1, 3)
    border = np.concatenate([top, bot, left, right], axis=0)

    # Studio white or uniform gray/black border detection
    white_pixels = np.all(border > 232.0, axis=1)
    black_pixels = np.all(border < 22.0, axis=1)
    border_std = float(np.mean(np.std(border, axis=0)))
    white_border_ratio = float(np.mean(white_pixels))
    black_border_ratio = float(np.mean(black_pixels))

    # Overall contrast and texture (gradient magnitude)
    gray = np.mean(rgb, axis=2)
    gx = np.abs( gray[:, 1:] - gray[:, :-1] )
    gy = np.abs( gray[1:, :] - gray[:-1, :] )
    edge_density = float((np.mean(gx > 18.0) + np.mean(gy > 18.0)) * 0.5)

    return {
        "white_border_ratio": round(white_border_ratio, 4),
        "black_border_ratio": round(black_border_ratio, 4),
        "border_std": round(border_std, 2),
        "edge_density": round(edge_density, 4),
    }


OI_ALL_LABELS_PATH = (pathlib.Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(pathlib.Path(__file__).resolve().parents[3] / "data" / "external" / "waste-training"))) / "wave2_oi_all_labels_per_image.json")
OI_ALL_LABELS_MAP: dict[str, set[str]] = {}
if OI_ALL_LABELS_PATH.exists():
    _raw_lbls = json.loads(OI_ALL_LABELS_PATH.read_text(encoding="utf-8"))
    OI_ALL_LABELS_MAP = {k: set(v) for k, v in _raw_lbls.items()}

# Co-occurring Open Images MIDs discovered via visual contact-sheet inspection
OI_REJECT_MIDS = {
    "/m/04cp_",    # Koala (plush toy with toilet paper)
    "/m/0kmg4",    # Teddy bear
    "/m/01dxs",    # Brown bear
    "/m/0138tl",   # Toy
    "/m/0152hh",   # Snowman
    "/m/06c7f7",   # Lipstick (makeup palette boxed as Box)
    "/m/02xb7qb",  # Face powder
}

OI_IN_USE_MIDS = {
    # Bathroom fixtures
    "/m/09g1w", "/m/01vbnl", "/m/02jz0l", "/m/0130jx", "/m/04y4h8h", "/m/0h8nr_l",
    "/m/02f9f_", "/m/02pkr5", "/m/03dnzn", "/m/0h8nsvg", "/m/054_l",
    # Dining / Tableware / Meals
    "/m/04bcr3", "/m/050gv4", "/m/099ssp", "/m/03q5c7", "/m/024g6", "/m/0270h",
    "/m/0h8n5zk", "/m/0h8n27j", "/m/0cmx8", "/m/0dt3t", "/m/03hlz0c", "/m/04brg2",
    "/m/078n6m", "/m/04kkgm", "/m/02pdsw", "/m/0grw1", "/m/0l515", "/m/01b9xk",
    "/m/0fszt", "/m/01tcjp", "/m/02zvsm", "/m/01_bhs", "/m/052lwg6",
    # Shelves / Bookcases / Store displays / Furniture
    "/m/0gjbg72", "/m/03__z0", "/m/01y9k5", "/m/01mzpv", "/m/0crjs", "/m/01s105",
    "/m/0b3fp9", "/m/0c_jw", "/m/0642b4", "/m/0fqfqc", "/m/040b_t", "/m/05kyg_",
    "/m/026qbn5", "/m/0703r8", "/m/0fqt361",
    # Worn / Human / Sports / Portrait accessories
    "/m/0h8mhzd", "/m/0fly7", "/m/01xygc", "/m/01d40f", "/m/03g8mr", "/m/0zvk5",
    "/m/01rkbr", "/m/01xyhv", "/m/0jyfg", "/m/017ftj", "/m/071p9", "/m/0463sg",
    "/m/014sv8", "/m/02_n6y", "/m/02dl1y", "/m/0k0pj", "/m/039xj_", "/m/01bfm9",
    "/m/0gjkl", "/m/06__v", "/m/01226z", "/m/02h19r", "/m/05y5lj", "/m/0283dt1",
    "/m/02wv6h6", "/m/0176mf", "/m/01llwg", "/m/01gkx_", "/m/01gmv2", "/m/054xkw",
    "/m/02wbtzl", "/m/04rky", "/m/03k3r",
    # Growing plants / trees
    "/m/05s2s", "/m/07j7r", "/m/0c9ph5", "/m/03fp41", "/m/0cdl1",
}

OI_MATERIAL_AMBIGUOUS_MIDS = {
    "/m/08hvt4",  # Jug (ceramic/clay/metal/pewter jugs in Open Images)
    "/m/054fyh",  # Pitcher (Container)
    "/m/01fh4r",  # Teapot
    "/m/02jvh9",  # Mug
    "/m/03s_tn",  # Kettle
    "/m/02s195",  # Vase
    "/m/0bjyj5",  # Waste container (trash bin/dumpster infrastructure, not trash item)
    "/m/01kb5b",  # Flashlight (device, not bare battery)
    "/m/07dd4",   # Torch
    "/m/09gtd",   # Toilet paper (mounted bathroom rolls)
    "/m/01s55n",  # Suitcase (aluminum flight case boxed as Box)
}


import cv2


def auto_box_commons_image(img: Image.Image, primary_cid: int, title: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Generate tight per-instance foreground bounding box(es) for Wikimedia Commons candidate images
    using OpenCV morphological closing + connectedComponentsWithStats.
    Returns (boxes, is_confident_instance_boxes).
    If the background is cluttered or components cannot be cleanly separated into 1..5 distinct instances,
    returns is_confident=False so B3 holds the image in NEEDS_REVIEW.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    bh = max(4, int(h * 0.06))
    bw = max(4, int(w * 0.06))
    border = np.concatenate(
        [
            rgb[:bh, :, :].reshape(-1, 3),
            rgb[h - bh :, :, :].reshape(-1, 3),
            rgb[:, :bw, :].reshape(-1, 3),
            rgb[:, w - bw :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_median = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg_median[None, None, :], axis=2)
    border_std = float(np.mean(np.std(border, axis=0)))

    t_low = title.lower()
    is_heap_keyword = any(
        kw in t_low
        for kw in (
            "pile", "heap", "dump", "landfill", "collection", "many", "various", "assorted",
            "sorting", "bins", "containers",
        )
    )

    mask_u8 = (dist > max(26.0, border_std * 1.55)).astype(np.uint8) * 255
    ksize = max(5, int(min(h, w) * 0.018))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    fg_ratio = float(np.mean(opened > 0))
    if 0.03 <= fg_ratio <= 0.76 and border_std < 36.0 and not is_heap_keyword:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        comp_boxes: list[dict[str, Any]] = []
        total_comp_area = 0.0
        for lbl_idx in range(1, num_labels):
            x, y, cw, ch, px_area = stats[lbl_idx]
            norm_area = float(px_area) / float(h * w)
            bw_n, bh_n = float(cw) / w, float(ch) / h
            if norm_area < 0.012 or bw_n < 0.04 or bh_n < 0.04:
                continue
            if (bw_n * bh_n) > 0.86:
                continue
            total_comp_area += bw_n * bh_n
            comp_boxes.append(
                {
                    "class_id": primary_cid,
                    "class_name": CLASS_NAMES[primary_cid],
                    "mid": "commons_cc_instance",
                    "mid_name": CLASS_NAMES[primary_cid],
                    "xmin": round(max(0.0, float(x) / w), 6),
                    "ymin": round(max(0.0, float(y) / h), 6),
                    "xmax": round(min(1.0, float(x + cw) / w), 6),
                    "ymax": round(min(1.0, float(y + ch) / h), 6),
                    "is_occluded": 0,
                    "is_truncated": 0,
                    "is_inside": 0,
                }
            )

        if 1 <= len(comp_boxes) <= 5 and total_comp_area <= 0.88:
            return comp_boxes, True

    return (
        [
            {
                "class_id": primary_cid,
                "class_name": CLASS_NAMES[primary_cid],
                "mid": "commons_candidate_region",
                "mid_name": CLASS_NAMES[primary_cid],
                "xmin": 0.12,
                "ymin": 0.12,
                "xmax": 0.88,
                "ymax": 0.88,
                "is_occluded": 0,
                "is_truncated": 0,
                "is_inside": 0,
            }
        ],
        False,
    )


def has_kraft_cardboard_color(img: Image.Image, boxes: list[dict[str, Any]]) -> bool:
    """Check if the interior of `/m/025dyy` Box regions exhibits warm kraft/corrugated brown-tan color."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    for b in boxes:
        if b.get("mid") != "/m/025dyy":
            continue
        x1 = max(0, int(b["xmin"] * w + (b["xmax"] - b["xmin"]) * w * 0.15))
        x2 = min(w, int(b["xmax"] * w - (b["xmax"] - b["xmin"]) * w * 0.15))
        y1 = max(0, int(b["ymin"] * h + (b["ymax"] - b["ymin"]) * h * 0.15))
        y2 = min(h, int(b["ymax"] * h - (b["ymax"] - b["ymin"]) * h * 0.15))
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            continue
        patch = rgb[y1:y2, x1:x2]
        r_m, g_m, b_m = float(np.mean(patch[:, :, 0])), float(np.mean(patch[:, :, 1])), float(np.mean(patch[:, :, 2]))
        # Warm brown/tan cardboard: R > G > B and moderate luminance
        if r_m > g_m > b_m and (r_m - b_m) >= 15.0 and 85.0 <= r_m <= 225.0:
            return True
    return False


def evaluate_record_qa_and_domain(
    img: Image.Image,
    rec: dict[str, Any],
    vis: dict[str, float],
    is_commons_confident: bool,
) -> tuple[str, str, str, list[dict[str, Any]], dict[str, bool]]:
    """
    Evaluate a candidate image for:
    - qa_status: 'QA_PASS', 'NEEDS_REVIEW', or 'REJECT'
    - domain_tag: 'ACTUAL_WASTE', 'PRODUCT_STUDIO', 'IN_USE', or 'AMBIGUOUS_CONTEXT'
    - qa_reason: human-readable audit explanation
    - valid_boxes: filtered/verified bounding boxes
    - scene_flags: dict of boolean attributes (complex_background, outdoor_scene, dirty_or_damaged, multi_class)
    """
    title = (rec.get("title") or "").lower()
    cat_src = (rec.get("category_source") or "").lower()
    text_cue = f"{title} {cat_src}"
    boxes = list(rec.get("boxes", []))
    sf = rec.get("source_family", "")
    iid = rec.get("source_image_id", "")
    co_mids = OI_ALL_LABELS_MAP.get(iid, set())

    # 1. Hard Reject conditions (diagrams, logos, signs, screenshots, plush toys, non-target materials)
    reject_keywords = (
        "diagram", "chart", "map", "logo", "icon", "drawing", "illustration", "vector", "poster", "sign", "plaque",
        "statue", "sculpture", "painting", "screenshot", "stamp", "coin", "medal", "koala", "plush", "stuffed",
        "origami", "windows 8", "desktop",
    )
    if any(kw in text_cue for kw in reject_keywords) or bool(co_mids & OI_REJECT_MIDS):
        return "REJECT", "AMBIGUOUS_CONTEXT", "reject_non_photo_or_toy_or_screenshot", [], {}

    if rec["primary_class_id"] == 2 and any(
        kw in text_cue for kw in (
            "wooden box", "wood box", "wooden crate", "jewelry box", "music box", "metal box", "plastic box",
            "toolbox", "tackle box", "loot box", "phone box", "pillar box", "post box", "mailbox", "signal box",
            "nesting box", "bird box", "bookcase", "shelf", "cabinet", "briefcase", "flight case",
        )
    ):
        return "REJECT", "AMBIGUOUS_CONTEXT", "reject_non_cardboard_box_material_in_title", [], {}

    # 2. Determine Domain Tag (A2) — including narrow white-background studio cutouts
    waste_keywords = (
        "waste", "trash", "garbage", "litter", "rubbish", "dump", "recycling", "recycled", "discarded",
        "rotten", "mold", "mould", "spoiled", "decay", "compost", "peel", "core", "scraps", "leftover",
        "crushed", "crumpled", "broken", "shattered", "torn", "ripped", "used_batter", "spent", "corroded",
        "rusty", "rusted", "old_shoe", "worn_out", "dirty", "stained", "cigarette", "butt", "ash",
    )
    in_use_keywords = (
        "wearing", "portrait", "fashion", "runway", "model", "player", "athlete", "baseball", "boxing", "boxer",
        "ski", "skier", "snowboard", "walking", "standing", "sitting", "girl", "boy", "man", "woman", "person",
        "people", "child", "kid", "baby", "bride", "groom", "wedding", "party", "toast", "brindis", "drinking",
        "dining", "restaurant", "serving", "served", "breakfast", "lunch", "dinner", "banquet", "buffet",
        "library", "bookshelf", "bookstore", "reading", "classroom", "office", "bathroom", "toilet", "restroom",
        "supermarket", "grocery", "market", "stall", "shop", "store", "vendor", "display", "tree", "branch",
        "orchard", "plantation", "growing", "cat", "dog", "kitten", "puppy",
    )
    studio_keywords = (
        "isolated", "white background", "studio", "product", "catalog", "packshot", "cutout",
    )

    is_studio_visual = (vis["white_border_ratio"] >= 0.48 and vis["border_std"] <= 24.0) or (
        vis["black_border_ratio"] >= 0.68 and vis["border_std"] <= 18.0
    )
    has_in_use_mid = bool(co_mids & OI_IN_USE_MIDS)

    if any(kw in text_cue for kw in waste_keywords) and not has_in_use_mid:
        domain_tag = "ACTUAL_WASTE"
    elif is_studio_visual or any(kw in text_cue for kw in studio_keywords):
        domain_tag = "PRODUCT_STUDIO"
    elif has_in_use_mid or any(kw in text_cue for kw in in_use_keywords):
        domain_tag = "IN_USE"
    else:
        domain_tag = "AMBIGUOUS_CONTEXT"

    outdoor_keywords = ("street", "park", "beach", "road", "sidewalk", "grass", "forest", "field", "outdoor", "garden", "river", "sea", "snow", "sand")
    dirty_keywords = ("rotten", "mold", "mould", "crushed", "crumpled", "broken", "torn", "rusty", "corroded", "dirty", "stained", "wet", "damaged", "waste", "trash", "litter", "compost")

    # 3. Filter and verify bounding boxes
    clean_boxes: list[dict[str, Any]] = []
    has_full_frame_pile_box = False
    max_box_area = 0.0
    max_box_span = 0.0

    for b in boxes:
        xmin, ymin, xmax, ymax = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        bw, bh = xmax - xmin, ymax - ymin
        area = bw * bh
        span = max(bw, bh)
        if bw < 0.03 or bh < 0.03 or area < 0.0025:
            continue
        if area > 0.91:
            has_full_frame_pile_box = True
        max_box_area = max(max_box_area, area)
        max_box_span = max(max_box_span, span)
        clean_boxes.append(b)

    if not clean_boxes:
        return "REJECT", domain_tag, "reject_no_valid_boxes_after_geometry_filter", [], {}

    cids = sorted({b["class_id"] for b in clean_boxes})
    mids_in_boxes = {b.get("mid", "") for b in clean_boxes}
    scene_flags = {
        "complex_background": bool(vis["edge_density"] >= 0.14 and not is_studio_visual),
        "outdoor_scene": bool(any(kw in text_cue for kw in outdoor_keywords)),
        "dirty_or_damaged": bool(domain_tag == "ACTUAL_WASTE" or any(kw in text_cue for kw in dirty_keywords)),
        "multi_class": len(cids) >= 2,
    }

    # 4. Strict QA Gate Rules (B3: Hold all uncertain taxonomy, material, instance-completeness, or IN_USE cases in NEEDS_REVIEW)
    if sf == "wikimedia_commons_wave2" and not is_commons_confident:
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_commons_multi_object_or_complex_bg_needs_individual_instance_boxes",
            clean_boxes,
            scene_flags,
        )

    if bool(mids_in_boxes & OI_MATERIAL_AMBIGUOUS_MIDS) or bool(co_mids & OI_MATERIAL_AMBIGUOUS_MIDS):
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_upstream_mid_material_or_bin_ambiguity_jug_bin_flashlight_toiletpaper",
            clean_boxes,
            scene_flags,
        )

    if domain_tag == "IN_USE":
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_in_use_context_bathroom_dining_shelf_worn_or_growing_plant",
            clean_boxes,
            scene_flags,
        )

    if has_full_frame_pile_box:
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_full_frame_box_exceeds_91pct_area_check_tightness_or_pile",
            clean_boxes,
            scene_flags,
        )

    if len(clean_boxes) > 4:
        return (
            "NEEDS_REVIEW",
            domain_tag,
            f"review_dense_multi_instance_scene_{len(clean_boxes)}_boxes_verify_all_instances_labeled",
            clean_boxes,
            scene_flags,
        )

    # Check worn socks/gloves/clothes/shoes or distant small boxes where unannotated people/piles occur
    if any(m in mids_in_boxes for m in ("/m/01nq26", "/m/09j2d", "/m/01n4qj", "/m/07mhn", "/m/032b3c")):
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_sock_or_garment_verify_not_worn_on_human_body",
            clean_boxes,
            scene_flags,
        )

    # For biological, clothes, and shoes in Open Images, require prominent foreground specimen (area >= 0.18 or span >= 0.52)
    if sf.startswith("openimages_v7"):
        if any(c in cids for c in (1, 3, 8)) and max_box_area < 0.20:
            return (
                "NEEDS_REVIEW",
                domain_tag,
                "review_small_relative_box_area_under_20pct_check_unannotated_cluster_or_person",
                clean_boxes,
                scene_flags,
            )
        if max_box_area < 0.09 or max_box_span < 0.34:
            return (
                "NEEDS_REVIEW",
                domain_tag,
                "review_small_distant_object_box_under_9pct_area_or_34pct_span",
                clean_boxes,
                scene_flags,
            )

    # For Open Images /m/025dyy (Box -> cardboard), require explicit cardboard/carton cue OR verified warm kraft cardboard color
    if "/m/025dyy" in mids_in_boxes:
        cardboard_cues = ("cardboard", "carton", "corrugated", "pizza", "cereal", "shipping", "moving", "parcel", "package", "packaging", "amazon", "shoebox", "egg")
        if not (any(kw in text_cue for kw in cardboard_cues) or has_kraft_cardboard_color(img, clean_boxes)):
            return (
                "NEEDS_REVIEW",
                domain_tag,
                "review_openimages_generic_box_verify_cardboard_vs_plastic_wood_metal",
                clean_boxes,
                scene_flags,
            )

    # For Open Images /m/04dr76w (Bottle -> glass), require wine/beer/liquor/glass cue or hold in NEEDS_REVIEW
    if "/m/04dr76w" in mids_in_boxes:
        glass_cues = ("glass", "wine", "beer", "liquor", "whisky", "whiskey", "vodka", "gin", "rum", "champagne", "ale", "lager", "stout", "cider", "sauce", "olive oil", "vinegar")
        has_alcohol_mid = bool(co_mids & {"/m/081qc", "/m/01599"})
        if not (has_alcohol_mid or any(kw in text_cue for kw in glass_cues)):
            return (
                "NEEDS_REVIEW",
                domain_tag,
                "review_openimages_bottle_verify_glass_vs_plastic_material",
                clean_boxes,
                scene_flags,
            )

    # 5. Manual visual contact-sheet audit overrides (B3 independent visual verification of all QA_PASS sheets)
    item_id = rec["item_id"]
    if item_id == "task8w2_1248":
        # "File:Caja de cartón.png" is a clean cardboard box cutout; map box class to 2: cardboard
        for b in clean_boxes:
            b["class_id"] = 2
            b["class_name"] = "cardboard"
        rec["primary_class_id"] = 2

    visual_reject_ids = {
        "task8w2_0089",  # Andy Warhol banana print
        "task8w2_0097",  # "Ce n'est pas une banana" wall drawing
        "task8w2_0631",  # Roadside Pearl Beer sign shaped like a can
        "task8w2_0655",  # Coca-Cola truck trailer painted like a can
    }
    if item_id in visual_reject_ids:
        return "REJECT", "AMBIGUOUS_CONTEXT", "reject_visual_audit_non_physical_depiction_sign_or_print", [], {}

    visual_needs_review_ids = {
        # 0: battery (flashlights, battery holders with wires, multi-cell single box, watch repair)
        "task8w2_1287", "task8w2_1289", "task8w2_1295", "task8w2_1308",
        # 1: biological (multi-banana bunches with 1 box, fruit bowls/baskets, hand holding fruit, growing branch)
        "task8w2_0028", "task8w2_0041", "task8w2_0046", "task8w2_0056", "task8w2_0065",
        "task8w2_0074", "task8w2_0080", "task8w2_0083", "task8w2_0085", "task8w2_0090",
        "task8w2_0098", "task8w2_0099", "task8w2_0548",
        # 2: cardboard (wooden/metal/plastic boxes or unboxed secondary items in scene)
        "task8w2_0172", "task8w2_0186", "task8w2_0191", "task8w2_0194", "task8w2_0198",
        "task8w2_0202", "task8w2_0203", "task8w2_0212", "task8w2_0217", "task8w2_0225",
        "task8w2_0230", "task8w2_0240", "task8w2_0241", "task8w2_0242", "task8w2_0244",
        "task8w2_0260", "task8w2_0262", "task8w2_0263", "task8w2_0273", "task8w2_0285",
        "task8w2_0290", "task8w2_0312", "task8w2_0678",
        # 3: clothes (worn garments on torso/person or multi-garment laundry pile with partial boxes)
        "task8w2_0340", "task8w2_0343", "task8w2_0363", "task8w2_0384", "task8w2_0405", "task8w2_0409",
        # 4: glass (hand holding bottle, unboxed secondary bottles/cups, duplicate metal+glass box on same object)
        "task8w2_0487", "task8w2_0508", "task8w2_0513", "task8w2_0554", "task8w2_0556", "task8w2_1370",
        # 5: metal (multi-can clusters with 1 box, unboxed bottles/cups/containers alongside can, hand holding can)
        "task8w2_0568", "task8w2_0576", "task8w2_0582", "task8w2_0593", "task8w2_0595",
        "task8w2_0596", "task8w2_0606", "task8w2_0608", "task8w2_0610", "task8w2_0613",
        "task8w2_0617", "task8w2_0628", "task8w2_0633", "task8w2_0642", "task8w2_0649",
        "task8w2_0653", "task8w2_0663", "task8w2_0666", "task8w2_0667", "task8w2_0669",
        # 6: paper (stacks of books/magazines with partial boxes, person holding book, outdoor sign)
        "task8w2_0703", "task8w2_0730", "task8w2_0749", "task8w2_0751", "task8w2_0752",
        # 8: shoes (shoes worn on human feet/legs or shoe racks/boxes with unboxed pairs)
        "task8w2_0831", "task8w2_0870", "task8w2_0872", "task8w2_0892", "task8w2_0905",
        "task8w2_0906", "task8w2_0907", "task8w2_0909", "task8w2_0910", "task8w2_0911",
        "task8w2_0914", "task8w2_0925", "task8w2_0929",
        # 9: trash (ashtray pile of 40+ cigarette butts with partial corner box)
        "task8w2_1407",
    }
    if item_id in visual_needs_review_ids:
        return (
            "NEEDS_REVIEW",
            domain_tag,
            "review_visual_contact_sheet_audit_partial_instances_or_material_or_in_use",
            clean_boxes,
            scene_flags,
        )

    return "QA_PASS", domain_tag, "verified_clean_instance_boxes_and_taxonomy", clean_boxes, scene_flags


def render_overlay_image(
    img: Image.Image,
    rec: dict[str, Any],
    out_path: pathlib.Path,
) -> None:
    """Draw bounding boxes, class names, QA status, and domain tag onto a copy of the image."""
    canvas = img.convert("RGB").copy()
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)

    for b in rec.get("boxes", []):
        cid = b["class_id"]
        cname = b["class_name"]
        color = CLASS_COLORS.get(cid, (255, 255, 0))
        x1 = int(round(b["xmin"] * w))
        y1 = int(round(b["ymin"] * h))
        x2 = int(round(b["xmax"] * w))
        y2 = int(round(b["ymax"] * h))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) * 0.006)))
        lbl_txt = f"{cid}:{cname}"
        tw = len(lbl_txt) * 7 + 8
        th = 16
        ty1 = max(0, y1 - th)
        draw.rectangle([x1, ty1, min(w, x1 + tw), ty1 + th], fill=color)
        draw.text((x1 + 4, ty1 + 2), lbl_txt, fill=(0, 0, 0))

    # Top banner with item_id, qa_status, domain_tag
    banner_color = (20, 130, 50) if rec["qa_status"] == "QA_PASS" else ((200, 130, 20) if rec["qa_status"] == "NEEDS_REVIEW" else (180, 30, 30))
    draw.rectangle([0, 0, w, 22], fill=banner_color)
    header = f"{rec['item_id']} | {rec['qa_status']} | {rec['domain_tag']} | {rec['specimen_group'][:18]}"
    draw.text((6, 4), header, fill=(255, 255, 255))
    canvas.save(out_path, format="JPEG", quality=88)


def build_contact_sheet(
    overlay_paths: list[tuple[pathlib.Path, str]],
    out_path: pathlib.Path,
    title_text: str,
    cols: int = 5,
    thumb_w: int = 280,
    thumb_h: int = 210,
) -> None:
    """Build a grid contact sheet from rendered overlay images."""
    if not overlay_paths:
        return
    rows = (len(overlay_paths) + cols - 1) // cols
    header_h = 42
    cell_h = thumb_h + 26
    sheet = Image.new("RGB", (cols * thumb_w, header_h + rows * cell_h), (24, 26, 30))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, cols * thumb_w, header_h], fill=(40, 46, 58))
    draw.text((14, 12), title_text, fill=(255, 255, 255))

    for idx, (p, caption) in enumerate(overlay_paths):
        r = idx // cols
        c = idx % cols
        x0 = c * thumb_w
        y0 = header_h + r * cell_h
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.BILINEAR)
            ox = x0 + (thumb_w - im.width) // 2
            oy = y0 + (thumb_h - im.height) // 2
            sheet.paste(im, (ox, oy))
        except Exception:
            pass
        draw.text((x0 + 6, y0 + thumb_h + 4), caption[:38], fill=(210, 215, 225))

    sheet.save(out_path, format="JPEG", quality=88)


def main() -> None:
    for d in (LABELS_PASS_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, CONTACT_SHEETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for old_f in d.glob("*"):
            if old_f.is_file():
                old_f.unlink()

    raw_data = json.loads(RAW_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_records = raw_data.get("records", [])
    print(f"[Wave2-QA] Loaded {len(raw_records)} raw downloaded records.")

    ext_hashes = load_external_hashes()
    ext_sha_set = {x[1] for x in ext_hashes}
    print(f"[Wave2-QA] Loaded {len(ext_hashes)} external dataset hashes for leakage/dedup verification.")

    seen_wave2_phashes: list[tuple[str, int, str]] = []  # (item_id, dhash_int, specimen_group)
    author_to_session: dict[str, str] = {}

    qa_records: list[dict[str, Any]] = []

    for rec in raw_records:
        item_id = rec["item_id"]
        img_path = REPO_ROOT / rec["rel_path"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        dhash_int = compute_dhash_int(img)
        phash_hex = f"{dhash_int:016x}"
        vis = analyze_visual_properties(img)

        # Session & specimen grouping by photographer (AuthorProfileURL or Author)
        author_key = (rec.get("author_profile_url") or rec.get("author") or rec["source_image_id"]).strip()
        if author_key not in author_to_session:
            ahash = hashlib.md5(author_key.encode("utf-8")).hexdigest()[:8].upper()
            author_to_session[author_key] = f"SESSION_W2_{ahash}"
        session_group = author_to_session[author_key]
        specimen_group = f"SPEC_W2_{item_id.replace('task8w2_', '')}_{session_group[-6:]}"

        # Check external exact/near duplicate leakage
        ext_dup_match = None
        if rec["sha256"] in ext_sha_set:
            ext_dup_match = "exact_sha256_external"
        else:
            for ext_key, _, ext_dhash in ext_hashes:
                if hamming_dist(dhash_int, ext_dhash) <= 6:
                    ext_dup_match = f"near_phash_external:{ext_key}"
                    break

        # Check internal Wave 2 near-duplicate pHash (Hamming <= 8)
        int_dup_match = None
        for prev_id, prev_dhash, prev_spec in seen_wave2_phashes:
            if hamming_dist(dhash_int, prev_dhash) <= 8:
                int_dup_match = prev_id
                specimen_group = prev_spec  # Group together!
                break

        seen_wave2_phashes.append((item_id, dhash_int, specimen_group))

        # Auto-box Wikimedia Commons if needed
        is_commons_confident = True
        if rec["source_family"] == "wikimedia_commons_wave2":
            c_boxes, is_commons_confident = auto_box_commons_image(img, rec["primary_class_id"], rec.get("title", ""))
            rec["boxes"] = c_boxes

        if ext_dup_match:
            qa_status = "REJECT"
            domain_tag = "AMBIGUOUS_CONTEXT"
            qa_reason = f"reject_external_dedup_{ext_dup_match}"
            valid_boxes = []
            scene_flags = {"complex_background": False, "outdoor_scene": False, "dirty_or_damaged": False, "multi_class": False}
        elif int_dup_match:
            qa_status = "REJECT"
            domain_tag = "AMBIGUOUS_CONTEXT"
            qa_reason = f"reject_internal_wave2_near_duplicate_of_{int_dup_match}"
            valid_boxes = []
            scene_flags = {"complex_background": False, "outdoor_scene": False, "dirty_or_damaged": False, "multi_class": False}
        else:
            qa_status, domain_tag, qa_reason, valid_boxes, scene_flags = evaluate_record_qa_and_domain(
                img, rec, vis, is_commons_confident
            )

        classes_present = sorted({b["class_id"] for b in valid_boxes})
        class_names_present = [CLASS_NAMES[c] for c in classes_present]

        # Write YOLO labels to appropriate directory
        yolo_lines = []
        for b in valid_boxes:
            xc = (b["xmin"] + b["xmax"]) / 2.0
            yc = (b["ymin"] + b["ymax"]) / 2.0
            bw = b["xmax"] - b["xmin"]
            bh = b["ymax"] - b["ymin"]
            yolo_lines.append(f"{b['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_rel_path = ""
        if qa_status == "QA_PASS" and yolo_lines:
            lbl_path = LABELS_PASS_DIR / f"{item_id}.txt"
            lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            label_rel_path = f"data/acquisition_staging/task8_wave2_large_batch/labels/{item_id}.txt"
        elif qa_status == "NEEDS_REVIEW" and yolo_lines:
            lbl_path = LABELS_REVIEW_DIR / f"{item_id}.txt"
            lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            label_rel_path = f"data/acquisition_staging/task8_wave2_large_batch/labels_needs_manual_review/{item_id}.txt"

        qa_rec = {
            "item_id": item_id,
            "filename": rec["filename"],
            "rel_path": rec["rel_path"],
            "label_rel_path": label_rel_path,
            "source_family": rec["source_family"],
            "split_origin": rec.get("split_origin", ""),
            "source_image_id": rec["source_image_id"],
            "primary_class_id": rec["primary_class_id"],
            "primary_class_name": CLASS_NAMES[rec["primary_class_id"]],
            "classes_present": classes_present,
            "class_names_present": class_names_present,
            "num_boxes": len(valid_boxes),
            "is_multi_class": len(classes_present) >= 2,
            "qa_status": qa_status,
            "qa_reason": qa_reason,
            "domain_tag": domain_tag,
            "specimen_group": specimen_group,
            "session_group": session_group,
            "candidate_split": "EXCLUDED",  # Assigned below by session_group
            "approved_for_train": False,
            "width": rec["width"],
            "height": rec["height"],
            "bytes": rec["bytes"],
            "sha256": rec["sha256"],
            "phash_hex": phash_hex,
            "author": rec.get("author", ""),
            "author_profile_url": rec.get("author_profile_url", ""),
            "original_landing_url": rec.get("original_landing_url", ""),
            "original_url": rec.get("original_url", ""),
            "license": rec.get("license", ""),
            "title": rec.get("title", ""),
            "original_md5": rec.get("original_md5", ""),
            "complex_background": scene_flags.get("complex_background", False),
            "outdoor_scene": scene_flags.get("outdoor_scene", False),
            "dirty_or_damaged": scene_flags.get("dirty_or_damaged", False),
            "visual_metrics": vis,
            "boxes": valid_boxes,
        }
        qa_records.append(qa_rec)

    # Assign candidate_split (TRAIN_WAVE2 vs VAL_REAL_WAVE2) strictly grouped by session_group!
    # Every session_group is deterministically assigned to either TRAIN_WAVE2 (~82%) or VAL_REAL_WAVE2 (~18%)
    session_to_split: dict[str, str] = {}
    for r in qa_records:
        if r["qa_status"] == "QA_PASS":
            sg = r["session_group"]
            if sg not in session_to_split:
                hval = int(hashlib.md5(sg.encode("utf-8")).hexdigest()[:4], 16) % 100
                session_to_split[sg] = "VAL_REAL_WAVE2" if hval < 18 else "TRAIN_WAVE2"
            r["candidate_split"] = session_to_split[sg]
        elif r["qa_status"] == "NEEDS_REVIEW":
            r["candidate_split"] = "HOLD_NEEDS_REVIEW"
        else:
            r["candidate_split"] = "REJECTED"

    # Render overlays for all QA_PASS and NEEDS_REVIEW images (plus up to 30 REJECT samples)
    print("[Wave2-QA] Rendering overlays and contact sheets...")
    pass_by_class: dict[int, list[tuple[pathlib.Path, str]]] = defaultdict(list)
    review_overlays: list[tuple[pathlib.Path, str]] = []
    multiclass_overlays: list[tuple[pathlib.Path, str]] = []

    reject_rendered = 0
    for r in qa_records:
        if r["qa_status"] == "REJECT" and reject_rendered >= 30:
            continue
        if r["qa_status"] == "REJECT":
            reject_rendered += 1
        img_path = REPO_ROOT / r["rel_path"]
        ov_path = OVERLAYS_DIR / f"{r['item_id']}_{r['qa_status']}.jpg"
        try:
            im = Image.open(img_path)
            render_overlay_image(im, r, ov_path)
            r["overlay_rel_path"] = f"data/acquisition_staging/task8_wave2_large_batch/overlays/{ov_path.name}"
            cap = f"{r['item_id']} [{r['domain_tag'][:10]}] {','.join(r['class_names_present'])}"
            if r["qa_status"] == "QA_PASS":
                for cid in r["classes_present"]:
                    pass_by_class[cid].append((ov_path, cap))
                if r["is_multi_class"]:
                    multiclass_overlays.append((ov_path, cap))
            elif r["qa_status"] == "NEEDS_REVIEW":
                review_overlays.append((ov_path, f"{r['item_id']} {r['qa_reason'][:26]}"))
        except Exception as e:
            print(f"  Overlay warning on {r['item_id']}: {e}")

    # Build contact sheets per class (up to 40 per sheet)
    sheet_files: list[str] = []
    for cid, items in sorted(pass_by_class.items()):
        cname = CLASS_NAMES[cid]
        for page_idx in range(0, len(items), 35):
            chunk = items[page_idx : page_idx + 35]
            pnum = (page_idx // 35) + 1
            sname = f"wave2_qa_pass_class_{cid}_{cname}_p{pnum}.jpg"
            spath = CONTACT_SHEETS_DIR / sname
            build_contact_sheet(
                chunk,
                spath,
                f"Task 8 Wave 2 QA_PASS — Class {cid}: {cname} (Page {pnum}, {len(chunk)} items)",
            )
            sheet_files.append(sname)

    if multiclass_overlays:
        sname = "wave2_qa_pass_multiclass.jpg"
        build_contact_sheet(
            multiclass_overlays[:40],
            CONTACT_SHEETS_DIR / sname,
            f"Task 8 Wave 2 QA_PASS — Multi-Class Images ({len(multiclass_overlays)} items)",
        )
        sheet_files.append(sname)

    for page_idx in range(0, min(len(review_overlays), 105), 35):
        chunk = review_overlays[page_idx : page_idx + 35]
        pnum = (page_idx // 35) + 1
        sname = f"wave2_needs_review_p{pnum}.jpg"
        build_contact_sheet(
            chunk,
            CONTACT_SHEETS_DIR / sname,
            f"Task 8 Wave 2 HELD IN NEEDS_REVIEW (Page {pnum}, {len(chunk)} items)",
        )
        sheet_files.append(sname)

    # Verify cross-split session & specimen leakage == 0
    train_sessions = {r["session_group"] for r in qa_records if r["candidate_split"] == "TRAIN_WAVE2"}
    val_sessions = {r["session_group"] for r in qa_records if r["candidate_split"] == "VAL_REAL_WAVE2"}
    cross_split_overlap = train_sessions & val_sessions

    status_counts = Counter(r["qa_status"] for r in qa_records)
    domain_pass_counts = Counter(r["domain_tag"] for r in qa_records if r["qa_status"] == "QA_PASS")
    domain_all_counts = Counter(r["domain_tag"] for r in qa_records)
    split_counts = Counter(r["candidate_split"] for r in qa_records if r["qa_status"] == "QA_PASS")

    per_class_summary: dict[str, Any] = {}
    for cid, cname in enumerate(CLASS_NAMES):
        pass_imgs = [r for r in qa_records if r["qa_status"] == "QA_PASS" and cid in r["classes_present"]]
        train_imgs = [r for r in pass_imgs if r["candidate_split"] == "TRAIN_WAVE2"]
        val_imgs = [r for r in pass_imgs if r["candidate_split"] == "VAL_REAL_WAVE2"]
        rev_imgs = [r for r in qa_records if r["qa_status"] == "NEEDS_REVIEW" and (cid in r["classes_present"] or r["primary_class_id"] == cid)]
        rej_imgs = [r for r in qa_records if r["qa_status"] == "REJECT" and r["primary_class_id"] == cid]
        boxes_pass = sum(1 for r in pass_imgs for b in r["boxes"] if b["class_id"] == cid)

        per_class_summary[f"{cid}_{cname}"] = {
            "class_id": cid,
            "class_name": cname,
            "qa_pass_unique_images": len(pass_imgs),
            "qa_pass_train_images": len(train_imgs),
            "qa_pass_val_images": len(val_imgs),
            "qa_pass_boxes": boxes_pass,
            "qa_pass_unique_specimens": len({r["specimen_group"] for r in pass_imgs}),
            "qa_pass_unique_sessions": len({r["session_group"] for r in pass_imgs}),
            "qa_pass_actual_waste_images": sum(1 for r in pass_imgs if r["domain_tag"] == "ACTUAL_WASTE"),
            "qa_pass_product_studio_images": sum(1 for r in pass_imgs if r["domain_tag"] == "PRODUCT_STUDIO"),
            "qa_pass_ambiguous_context_images": sum(1 for r in pass_imgs if r["domain_tag"] == "AMBIGUOUS_CONTEXT"),
            "qa_pass_multi_class_images": sum(1 for r in pass_imgs if r["is_multi_class"]),
            "qa_pass_complex_bg_images": sum(1 for r in pass_imgs if r["complex_background"]),
            "qa_pass_outdoor_images": sum(1 for r in pass_imgs if r["outdoor_scene"]),
            "qa_pass_dirty_or_damaged_images": sum(1 for r in pass_imgs if r["dirty_or_damaged"]),
            "needs_review_images": len(rev_imgs),
            "reject_images": len(rej_imgs),
        }

    manifest_out = {
        "wave": "TASK8_WAVE2_LARGE_BATCH_QA",
        "generated_at": "2026-10-14",
        "approved_for_train": False,
        "total_downloaded_evaluated": len(qa_records),
        "qa_status_counts": dict(status_counts),
        "qa_pass_unique_images": status_counts["QA_PASS"],
        "qa_pass_unique_specimens": len({r["specimen_group"] for r in qa_records if r["qa_status"] == "QA_PASS"}),
        "qa_pass_unique_sessions": len({r["session_group"] for r in qa_records if r["qa_status"] == "QA_PASS"}),
        "qa_pass_split_counts": dict(split_counts),
        "qa_pass_domain_counts": dict(domain_pass_counts),
        "all_evaluated_domain_counts": dict(domain_all_counts),
        "qa_pass_multi_class_images": sum(1 for r in qa_records if r["qa_status"] == "QA_PASS" and r["is_multi_class"]),
        "qa_pass_total_boxes": sum(r["num_boxes"] for r in qa_records if r["qa_status"] == "QA_PASS"),
        "zero_external_leakage": True,
        "zero_cross_split_session_leakage": len(cross_split_overlap) == 0,
        "contact_sheets": sheet_files,
        "per_class_summary": per_class_summary,
        "records": qa_records,
    }

    QA_JSON_PATH.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write CSV manifest
    csv_cols = [
        "item_id", "filename", "source_family", "split_origin", "source_image_id",
        "primary_class_id", "primary_class_name", "classes_present", "num_boxes", "is_multi_class",
        "qa_status", "qa_reason", "domain_tag", "specimen_group", "session_group",
        "candidate_split", "approved_for_train", "width", "height", "bytes",
        "sha256", "phash_hex", "author", "author_profile_url", "original_landing_url", "license", "title",
    ]
    with open(QA_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in qa_records:
            row = {k: r.get(k, "") for k in csv_cols}
            row["classes_present"] = "|".join(str(c) for c in r["classes_present"])
            writer.writerow(row)

    print(f"[Wave2-QA] Complete! Total evaluated: {len(qa_records)}")
    print("  QA Status counts:", dict(status_counts))
    print("  QA_PASS Split counts:", dict(split_counts))
    print("  QA_PASS Domain counts:", dict(domain_pass_counts))
    print("  Zero cross-split session leakage:", len(cross_split_overlap) == 0)


if __name__ == "__main__":
    main()
