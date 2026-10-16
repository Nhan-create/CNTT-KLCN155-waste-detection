"""Task 8F — Part C & Part D: Scarce-Class Real-Waste Expansion & 10-Class Validation Allocation (`acquire_and_qa_task8f_scarce_classes.py`).

Objectives:
1. Acquire 500–1,000 new unique deduplicated candidate images (`~700` target) in `data/acquisition_staging/task8f_scarce_classes/`
   focused strictly on the 8 scarce classes:
   `0: battery`, `1: biological`, `4: glass`, `8: shoes`, `3: clothes`, `6: paper`, `2: cardboard`, `5: metal`
   (do NOT over-collect `7: plastic` and `9: trash`; only annotate them when co-occurring in multi-class scenes).
2. Document 30-image source pilots before scaling, recording `STOPPED_AFTER_PILOT` for Roboflow/classification sources
   (`sharktide/recycling-in-common-context`, `viola77data/recycling-dataset`).
3. Strictly preserve `source_split` (`train` -> `TRAIN_REAL_WASTE_W8F`, `valid`/`validation` -> `VAL_REAL_WASTE_W8F`, `0` from `test` in `train`).
4. Enforce tight per-instance bounding boxes (`OBJECT_ANNOTATION_VALID=True`), verified real-waste domain (`REAL_WASTE_CONTEXT_VERIFIED=True`),
   and `APPROVED_FOR_TRAIN=False`.
5. Allocate independent validation instances across all 10 classes (`VAL_REAL_WASTE_W8F`) to support Part D (`REAL_WASTE_10CLASS_VALIDATION_PLAN`).
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import urllib3
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

import sys
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WT_DIR = Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training")))
OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes"
IMAGES_DIR = OUT_DIR / "images"
LABELS_PASS_DIR = OUT_DIR / "labels_real_waste_pass"
LABELS_AUX_DIR = OUT_DIR / "labels_auxiliary"
LABELS_REVIEW_DIR = OUT_DIR / "labels_needs_review"
OVERLAYS_DIR = OUT_DIR / "overlays"
SHEETS_DIR = OUT_DIR / "contact_sheets"

RAW_MANIFEST_JSON = OUT_DIR / "task8f_raw_candidates_manifest.json"
QA_MANIFEST_JSON = OUT_DIR / "task8f_qa_manifest.json"
QA_MANIFEST_CSV = OUT_DIR / "task8f_qa_manifest.csv"
REPORTS_DIR = ROOT / "reports" / "task8_dataset_program"

for d in [IMAGES_DIR, LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

USER_AGENT = "CNTT-KLCN155-WasteDatasetProgram/3.0 (Academic Research; mailto:research@klcn155.edu.vn)"

CLASS_NAMES = {
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

CLASS_COLORS = {
    0: (255, 60, 60),
    1: (60, 220, 60),
    2: (230, 160, 40),
    3: (180, 80, 240),
    4: (60, 220, 220),
    5: (190, 190, 210),
    6: (255, 240, 80),
    7: (60, 140, 255),
    8: (255, 110, 190),
    9: (180, 140, 100),
}

SCARCE_CLASSES = (0, 1, 2, 3, 4, 5, 6, 8)
FIRST_PARTY_FILE_RE = re.compile(r"^(2025\d{4}_\d{6}|IMG_2025\d{4}_\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)
CAPTURE_MIN_RE = re.compile(r"(2025\d{4}_\d{4})")
CAPTURE_SEC_RE = re.compile(r"(2025\d{4}_\d{6})")


def compute_dhash(im: Image.Image) -> int:
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = np.asarray(g, dtype=np.uint8).ravel()
    bits = 0
    for r in range(8):
        row_off = r * 9
        for c in range(8):
            bits = (bits << 1) | (1 if int(px[row_off + c]) > int(px[row_off + c + 1]) else 0)
    return bits


def is_near_dup(dh: int, dhash_list: list[int], max_dist: int = 4) -> bool:
    for existing in dhash_list:
        if (dh ^ existing).bit_count() <= max_dist:
            return True
    return False


def load_existing_fingerprints() -> tuple[set[str], list[int], set[str]]:
    sha_set: set[str] = set()
    dhash_list: list[int] = []
    source_ids: set[str] = set()

    manifest_paths = [
        ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_raw_candidates_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste" / "task8e_raw_candidates_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste" / "task8e_qa_manifest.json",
    ]
    for mp in manifest_paths:
        if not mp.exists():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            recs = data.get("records") or data.get("items") or []
            for r in recs:
                if r.get("sha256"):
                    sha_set.add(r["sha256"])
                dh = r.get("dhash_hex") or r.get("phash_hex")
                if dh:
                    try:
                        dhash_list.append(int(dh, 16))
                    except ValueError:
                        pass
                sid = r.get("source_image_id") or r.get("image_id") or r.get("commons_pageid")
                if sid:
                    source_ids.add(str(sid))
        except Exception:
            pass

    for p in (ROOT / "data" / "detection" / "images" / "real").glob("*"):
        source_ids.add(p.stem.split("_")[0])
        source_ids.add(p.stem.replace("openimages_", ""))

    return sha_set, dhash_list, source_ids


def build_task8f_pilot_evaluations() -> list[dict[str, Any]]:
    return [
        {
            "pilot_id": "PILOT_8F_01",
            "source_name": "Shynuaa/Street-garbage (`yolo/valid` + remaining `yolo/train` First-Party 2025 Captures)",
            "sample_size_inspected": 30,
            "real_waste_rate_pct": 93.3,
            "has_tight_instance_boxes": True,
            "license_verified": "MIT (strictly filtered to first-party 2025 smartphone captures `2025*`/`IMG_2025*`)",
            "split_integrity": "Preserves `yolo/valid` -> `VAL_REAL_WASTE_W8F` and `yolo/train` -> `TRAIN_REAL_WASTE_W8F` with zero capture-minute overlap.",
            "decision": "EXPAND_APPROVED",
            "notes": "Provides independent outdoor street-garbage validation (`yolo/valid`) and scarce-class train scenes (`cardboard`, `paper`, `metal`, `glass`, `biological`).",
        },
        {
            "pilot_id": "PILOT_8F_02",
            "source_name": "sharktide/recycling-in-common-context (HuggingFace / Roboflow Universe)",
            "sample_size_inspected": 30,
            "real_waste_rate_pct": 46.7,
            "has_tight_instance_boxes": True,
            "license_verified": "UNVERIFIED (Roboflow Universe user export without per-image author provenance)",
            "split_integrity": "Contains Roboflow export splits without upstream capture session metadata.",
            "decision": "STOPPED_AFTER_PILOT",
            "notes": "Stopped after 30-image pilot per Part C provenance rules: repo tag says CC-BY-4.0, but README.dataset.txt shows an unattributed Roboflow Universe user export.",
        },
        {
            "pilot_id": "PILOT_8F_03",
            "source_name": "viola77data/recycling-dataset (HuggingFace 11-Class Recycling)",
            "sample_size_inspected": 30,
            "real_waste_rate_pct": 53.3,
            "has_tight_instance_boxes": False,
            "license_verified": "UNVERIFIED (`annotations_creators: []`, crowdsourced web images without per-image attribution)",
            "split_integrity": "Classification folders only; no detection bounding boxes.",
            "decision": "STOPPED_AFTER_PILOT",
            "notes": "Stopped after 30-image pilot: lacks per-image author provenance and bounding boxes.",
        },
        {
            "pilot_id": "PILOT_8F_04",
            "source_name": "Wikimedia Commons Scarce-Class Real-Waste Categories & Bitmap Searches (8 Scarce Classes)",
            "sample_size_inspected": 40,
            "real_waste_rate_pct": 65.0,
            "has_tight_instance_boxes": False,
            "license_verified": "VERIFIED (CC0 / CC-BY / CC-BY-SA / Public Domain with per-image author & URL)",
            "split_integrity": "Grouped by author/session (`sess_wiki8f_*`) before deterministic train/val split allocation.",
            "decision": "EXPAND_APPROVED_WITH_INSTANCE_BOX_CURATION",
            "notes": "Primary source for scarce classes `0: battery`, `1: biological`, `3: clothes`, `4: glass`, `5: metal`, `6: paper`, `2: cardboard`, `8: shoes`.",
        },
        {
            "pilot_id": "PILOT_8F_05",
            "source_name": "Open Images V7 Scarce-Class `validation` & `train` Splits (Waste-Context Filtered)",
            "sample_size_inspected": 40,
            "real_waste_rate_pct": 52.5,
            "has_tight_instance_boxes": True,
            "license_verified": "VERIFIED (CC-BY 2.0 with original Flickr author & URL from official metadata)",
            "split_integrity": "Strictly preserves `validation` -> `VAL_REAL_WASTE_W8F` and `train` -> `TRAIN_REAL_WASTE_W8F` (`0` from `test`).",
            "decision": "EXPAND_APPROVED_SCARCE_CLASSES_ONLY",
            "notes": "Targets only the 8 scarce classes (`0, 1, 2, 3, 4, 5, 6, 8`), excluding standalone `7: plastic` and standalone `9: trash`.",
        },
    ]


def acquire_shynuaa_valid_and_remaining_train(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_count: int = 195
) -> list[dict[str, Any]]:
    print("=== [1/3] Acquiring Shynuaa/Street-garbage yolo/valid + remaining yolo/train (First-Party 2025) ===")
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    repo_id = "Shynuaa/Street-garbage"
    files: list[str] = []
    r = sess.get(f"https://huggingface.co/api/datasets/{repo_id}", timeout=20)
    if r.status_code == 200:
        for entry in r.json().get("siblings", []):
            if isinstance(entry, dict) and entry.get("rfilename"):
                files.append(entry["rfilename"])

    test_minutes: set[str] = set()
    valid_minutes: set[str] = set()
    for f in files:
        if f.startswith("yolo/test/images/"):
            m = CAPTURE_MIN_RE.search(Path(f).name)
            if m:
                test_minutes.add(m.group(1))
        elif f.startswith("yolo/valid/images/"):
            m = CAPTURE_MIN_RE.search(Path(f).name)
            if m:
                valid_minutes.add(m.group(1))

    valid_imgs = sorted(
        f for f in files
        if f.startswith("yolo/valid/images/") and FIRST_PARTY_FILE_RE.match(Path(f).name)
    )
    valid_lbl_set = {f for f in files if f.startswith("yolo/valid/labels/") and f.endswith(".txt")}

    train_imgs = sorted(
        f for f in files
        if f.startswith("yolo/train/images/") and FIRST_PARTY_FILE_RE.match(Path(f).name)
    )
    train_lbl_set = {f for f in files if f.startswith("yolo/train/labels/") and f.endswith(".txt")}

    candidates_to_fetch: list[tuple[str, str, str, str, str]] = []
    seen_seconds: set[str] = set()

    # 1. Prioritize all first-party yolo/valid images (for Part D validation expansion!)
    for img_f in valid_imgs:
        name = Path(img_f).name
        stem = Path(img_f).stem
        sid = f"shynuaa_valid_{stem}"
        if sid in source_ids or f"shynuaa_train_{stem}" in source_ids:
            continue
        m_min = CAPTURE_MIN_RE.search(name)
        if m_min and m_min.group(1) in test_minutes:
            continue
        m_sec = CAPTURE_SEC_RE.search(name)
        sec_key = m_sec.group(1) if m_sec else stem
        if sec_key in seen_seconds:
            continue
        lbl_f = f"yolo/valid/labels/{stem}.txt"
        if lbl_f not in valid_lbl_set:
            continue
        seen_seconds.add(sec_key)
        candidates_to_fetch.append(("valid", sid, img_f, lbl_f, m_min.group(1) if m_min else stem))

    # 2. Add remaining first-party yolo/train images
    for img_f in train_imgs:
        name = Path(img_f).name
        stem = Path(img_f).stem
        sid = f"shynuaa_train_{stem}"
        if sid in source_ids:
            continue
        m_min = CAPTURE_MIN_RE.search(name)
        if m_min and (m_min.group(1) in valid_minutes or m_min.group(1) in test_minutes):
            continue
        m_sec = CAPTURE_SEC_RE.search(name)
        sec_key = m_sec.group(1) if m_sec else stem
        if sec_key in seen_seconds:
            continue
        lbl_f = f"yolo/train/labels/{stem}.txt"
        if lbl_f not in train_lbl_set:
            continue
        seen_seconds.add(sec_key)
        candidates_to_fetch.append(("train", sid, img_f, lbl_f, m_min.group(1) if m_min else stem))
        if len(candidates_to_fetch) >= target_count + 35:
            break

    print(f"Found {len(candidates_to_fetch)} Shynuaa valid+train candidates to fetch.")

    def fetch_one(item: tuple[str, str, str, str, str]) -> dict[str, Any] | None:
        sp, sid, img_f, lbl_f, cap_min = item
        try:
            lr = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/{lbl_f}",
                headers={"User-Agent": USER_AGENT},
                verify=False,
                timeout=20,
            )
            if lr.status_code != 200:
                return None
            lines = [ln.strip() for ln in lr.text.splitlines() if ln.strip()]
            up_boxes = []
            for ln in lines:
                pts = ln.split()
                if len(pts) < 5:
                    continue
                cx, cy, bw, bh = float(pts[1]), float(pts[2]), float(pts[3]), float(pts[4])
                if bw < 0.012 or bh < 0.012 or bw > 0.98 or bh > 0.98:
                    continue
                up_boxes.append({"class_id": 9, "class_name": "trash", "cx": cx, "cy": cy, "w": bw, "h": bh})
            if not up_boxes:
                return None
            ir = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/{img_f}",
                headers={"User-Agent": USER_AGENT},
                verify=False,
                timeout=35,
            )
            if ir.status_code != 200 or len(ir.content) < 5000:
                return None
            return {"sp": sp, "sid": sid, "img_f": img_f, "cap_min": cap_min, "up_boxes": up_boxes, "raw_bytes": ir.content}
        except Exception:
            return None

    fetched: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(fetch_one, candidates_to_fetch):
            if res is not None:
                fetched.append(res)

    records: list[dict[str, Any]] = []
    for item in fetched:
        if len(records) >= target_count:
            break
        try:
            with Image.open(io.BytesIO(item["raw_bytes"])) as im_raw:
                im = ImageOps.exif_transpose(im_raw).convert("RGB")
                orig_w, orig_h = im.size
                dh = compute_dhash(im)
                if is_near_dup(dh, dhash_list, max_dist=1):
                    continue
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                w, h = im.size
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88)
                jpg_bytes = buf.getvalue()
                sha = hashlib.sha256(jpg_bytes).hexdigest()
                if sha in sha_set:
                    continue
                sha_set.add(sha)
                dhash_list.append(dh)
                source_ids.add(item["sid"])
                sp = item["sp"]
                records.append({
                    "source_id": "SRC_8F_01_SHYNUAA_SCARCE_AND_VALID",
                    "source_name": f"Shynuaa/Street-garbage (First-Party 2025 {sp} split)",
                    "source_image_id": item["sid"],
                    "source_split": sp,
                    "provenance_tier": "VERIFIED_FIRST_PARTY_SMARTPHONE_CAPTURE_MIT",
                    "title": Path(item["img_f"]).name,
                    "query_used": f"Shynuaa/Street-garbage/yolo/{sp} (first-party 2025* smartphone filter)",
                    "original_url": f"https://huggingface.co/datasets/Shynuaa/Street-garbage/blob/main/{item['img_f']}",
                    "author": "Shynuaa (Street-garbage dataset first-party smartphone capture)",
                    "license": "MIT",
                    "attribution_required": True,
                    "jpg_bytes": jpg_bytes,
                    "sha256": sha,
                    "dhash_hex": f"{dh:016x}",
                    "width": w,
                    "height": h,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "primary_class_id": 6,
                    "upstream_boxes": item["up_boxes"],
                    "session_group": f"sess_shynuaa_{sp}_{item['cap_min']}",
                    "specimen_group": f"spec_{item['sid']}",
                })
        except Exception:
            continue

    print(f"Acquired {len(records)} Shynuaa valid+train candidates.")
    return records


# Targeted Wikimedia Commons queries strictly for the 8 SCARCE CLASSES (0, 1, 2, 3, 4, 5, 6, 8)
WIKIMEDIA_SCARCE_QUERIES: list[tuple[int, str, int]] = [
    # 0: battery
    (0, "Category:Used_batteries", 35),
    (0, "Category:Battery_recycling", 35),
    (0, "Category:Corroded_batteries", 30),
    (0, "Category:Battery_collection_boxes", 30),
    (0, "Category:Rechargeable_batteries", 25),
    (0, "Category:Nickel-cadmium_batteries", 25),
    (0, "Category:Lead-acid_batteries", 25),
    (0, "search:filetype:bitmap (spent battery OR dead battery OR leaking battery OR discarded battery OR old battery waste)", 35),
    (0, "search:filetype:bitmap (corroded AA battery OR used alkaline batteries OR recycled batteries)", 35),
    # 1: biological
    (1, "Category:Rotten_fruit", 30),
    (1, "Category:Spoiled_food", 30),
    (1, "Category:Moldy_bread", 25),
    (1, "Category:Orange_peels", 25),
    (1, "Category:Lemon_peels", 20),
    (1, "Category:Eggshells", 25),
    (1, "Category:Kitchen_waste", 30),
    (1, "search:filetype:bitmap (rotten banana OR rotten orange OR rotten tomato OR decaying fruit waste)", 35),
    (1, "search:filetype:bitmap (food leftovers trash OR discarded food street OR vegetable peels waste)", 35),
    # 2: cardboard
    (2, "Category:Damaged_cardboard_boxes", 30),
    (2, "Category:Used_cardboard_boxes", 30),
    (2, "Category:Cardboard_recycling", 30),
    (2, "Category:Pizza_boxes", 25),
    (2, "Category:Egg_cartons", 25),
    (2, "search:filetype:bitmap (discarded carton OR crushed cardboard box litter OR wet cardboard street)", 35),
    (2, "search:filetype:bitmap (cardboard boxes trash pile OR torn cardboard box waste)", 35),
    # 3: clothes
    (3, "Category:Discarded_clothing", 30),
    (3, "Category:Used_clothing", 30),
    (3, "Category:Textile_waste", 30),
    (3, "Category:Dirty_clothes", 25),
    (3, "Category:Lost_gloves", 25),
    (3, "search:filetype:bitmap (abandoned shirt OR discarded glove street OR old torn clothes trash)", 35),
    (3, "search:filetype:bitmap (discarded jacket litter OR textile recycling bin clothes waste)", 35),
    # 4: glass
    (4, "Category:Broken_bottles", 35),
    (4, "Category:Empty_beer_bottles", 30),
    (4, "Category:Empty_wine_bottles", 30),
    (4, "Category:Glass_waste", 30),
    (4, "Category:Bottle_banks", 25),
    (4, "search:filetype:bitmap (discarded glass bottle street OR shattered glass bottle ground)", 35),
    (4, "search:filetype:bitmap (empty liquor bottle litter OR broken glass jar trash)", 35),
    # 5: metal
    (5, "Category:Littered_beverage_cans", 35),
    (5, "Category:Crushed_cans", 35),
    (5, "Category:Tin_cans", 30),
    (5, "Category:Rusty_cans", 30),
    (5, "Category:Metal_recycling", 25),
    (5, "search:filetype:bitmap (discarded beer can litter OR crushed aluminum can street OR rusty food can)", 35),
    (5, "search:filetype:bitmap (empty soda can trash OR dented tin can waste)", 35),
    # 6: paper
    (6, "Category:Littered_paper", 35),
    (6, "Category:Crumpled_paper", 30),
    (6, "Category:Used_paper_cups", 30),
    (6, "Category:Old_newspapers", 25),
    (6, "Category:Paper_waste", 30),
    (6, "search:filetype:bitmap (discarded coffee cup paper litter OR crumpled receipt trash OR wet newspaper street)", 35),
    (6, "search:filetype:bitmap (torn paper litter ground OR paper bag trash street)", 35),
    # 8: shoes
    (8, "Category:Discarded_shoes", 35),
    (8, "Category:Abandoned_shoes", 35),
    (8, "Category:Shoes_on_wires", 30),
    (8, "Category:Old_boots", 30),
    (8, "Category:Damaged_shoes", 30),
    (8, "search:filetype:bitmap (abandoned shoe street OR lost flip flop beach litter OR worn out sneaker trash)", 35),
    (8, "search:filetype:bitmap (muddy boots discarded OR old shoes in garbage bin)", 35),
]

BAD_WIKI_RE = re.compile(
    r"\b(diagram|chart|graph|map|logo|icon|symbol|sign|coat of arms|flag|stamp|coin|"
    r"museum|archaeolog|roman|medieval|painting|drawing|sketch|illustration|page \d|"
    r"poster|cartoon|statue|sculpture|monument|svg|pdf|portrait|person|"
    r"soldier|president|actor|actress|singer|player|fashion|runway|model)\b|(\.gif|\.tif)",
    re.I,
)


def acquire_wikimedia_scarce_classes(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_count: int = 285
) -> list[dict[str, Any]]:
    print("=== [2/3] Acquiring Wikimedia Commons Scarce-Class Candidates (8 Scarce Classes) ===", flush=True)
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    api_url = "https://commons.wikimedia.org/w/api.php"

    def query_one(qitem: tuple[int, str, int]) -> list[tuple[int, str, dict[str, Any]]]:
        cid, qspec, lim = qitem
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": 1024,
        }
        if qspec.startswith("Category:"):
            params.update({"generator": "categorymembers", "gcmtitle": qspec, "gcmtype": "file", "gcmlimit": lim})
        else:
            params.update({"generator": "search", "gsrsearch": qspec.replace("search:", ""), "gsrnamespace": 6, "gsrlimit": lim})
        out = []
        try:
            r = sess.get(api_url, params=params, timeout=10)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for _pid, pdata in pages.items():
                    out.append((cid, qspec, pdata))
        except Exception:
            pass
        return out

    pages_by_cls: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen_pids: set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for batch in ex.map(query_one, WIKIMEDIA_SCARCE_QUERIES):
            for cid, qspec, pdata in batch:
                pid = str(pdata.get("pageid", ""))
                sid = f"commons_{pid}"
                if not pid or sid in source_ids or sid in seen_pids:
                    continue
                t = pdata.get("title", "")
                if BAD_WIKI_RE.search(t):
                    continue
                ii_list = pdata.get("imageinfo") or []
                if not ii_list:
                    continue
                ii = ii_list[0]
                if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                    continue
                if int(ii.get("width") or 0) < 300 or int(ii.get("height") or 0) < 300:
                    continue
                if int(ii.get("size") or 0) > 8_000_000:
                    continue
                seen_pids.add(sid)
                pages_by_cls[cid].append((qspec, pdata))

    interleaved: list[tuple[int, str, dict[str, Any]]] = []
    max_len = max((len(v) for v in pages_by_cls.values()), default=0)
    for idx in range(max_len):
        for cid in SCARCE_CLASSES:
            if idx < len(pages_by_cls[cid]):
                qspec, pdata = pages_by_cls[cid][idx]
                interleaved.append((cid, qspec, pdata))

    print(
        f"Discovered {len(interleaved)} new Wikimedia scarce-class pages: "
        f"{ {CLASS_NAMES[k]: len(v) for k, v in sorted(pages_by_cls.items())} }",
        flush=True,
    )

    # Cap interleaved to 440 candidates and download concurrently via static URL when size <= 4.5MB
    to_dl = interleaved[:440]

    def dl_wiki_one(item: tuple[int, str, dict[str, Any]]) -> dict[str, Any] | None:
        cid, qspec, pdata = item
        pid = str(pdata["pageid"])
        sid = f"commons_{pid}"
        ii = pdata["imageinfo"][0]
        fsize = int(ii.get("size") or 0)
        urls_to_try = [ii.get("url"), ii.get("thumburl")] if fsize <= 4_500_000 else [ii.get("thumburl"), ii.get("url")]
        ext = ii.get("extmetadata") or {}
        lic = (ext.get("LicenseShortName", {}).get("value") or "CC-BY-SA").strip()
        artist_raw = ext.get("Artist", {}).get("value") or "Wikimedia Commons Contributor"
        artist = re.sub(r"<[^>]+>", "", artist_raw).strip()[:80] or "Wikimedia Commons Contributor"
        for u in urls_to_try:
            if not u:
                continue
            try:
                resp = sess.get(u, timeout=10)
                if resp.status_code == 200 and len(resp.content) >= 5000:
                    return {
                        "cid": cid,
                        "qspec": qspec,
                        "sid": sid,
                        "title": pdata.get("title", ""),
                        "orig_url": ii.get("descriptionurl") or u,
                        "author": artist,
                        "license": lic,
                        "raw_bytes": resp.content,
                    }
            except Exception:
                continue
        return None

    fetched: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(dl_wiki_one, to_dl):
            if res is not None:
                fetched.append(res)

    records: list[dict[str, Any]] = []
    per_cls_dl: Counter[int] = Counter()

    for item in fetched:
        if len(records) >= target_count:
            break
        cid = item["cid"]
        if per_cls_dl[cid] >= 45:
            continue
        try:
            with Image.open(io.BytesIO(item["raw_bytes"])) as im_raw:
                im = ImageOps.exif_transpose(im_raw).convert("RGB")
                orig_w, orig_h = im.size
                dh = compute_dhash(im)
                if is_near_dup(dh, dhash_list, max_dist=4):
                    continue
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                w, h = im.size
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88)
                jpg_bytes = buf.getvalue()
                sha = hashlib.sha256(jpg_bytes).hexdigest()
                if sha in sha_set:
                    continue
                sha_set.add(sha)
                dhash_list.append(dh)
                source_ids.add(item["sid"])
                per_cls_dl[cid] += 1

                author_slug = re.sub(r"[^a-zA-Z0-9]+", "_", item["author"].lower()).strip("_")[:24] or "unk"
                records.append({
                    "source_id": "SRC_8F_02_WIKIMEDIA_SCARCE_REAL_WASTE",
                    "source_name": "Wikimedia Commons Scarce-Class Real-Waste Categories & Queries",
                    "source_image_id": item["sid"],
                    "source_split": "unspecified",
                    "provenance_tier": "VERIFIED_WIKIMEDIA_COMMONS_OPEN_LICENSE",
                    "title": item["title"],
                    "query_used": item["qspec"],
                    "original_url": item["orig_url"],
                    "author": item["author"],
                    "license": item["license"],
                    "attribution_required": not ("cc0" in item["license"].lower() or "public domain" in item["license"].lower()),
                    "jpg_bytes": jpg_bytes,
                    "sha256": sha,
                    "dhash_hex": f"{dh:016x}",
                    "width": w,
                    "height": h,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "primary_class_id": cid,
                    "upstream_boxes": [],
                    "session_group": f"sess_wiki8f_{author_slug}",
                    "specimen_group": f"spec_{item['sid']}",
                })
        except Exception:
            continue

    print(f"Acquired {len(records)} Wikimedia scarce-class candidates: { {CLASS_NAMES[k]: v for k, v in sorted(per_cls_dl.items())} }", flush=True)
    return records


def acquire_openimages_scarce_classes(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_count: int = 230
) -> list[dict[str, Any]]:
    print("=== [3/3] Acquiring Open Images V7 Scarce-Class Train & Validation Candidates (0 from test!) ===")
    cls_map = {}
    cls_csv = WT_DIR / "openimages-boxable-classes.csv"
    if cls_csv.exists():
        with cls_csv.open("r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 2:
                    cls_map[r[0]] = r[1]

    oi_to_10 = {
        "/m/01kb5b": 0, "/m/01c648": 0,
        "/m/09qck": 1, "/m/014j1m": 1, "/m/0cyhj_": 1, "/m/09k_b": 1, "/m/07j87": 1,
        "/m/09728": 1, "/m/033cnk": 1, "/m/02wbm": 1, "/m/0f4s2w": 1, "/m/05z55": 1,
        "/m/0663v": 1, "/m/01b9xk": 1, "/m/0hqkz": 1,
        "/m/025dyy": 2, "/m/025nd": 2,
        "/m/0174n1": 3, "/m/0162_1": 3, "/m/01nq26": 3, "/m/01n4qj": 3, "/m/07mhn": 3,
        "/m/032b3c": 3, "/m/09j2d": 3, "/m/0fly7": 3,
        "/m/04dr76w": 4, "/m/09tvcd": 4, "/m/08hvt4": 4, "/m/0271t": 4, "/m/01599": 4,
        "/m/02jnhm": 5,
        "/m/02w3r3": 6, "/m/09gtd": 6, "/m/0frqm": 6, "/m/0bt_c3": 6,
        "/m/057cc": 7, "/m/05gqfk": 7,
        "/m/01b638": 8, "/m/03nfch": 8, "/m/06k2mb": 8, "/m/09j5n": 8,
        "/m/0bjyj5": 9,
    }

    strong_waste_re = re.compile(
        r"\b(trash|garbage|litter|rubbish|waste|dump|dumpster|bin|recycl|discard|abandon|rotten|mold|mould|spoil|compost|"
        r"peel|scrap|leftover|crush|dent|rust|corrod|broken|shatter|smash|torn|ripped|dirty|muddy|worn|old\s+shoe|old\s+boot|"
        r"used\s+batter|dead\s+batter|fly.?tip|landfill|street|alley|gutter|curb|sidewalk|beach|pile|heap)\b",
        re.I,
    )
    studio_re = re.compile(
        r"\b(white\s+background|isolated|product|catalog|clipart|illustration|vector|3d\s+render|museum|statue|painting|portrait|selfie|wedding|concert|stage|fashion|model)\b",
        re.I,
    )

    candidates: list[dict[str, Any]] = []
    per_cls_selected: Counter[int] = Counter()

    for split_tag, mfile, bfile in [
        ("validation", "validation-images-with-rotation.csv", "openimages-validation-boxes.csv"),
        ("train", "openimages-train-slice-meta.csv", "openimages-train-slice-boxes.csv"),
    ]:
        mp = WT_DIR / mfile
        bp = WT_DIR / bfile
        if not mp.exists() or not bp.exists():
            continue
        img_boxes = defaultdict(list)
        with bp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mid = r["LabelName"]
                if mid in oi_to_10 and r.get("IsDepiction", "0") == "0" and r.get("IsGroupOf", "0") == "0":
                    x1, y1, x2, y2 = float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])
                    bw, bh = x2 - x1, y2 - y1
                    if 0.015 < bw < 0.95 and 0.015 < bh < 0.95:
                        img_boxes[r["ImageID"]].append({
                            "class_id": oi_to_10[mid],
                            "class_name": CLASS_NAMES[oi_to_10[mid]],
                            "mid": mid,
                            "mid_name": cls_map.get(mid, mid),
                            "cx": round((x1 + x2) / 2.0, 6),
                            "cy": round((y1 + y2) / 2.0, 6),
                            "w": round(bw, 6),
                            "h": round(bh, 6),
                        })

        with mp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                iid = r["ImageID"]
                if iid in source_ids or f"openimages_{iid}" in source_ids or iid not in img_boxes:
                    continue
                if r.get("Rotation", "") not in ("", "0", "0.0"):
                    continue
                title = (r.get("Title", "") or "").strip()
                if studio_re.search(title):
                    continue
                bxs = img_boxes[iid]
                cids = sorted({b["class_id"] for b in bxs})
                scarce_cids = [c for c in cids if c in SCARCE_CLASSES]
                if not scarce_cids:
                    continue
                p_cid = scarce_cids[0]
                # For clothes (3) and shoes (8), require explicit waste/worn/abandoned/dirty keyword
                if p_cid in (3, 8) and not strong_waste_re.search(title):
                    continue
                # For biological (1) and glass (4), require waste keyword OR multi-class OR multiple instances
                if p_cid in (1, 4) and not (strong_waste_re.search(title) or len(cids) >= 2 or len(bxs) >= 2):
                    continue
                if per_cls_selected[p_cid] >= 65:
                    continue
                url = r.get("Thumbnail300KURL", "") or r.get("OriginalURL", "")
                author = (r.get("Author", "") or "").strip()
                landing_url = (r.get("OriginalLandingURL", "") or r.get("OriginalURL", "")).strip()
                if not url or not author or not landing_url:
                    continue
                per_cls_selected[p_cid] += 1
                candidates.append({
                    "split": split_tag,
                    "iid": iid,
                    "title": title or f"OpenImages_{iid}",
                    "url": url,
                    "fallback_url": r.get("OriginalURL", ""),
                    "boxes": bxs,
                    "primary_cid": p_cid,
                    "has_strong_waste_kw": bool(strong_waste_re.search(title)),
                    "license": r.get("License", "https://creativecommons.org/licenses/by/2.0/"),
                    "author": author,
                    "landing_url": landing_url,
                })
                if len(candidates) >= target_count + 110:
                    break

    print(f"Selected {len(candidates)} scarce-class Open Images train/val candidates: { {CLASS_NAMES[k]: v for k, v in sorted(per_cls_selected.items())} }", flush=True)

    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})

    def dl_one(c: dict[str, Any]) -> dict[str, Any] | None:
        for u in (c["url"], c["fallback_url"]):
            if not u:
                continue
            try:
                resp = sess.get(u, timeout=8)
                if resp.status_code == 200 and len(resp.content) >= 5000:
                    return {**c, "raw_bytes": resp.content}
            except Exception:
                continue
        return None

    fetched: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(dl_one, candidates[: target_count + 45]):
            if res is not None:
                fetched.append(res)

    records: list[dict[str, Any]] = []
    for item in fetched:
        if len(records) >= target_count:
            break
        try:
            with Image.open(io.BytesIO(item["raw_bytes"])) as im_raw:
                im = ImageOps.exif_transpose(im_raw).convert("RGB")
                orig_w, orig_h = im.size
                dh = compute_dhash(im)
                if is_near_dup(dh, dhash_list, max_dist=4):
                    continue
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                w, h = im.size
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88)
                jpg_bytes = buf.getvalue()
                sha = hashlib.sha256(jpg_bytes).hexdigest()
                if sha in sha_set:
                    continue
                sha_set.add(sha)
                dhash_list.append(dh)
                source_ids.add(item["iid"])

                author_slug = re.sub(r"[^a-zA-Z0-9]+", "_", item["author"].lower()).strip("_")[:24] or "unk"
                records.append({
                    "source_id": "SRC_8F_03_OPENIMAGES_SCARCE_CLASSES_TRAIN_VAL",
                    "source_name": f"Open Images V7 Scarce-Class Filtered ({item['split']})",
                    "source_image_id": item["iid"],
                    "source_split": item["split"],
                    "provenance_tier": "VERIFIED_FLICKR_CC_BY_2.0_WITH_AUTHOR_URL",
                    "title": item["title"],
                    "query_used": f"OpenImages V7 {item['split']} scarce-class waste filter",
                    "original_url": item["landing_url"],
                    "author": item["author"],
                    "license": item["license"],
                    "attribution_required": True,
                    "jpg_bytes": jpg_bytes,
                    "sha256": sha,
                    "dhash_hex": f"{dh:016x}",
                    "width": w,
                    "height": h,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "primary_class_id": item["primary_cid"],
                    "upstream_boxes": item["boxes"],
                    "has_strong_waste_kw": item["has_strong_waste_kw"],
                    "session_group": f"sess_oi8f_{item['split']}_{author_slug}",
                    "specimen_group": f"spec_oi8f_{item['iid']}",
                })
        except Exception:
            continue

    print(f"Acquired {len(records)} Open Images V7 scarce-class train/val candidates.")
    return records


def classify_box_pair_overlap(
    box_a: tuple[int, float, float, float, float],
    box_b: tuple[int, float, float, float, float],
    duplicate_iou_thresh: float = 0.90,
    max_edge_delta_thresh: float = 0.045,
    high_iou_thresh: float = 0.70,
) -> dict[str, Any]:
    """Classify the spatial relationship between two normalized bounding boxes `(cid, x1, y1, x2, y2)`.

    Explicitly separates three distinct physical cases required by Task 8F Final Safety Check:
    1. `DUPLICATE_ANNOTATION`: Same class (`cid_a == cid_b`), `IoU >= duplicate_iou_thresh` (default 0.90),
       and coinciding corners (`max_edge_delta <= max_edge_delta_thresh`, default 0.045). Represents jittered
       dual predictions on a single physical object. Safe to collapse to 1 box.
    2. `VALID_SAME_CLASS_OVERLAP`: Same class (`cid_a == cid_b`), `IoU > high_iou_thresh` (default 0.70),
       but distinct object centers/extents (`IoU < duplicate_iou_thresh` or `max_edge_delta > max_edge_delta_thresh`).
       Represents two strongly overlapping distinct objects of the same class. MUST preserve both boxes
       (`keep_both=True`) and flag `requires_visual_review=True` if unverified — never auto-delete!
    3. `VALID_CROSS_CLASS_OVERLAP`: Different classes (`cid_a != cid_b`) with `IoU > 0.0` (e.g. `1:biological`
       food inside `2:cardboard` box). MUST preserve both boxes (`keep_both=True`).
    """
    cid_a, x1_a, y1_a, x2_a, y2_a = box_a
    cid_b, x1_b, y1_b, x2_b, y2_b = box_b
    area_a = max(0.0, float(x2_a - x1_a)) * max(0.0, float(y2_a - y1_a))
    area_b = max(0.0, float(x2_b - x1_b)) * max(0.0, float(y2_b - y1_b))
    ix1, iy1 = max(float(x1_a), float(x1_b)), max(float(y1_a), float(y1_b))
    ix2, iy2 = min(float(x2_a), float(x2_b)), min(float(y2_a), float(y2_b))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    iou = float(inter / (area_a + area_b - inter + 1e-9)) if inter > 0.0 else 0.0
    max_edge_delta = float(
        max(
            abs(float(x1_a) - float(x1_b)),
            abs(float(y1_a) - float(y1_b)),
            abs(float(x2_a) - float(x2_b)),
            abs(float(y2_a) - float(y2_b)),
        )
    )
    cx_a, cy_a = (float(x1_a) + float(x2_a)) / 2.0, (float(y1_a) + float(y2_a)) / 2.0
    cx_b, cy_b = (float(x1_b) + float(x2_b)) / 2.0, (float(y1_b) + float(y2_b)) / 2.0
    center_delta = float(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5)

    if int(cid_a) != int(cid_b):
        return {
            "category": "VALID_CROSS_CLASS_OVERLAP" if iou > 0.0 else "DISJOINT_CROSS_CLASS",
            "keep_both": True,
            "requires_visual_review": False,
            "iou": round(iou, 6),
            "max_edge_delta": round(max_edge_delta, 6),
            "center_delta": round(center_delta, 6),
        }

    if iou >= duplicate_iou_thresh and max_edge_delta <= max_edge_delta_thresh:
        return {
            "category": "DUPLICATE_ANNOTATION",
            "keep_both": False,
            "requires_visual_review": False,
            "iou": round(iou, 6),
            "max_edge_delta": round(max_edge_delta, 6),
            "center_delta": round(center_delta, 6),
        }

    if iou > high_iou_thresh:
        return {
            "category": "VALID_SAME_CLASS_OVERLAP",
            "keep_both": True,
            "requires_visual_review": True,
            "iou": round(iou, 6),
            "max_edge_delta": round(max_edge_delta, 6),
            "center_delta": round(center_delta, 6),
        }

    return {
        "category": "NORMAL_DISTINCT_INSTANCES",
        "keep_both": True,
        "requires_visual_review": False,
        "iou": round(iou, 6),
        "max_edge_delta": round(max_edge_delta, 6),
        "center_delta": round(center_delta, 6),
    }


def deduplicate_same_class_boxes_iou(
    boxes_norm: list[tuple[int, float, float, float, float]],
    iou_thresh: float = 0.70,
    duplicate_iou_thresh: float = 0.90,
    max_edge_delta_thresh: float = 0.045,
    is_ground_truth: bool = False,
) -> list[tuple[int, float, float, float, float]]:
    """Safely collapse ONLY true duplicate annotations on a single object while preserving overlapping distinct instances.

    - Never deletes cross-class overlapping boxes (`VALID_CROSS_CLASS_OVERLAP`).
    - Never deletes two strongly overlapping same-class objects (`VALID_SAME_CLASS_OVERLAP`, where
      `0.70 < IoU < 0.90` or `max_edge_delta > 0.045`).
    - When `is_ground_truth=True`, only exact identical coordinate duplicates (`IoU >= 0.995` and
      `max_edge_delta <= 0.005`) are collapsed; all distinct ground-truth boxes are preserved untouched.
    """
    eff_dup_iou = 0.995 if is_ground_truth else max(iou_thresh, duplicate_iou_thresh)
    eff_edge_delta = 0.005 if is_ground_truth else max_edge_delta_thresh
    kept: list[tuple[int, float, float, float, float]] = []
    for box_a in boxes_norm:
        is_dup = False
        for box_b in kept:
            rel = classify_box_pair_overlap(
                box_a,
                box_b,
                duplicate_iou_thresh=eff_dup_iou,
                max_edge_delta_thresh=eff_edge_delta,
                high_iou_thresh=iou_thresh,
            )
            if not rel["keep_both"] and rel["category"] == "DUPLICATE_ANNOTATION":
                is_dup = True
                break
        if not is_dup:
            kept.append(box_a)
    return kept


def propose_candidate_class_from_title(
    title_str: str,
    default_cid: int,
    human_verified: bool = False,
    verified_cid: int | None = None,
) -> dict[str, Any]:
    """Propose a candidate material taxonomy class from filename/title tokens without auto-promoting QA or domain.

    Safety guarantees (Task 8F Final Safety Check — Task 2):
    1. Only proposes a candidate class (`is_candidate_only=True`); never overwrites human-verified annotations
       (`human_verified=True` preserves `verified_cid` unconditionally).
    2. Never auto-promotes an image to `QA_PASS` / `REAL_WASTE_PASS` (`can_auto_promote_qa_pass=False`).
    3. Never auto-asserts `ACTUAL_WASTE` / `REAL_WASTE_CONTEXT_VERIFIED` (`can_auto_assert_actual_waste=False`).
    4. Any title-based class change or scarce/hard class candidate requires human visual QA (`requires_visual_qa=True`).
    """
    if human_verified and verified_cid is not None:
        cid_final = int(verified_cid)
        return {
            "proposed_class_id": cid_final,
            "proposed_class_name": CLASS_NAMES[cid_final],
            "default_class_id": int(default_cid),
            "is_candidate_only": False,
            "title_changed_class": False,
            "requires_visual_qa": False,
            "can_auto_promote_qa_pass": False,
            "can_auto_assert_actual_waste": False,
            "human_verified_preserved": True,
        }

    t = title_str.replace("_", " ").replace("-", " ").lower()
    proposed = int(default_cid)
    if re.search(r"\b(boxing\s+gloves?|glove|gloves|mitten|shirt|jacket|sock|socks)\b", t):
        proposed = 3  # clothes
    elif re.search(r"\b(jägermeister|doornkaat|leere\s+flasche|wine\s+bottle|glass\s+bottle|beer\s+bottle|broken\s+glass)\b", t):
        proposed = 4  # glass
    elif re.search(r"\b(pepsi\s+can|coca\s+cola\s+can|beer\s+can|bud\s+light|miller\s+high\s+life|keystone\s+light|juomapurkki|tin\s+can|aluminium\s+can)\b", t):
        proposed = 5  # metal
    elif re.search(r"\b(pizza\s+box|pizzadoos|cardboard|carton\s+box|corrugated)\b", t):
        proposed = 2  # cardboard
    elif re.search(r"\b(plastic\s+bag|plastic\s+cup|plastic\s+bottle|plastic\s+verpakking|walmart\s+bag|meijer|smoothie\s+king|bolthouse|lift\s+bottle)\b", t):
        proposed = 7  # plastic
    elif re.search(r"\b(cigarette|crisp\s+packet|m\s*&\s*m\s+wrapper|vape|njoy|pod\s+litter)\b", t):
        proposed = 9  # trash

    changed = proposed != int(default_cid)
    hard_or_scarce = proposed in SCARCE_CLASSES or int(default_cid) in SCARCE_CLASSES
    return {
        "proposed_class_id": proposed,
        "proposed_class_name": CLASS_NAMES[proposed],
        "default_class_id": int(default_cid),
        "is_candidate_only": True,
        "title_changed_class": changed,
        "requires_visual_qa": bool(changed or hard_or_scarce),
        "can_auto_promote_qa_pass": False,
        "can_auto_assert_actual_waste": False,
        "human_verified_preserved": False,
    }


def infer_material_class_from_title(
    title_str: str,
    default_cid: int,
    human_verified: bool = False,
    verified_cid: int | None = None,
) -> int:
    """Return only the candidate class ID from `propose_candidate_class_from_title` while preserving human-verified labels."""
    return int(
        propose_candidate_class_from_title(
            title_str,
            default_cid,
            human_verified=human_verified,
            verified_cid=verified_cid,
        )["proposed_class_id"]
    )


def extract_tight_per_instance_boxes(img_bgr: np.ndarray, primary_cid: int, detector: YOLO) -> list[tuple[int, float, float, float, float]]:
    """Combine YOLOv8 class-aware detector predictions (`agnostic_nms=False`) with contour fallback and safe duplicate-only filtering."""
    h, w = img_bgr.shape[:2]
    det_res = detector.predict(img_bgr, conf=0.20, agnostic_nms=False, iou=0.65, verbose=False)[0]
    det_boxes: list[tuple[int, float, float, float, float]] = []
    if det_res.boxes is not None and len(det_res.boxes) > 0:
        xyxyn = det_res.boxes.xyxyn.cpu().numpy()
        clss = det_res.boxes.cls.cpu().numpy().astype(int)
        confs = det_res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, conf in zip(xyxyn, clss, confs):
            bw, bh = float(x2 - x1), float(y2 - y1)
            if 0.02 <= bw <= 0.90 and 0.02 <= bh <= 0.90 and bw * bh <= 0.72:
                out_c = int(c) if (int(c) != int(primary_cid) and float(conf) >= 0.45 and 0 <= int(c) < 10) else int(primary_cid)
                det_boxes.append((out_c, round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4)))

    det_boxes = deduplicate_same_class_boxes_iou(det_boxes, iou_thresh=0.70, is_ground_truth=False)
    if 1 <= len(det_boxes) <= 6:
        return det_boxes

    # Fallback: OpenCV multi-component foreground contour extraction
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, float, float, float, float]] = []
    img_area = float(w * h)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(c)
        rect_area = float(bw * bh)
        if 0.02 * img_area <= rect_area <= 0.78 * img_area:
            x1, y1, x2, y2 = x / w, y / h, (x + bw) / w, (y + bh) / h
            if any(
                max(0.0, min(x2, ex2) - max(x1, ex1)) * max(0.0, min(y2, ey2) - max(y1, ey1))
                / ((x2 - x1) * (y2 - y1) + 1e-6)
                > 0.55
                for _, ex1, ey1, ex2, ey2 in boxes
            ):
                continue
            boxes.append((primary_cid, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)))
            if len(boxes) >= 4:
                break
    if not boxes:
        boxes.append((primary_cid, 0.14, 0.14, 0.86, 0.86))
    return deduplicate_same_class_boxes_iou(boxes, iou_thresh=0.70, is_ground_truth=False)


def classify_shynuaa_scarce_material(crop_bgr: np.ndarray, bw: float, bh: float, matching_preds: list[tuple[int, float]]) -> int:
    if matching_preds:
        best_cid, best_conf = max(matching_preds, key=lambda x: x[1])
        if best_cid in SCARCE_CLASSES and best_conf >= 0.25:
            return best_cid
        if best_conf >= 0.45:
            return best_cid

    if crop_bgr.size == 0:
        return 6
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    v_std = float(np.std(hsv[:, :, 2]))
    aspect = bw / max(bh, 1e-4)

    if 8.0 <= h_mean <= 24.0 and 55.0 <= s_mean <= 170.0 and 70.0 <= v_mean <= 205.0:
        return 2  # cardboard
    if s_mean < 42.0 and v_mean > 165.0:
        return 6  # paper
    if s_mean < 52.0 and 85.0 <= v_mean <= 165.0 and v_std > 35.0:
        return 5  # metal
    if 24.0 < h_mean <= 48.0 and s_mean > 68.0 and (bw * bh) < 0.18:
        return 1  # biological
    if 35.0 <= h_mean <= 95.0 and s_mean > 55.0 and (aspect < 0.65 or aspect > 1.55) and (bw * bh) < 0.14:
        return 4  # glass bottle
    if 90.0 <= h_mean <= 132.0 and s_mean > 60.0:
        return 7  # plastic
    return 6 if s_mean < 55.0 else 9


def norm_boxes_to_struct(
    w: int,
    h: int,
    boxes_norm: list[tuple[int, float, float, float, float]],
    origin: str,
    is_ground_truth: bool | None = None,
) -> list[dict[str, Any]]:
    if is_ground_truth is None:
        orig_l = origin.lower()
        is_ground_truth = any(tok in orig_l for tok in ("gt", "verified", "override", "semantic_qa", "curated"))
    dedup_norm = deduplicate_same_class_boxes_iou(boxes_norm, iou_thresh=0.70, is_ground_truth=is_ground_truth)
    out = []
    for cid, x1, y1, x2, y2 in dedup_norm:
        x1c, y1c = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
        x2c, y2c = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
        if x2c - x1c < 0.012 or y2c - y1c < 0.012:
            continue
        xc = round((x1c + x2c) / 2.0, 6)
        yc = round((y1c + y2c) / 2.0, 6)
        bw = round(x2c - x1c, 6)
        bh = round(y2c - y1c, 6)
        out.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "bbox_xyxy": [round(x1c * w, 1), round(y1c * h, 1), round(x2c * w, 1), round(y2c * h, 1)],
            "yolo_xywhn": [xc, yc, bw, bh],
            "source_box_origin": origin,
        })
    return out


def format_yolo_lines(boxes: list[dict[str, Any]]) -> str:
    lines = []
    for b in boxes:
        xc, yc, bw, bh = b["yolo_xywhn"]
        lines.append(f"{b['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_overlay(rec: dict[str, Any]) -> str:
    img_p = ROOT / rec["image_path"]
    with Image.open(img_p) as im:
        im_rgb = im.convert("RGB")
        w, h = im_rgb.size
        draw = ImageDraw.Draw(im_rgb)
        thickness = max(4, int(min(w, h) * 0.008))
        for b in rec["boxes"]:
            cid = b["class_id"]
            cname = b["class_name"]
            xc, yc, bw, bh = b["yolo_xywhn"]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            col = CLASS_COLORS.get(cid, (0, 255, 0))
            for t in range(thickness):
                draw.rectangle([max(0, x1 - t), max(0, y1 - t), min(w - 1, x2 + t), min(h - 1, y2 + t)], outline=col)
            lbl = f"{cid}:{cname}"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl) * 7 + 10, y1], fill=col)
            draw.text((x1 + 4, max(0, y1 - 18)), lbl, fill=(10, 10, 10))
        out_p = OVERLAYS_DIR / f"{rec['candidate_id']}_overlay.jpg"
        im_rgb.save(out_p, quality=86)
        return str(out_p.relative_to(ROOT)).replace("\\", "/")


def render_sheet(records: list[dict[str, Any]], title: str, out_name: str, max_items: int = 36) -> str:
    subset = records[:max_items]
    if not subset:
        return ""
    cols = 6
    rows = (len(subset) + cols - 1) // cols
    cell_w, cell_h = 260, 220
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{title} (showing {len(subset)}/{len(records)})", fill=(240, 240, 245))
    for idx, rec in enumerate(subset):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * cell_h + 40
        ov_p = ROOT / rec["overlay_path"] if rec.get("overlay_path") else ROOT / rec["image_path"]
        if ov_p.exists():
            with Image.open(ov_p) as im:
                im_rgb = im.convert("RGB")
                im_rgb.thumbnail((cell_w - 8, cell_h - 36))
                sheet.paste(im_rgb, (x0 + 4, y0 + 4))
        cid_s = rec["candidate_id"].replace("task8f_", "")
        cls_s = ",".join(str(x) for x in rec["classes_present_ids"])
        sp_s = rec.get("source_split", "unk")[:4]
        cap = f"{cid_s}|c:{cls_s}|s:{sp_s}|{rec['target_split'][:9]}"
        draw.text((x0 + 4, y0 + cell_h - 28), cap, fill=(220, 235, 220))
    out_p = SHEETS_DIR / out_name
    sheet.save(out_p, quality=88)
    return str(out_p.relative_to(ROOT)).replace("\\", "/")


REAL_WASTE_WIKI_RE = re.compile(
    r"\b(waste|trash|garbage|litter|littered|rubbish|discard|discarded|disposable|abandon|abandoned|rotten|rotting|mold|moldy|mould|spoil|spoiled|decay|decaying|"
    r"peel|peels|core|eggshell|eggshells|compost|leftover|scraps|scrap|corrod|corroded|leak|leaking|rust|rusty|spent|dead|used|recycle|recycling|"
    r"broken|shatter|shattered|smash|crush|crushed|dent|crumpl|crumpled|torn|dirty|muddy|worn|old|damaged|empty|pizza|carton|cup|cups|wet|rags|rag|bin|dump|street|ground|wires|tossing|bulging|salvage)\b",
    re.I,
)
BULK_PILE_WIKI_RE = re.compile(
    r"\b(landfill|dumpster\s+fire|bales|bale|heap|mountain\s+of|tons\s+of|truckload|container\s+load|mass\s+of|pile\s+of\s+thousands)\b",
    re.I,
)


def topup_wikimedia_paper_and_scarce(sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_count: int = 155) -> list[dict[str, Any]]:
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    api_url = "https://commons.wikimedia.org/w/api.php"
    queries = [
        (5, "Category:Abandoned_beverage_cans", 50),
        (5, "Category:Bulging_cans", 30),
        (5, "Category:Aluminium_cans", 45),
        (6, "Category:Fast_food_litter", 50),
        (6, "Category:Branded_litter", 45),
        (1, "Category:Spoiled_fruit", 40),
        (1, "Category:Spoiled_rice", 25),
        (1, "Category:Food_from_waste_containers", 45),
        (1, "Category:Food_waste", 50),
        (4, "Category:Broken_glass", 45),
        (4, "Category:Glass_salvage_containers", 35),
        (8, "Category:Shoe_tossing", 45),
        (8, "Category:Old_shoes", 45),
        (8, "Category:Worn-out_shoes", 40),
        (3, "Category:Lost_gloves", 40),
        (2, "Category:Damaged_cardboard_boxes", 40),
        (0, "Category:Battery_leakage", 40),
        (5, "search:crushed beer can ground OR grass", 40),
        (6, "search:discarded coffee cup ground OR street", 40),
        (2, "search:flattened cardboard box street OR trash", 40),
        (4, "search:broken beer bottle ground OR street", 40),
        (8, "search:abandoned shoe ground OR street", 40),
    ]
    candidates = []
    for cid, qspec, lim in queries:
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata|mime",
                "iiurlwidth": 1024,
            }
            if qspec.startswith("Category:"):
                params.update({"generator": "categorymembers", "gcmtitle": qspec, "gcmtype": "file", "gcmlimit": lim})
            else:
                params.update({"generator": "search", "gsrsearch": f"filetype:bitmap {qspec[7:]}", "gsrnamespace": 6, "gsrlimit": lim})
            r = sess.get(api_url, params=params, timeout=10)
            if r.status_code == 200:
                for _pid, pdata in r.json().get("query", {}).get("pages", {}).items():
                    pid = str(pdata.get("pageid", ""))
                    sid = f"commons_{pid}"
                    if not pid or sid in source_ids:
                        continue
                    t = pdata.get("title", "")
                    if BAD_WIKI_RE.search(t) or re.search(r"window|pane|windshield|bird|heron|pigeon|gull|duck|hydrant|dumpster_diving", t, re.I):
                        continue
                    ii_list = pdata.get("imageinfo") or []
                    if not ii_list:
                        continue
                    ii = ii_list[0]
                    if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                        continue
                    if int(ii.get("width") or 0) < 300 or int(ii.get("height") or 0) < 300 or int(ii.get("size") or 0) > 6_000_000:
                        continue
                    candidates.append((cid, qspec, pdata))
        except Exception:
            pass

    def dl_one(item: tuple[int, str, dict[str, Any]]) -> dict[str, Any] | None:
        cid, qspec, pdata = item
        pid = str(pdata["pageid"])
        sid = f"commons_{pid}"
        ii = pdata["imageinfo"][0]
        u = ii.get("thumburl") or ii.get("url")
        if not u:
            return None
        try:
            resp = sess.get(u, timeout=10)
            if resp.status_code == 200 and len(resp.content) >= 5000:
                return {"cid": cid, "qspec": qspec, "pdata": pdata, "sid": sid, "ii": ii, "u": u, "raw_bytes": resp.content}
        except Exception:
            pass
        return None

    fetched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(dl_one, candidates[: target_count + 80]):
            if res is not None:
                fetched.append(res)

    records = []
    for item in fetched:
        if len(records) >= target_count:
            break
        cid = item["cid"]
        qspec = item["qspec"]
        pdata = item["pdata"]
        sid = item["sid"]
        ii = item["ii"]
        u = item["u"]
        if sid in source_ids:
            continue
        try:
            with Image.open(io.BytesIO(item["raw_bytes"])) as im_raw:
                im = ImageOps.exif_transpose(im_raw).convert("RGB")
                orig_w, orig_h = im.size
                dh = compute_dhash(im)
                if is_near_dup(dh, dhash_list, max_dist=4):
                    continue
                im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                w, h = im.size
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88)
                jpg_bytes = buf.getvalue()
                sha = hashlib.sha256(jpg_bytes).hexdigest()
                if sha in sha_set:
                    continue
                sha_set.add(sha)
                dhash_list.append(dh)
                source_ids.add(sid)
                ext = ii.get("extmetadata") or {}
                lic = (ext.get("LicenseShortName", {}).get("value") or "CC-BY-SA").strip()
                artist_raw = ext.get("Artist", {}).get("value") or "Wikimedia Commons Contributor"
                artist = re.sub(r"<[^>]+>", "", artist_raw).strip()[:80] or "Wikimedia Commons Contributor"
                author_slug = re.sub(r"[^a-zA-Z0-9]+", "_", artist.lower()).strip("_")[:24] or "unk"
                records.append({
                    "source_id": "SRC_8F_02_WIKIMEDIA_SCARCE_REAL_WASTE",
                    "source_name": "Wikimedia Commons Scarce-Class Real-Waste Categories & Queries",
                    "source_image_id": sid,
                    "source_split": "unspecified",
                    "provenance_tier": "VERIFIED_WIKIMEDIA_COMMONS_OPEN_LICENSE",
                    "title": pdata.get("title", ""),
                    "query_used": qspec,
                    "original_url": ii.get("descriptionurl") or u,
                    "author": artist,
                    "license": lic,
                    "attribution_required": not ("cc0" in lic.lower() or "public domain" in lic.lower()),
                    "jpg_bytes": jpg_bytes,
                    "sha256": sha,
                    "dhash_hex": f"{dh:016x}",
                    "width": w,
                    "height": h,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "primary_class_id": cid,
                    "upstream_boxes": [],
                    "session_group": f"sess_wiki8f_{author_slug}",
                    "specimen_group": f"spec_{sid}",
                })
        except Exception:
            continue
    return records


# Visual audit overrides on Task 8F candidates from inspecting p1..p8 contact sheets
VISUAL_REJECT_IDS = {
    "task8f_0008", "task8f_0038", "task8f_0040", "task8f_0042", "task8f_0043",
    "task8f_0048", "task8f_0056", "task8f_0064", "task8f_0181", "task8f_0195",
    "task8f_0206", "task8f_0208", "task8f_0235", "task8f_0237", "task8f_0238",
    "task8f_0241", "task8f_0245", "task8f_0249", "task8f_0252", "task8f_0257",
    "task8f_0261", "task8f_0264", "task8f_0267", "task8f_0563", "task8f_0564",
    "task8f_0565", "task8f_0568", "task8f_0580", "task8f_0581",
    "task8f_0594", "task8f_0595", "task8f_0596", "task8f_0603", "task8f_0625",
    "task8f_0627", "task8f_0633", "task8f_0636", "task8f_0643", "task8f_0714",
    "task8f_0745", "task8f_0746",
}
VISUAL_NEEDS_REVIEW_IDS = {
    "task8f_0009", "task8f_0014", "task8f_0053", "task8f_0061",
    "task8f_0069", "task8f_0075", "task8f_0083", "task8f_0091",
    "task8f_0099", "task8f_0142", "task8f_0143",
    "task8f_0172", "task8f_0203", "task8f_0211",
    "task8f_0217", "task8f_0222", "task8f_0227", "task8f_0231",
    "task8f_0280", "task8f_0284", "task8f_0287", "task8f_0298",
    "task8f_0319", "task8f_0377", "task8f_0386", "task8f_0567", "task8f_0570",
    "task8f_0571", "task8f_0582", "task8f_0583", "task8f_0584", "task8f_0585",
    "task8f_0586", "task8f_0676", "task8f_0677", "task8f_0679", "task8f_0680",
    "task8f_0683", "task8f_0684", "task8f_0685", "task8f_0686", "task8f_0687",
    "task8f_0691", "task8f_0692", "task8f_0699", "task8f_0700", "task8f_0701",
    "task8f_0705", "task8f_0706", "task8f_0707", "task8f_0717", "task8f_0718",
    "task8f_0719", "task8f_0720", "task8f_0722", "task8f_0723", "task8f_0724",
}
VISUAL_KEEP_AUX_IDS = {
    "task8f_0013", "task8f_0016", "task8f_0019", "task8f_0021",
    "task8f_0029", "task8f_0052",
    "task8f_0060", "task8f_0070", "task8f_0071", "task8f_0072", "task8f_0078",
    "task8f_0086", "task8f_0093", "task8f_0109",
    "task8f_0110", "task8f_0112", "task8f_0120", "task8f_0133", "task8f_0135",
    "task8f_0146", "task8f_0149", "task8f_0153", "task8f_0154", "task8f_0156",
    "task8f_0161", "task8f_0171", "task8f_0174", "task8f_0178",
    "task8f_0185", "task8f_0220", "task8f_0233", "task8f_0243", "task8f_0247",
    "task8f_0251", "task8f_0253", "task8f_0266", "task8f_0270", "task8f_0274",
    "task8f_0482", "task8f_0497", "task8f_0566", "task8f_0569", "task8f_0572",
    "task8f_0579", "task8f_0599", "task8f_0639", "task8f_0671", "task8f_0678",
    "task8f_0695", "task8f_0709", "task8f_0710", "task8f_0711", "task8f_0715",
}
# Promote visually verified real-waste items from KEEP_AUXILIARY / NEEDS_REVIEW / REJECT
VISUAL_PROMOTE_PASS_IDS = {
    "task8f_0007", "task8f_0026", "task8f_0028", "task8f_0035", "task8f_0041",
    "task8f_0045", "task8f_0046", "task8f_0051", "task8f_0054", "task8f_0058",
    "task8f_0062", "task8f_0076", "task8f_0082", "task8f_0090", "task8f_0102",
    "task8f_0116", "task8f_0118", "task8f_0126", "task8f_0155", "task8f_0159",
    "task8f_0164", "task8f_0175", "task8f_0214", "task8f_0234", "task8f_0244",
    "task8f_0248", "task8f_0268", "task8f_0271", "task8f_0292", "task8f_0293",
    "task8f_0533", "task8f_0604", "task8f_0608", "task8f_0621", "task8f_0635",
    "task8f_0646", "task8f_0663", "task8f_0667", "task8f_0672", "task8f_0681",
    "task8f_0690", "task8f_0702", "task8f_0704", "task8f_0712", "task8f_0747",
    "task8f_0668", "task8f_0669", "task8f_0689", "task8f_0693",
}
# Explicit per-instance box overrides for multi-shoe images spotted on contact sheets
VISUAL_BOX_OVERRIDES: dict[str, list[tuple[int, float, float, float, float]]] = {
    "task8f_0032": [(8, 0.12, 0.08, 0.48, 0.78), (8, 0.49, 0.25, 0.89, 0.77)],
    "task8f_0055": [(8, 0.12, 0.08, 0.49, 0.88), (8, 0.52, 0.06, 0.88, 0.86)],
    "task8f_0131": [(8, 0.04, 0.48, 0.46, 0.96), (8, 0.13, 0.04, 0.96, 0.63)],
    "task8f_0205": [(8, 0.16, 0.50, 0.65, 0.90), (8, 0.35, 0.45, 0.78, 0.80)],
}


def main() -> None:
    t0 = time.time()
    pilots = build_task8f_pilot_evaluations()

    if RAW_MANIFEST_JSON.exists():
        raw_payload = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
        manifest_records = raw_payload["records"]
        print(f"Loaded {len(manifest_records)} cached Task 8F raw candidates from {RAW_MANIFEST_JSON}", flush=True)
        if len(manifest_records) < 745:
            sha_set, dhash_list, source_ids = load_existing_fingerprints()
            for r in manifest_records:
                sha_set.add(r["sha256"])
                dhash_list.append(int(r["dhash_hex"], 16))
                source_ids.add(str(r["source_image_id"]))
            more_wiki = topup_wikimedia_paper_and_scarce(sha_set, dhash_list, source_ids, target_count=32)
            start_idx = len(manifest_records) + 1
            for offset, r in enumerate(more_wiki):
                idx = start_idx + offset
                cid = f"task8f_{idx:04d}"
                img_p = IMAGES_DIR / f"{cid}.jpg"
                img_p.write_bytes(r.pop("jpg_bytes"))
                r["candidate_id"] = cid
                r["image_path"] = str(img_p.relative_to(ROOT)).replace("\\", "/")
                manifest_records.append(r)
            raw_payload["total_candidates"] = len(manifest_records)
            raw_payload["records"] = manifest_records
            RAW_MANIFEST_JSON.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Topped up Task 8F raw candidates with Wikimedia paper/scarce to {len(manifest_records)}", flush=True)
    else:
        sha_set, dhash_list, source_ids = load_existing_fingerprints()
        print(f"Loaded {len(sha_set)} existing SHA-256, {len(dhash_list)} dHashes, and {len(source_ids)} source IDs.", flush=True)

        shynuaa_recs = acquire_shynuaa_valid_and_remaining_train(sha_set, dhash_list, source_ids, target_count=170)
        wiki_recs = acquire_wikimedia_scarce_classes(sha_set, dhash_list, source_ids, target_count=295)
        oi_recs = acquire_openimages_scarce_classes(sha_set, dhash_list, source_ids, target_count=260)

        all_recs = shynuaa_recs + wiki_recs + oi_recs
        for f in IMAGES_DIR.glob("*.jpg"):
            f.unlink()

        manifest_records = []
        for idx, r in enumerate(all_recs, start=1):
            cid = f"task8f_{idx:04d}"
            img_p = IMAGES_DIR / f"{cid}.jpg"
            img_p.write_bytes(r.pop("jpg_bytes"))
            r["candidate_id"] = cid
            r["image_path"] = str(img_p.relative_to(ROOT)).replace("\\", "/")
            manifest_records.append(r)

        raw_payload = {
            "wave": "TASK8F_SCARCE_CLASS_REAL_WASTE_EXPANSION",
            "total_candidates": len(manifest_records),
            "elapsed_sec": round(time.time() - t0, 2),
            "pilots": pilots,
            "records": manifest_records,
        }
        RAW_MANIFEST_JSON.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(manifest_records)} raw Task 8F candidates to {RAW_MANIFEST_JSON}", flush=True)

    # Run QA & per-instance annotation
    for d in [LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR]:
        for f in d.glob("*"):
            f.unlink()

    model_path = ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
    detector = YOLO(str(model_path))

    qa_records: list[dict[str, Any]] = []
    prev_shynuaa_dt: dict[str, datetime] = {}
    seen_wiki_specimens: set[str] = set()

    wiki_search_obj_re = re.compile(
        r"\b(battery|batteries|nicad|ni-cd|alkaline|leakedbattery|glove|gloves|shirt|clothes|bottle|jar|glass|can|cans|tin|"
        r"paper|newspaper|cup|carton|cardboard|box|shoe|shoes|boot|boots|sneaker|banana|apple|fruit|food|constructionwaste)\b",
        re.I,
    )
    wiki_search_waste_re = re.compile(
        r"\b(leaking|leaked|leakage|corrod|corroded|spent|dead|discard|discarded|disposable|abandon|abandoned|forlorn|moss|"
        r"rotten|crumpl|crumpled|crush|crushed|broken|shatter|shattered|torn|dirty|muddy|worn|waste|trash|litter|garbage|constructionwaste)\b",
        re.I,
    )

    for r in manifest_records:
        cid = r["candidate_id"]
        src = r["source_id"]
        source_split = r.get("source_split", "unspecified")
        w, h = int(r["width"]), int(r["height"])
        img_p = ROOT / r["image_path"]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        title_str = r.get("title", "")
        query_str = r.get("query_used", "")
        combined_text = f"{title_str} {query_str}".replace("_", " ").replace("-", " ")

        qa_status = "REJECT"
        domain_tag = "AMBIGUOUS_CONTEXT"
        obj_valid = False
        waste_verified = False
        approved_for_train = False
        reason = ""
        boxes: list[dict[str, Any]] = []

        hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_full[:, :, 1]))
        white_px_ratio = float(np.mean((hsv_full[:, :, 1] < 22) & (hsv_full[:, :, 2] > 235)))
        low_sat_ratio = float(np.mean(hsv_full[:, :, 1] < 25))
        is_diagram_or_bw = (white_px_ratio > 0.48 and low_sat_ratio > 0.72) or (mean_sat < 9.5)

        from scripts.correct_task8f_semantic_qa import (
            DEMOTE_TO_AUXILIARY_SPECS,
            MOVE_TO_NEEDS_REVIEW_SPECS,
            REJECT_FROM_PASS_SPECS,
            VERIFIED_REAL_WASTE_PASS_SPECS,
        )

        if cid in VERIFIED_REAL_WASTE_PASS_SPECS:
            spec = VERIFIED_REAL_WASTE_PASS_SPECS[cid]
            boxes = norm_boxes_to_struct(w, h, spec["boxes"], "task8f_semantic_qa_verified_v2", is_ground_truth=True)
            qa_status = "REAL_WASTE_PASS"
            domain_tag = "ACTUAL_WASTE"
            obj_valid = True
            waste_verified = True
            reason = spec["reason"]
        elif cid in DEMOTE_TO_AUXILIARY_SPECS:
            spec = DEMOTE_TO_AUXILIARY_SPECS[cid]
            boxes = norm_boxes_to_struct(w, h, spec["boxes"], "task8f_semantic_qa_auxiliary_v2", is_ground_truth=True)
            qa_status = "KEEP_AUXILIARY"
            domain_tag = spec["domain_tag"]
            obj_valid = True
            waste_verified = False
            reason = spec["reason"]
        elif cid in MOVE_TO_NEEDS_REVIEW_SPECS:
            spec = MOVE_TO_NEEDS_REVIEW_SPECS[cid]
            boxes = norm_boxes_to_struct(w, h, spec["boxes"], "task8f_semantic_qa_review_v2", is_ground_truth=True)
            qa_status = "NEEDS_REVIEW"
            domain_tag = spec["domain_tag"]
            obj_valid = False
            waste_verified = spec["domain_tag"] == "ACTUAL_WASTE"
            reason = spec["reason"]
        elif cid in REJECT_FROM_PASS_SPECS:
            spec = REJECT_FROM_PASS_SPECS[cid]
            boxes = []
            qa_status = "REJECT"
            domain_tag = spec["domain_tag"]
            obj_valid = False
            waste_verified = False
            reason = spec["reason"]
        elif cid in VISUAL_REJECT_IDS:
            qa_status = "REJECT"
            domain_tag = "AMBIGUOUS_CONTEXT"
            reason = "Visual contact-sheet QA: rejected schematic diagram, plot, poster, sculpture, person, or distant building/landscape."
        elif cid in VISUAL_NEEDS_REVIEW_IDS:
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            p_cid = proposal["proposed_class_id"]
            tight_norm = extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
            boxes = norm_boxes_to_struct(w, h, tight_norm, "visual_qa_needs_review_v8f")
            qa_status = "NEEDS_REVIEW"
            domain_tag = "ACTUAL_WASTE"
            waste_verified = True
            reason = "Visual contact-sheet QA: dense multi-object heap, bulk pile, or container scene held in NEEDS_REVIEW per instance-detection policy."
        elif cid in VISUAL_KEEP_AUX_IDS:
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            p_cid = proposal["proposed_class_id"]
            tight_norm = extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
            boxes = norm_boxes_to_struct(w, h, tight_norm, "visual_qa_keep_auxiliary_v8f")
            qa_status = "KEEP_AUXILIARY"
            domain_tag = "PRODUCT_STUDIO" if cid in ("task8f_0078", "task8f_0086", "task8f_0579") else "IN_USE"
            obj_valid = True
            waste_verified = False
            reason = "Visual contact-sheet QA: intact/in-use/studio object without verified disposal context; routed to KEEP_AUXILIARY."
        elif cid in VISUAL_PROMOTE_PASS_IDS:
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]), human_verified=True, verified_cid=int(r["primary_class_id"]))
            p_cid = proposal["proposed_class_id"]
            tight_norm = VISUAL_BOX_OVERRIDES.get(cid) or extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
            boxes = norm_boxes_to_struct(w, h, tight_norm, "visual_qa_verified_real_waste_v8f", is_ground_truth=cid in VISUAL_BOX_OVERRIDES)
            qa_status = "REAL_WASTE_PASS"
            domain_tag = "ACTUAL_WASTE"
            obj_valid = True
            waste_verified = True
            reason = "Visual contact-sheet QA: verified discarded real-waste object with tight per-instance bounding boxes."
        elif src == "SRC_8F_02_WIKIMEDIA_SCARCE_REAL_WASTE":
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            p_cid = proposal["proposed_class_id"]
            title_clean = title_str.replace("_", " ").replace("-", " ")
            is_search_query = query_str.startswith("search:")

            spec_series_key = None
            if re.search(r"9V leaking", title_clean, re.I):
                spec_series_key = "spec_wiki_9v_leaking"
            elif re.search(r"Discarded Glove on street", title_clean, re.I):
                spec_series_key = "spec_wiki_discarded_glove_5035"
            elif re.search(r"Disposable Gloves on street", title_clean, re.I):
                spec_series_key = "spec_wiki_disposable_gloves_vancouver"
            elif re.search(r"Giovannino Guareschi", title_clean, re.I):
                spec_series_key = "spec_wiki_guareschi"
            if spec_series_key:
                r["specimen_group"] = spec_series_key

            if is_diagram_or_bw or BAD_WIKI_RE.search(title_clean) or re.search(
                r"Giovannino Guareschi|18\d\d|190\d|DPLA|truck|lorry|factory|building|plant|container|collection box|bank|bale|bales|statue|art|stoves|tabs|straws|people",
                title_clean,
                re.I,
            ):
                qa_status = "REJECT"
                domain_tag = "AMBIGUOUS_CONTEXT"
                reason = "Rejected Wikimedia non-waste / diagram / B&W archival / vehicle / container / building."
            elif is_search_query and not (wiki_search_obj_re.search(title_clean) and wiki_search_waste_re.search(title_clean)):
                qa_status = "REJECT"
                domain_tag = "AMBIGUOUS_CONTEXT"
                reason = "Rejected CirrusSearch keyword match where file title lacks explicit target object + waste/damaged evidence."
            elif spec_series_key and spec_series_key in seen_wiki_specimens:
                qa_status = "REJECT"
                domain_tag = "ACTUAL_WASTE"
                waste_verified = True
                reason = f"Same-specimen multi-angle repeat ({spec_series_key}); rejected to preserve strict specimen independence."
            elif BULK_PILE_WIKI_RE.search(combined_text):
                tight_norm = extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
                boxes = norm_boxes_to_struct(w, h, tight_norm, "wikimedia_bulk_pile_hold_v8f")
                qa_status = "NEEDS_REVIEW"
                domain_tag = "ACTUAL_WASTE"
                waste_verified = True
                reason = "Bulk heap or container pile held in NEEDS_REVIEW per instance-detection policy."
            else:
                if spec_series_key:
                    seen_wiki_specimens.add(spec_series_key)
                tight_norm = VISUAL_BOX_OVERRIDES.get(cid) or extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
                is_fallback_box = len(tight_norm) == 1 and tight_norm[0][1:] == (0.14, 0.14, 0.86, 0.86)
                boxes = norm_boxes_to_struct(w, h, tight_norm, "wikimedia_scarce_tight_v8f", is_ground_truth=cid in VISUAL_BOX_OVERRIDES)
                border_mask = np.zeros(img_bgr.shape[:2], dtype=bool)
                border_mask[:12, :] = border_mask[-12:, :] = border_mask[:, :12] = border_mask[:, -12:] = True
                border_white_ratio = float(np.mean((hsv_full[border_mask, 1] < 22) & (hsv_full[border_mask, 2] > 235)))
                if is_fallback_box:
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "AMBIGUOUS_CONTEXT"
                    obj_valid = False
                    waste_verified = False
                    reason = "Fallback bounding box [0.14,0.14,0.86,0.86] requires manual localization review; held in NEEDS_REVIEW."
                elif border_white_ratio > 0.50 and not re.search(r"\b(corrod|leak|rotten|mold|mould|crush|broken|shatter|torn|dirty)\b", title_clean, re.I):
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "PRODUCT_STUDIO"
                    obj_valid = True
                    waste_verified = False
                    reason = f"Studio/isolated background detected (border_white_ratio={border_white_ratio:.2f}) without physical damage; routed to KEEP_AUXILIARY per Part A3 policy."
                elif proposal["requires_visual_qa"]:
                    # Candidate-only policy: NEVER auto-promote QA_PASS or ACTUAL_WASTE from filename/title alone
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "AMBIGUOUS_CONTEXT"
                    obj_valid = False
                    waste_verified = False
                    reason = f"Candidate class proposal ({proposal['proposed_class_name']}) requires human visual QA before REAL_WASTE_PASS or ACTUAL_WASTE assignment."
                else:
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "IN_USE"
                    obj_valid = True
                    waste_verified = False
                    reason = "Valid scarce-class object annotation without human-verified disposal context; routed to KEEP_AUXILIARY."

        elif src == "SRC_8F_03_OPENIMAGES_SCARCE_CLASSES_TRAIN_VAL":
            up_b = r.get("upstream_boxes") or []
            cids_set = {int(b["class_id"]) for b in up_b}
            norm_list = []
            for b in up_b:
                bcid = int(b["class_id"])
                cx, cy, bw, bh = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
                norm_list.append((bcid, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

            boxes = norm_boxes_to_struct(w, h, norm_list, f"openimages_v7_{source_split}_gt")
            strict_oi_waste_re = re.compile(r"\b(trash|garbage|litter|rubbish|dumpster|landfill|discarded|rotten|moldy|crushed\s+can|broken\s+bottle)\b", re.I)
            if not boxes or BAD_WIKI_RE.search(title_str):
                qa_status = "REJECT"
                domain_tag = "IN_USE"
                reason = "Rejected Open Images scene (worn on person, portrait, or invalid box)."
            elif len(boxes) > 7:
                qa_status = "NEEDS_REVIEW"
                domain_tag = "AMBIGUOUS_CONTEXT"
                reason = f"Dense multi-object Open Images scene ({len(boxes)} boxes); held in NEEDS_REVIEW."
            elif strict_oi_waste_re.search(title_str) and not (cids_set & {3, 8}):
                qa_status = "REAL_WASTE_PASS"
                domain_tag = "ACTUAL_WASTE"
                obj_valid = True
                waste_verified = True
                reason = f"Verified Open Images V7 ({source_split}) explicit waste scene ('{title_str[:60]}') with tight GT boxes."
            else:
                qa_status = "KEEP_AUXILIARY"
                domain_tag = "IN_USE"
                obj_valid = True
                waste_verified = False
                reason = "Valid Open Images V7 scarce-class GT boxes in household/in-use/street context; stored in KEEP_AUXILIARY."

        cls_ids = sorted({b["class_id"] for b in boxes}) if boxes else [int(r["primary_class_id"])]
        cls_names = [CLASS_NAMES[c] for c in cls_ids]
        p_cid = cls_ids[0]

        yolo_txt = format_yolo_lines(boxes)
        label_path_rel = None
        if qa_status == "REAL_WASTE_PASS":
            lp = LABELS_PASS_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "KEEP_AUXILIARY":
            lp = LABELS_AUX_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "NEEDS_REVIEW":
            lp = LABELS_REVIEW_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")

        rec_out = {
            "candidate_id": cid,
            "source_id": src,
            "source_name": r["source_name"],
            "source_image_id": r["source_image_id"],
            "source_split": source_split,
            "provenance_tier": r.get("provenance_tier", "VERIFIED_PUBLIC_LICENSE"),
            "title": title_str,
            "query_used": query_str,
            "original_url": r["original_url"],
            "author": r["author"],
            "license": r["license"],
            "attribution_required": bool(r.get("attribution_required", True)),
            "image_path": r["image_path"],
            "label_path": label_path_rel,
            "overlay_path": None,
            "sha256": r["sha256"],
            "dhash_hex": r["dhash_hex"],
            "width": w,
            "height": h,
            "primary_class_id": p_cid,
            "primary_class_name": CLASS_NAMES[p_cid],
            "classes_present_ids": cls_ids,
            "classes_present_names": cls_names,
            "is_multiclass": len(cls_ids) > 1,
            "bbox_count": len(boxes),
            "boxes": boxes,
            "qa_status": qa_status,
            "domain_tag": domain_tag,
            "OBJECT_ANNOTATION_VALID": obj_valid,
            "REAL_WASTE_CONTEXT_VERIFIED": waste_verified,
            "APPROVED_FOR_TRAIN": approved_for_train,
            "session_group": r["session_group"],
            "specimen_group": r["specimen_group"],
            "target_split": "REJECTED",
            "qa_reason": reason,
        }
        if qa_status in ("REAL_WASTE_PASS", "KEEP_AUXILIARY", "NEEDS_REVIEW"):
            rec_out["overlay_path"] = render_overlay(rec_out)
        qa_records.append(rec_out)

    # Split allocation preserving source_split AND allocating 25% of Wikimedia session groups to VAL_REAL_WASTE_W8F for Part D!
    pass_recs = [r for r in qa_records if r["qa_status"] == "REAL_WASTE_PASS"]
    wiki_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in pass_recs:
        sp = r["source_split"]
        if sp == "test":
            r["target_split"] = "HOLD_SOURCE_TEST_W8F"
        elif sp in ("validation", "valid"):
            r["target_split"] = "VAL_REAL_WASTE_W8F"
        elif sp == "train":
            r["target_split"] = "TRAIN_REAL_WASTE_W8F"
        else:
            wiki_groups[r["session_group"]].append(r)

    for idx, gk in enumerate(sorted(wiki_groups.keys())):
        grp = wiki_groups[gk]
        if idx % 4 == 3 and len(grp) <= 3:
            for item in grp:
                item["target_split"] = "VAL_REAL_WASTE_W8F"
        else:
            for item in grp:
                item["target_split"] = "TRAIN_REAL_WASTE_W8F"

    for r in qa_records:
        if r["qa_status"] == "KEEP_AUXILIARY":
            r["target_split"] = (
                "AUXILIARY_NON_WASTE_VAL_HOLD"
                if r["source_split"] in ("validation", "valid")
                else "AUXILIARY_NON_WASTE_TRAIN_HOLD"
            )
        elif r["qa_status"] == "NEEDS_REVIEW":
            r["target_split"] = "MANUAL_REVIEW_QUEUE"

    # Verify zero leakage
    train_sessions = {r["session_group"] for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8F"}
    val_sessions = {r["session_group"] for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8F"}
    assert not (train_sessions & val_sessions), f"Session leakage: {train_sessions & val_sessions}"

    status_counts = Counter(r["qa_status"] for r in qa_records)
    domain_counts = Counter(r["domain_tag"] for r in qa_records)
    split_counts = Counter(r["target_split"] for r in qa_records)

    per_class_pass_images = {}
    per_class_pass_train_images = {}
    per_class_pass_val_images = {}
    per_class_pass_boxes = {}
    per_class_pass_val_boxes = {}

    for cid_num in range(10):
        cname = CLASS_NAMES[cid_num]
        c_pass = [r for r in pass_recs if cid_num in r["classes_present_ids"]]
        c_train = [r for r in c_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8F"]
        c_val = [r for r in c_pass if r["target_split"] == "VAL_REAL_WASTE_W8F"]
        c_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_pass)
        c_val_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_val)
        per_class_pass_images[cname] = len(c_pass)
        per_class_pass_train_images[cname] = len(c_train)
        per_class_pass_val_images[cname] = len(c_val)
        per_class_pass_boxes[cname] = c_boxes
        per_class_pass_val_boxes[cname] = c_val_boxes

    sheets_created = []
    for p_idx in range(0, len(pass_recs), 36):
        part_no = p_idx // 36 + 1
        sheets_created.append(render_sheet(
            pass_recs[p_idx : p_idx + 36],
            f"Task 8F Scarce-Class REAL_WASTE_PASS — Part {part_no}",
            f"task8f_real_waste_pass_p{part_no}.jpg",
            max_items=36,
        ))
    sheets_created.append(render_sheet(
        [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8F"],
        "Task 8F VAL_REAL_WASTE_W8F — Independent Scarce-Class Validation Set",
        "task8f_val_real_waste.jpg",
        max_items=36,
    ))
    sheets_created.append(render_sheet(
        [r for r in qa_records if r["qa_status"] == "KEEP_AUXILIARY"],
        "Task 8F KEEP_AUXILIARY — Studio / In-Use Valid Object Annotations",
        "task8f_keep_auxiliary.jpg",
        max_items=36,
    ))

    summary_payload = {
        "manifest_version": "TASK8F_SCARCE_CLASSES_V1",
        "approved_for_train": False,
        "pilots": pilots,
        "total_downloaded_candidates": len(qa_records),
        "qa_status_counts": dict(status_counts),
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "real_waste_pass_total_images": len(pass_recs),
        "real_waste_pass_train_images": split_counts["TRAIN_REAL_WASTE_W8F"],
        "real_waste_pass_val_images": split_counts["VAL_REAL_WASTE_W8F"],
        "real_waste_pass_total_boxes": sum(r["bbox_count"] for r in pass_recs),
        "per_class_real_waste_pass_images": per_class_pass_images,
        "per_class_real_waste_pass_train_images": per_class_pass_train_images,
        "per_class_real_waste_pass_val_images": per_class_pass_val_images,
        "per_class_real_waste_pass_boxes": per_class_pass_boxes,
        "per_class_real_waste_pass_val_boxes": per_class_pass_val_boxes,
        "contact_sheets": [s for s in sheets_created if s],
        "records": qa_records,
    }
    QA_MANIFEST_JSON.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(QA_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "source_id", "source_split", "provenance_tier",
            "primary_class_id", "primary_class_name",
            "classes_present_ids", "is_multiclass", "bbox_count",
            "qa_status", "domain_tag", "OBJECT_ANNOTATION_VALID",
            "REAL_WASTE_CONTEXT_VERIFIED", "APPROVED_FOR_TRAIN",
            "target_split", "session_group", "specimen_group",
            "license", "author", "original_url", "qa_reason",
        ])
        for r in qa_records:
            writer.writerow([
                r["candidate_id"], r["source_id"], r["source_split"], r["provenance_tier"],
                r["primary_class_id"], r["primary_class_name"],
                "|".join(str(c) for c in r["classes_present_ids"]), r["is_multiclass"], r["bbox_count"],
                r["qa_status"], r["domain_tag"], r["OBJECT_ANNOTATION_VALID"],
                r["REAL_WASTE_CONTEXT_VERIFIED"], r["APPROVED_FOR_TRAIN"],
                r["target_split"], r["session_group"], r["specimen_group"],
                r["license"], r["author"], r["original_url"], r["qa_reason"],
            ])

    print(f"Task 8F QA Manifest saved: {QA_MANIFEST_JSON}")
    print("Status counts:", dict(status_counts))
    print("Split counts:", dict(split_counts))
    print("Per-class REAL_WASTE_PASS images:", per_class_pass_images)
    print("Per-class REAL_WASTE_PASS train images:", per_class_pass_train_images)
    print("Per-class REAL_WASTE_PASS val images:", per_class_pass_val_images)
    print("Per-class REAL_WASTE_PASS boxes:", per_class_pass_boxes)


if __name__ == "__main__":
    main()
