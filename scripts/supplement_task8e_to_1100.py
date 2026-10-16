"""Supplement `task8e_raw_candidates_manifest.json` from 575 to ~1,100 unique deduplicated candidates.

Reuses the 575 candidates already downloaded in `data/acquisition_staging/task8e_scale_real_waste/`
and appends ~525 additional deduplicated candidates from:
1. Remaining first-party `2025*` smartphone captures in `Shynuaa/Street-garbage/yolo/train` (`source_split="train"`).
2. `Open Images V7` `train` and `validation` splits (`data/external/waste-training/openimages-train-slice-*.csv` & `validation-*.csv`, `0` from `test`!).
3. `Wikimedia Commons` explicit waste/decay/litter searches across classes `1..9` (throttled to avoid Wikimedia thumbnail 429 rate-limits).
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
import urllib3
from PIL import Image, ImageOps

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parents[1]
WT_DIR = Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training")))
OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste"
IMAGES_DIR = OUT_DIR / "images"
RAW_MANIFEST_JSON = OUT_DIR / "task8e_raw_candidates_manifest.json"

USER_AGENT = "CNTT-KLCN155-WasteDetection/2.0 (Academic Research; mailto:research@klcn155.edu.vn)"

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

FIRST_PARTY_FILE_RE = re.compile(r"^(2025\d{4}_\d{6}|IMG_2025\d{4}_\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)
CAPTURE_MIN_RE = re.compile(r"(2025\d{4}_\d{4})")


def compute_dhash(im: Image.Image) -> int:
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        row_off = r * 9
        for c in range(8):
            bits = (bits << 1) | (1 if px[row_off + c] > px[row_off + c + 1] else 0)
    return bits


def is_near_dup(dh: int, dhash_list: list[int], max_dist: int = 4) -> bool:
    for existing in dhash_list:
        if (dh ^ existing).bit_count() <= max_dist:
            return True
    return False


def load_all_fingerprints(existing_8e: list[dict[str, Any]]) -> tuple[set[str], list[int], set[str]]:
    sha_set: set[str] = set()
    dhash_list: list[int] = []
    source_ids: set[str] = set()

    manifest_paths = [
        ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_raw_candidates_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json",
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

    for r in existing_8e:
        if r.get("sha256"):
            sha_set.add(r["sha256"])
        if r.get("dhash_hex"):
            dhash_list.append(int(r["dhash_hex"], 16))
        if r.get("source_image_id"):
            source_ids.add(str(r["source_image_id"]))

    for p in (ROOT / "data" / "detection" / "images" / "real").glob("*"):
        source_ids.add(p.stem.split("_")[0])

    return sha_set, dhash_list, source_ids


def fetch_more_shynuaa_train(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_add: int = 85
) -> list[dict[str, Any]]:
    print("=== [A] Fetching remaining first-party Shynuaa/Street-garbage yolo/train candidates ===")
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    repo_id = "Shynuaa/Street-garbage"
    files: list[str] = []
    for sub in ["yolo/train/images", "yolo/train/labels", "yolo/valid/images", "yolo/test/images"]:
        r = sess.get(f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{sub}?limit=1000", timeout=20)
        if r.status_code == 200:
            for entry in r.json():
                if isinstance(entry, dict) and entry.get("path"):
                    files.append(entry["path"])

    holdout_minutes: set[str] = set()
    for f in files:
        if f.startswith("yolo/valid/images/") or f.startswith("yolo/test/images/"):
            m = CAPTURE_MIN_RE.search(Path(f).name)
            if m:
                holdout_minutes.add(m.group(1))

    train_imgs = sorted(
        f for f in files
        if f.startswith("yolo/train/images/") and FIRST_PARTY_FILE_RE.match(Path(f).name)
    )
    train_lbl_set = {f for f in files if f.startswith("yolo/train/labels/") and f.endswith(".txt")}

    candidates_to_fetch: list[tuple[str, str, str, str]] = []
    for img_f in train_imgs:
        name = Path(img_f).name
        stem = Path(img_f).stem
        sid = f"shynuaa_train_{stem}"
        if sid in source_ids:
            continue
        m_min = CAPTURE_MIN_RE.search(name)
        if m_min and m_min.group(1) in holdout_minutes:
            continue
        lbl_f = f"yolo/train/labels/{stem}.txt"
        if lbl_f not in train_lbl_set:
            continue
        candidates_to_fetch.append((sid, img_f, lbl_f, m_min.group(1) if m_min else stem))
        if len(candidates_to_fetch) >= target_add + 40:
            break

    print(f"Found {len(candidates_to_fetch)} additional Shynuaa train candidates.")

    def fetch_one(item: tuple[str, str, str, str]) -> dict[str, Any] | None:
        sid, img_f, lbl_f, cap_min = item
        try:
            lr = sess.get(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{lbl_f}", timeout=15)
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
            ir = sess.get(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{img_f}", timeout=20)
            if ir.status_code != 200 or len(ir.content) < 5000:
                return None
            return {"sid": sid, "img_f": img_f, "cap_min": cap_min, "up_boxes": up_boxes, "raw_bytes": ir.content}
        except Exception:
            return None

    fetched: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(fetch_one, candidates_to_fetch):
            if res is not None:
                fetched.append(res)

    records: list[dict[str, Any]] = []
    for item in fetched:
        if len(records) >= target_add:
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
                source_ids.add(item["sid"])
                records.append({
                    "source_id": "SRC_8E_01_SHYNUAA_STREET_GARBAGE_TRAIN",
                    "source_name": "Shynuaa/Street-garbage (HuggingFace First-Party 2025 Train Split)",
                    "source_image_id": item["sid"],
                    "source_split": "train",
                    "provenance_tier": "VERIFIED_FIRST_PARTY_SMARTPHONE_CAPTURE_MIT",
                    "title": Path(item["img_f"]).name,
                    "query_used": "Shynuaa/Street-garbage/yolo/train (first-party 2025* smartphone filter)",
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
                    "primary_class_id": 9,
                    "upstream_boxes": item["up_boxes"],
                    "session_group": f"sess_shynuaa_train_{item['cap_min']}",
                    "specimen_group": f"spec_{item['sid']}",
                })
        except Exception:
            continue

    print(f"Added {len(records)} additional Shynuaa train candidates.")
    return records


def fetch_openimages_train_val(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_add: int = 330
) -> list[dict[str, Any]]:
    print("=== [B] Fetching Open Images V7 Train & Validation Candidates from data/external/waste-training (0 from test!) ===")
    cls_map = {}
    cls_csv = WT_DIR / "openimages-boxable-classes.csv"
    if cls_csv.exists():
        with cls_csv.open("r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 2:
                    cls_map[r[0]] = r[1]

    oi_to_10 = {
        "/m/01kb5b": 0,
        "/m/01c648": 0,
        "/m/09qck": 1,
        "/m/014j1m": 1,
        "/m/0cyhj_": 1,
        "/m/09k_b": 1,
        "/m/07j87": 1,
        "/m/09728": 1,
        "/m/033cnk": 1,
        "/m/02wbm": 1,
        "/m/0f4s2w": 1,
        "/m/05z55": 1,
        "/m/0663v": 1,
        "/m/01b9xk": 1,
        "/m/0hqkz": 1,
        "/m/025dyy": 2,
        "/m/025nd": 2,
        "/m/0174n1": 3,
        "/m/0162_1": 3,
        "/m/01nq26": 3,
        "/m/01n4qj": 3,
        "/m/07mhn": 3,
        "/m/032b3c": 3,
        "/m/09j2d": 3,
        "/m/0fly7": 3,
        "/m/04dr76w": 4,
        "/m/09tvcd": 4,
        "/m/08hvt4": 4,
        "/m/0271t": 4,
        "/m/01599": 4,
        "/m/02jnhm": 5,
        "/m/02w3r3": 6,
        "/m/09gtd": 6,
        "/m/0frqm": 6,
        "/m/0bt_c3": 6,
        "/m/057cc": 7,
        "/m/05gqfk": 7,
        "/m/01b638": 8,
        "/m/03nfch": 8,
        "/m/06k2mb": 8,
        "/m/09j5n": 8,
        "/m/0bjyj5": 9,
    }

    waste_re = re.compile(
        r"\b(trash|garbage|litter|rubbish|waste|dump|dumpster|bin|recycl|discard|abandon|rotten|mold|mould|spoil|compost|"
        r"peel|scrap|leftover|crush|dent|rust|corrod|broken|shatter|smash|torn|ripped|dirty|muddy|worn|old\s+shoe|old\s+boot|"
        r"used\s+batter|dead\s+batter|cigarette|butt|fly.?tip|landfill|street|alley|gutter|curb|sidewalk|beach|pile|heap|"
        r"cardboard|box|can|tin|bottle|jar|cup|bag|wrapper|tissue|napkin|newspaper|paper)\b",
        re.I,
    )
    studio_re = re.compile(
        r"\b(white\s+background|isolated|product|catalog|clipart|illustration|vector|3d\s+render|museum|statue|painting)\b",
        re.I,
    )

    candidates: list[dict[str, Any]] = []
    per_cls_selected: Counter[int] = Counter()

    # STRICTLY train and validation splits only (NEVER test!)
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
                    if 0.015 < bw < 0.96 and 0.015 < bh < 0.96:
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
                # Avoid standalone class 3 (Clothing on people) or standalone class 8 (Footwear on people) unless explicit waste title
                if cids in ([3], [8], [3, 8]) and not re.search(r"\b(trash|garbage|litter|rubbish|waste|discard|abandon|old|worn|dirty|muddy)\b", title, re.I):
                    continue
                if not (waste_re.search(title) or len(cids) >= 2 or cids[0] in (0, 2, 5, 6, 9)):
                    continue
                url = r.get("Thumbnail300KURL", "") or r.get("OriginalURL", "")
                if not url:
                    continue
                p_cid = cids[0]
                if per_cls_selected[p_cid] >= 65:
                    continue
                per_cls_selected[p_cid] += 1
                author = (r.get("Author", "") or "").strip()
                landing_url = (r.get("OriginalLandingURL", "") or r.get("OriginalURL", "")).strip()
                if not author or not landing_url:
                    continue
                candidates.append({
                    "split": split_tag,
                    "iid": iid,
                    "title": title or f"OpenImages_{iid}",
                    "url": url,
                    "fallback_url": r.get("OriginalURL", ""),
                    "boxes": bxs,
                    "primary_cid": p_cid,
                    "license": r.get("License", "https://creativecommons.org/licenses/by/2.0/"),
                    "author": author,
                    "landing_url": landing_url,
                })
                if len(candidates) >= target_add + 120:
                    break

    print(f"Selected {len(candidates)} Open Images V7 train/validation candidates across classes: {dict(per_cls_selected)}")

    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})

    def dl_one(c: dict[str, Any]) -> dict[str, Any] | None:
        for u in (c["url"], c["fallback_url"]):
            if not u:
                continue
            try:
                resp = sess.get(u, timeout=12)
                if resp.status_code == 200 and len(resp.content) >= 5000:
                    return {**c, "raw_bytes": resp.content}
            except Exception:
                continue
        return None

    fetched: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for res in ex.map(dl_one, candidates):
            if res is not None:
                fetched.append(res)
                if len(fetched) >= target_add + 30:
                    break

    records: list[dict[str, Any]] = []
    for item in fetched:
        if len(records) >= target_add:
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
                    "source_id": "SRC_8E_03_OPENIMAGES_TRAIN_VAL_REAL_WASTE",
                    "source_name": f"Open Images V7 Waste-Context Filtered ({item['split']})",
                    "source_image_id": item["iid"],
                    "source_split": item["split"],
                    "provenance_tier": "VERIFIED_FLICKR_CC_BY_2.0_WITH_AUTHOR_URL",
                    "title": item["title"],
                    "query_used": f"OpenImages V7 {item['split']} waste-context filter",
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
                    "session_group": f"sess_oi8e_{item['split']}_{author_slug}",
                    "specimen_group": f"spec_oi8e_{item['iid']}",
                })
        except Exception:
            continue

    print(f"Added {len(records)} Open Images V7 train/validation candidates.")
    return records


WIKIMEDIA_CLASS1_TO_9_SEARCHES: list[tuple[int, str, int]] = [
    # 1: biological
    (1, "Category:Food_waste", 25),
    (1, "Category:Rotting_food", 20),
    (1, "Category:Moldy_food", 20),
    (1, "Category:Banana_peel", 20),
    (1, "Category:Compost", 20),
    (1, "search:filetype:bitmap (rotten fruit OR moldy bread OR food scraps OR apple core OR orange peel)", 25),
    # 2: cardboard
    (2, "Category:Cardboard_boxes", 25),
    (2, "Category:Paper_salvage_containers", 20),
    (2, "search:filetype:bitmap (discarded cardboard OR crushed cardboard box OR wet cardboard OR pizza box trash)", 25),
    # 3: clothes
    (3, "Category:Rags", 20),
    (3, "Category:Textiles_collection_boxes", 15),
    (3, "search:filetype:bitmap (discarded glove OR abandoned clothes OR dirty rag OR old sock litter)", 25),
    # 4: glass
    (4, "Category:Broken_glass", 25),
    (4, "Category:Glass_recycling", 20),
    (4, "search:filetype:bitmap (broken glass bottle OR shattered bottle OR empty beer bottle litter)", 25),
    # 5: metal
    (5, "Category:Aluminium_cans", 25),
    (5, "Category:Scrap_metal", 20),
    (5, "search:filetype:bitmap (crushed soda can OR crushed beer can OR rusty tin can litter)", 25),
    # 6: paper
    (6, "Category:Paper_recycling", 25),
    (6, "Category:Waste_paper", 20),
    (6, "search:filetype:bitmap (crumpled paper litter OR discarded paper cup OR torn newspaper trash)", 25),
    # 7: plastic
    (7, "Category:Plastic_waste", 25),
    (7, "search:filetype:bitmap (crushed plastic bottle litter OR plastic bag trash street)", 20),
    # 8: shoes
    (8, "Category:Abandoned_shoes", 25),
    (8, "Category:Worn-out_shoes", 25),
    (8, "search:filetype:bitmap (discarded shoe OR old worn boots OR abandoned sneaker litter)", 25),
    # 9: trash
    (9, "Category:Litter", 25),
    (9, "Category:Illegal_dumping", 25),
]

BAD_WIKI_RE = re.compile(
    r"(diagram|chart|graph|map|logo|icon|symbol|sign|coat of arms|flag|stamp|coin|"
    r"museum|archaeolog|roman|medieval|painting|drawing|sketch|illustration|page \d|"
    r"poster|cartoon|statue|sculpture|monument|svg|pdf|\.gif|\.tif)",
    re.I,
)


def fetch_more_wikimedia_classes_1_to_9(
    sha_set: set[str], dhash_list: list[int], source_ids: set[str], target_add: int = 160
) -> list[dict[str, Any]]:
    print("=== [C] Fetching Wikimedia Commons Candidates for Classes 1..9 (Throttled) ===")
    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    api_url = "https://commons.wikimedia.org/w/api.php"

    pages_by_cls: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen_pids: set[str] = set()

    for cid, qspec, lim in WIKIMEDIA_CLASS1_TO_9_SEARCHES:
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

        try:
            r = sess.get(api_url, params=params, timeout=15)
            if r.status_code != 200:
                continue
            pages = r.json().get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                sid = f"commons_{pid}"
                if sid in source_ids or sid in seen_pids:
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
                seen_pids.add(sid)
                pages_by_cls[cid].append((qspec, pdata))
        except Exception:
            continue

    # Interleave round-robin across classes 1..9 so every class gets balanced downloads!
    interleaved: list[tuple[int, str, dict[str, Any]]] = []
    max_len = max((len(v) for v in pages_by_cls.values()), default=0)
    for idx in range(max_len):
        for cid in range(1, 10):
            if idx < len(pages_by_cls[cid]):
                qspec, pdata = pages_by_cls[cid][idx]
                interleaved.append((cid, qspec, pdata))

    print(f"Interleaved {len(interleaved)} Wikimedia candidate pages across classes 1..9.")

    records: list[dict[str, Any]] = []
    for cid, qspec, pdata in interleaved:
        if len(records) >= target_add:
            break
        pid = str(pdata["pageid"])
        sid = f"commons_{pid}"
        ii = pdata["imageinfo"][0]
        dl_url = ii.get("thumburl") or ii.get("url")
        if not dl_url:
            continue
        ext = ii.get("extmetadata") or {}
        lic = (ext.get("LicenseShortName", {}).get("value") or "CC-BY-SA").strip()
        artist_raw = ext.get("Artist", {}).get("value") or "Wikimedia Commons Contributor"
        artist = re.sub(r"<[^>]+>", "", artist_raw).strip()[:80] or "Wikimedia Commons Contributor"
        try:
            time.sleep(0.12)
            resp = sess.get(dl_url, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 5000:
                continue
            with Image.open(io.BytesIO(resp.content)) as im_raw:
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

                author_slug = re.sub(r"[^a-zA-Z0-9]+", "_", artist.lower()).strip("_")[:24] or "unk"
                records.append({
                    "source_id": "SRC_8E_02_WIKIMEDIA_REAL_WASTE_CATEGORIES",
                    "source_name": "Wikimedia Commons Explicit Real-Waste Categories & Queries",
                    "source_image_id": sid,
                    "source_split": "unspecified",
                    "provenance_tier": "VERIFIED_WIKIMEDIA_COMMONS_OPEN_LICENSE",
                    "title": pdata.get("title", ""),
                    "query_used": qspec,
                    "original_url": ii.get("descriptionurl") or dl_url,
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
                    "session_group": f"sess_wiki8e_{author_slug}",
                    "specimen_group": f"spec_{sid}",
                })
        except Exception:
            continue

    print(f"Added {len(records)} balanced Wikimedia Commons candidates across classes 1..9.")
    return records


def main() -> None:
    t0 = time.time()
    existing_data = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    existing_records: list[dict[str, Any]] = existing_data["records"]
    print(f"Loaded {len(existing_records)} existing Task 8E candidates.")

    sha_set, dhash_list, source_ids = load_all_fingerprints(existing_records)

    more_shynuaa = fetch_more_shynuaa_train(sha_set, dhash_list, source_ids, target_add=85)
    oi_recs = fetch_openimages_train_val(sha_set, dhash_list, source_ids, target_add=330)
    more_wiki = fetch_more_wikimedia_classes_1_to_9(sha_set, dhash_list, source_ids, target_add=160)

    new_recs = more_shynuaa + oi_recs + more_wiki
    start_idx = len(existing_records) + 1
    for offset, r in enumerate(new_recs):
        idx = start_idx + offset
        cid = f"task8e_{idx:04d}"
        img_p = IMAGES_DIR / f"{cid}.jpg"
        img_p.write_bytes(r.pop("jpg_bytes"))
        r["candidate_id"] = cid
        r["image_path"] = str(img_p.relative_to(ROOT)).replace("\\", "/")
        existing_records.append(r)

    src_counts = Counter(r["source_id"] for r in existing_records)
    split_counts = Counter(r["source_split"] for r in existing_records)
    cls_counts = Counter(CLASS_NAMES[r["primary_class_id"]] for r in existing_records)

    payload = {
        "wave": "TASK8E_SCALE_REAL_WASTE_ACQUISITION",
        "total_candidates": len(existing_records),
        "elapsed_sec": round(time.time() - t0, 2),
        "source_counts": dict(src_counts),
        "source_split_counts": dict(split_counts),
        "primary_class_counts": dict(cls_counts),
        "records": existing_records,
    }
    RAW_MANIFEST_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {RAW_MANIFEST_JSON} with {len(existing_records)} total candidates!")
    print("Source breakdown:", dict(src_counts))
    print("Source split breakdown:", dict(split_counts))
    print("Primary class breakdown:", dict(cls_counts))


if __name__ == "__main__":
    main()
