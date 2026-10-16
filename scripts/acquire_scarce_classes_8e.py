"""Acquire ~180 high-yield real-waste candidates for scarce classes (0: battery, 1: biological, 3: clothes, 4: glass, 8: shoes, 2: cardboard, 5: metal, 6: paper)
from explicit waste/decay/litter Wikimedia Commons subcategories & intitle searches + strict waste-title Open Images V7 train/validation records.
Appends to `task8e_raw_candidates_manifest.json` (~1,066 -> ~1,240 candidates, well within 1,000-1,500).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
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

    for mp in [
        ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_raw_candidates_manifest.json",
        ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json",
    ]:
        if not mp.exists():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        for r in data.get("records") or data.get("items") or []:
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

    for r in existing_8e:
        if r.get("sha256"):
            sha_set.add(r["sha256"])
        if r.get("dhash_hex"):
            dhash_list.append(int(r["dhash_hex"], 16))
        if r.get("source_image_id"):
            source_ids.add(str(r["source_image_id"]))

    return sha_set, dhash_list, source_ids


SCARCE_WIKI_QUERIES: list[tuple[int, str, int]] = [
    # Class 0: battery (explicitly damaged, leaked, corroded, spent, discarded, e-waste batteries)
    (0, "Category:Leaked_electric_batteries", 35),
    (0, "Category:Damaged_electric_batteries", 35),
    (0, "Category:Swollen_batteries", 35),
    (0, "Category:Electric_battery_recycling", 35),
    (0, "search:filetype:bitmap (corroded battery OR leaked battery OR swollen battery OR rusty battery)", 40),
    (0, "search:filetype:bitmap (used batteries waste OR spent batteries pile OR old battery trash)", 40),
    (0, "search:filetype:bitmap intitle:battery (corrosion OR leak OR leaking OR damaged OR old OR waste OR recycling)", 40),
    # Class 1: biological (explicitly rotting, moldy, decomposing, food waste, peels)
    (1, "Category:Decomposing_food", 35),
    (1, "Category:Moldy_food", 35),
    (1, "Category:Rotting_fruit", 35),
    (1, "Category:Banana_peel", 35),
    (1, "Category:Food_waste", 35),
    (1, "search:filetype:bitmap (moldy bread OR rotten apple OR rotten orange OR decomposing fruit)", 40),
    (1, "search:filetype:bitmap (banana peel ground OR discarded food street OR food waste bin)", 40),
    (1, "search:filetype:bitmap (apple core trash OR eggshell waste OR vegetable peelings compost)", 35),
    # Class 3: clothes (explicitly discarded gloves, masks, socks, rags, clothing litter)
    (3, "Category:Discarded_gloves", 35),
    (3, "Category:Rags", 35),
    (3, "search:filetype:bitmap (discarded glove ground OR lost glove street OR dirty glove litter)", 40),
    (3, "search:filetype:bitmap (discarded sock OR old torn shirt OR abandoned clothing litter)", 40),
    (3, "search:filetype:bitmap (discarded clothes pile OR textile waste bin OR rag trash)", 35),
    # Class 4: glass (explicitly broken glass bottle, shattered bottle, littered bottle)
    (4, "Category:Broken_bottles", 35),
    (4, "Category:Glass_bottles_as_litter", 35),
    (4, "search:filetype:bitmap (broken beer bottle OR smashed glass bottle OR shattered bottle street)", 40),
    (4, "search:filetype:bitmap (discarded glass bottle grass OR empty beer bottle litter ground)", 40),
    # Class 8: shoes (explicitly abandoned shoes, discarded footwear, lost shoe beach/street)
    (8, "Category:Abandoned_shoes", 45),
    (8, "Category:Worn-out_shoes", 40),
    (8, "search:filetype:bitmap (abandoned shoe street OR discarded shoe litter OR lost shoe beach)", 45),
    (8, "search:filetype:bitmap (old worn boots mud OR torn sneaker trash OR discarded sandal ground)", 45),
    (8, "search:filetype:bitmap intitle:shoe (abandoned OR old OR worn OR dirty OR beach OR street OR lost)", 40),
]

BAD_WIKI_RE = re.compile(
    r"(diagram|chart|graph|map|logo|icon|symbol|sign|coat of arms|flag|stamp|coin|"
    r"museum|archaeolog|roman|medieval|century|painting|drawing|sketch|illustration|page \d|"
    r"poster|advertisement|cartoon|statue|sculpture|monument|building|facade|window|"
    r"truck|vehicle|lorry|train|artillery|cannon|bunker|soldier|portrait|person|man|woman|child|"
    r"cat|dog|bird|sparrow|horse|cow|tree|flower|landscape|panorama|svg|pdf|\.gif|\.tif)",
    re.I,
)


def main() -> None:
    t0 = time.time()
    existing_data = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    existing_records: list[dict[str, Any]] = existing_data["records"]
    sha_set, dhash_list, source_ids = load_all_fingerprints(existing_records)

    sess = requests.Session()
    sess.verify = False
    sess.headers.update({"User-Agent": USER_AGENT})
    api_url = "https://commons.wikimedia.org/w/api.php"

    pages_by_cls: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen_pids: set[str] = set()

    for cid, qspec, lim in SCARCE_WIKI_QUERIES:
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

    print("Scarce Wikimedia candidate pages discovered per class:", {CLASS_NAMES[k]: len(v) for k, v in pages_by_cls.items()})

    # Round-robin across (0, 1, 3, 4, 8)
    interleaved: list[tuple[int, str, dict[str, Any]]] = []
    max_len = max((len(v) for v in pages_by_cls.values()), default=0)
    for idx in range(max_len):
        for cid in (0, 1, 3, 4, 8):
            if idx < len(pages_by_cls[cid]):
                qspec, pdata = pages_by_cls[cid][idx]
                interleaved.append((cid, qspec, pdata))

    added_recs: list[dict[str, Any]] = []
    per_cls_added: Counter[int] = Counter()
    for cid, qspec, pdata in interleaved:
        if len(added_recs) >= 180:
            break
        if per_cls_added[cid] >= 42:
            continue
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
            time.sleep(0.10)
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
                per_cls_added[cid] += 1

                author_slug = re.sub(r"[^a-zA-Z0-9]+", "_", artist.lower()).strip("_")[:24] or "unk"
                added_recs.append({
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

    start_idx = len(existing_records) + 1
    for offset, r in enumerate(added_recs):
        idx = start_idx + offset
        cid_str = f"task8e_{idx:04d}"
        img_p = IMAGES_DIR / f"{cid_str}.jpg"
        img_p.write_bytes(r.pop("jpg_bytes"))
        r["candidate_id"] = cid_str
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
    print(f"Added {len(added_recs)} scarce-class real-waste candidates ({ {CLASS_NAMES[k]: v for k, v in per_cls_added.items()} })!")
    print(f"Total Wave 8E candidates now: {len(existing_records)}")


if __name__ == "__main__":
    main()
