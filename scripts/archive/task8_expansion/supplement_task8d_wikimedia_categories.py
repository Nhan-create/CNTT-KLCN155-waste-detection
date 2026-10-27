"""Supplement Task 8D (`SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES`) with curated Wikimedia Commons
waste/recycling categories and `filetype:bitmap intitle:<item>` waste queries across all 10 classes.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import urllib3
from PIL import Image, ImageDraw, ImageOps

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste"
IMAGES_DIR = OUT_DIR / "images"
INSPECT_DIR = OUT_DIR / "inspection_sheets"
MANIFEST_PATH = OUT_DIR / "task8d_raw_candidates_manifest.json"

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({
    "User-Agent": "CNTT-KLCN155-WasteDetection/2.0 (Academic Research; mailto:research@klcn155.edu.vn)"
})

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


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    diff = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            diff = (diff << 1) | (1 if pixels[row_start + col] > pixels[row_start + col + 1] else 0)
    return diff


def hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


BAD_TITLE_RE = re.compile(
    r"(diagram|chart|graph|map|logo|icon|symbol|sign|coat of arms|flag|stamp|coin|"
    r"museum|archaeolog|roman|medieval|bronze age|century|painting|drawing|sketch|"
    r"illustration|engraving|page \d|dpla|book|cyclopedia|memoir|herald|newspaper|"
    r"poster|advertisement|cartoon|caricature|statue|sculpture|monument|building|"
    r"truck|vehicle|lorry|train|locomotive|artillery|cannon|bunker|fortress|ruin|"
    r"landscape|aerial|panorama|portrait|person|soldier|marine|bird|sparrow|dog|"
    r"cat|chicken|hen|mushroom|fungi|tree|flower|plant|berry|berries|pepper)",
    re.I,
)

# Curated Wikimedia Commons Waste Categories: (category_title, class_id, max_take)
WASTE_CATEGORIES: list[tuple[str, int, int]] = [
    ("Category:Leaked_electric_batteries", 0, 14),
    ("Category:Electric_battery_recycling", 0, 12),
    ("Category:Electric_battery_recycling_containers", 0, 8),
    ("Category:Banana_peel", 1, 12),
    ("Category:Food_waste", 1, 12),
    ("Category:Compost", 1, 8),
    ("Category:Cardboard_boxes", 2, 10),
    ("Category:Paper_salvage_containers", 2, 8),
    ("Category:Rags", 3, 10),
    ("Category:Textiles_collection_boxes", 3, 6),
    ("Category:Broken_glass", 4, 12),
    ("Category:Glass_recycling", 4, 8),
    ("Category:Aluminium_cans", 5, 12),
    ("Category:Scrap_metal", 5, 8),
    ("Category:Paper_recycling", 6, 12),
    ("Category:Litter", 9, 10),
]

# Targeted bitmap intitle searches: (query, class_id, max_take)
BITMAP_SEARCHES: list[tuple[str, int, int]] = [
    ("filetype:bitmap intitle:battery (corroded OR leaked OR rusty OR old OR used OR recycling OR waste OR discarded)", 0, 10),
    ("filetype:bitmap intitle:rotten (apple OR banana OR orange OR fruit OR tomato OR food)", 1, 10),
    ("filetype:bitmap intitle:moldy OR intitle:mouldy (bread OR fruit OR food OR orange OR lemon)", 1, 10),
    ("filetype:bitmap intitle:cardboard (trash OR waste OR discarded OR recycling OR wet OR crushed OR litter)", 2, 10),
    ("filetype:bitmap intitle:clothes OR intitle:clothing OR intitle:shirt OR intitle:sock (discarded OR abandoned OR trash OR litter OR rags OR torn)", 3, 12),
    ("filetype:bitmap intitle:bottle (broken OR smashed OR litter OR discarded OR trash OR beach)", 4, 10),
    ("filetype:bitmap intitle:can (crushed OR rusty OR litter OR discarded OR flattened OR beverage)", 5, 10),
    ("filetype:bitmap intitle:paper (crumpled OR torn OR waste OR litter OR shredded OR discarded)", 6, 12),
    ("filetype:bitmap intitle:shoe OR intitle:shoes (abandoned OR discarded OR litter OR beach OR trash OR worn OR derelict OR muddy)", 8, 16),
    ("filetype:bitmap intitle:boot OR intitle:boots OR intitle:sneaker OR intitle:sandal (abandoned OR discarded OR worn OR old OR muddy OR litter)", 8, 10),
]


def query_category_files(cat: str, limit: int = 40) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": cat,
        "gcmtype": "file",
        "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1024,
        "format": "json",
    }
    try:
        r = SESSION.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=20)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            pages = r.json().get("query", {}).get("pages", {})
            return list(pages.values())
    except Exception:
        pass
    return []


def query_bitmap_search(q: str, limit: int = 25) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": q,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1024,
        "format": "json",
    }
    try:
        r = SESSION.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=20)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            pages = r.json().get("query", {}).get("pages", {})
            return list(pages.values())
    except Exception:
        pass
    return []


def render_new_inspection_sheets(records: list[dict[str, Any]], start_idx: int) -> list[str]:
    subset = records[start_idx:]
    per_page = 30
    cols = 6
    cell_w, cell_h = 260, 220
    paths = []
    for p_idx in range(0, len(subset), per_page):
        chunk = subset[p_idx : p_idx + per_page]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 36), (18, 20, 24))
        draw = ImageDraw.Draw(sheet)
        page_num = (start_idx // per_page) + (p_idx // per_page) + 1
        draw.text(
            (10, 8),
            f"Task 8D Visual Inspection Sheet p{page_num:02d} ({chunk[0]['candidate_id']} .. {chunk[-1]['candidate_id']})",
            fill=(240, 240, 240),
        )
        for i, rec in enumerate(chunk):
            r = i // cols
            c = i % cols
            x0 = c * cell_w
            y0 = r * cell_h + 36
            img_path = ROOT / rec["image_path"]
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                im_rgb.thumbnail((cell_w - 8, cell_h - 34))
                sheet.paste(im_rgb, (x0 + 4, y0 + 4))
            cid_short = rec["candidate_id"].replace("task8d_", "")
            p_cls = rec["primary_class_id"]
            p_name = rec["primary_class_name"][:4]
            title_short = (rec.get("title") or "")[:20]
            caption = f"{cid_short}|{p_cls}:{p_name}|{title_short}"
            draw.text((x0 + 4, y0 + cell_h - 26), caption, fill=(220, 230, 240))
        out_p = INSPECT_DIR / f"task8d_inspect_p{page_num:02d}.jpg"
        sheet.save(out_p, quality=88)
        paths.append(str(out_p.relative_to(ROOT)).replace("\\", "/"))
    return paths


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = manifest["records"]
    start_idx = len(records)

    existing_sha256: set[str] = set()
    existing_dhashes: list[int] = []
    existing_titles: set[str] = set()
    specimen_stems: set[str] = set()

    # Load all repo hashes + current task8d hashes
    auth_p = ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if auth_p.exists():
        for v in json.loads(auth_p.read_text(encoding="utf-8")).values():
            if isinstance(v, dict) and v.get("sha256"):
                existing_sha256.add(v["sha256"])

    for w_p in [
        ROOT / "data/acquisition_staging/task8a_multiclass_batch/task8_wave1_qa_manifest.json",
        ROOT / "data/acquisition_staging/task8_wave2_large_batch/task8_wave2_raw_manifest.json",
    ]:
        if w_p.exists():
            for r in json.loads(w_p.read_text(encoding="utf-8")).get("records", []):
                if r.get("sha256"):
                    existing_sha256.add(r["sha256"])
                if r.get("title"):
                    existing_titles.add(r["title"])

    for r in records:
        if r.get("sha256"):
            existing_sha256.add(r["sha256"])
        if r.get("dhash_int") is not None:
            existing_dhashes.append(int(r["dhash_int"]))
        if r.get("title"):
            existing_titles.add(r["title"])

    candidate_Queue: list[tuple[dict[str, Any], int, str]] = []
    for cat, cid, limit in WASTE_CATEGORIES:
        pages = query_category_files(cat, limit=40)
        for p in pages:
            candidate_Queue.append((p, cid, cat))
        time.sleep(0.08)

    for q, cid, limit in BITMAP_SEARCHES:
        pages = query_bitmap_search(q, limit=25)
        for p in pages:
            candidate_Queue.append((p, cid, q[:45]))
        time.sleep(0.08)

    print(f"Total raw page metadata entries fetched: {len(candidate_Queue)}")

    per_class_added = {k: 0 for k in CLASS_NAMES}
    per_class_cap = {0: 18, 1: 18, 2: 14, 3: 14, 4: 14, 5: 14, 6: 14, 7: 6, 8: 18, 9: 8}

    next_num = len(records) + 1
    added = 0

    for page, cid, query_tag in candidate_Queue:
        if per_class_added[cid] >= per_class_cap[cid]:
            continue
        title = page.get("title", "")
        if not title or title in existing_titles:
            continue
        if BAD_TITLE_RE.search(title):
            continue
        # Avoid multiple photos of same specimen series (e.g. Norway 2025 IMG 1771, 1785...)
        stem_clean = re.sub(r"\d+", "#", title.lower())[:38]
        if stem_clean in specimen_stems:
            continue

        ii_list = page.get("imageinfo", [])
        if not ii_list:
            continue
        ii = ii_list[0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        w_orig = int(ii.get("width", 0))
        h_orig = int(ii.get("height", 0))
        if w_orig < 260 or h_orig < 260:
            continue

        ext = ii.get("extmetadata", {})
        lic = (ext.get("LicenseShortName", {}) or {}).get("value", "CC BY-SA")
        artist_raw = (ext.get("Artist", {}) or {}).get("value", "Wikimedia Commons contributor")
        artist = re.sub(r"<[^>]+>", "", artist_raw).strip() or "Wikimedia Commons contributor"
        dl_url = ii.get("thumburl") or ii.get("url")
        if not dl_url:
            continue

        try:
            resp = SESSION.get(dl_url, timeout=20)
            if resp.status_code != 200:
                continue
            raw_bytes = resp.content
            with Image.open(io.BytesIO(raw_bytes)) as im_in:
                im_norm = ImageOps.exif_transpose(im_in).convert("RGB")
                buf = io.BytesIO()
                im_norm.save(buf, format="JPEG", quality=92)
                norm_bytes = buf.getvalue()
                w, h = im_norm.size
                dhash_val = compute_dhash(im_norm)
        except Exception:
            continue

        sha = hashlib.sha256(norm_bytes).hexdigest()
        if sha in existing_sha256:
            continue
        if any(hamming_dist(dhash_val, dh) <= 6 for dh in existing_dhashes):
            continue

        existing_sha256.add(sha)
        existing_dhashes.append(dhash_val)
        existing_titles.add(title)
        specimen_stems.add(stem_clean)

        cand_id = f"task8d_{next_num:04d}"
        next_num += 1
        img_rel = f"data/acquisition_staging/task8d_high_yield_real_waste/images/{cand_id}.jpg"
        (ROOT / img_rel).write_bytes(norm_bytes)

        rec = {
            "candidate_id": cand_id,
            "source_id": "SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES",
            "source_name": "Wikimedia Commons Waste Categories & Bitmap Queries",
            "source_image_id": f"commons_{page.get('pageid', cand_id)}",
            "title": title,
            "query_used": query_tag,
            "original_url": ii.get("url", dl_url),
            "author": artist[:120],
            "license": lic,
            "attribution_required": not ("cc0" in lic.lower() or "public domain" in lic.lower()),
            "image_path": img_rel,
            "sha256": sha,
            "dhash_hex": f"{dhash_val:016x}",
            "dhash_int": dhash_val,
            "width": w,
            "height": h,
            "primary_class_id": cid,
            "primary_class_name": CLASS_NAMES[cid],
            "classes_present_ids": [cid],
            "classes_present_names": [CLASS_NAMES[cid]],
            "is_multiclass": False,
            "proposed_boxes": [
                {
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "bbox_xyxy": [round(0.08 * w, 1), round(0.08 * h, 1), round(0.92 * w, 1), round(0.92 * h, 1)],
                    "yolo_xywhn": [0.5, 0.5, 0.84, 0.84],
                    "source_box_origin": "wikimedia_pending_manual_tight_box",
                }
            ],
        }
        records.append(rec)
        per_class_added[cid] += 1
        added += 1

    manifest["total_candidates"] = len(records)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    new_sheets = render_new_inspection_sheets(records, start_idx)
    print(f"Added {added} new Wikimedia waste-category candidates (total={len(records)})")
    print("Per class added:", per_class_added)
    print("New inspection sheets:", new_sheets)


if __name__ == "__main__":
    main()
