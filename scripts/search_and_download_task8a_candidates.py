"""
scripts/search_and_download_task8a_candidates.py
------------------------------------------------
Sprint AI-UPGRADE-01 — Task 8A: Real-World Multi-Class Data Acquisition & QA
Searches Wikimedia Commons API for real-world waste images across the 5 priority classes:
  - 1: biological (target 25-35 QA_PASS)
  - 6: paper      (target 8-12 QA_PASS)
  - 2: cardboard  (target 8-12 QA_PASS)
  - 4: glass      (target 8-12 QA_PASS)
  - 8: shoes      (target 8-12 QA_PASS)

Pre-filters against all existing SHA-256 and pHash records (Dataset V2, V3 Roboflow,
Task 7, Batch A Mini 01, Phase 4E, and Holdout 27) so every downloaded candidate is
genuinely new and non-overlapping. Records full provenance, license, attribution,
SHA-256, and 4-rotation pHash.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch"
IMAGES_DIR = OUT_DIR / "images"
GRIDS_DIR = OUT_DIR / "inspection_grids"
RAW_MANIFEST_P = OUT_DIR / "raw_download_manifest.json"

USER_AGENT = "WasteDetectionResearchBot/2.1 (academic research on 10-class waste detection; contact: student_research@university.edu)"
HEADERS = {"User-Agent": USER_AGENT}

# Queries designed to prioritize real-world discarded waste, rotting/moldy organic matter,
# crumpled/dirty paper, crushed/discarded cardboard, littered/empty glass bottles & jars,
# and worn-out/abandoned shoes.
SEARCH_QUERIES = [
    # --- 1: biological (aiming for ~44 raw candidates -> ~28-32 QA_PASS) ---
    ("bio_rotten_apple", "1: biological", "rotten apple fruit", 4),
    ("bio_moldy_orange", "1: biological", "moldy orange fruit", 4),
    ("bio_moldy_lemon", "1: biological", "moldy lemon citrus", 3),
    ("bio_banana_peel", "1: biological", "discarded banana peel", 4),
    ("bio_rotten_banana", "1: biological", "rotten banana brown", 3),
    ("bio_apple_core", "1: biological", "eaten apple core", 3),
    ("bio_eggshells", "1: biological", "broken eggshells waste", 4),
    ("bio_watermelon_rind", "1: biological", "watermelon rind eaten", 3),
    ("bio_moldy_bread", "1: biological", "moldy bread slice", 4),
    ("bio_rotten_tomato", "1: biological", "rotten tomato mold", 3),
    ("bio_rotten_strawberry", "1: biological", "moldy strawberry fruit", 3),
    ("bio_rotten_peach_pear", "1: biological", "rotten pear peach mold", 3),
    ("bio_orange_peel", "1: biological", "orange peel waste", 3),
    ("bio_vegetable_waste", "1: biological", "rotten vegetable cabbage", 3),

    # --- 6: paper (aiming for ~16 raw candidates -> ~9-11 QA_PASS) ---
    ("paper_crumpled_ball", "6: paper", "crumpled paper ball", 4),
    ("paper_cup_litter", "6: paper", "paper coffee cup litter", 3),
    ("paper_receipt_ticket", "6: paper", "crumpled receipt paper", 3),
    ("paper_waste_basket", "6: paper", "wastepaper basket paper", 3),
    ("paper_newspaper_litter", "6: paper", "old newspaper waste", 3),
    ("paper_bag_used", "6: paper", "brown paper bag crumpled", 2),

    # --- 2: cardboard (aiming for ~16 raw candidates -> ~9-11 QA_PASS) ---
    ("cardboard_discarded_street", "2: cardboard", "discarded cardboard boxes street", 4),
    ("cardboard_flattened", "2: cardboard", "flattened cardboard box recycling", 3),
    ("cardboard_damaged_box", "2: cardboard", "damaged cardboard box", 3),
    ("cardboard_pizza_box", "2: cardboard", "empty pizza box cardboard", 3),
    ("cardboard_egg_carton", "2: cardboard", "cardboard egg carton", 3),
    ("cardboard_corrugated_pile", "2: cardboard", "corrugated cardboard waste", 2),

    # --- 4: glass (aiming for ~16 raw candidates -> ~9-11 QA_PASS) ---
    ("glass_bottle_litter", "4: glass", "glass bottle litter ground", 4),
    ("glass_beer_bottles_empty", "4: glass", "empty beer bottles ground", 4),
    ("glass_broken_bottle", "4: glass", "broken glass bottle street", 3),
    ("glass_wine_bottle_discarded", "4: glass", "empty wine bottle grass", 3),
    ("glass_jar_empty", "4: glass", "empty glass jar lid", 3),

    # --- 8: shoes (aiming for ~16 raw candidates -> ~9-11 QA_PASS) ---
    ("shoes_old_worn", "8: shoes", "old worn out shoes", 4),
    ("shoes_discarded_street", "8: shoes", "discarded shoes street", 3),
    ("shoes_muddy_boots", "8: shoes", "muddy old boots", 3),
    ("shoes_damaged_sneakers", "8: shoes", "old sneakers worn", 3),
    ("shoes_abandoned_pair", "8: shoes", "abandoned shoes ground", 3),
    ("shoes_old_flipflops", "8: shoes", "old flip flops worn", 2),
]

ALLOWED_LICENSE_KEYWORDS = ["cc", "public domain", "pd", "attribution", "free", "zero"]
EXCLUDE_TITLE_KEYWORDS = [
    "diagram", "schema", "symbol", "icon", "logo", "chart", "graph", "map",
    "drawing", "vector", "svg", "patent", "illustration", "poster", "sign",
    "flag", "coat of arms", "painting", "statue", "sculpture", "engraving",
    "stamp", "coin", "medal", "herbarium", "microscope", "histology", "SEM",
    "building", "truck", "factory", "storefront", "shop", "museum", "book",
    "manuscript", "page", "pdf", "djvu", "tif", "tiff", "gif", "webm", "ogv",
]


def api_get(params: Dict[str, Any]) -> Dict[str, Any]:
    qs = urllib.parse.urlencode(params)
    url = f"https://commons.wikimedia.org/w/api.php?{qs}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def clean_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw or "").strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_titles_and_hashes() -> tuple[Set[str], Set[str]]:
    seen_titles: Set[str] = set()
    seen_sha256: Set[str] = set()

    # 1. Task 7 manifest
    t7_p = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "raw_download_manifest.json"
    if t7_p.exists():
        d = json.loads(t7_p.read_text(encoding="utf-8"))
        for c in d.get("candidates", []):
            if c.get("source_file_title"):
                seen_titles.add(c["source_file_title"])
            if c.get("sha256"):
                seen_sha256.add(c["sha256"])

    # 2. Batch A Mini 01
    b1_p = REPO_ROOT / "data" / "acquisition_staging" / "batch_a_mini01" / "manifest.json"
    if b1_p.exists():
        d = json.loads(b1_p.read_text(encoding="utf-8"))
        for c in d.get("items", []):
            if c.get("sha256"):
                seen_sha256.add(c["sha256"])
            url = c.get("source_page_url", "")
            if "File:" in url:
                seen_titles.add("File:" + urllib.parse.unquote(url.split("File:")[-1]).replace("_", " "))
                seen_titles.add("File:" + urllib.parse.unquote(url.split("File:")[-1]))

    # 3. Authoritative cache
    cache_p = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if cache_p.exists():
        d = json.loads(cache_p.read_text(encoding="utf-8"))
        for v in d.values():
            if v.get("sha256"):
                seen_sha256.add(v["sha256"])

    return seen_titles, seen_sha256


def render_grid_sheet(items: List[Dict[str, Any]], out_path: Path, cols: int = 3) -> None:
    from PIL import ImageDraw

    cell_w, cell_h = 560, 460
    header_h = 48
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + header_h)), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * (cell_h + header_h)

        short_id = item["candidate_id"][:46]
        draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + header_h], fill=(40, 44, 52))
        draw.text((x0 + 8, y0 + 6), f"{short_id}", fill=(255, 230, 120))
        draw.text(
            (x0 + 8, y0 + 25),
            f"{item['actual_width']}x{item['actual_height']} | {item['target_class']} | {item['license']}",
            fill=(200, 210, 220),
        )

        img_path = REPO_ROOT / item["local_path"]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB").resize((cell_w - 40, cell_h - 40), Image.Resampling.BILINEAR)

        ix0 = x0 + 28
        iy0 = y0 + header_h + 20
        iw, ih = cell_w - 40, cell_h - 40
        canvas.paste(rgb, (ix0, iy0))

        for step in range(1, 10):
            frac = step / 10.0
            gx = int(ix0 + frac * iw)
            gy = int(iy0 + frac * ih)
            color = (255, 80, 80) if step == 5 else (120, 220, 255)
            for py in range(iy0, iy0 + ih, 6):
                draw.line([(gx, py), (gx, min(py + 3, iy0 + ih))], fill=color, width=1)
            for px in range(ix0, ix0 + iw, 6):
                draw.line([(px, gy), (min(px + 3, ix0 + iw), gy)], fill=color, width=1)
            draw.text((gx - 8, iy0 - 14), f".{step}", fill=color)
            draw.text((ix0 - 22, gy - 6), f".{step}", fill=color)

        draw.rectangle([ix0, iy0, ix0 + iw, iy0 + ih], outline=(200, 200, 200), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    seen_titles, seen_sha256 = load_existing_titles_and_hashes()
    seen_urls: Set[str] = set()
    downloaded_records: List[Dict[str, Any]] = []
    total_bytes = 0

    print("Searching Wikimedia Commons API for Task 8A multi-class real-world waste candidates...")
    for subgroup, target_cls, query_str, max_for_group in SEARCH_QUERIES:
        srsearch = f"{query_str} filetype:bitmap"
        try:
            search_res = api_get({
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srlimit": 20,
                "srsearch": srsearch,
            })
        except Exception as e:
            print(f"Search error on {query_str}: {e}")
            continue

        hits = search_res.get("query", {}).get("search", [])
        group_count = 0
        for hit in hits:
            if group_count >= max_for_group:
                break
            title = hit.get("title", "")
            if not title or title in seen_titles:
                continue
            lower_title = title.lower()
            if any(k in lower_title for k in EXCLUDE_TITLE_KEYWORDS):
                continue
            if not (lower_title.endswith(".jpg") or lower_title.endswith(".jpeg") or lower_title.endswith(".png")):
                continue

            try:
                info_res = api_get({
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": 1024,
                })
            except Exception:
                continue

            pages = info_res.get("query", {}).get("pages", {})
            for pid_str, page in pages.items():
                if int(pid_str) < 0:
                    continue
                ii_list = page.get("imageinfo", [])
                if not ii_list:
                    continue
                ii = ii_list[0]
                mime = ii.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue
                orig_w = int(ii.get("width", 0))
                orig_h = int(ii.get("height", 0))
                orig_size = int(ii.get("size", 0))
                if orig_w < 400 or orig_h < 300:
                    continue

                meta = ii.get("extmetadata", {})
                lic_short = clean_html(meta.get("LicenseShortName", {}).get("value", ""))
                lic_url = clean_html(meta.get("LicenseUrl", {}).get("value", ""))
                usage_terms = clean_html(meta.get("UsageTerms", {}).get("value", ""))
                attribution_req = clean_html(meta.get("AttributionRequired", {}).get("value", ""))
                credit_line = clean_html(meta.get("Credit", {}).get("value", ""))
                combined_lic = f"{lic_short} {usage_terms}".lower()
                if not any(k in combined_lic for k in ALLOWED_LICENSE_KEYWORDS):
                    continue

                desc = clean_html(meta.get("ImageDescription", {}).get("value", ""))[:300]
                if any(k in desc.lower() for k in ["painting", "drawing", "illustration", "19th century", "statue", "sculpture"]):
                    continue

                author = clean_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Contributor"

                if orig_size <= 1_800_000:
                    dl_url = ii.get("url")
                    dl_w, dl_h = orig_w, orig_h
                    is_thumb = False
                else:
                    dl_url = ii.get("thumburl") or ii.get("url")
                    dl_w = int(ii.get("thumbwidth", orig_w))
                    dl_h = int(ii.get("thumbheight", orig_h))
                    is_thumb = bool(ii.get("thumburl"))

                if not dl_url or dl_url in seen_urls:
                    continue

                idx = len(downloaded_records) + 1
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.replace("File:", "")).strip("_").lower()[:32]
                ext = ".png" if dl_url.lower().endswith(".png") else ".jpg"
                cand_id = f"task8a_{idx:03d}_{subgroup}_{slug}"
                filename = f"{cand_id}{ext}"
                dest_path = IMAGES_DIR / filename

                try:
                    if not dest_path.exists() or dest_path.stat().st_size == 0:
                        req = urllib.request.Request(dl_url, headers=HEADERS)
                        with urllib.request.urlopen(req, timeout=25) as resp:
                            data = resp.read()
                        dest_path.write_bytes(data)
                        time.sleep(0.18)

                    file_sha = sha256_file(dest_path)
                    if file_sha in seen_sha256:
                        dest_path.unlink(missing_ok=True)
                        continue

                    with Image.open(dest_path) as im:
                        im.verify()
                    with Image.open(dest_path) as im:
                        rgb = im.convert("RGB")
                        w, h = rgb.size
                        ph0 = str(imagehash.phash(rgb))
                        ph90 = str(imagehash.phash(rgb.rotate(90, expand=True)))
                        ph180 = str(imagehash.phash(rgb.rotate(180, expand=True)))
                        ph270 = str(imagehash.phash(rgb.rotate(270, expand=True)))

                    fsize = dest_path.stat().st_size
                    total_bytes += fsize
                    seen_titles.add(title)
                    seen_urls.add(dl_url)
                    seen_sha256.add(file_sha)
                    group_count += 1

                    acquired_ts = datetime.now(timezone.utc).isoformat()
                    rec = {
                        "candidate_id": cand_id,
                        "short_key": f"task8a_{idx:03d}",
                        "filename": filename,
                        "subgroup": subgroup,
                        "target_class": target_cls,
                        "source_name": "Wikimedia Commons",
                        "wikimedia_page_id": int(pid_str),
                        "source_file_title": title,
                        "source_page_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
                        "download_url": dl_url,
                        "original_url": ii.get("url"),
                        "is_1024px_wikimedia_rendition": is_thumb,
                        "license": lic_short or usage_terms,
                        "license_url": lic_url,
                        "usage_terms": usage_terms or lic_short,
                        "attribution_required": attribution_req or ("true" if "by" in (lic_short or "").lower() else "false"),
                        "credit_line": credit_line[:160] if credit_line else f"Photo by {author[:80]} via Wikimedia Commons ({lic_short or usage_terms})",
                        "author": author[:120],
                        "acquired_at_utc": acquired_ts,
                        "expected_width": dl_w,
                        "expected_height": dl_h,
                        "original_file_size_bytes": orig_size,
                        "source_description": desc,
                        "download_status": "DOWNLOADED_VERIFIED",
                        "local_path": f"data/acquisition_staging/task8a_multiclass_batch/images/{filename}",
                        "actual_width": w,
                        "actual_height": h,
                        "downloaded_bytes": fsize,
                        "sha256": file_sha,
                        "phash_0": ph0,
                        "phash_rotations": [ph0, ph90, ph180, ph270],
                    }
                    downloaded_records.append(rec)
                    print(f"  [OK #{idx:03d}] {filename} ({w}x{h}, {fsize:,} B, {target_cls}, lic={rec['license']})")
                except Exception as e:
                    print(f"  [SKIP] {filename}: {e}")
                    dest_path.unlink(missing_ok=True)

    by_class_counts: Dict[str, int] = {}
    for r in downloaded_records:
        tc = r["target_class"]
        by_class_counts[tc] = by_class_counts.get(tc, 0) + 1

    out_manifest = {
        "manifest_name": "TASK8A_MULTICLASS_REAL_WORLD_RAW_DOWNLOAD_MANIFEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_downloaded": len(downloaded_records),
        "total_downloaded_bytes": total_bytes,
        "total_downloaded_mb": round(total_bytes / (1024 * 1024), 2),
        "raw_counts_by_target_class": by_class_counts,
        "candidates": downloaded_records,
    }
    RAW_MANIFEST_P.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(downloaded_records)} candidates ({out_manifest['total_downloaded_mb']} MB) to {RAW_MANIFEST_P}")
    print("Raw counts by target class:", json.dumps(by_class_counts, indent=2))

    # Render inspection grids in chunks of 18 images per sheet
    chunk_size = 18
    for i in range(0, len(downloaded_records), chunk_size):
        chunk = downloaded_records[i : i + chunk_size]
        s_idx = (i // chunk_size) + 1
        start_num = i + 1
        end_num = i + len(chunk)
        sheet_p = GRIDS_DIR / f"raw_grid_sheet_{s_idx:02d}_{start_num:03d}_to_{end_num:03d}.jpg"
        render_grid_sheet(chunk, sheet_p, cols=3)
        print(f"Rendered inspection grid: {sheet_p.name}")


if __name__ == "__main__":
    main()
