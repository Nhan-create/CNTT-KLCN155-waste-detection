# Mô tả: [Lưu trữ — Kịch bản mở rộng & kiểm định dữ liệu đa lớp (Task 8)] Tìm kiếm, tải về và mở rộng mẫu ảnh thực tế bổ sung cho `expand_task8a_candidates`.
"""
scripts/expand_task8a_candidates.py
-----------------------------------
Uses Wikimedia Commons `generator=categorymembers` on verified real-world waste
photographic categories to expand `data/acquisition_staging/task8a_multiclass_batch/raw_download_manifest.json`
across all 5 priority classes:
  - 1: biological (`Category:Banana peel`, `Category:Apple cores`, `Category:Orange peels`, `Category:Egg shells`, `Category:Food waste`)
  - 6: paper (`Category:Crumpled paper`, `Category:Paper coffee cups`, `Category:Paper cups`)
  - 2: cardboard (`Category:Pizza boxes`, `Category:Pizza trash cans`, `Category:Cardboard boxes`, `Category:Banana boxes`)
  - 4: glass (`Category:Broken glass bottles`, `Category:Broken bottles`, `Category:Broken glass bottles on beaches`, `Category:Empty bottles`)
  - 8: shoes (`Category:Abandoned shoes`, `Category:Abandoned shoes in London`, `Category:Abandoned boots`, `Category:Used shoes`, `Category:Dirty shoes`, `Category:Muddy shoes`)
"""

from __future__ import annotations

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

from scripts.search_and_download_task8a_candidates import (
    ALLOWED_LICENSE_KEYWORDS,
    EXCLUDE_TITLE_KEYWORDS,
    GRIDS_DIR,
    IMAGES_DIR,
    RAW_MANIFEST_P,
    REPO_ROOT,
    clean_html,
    load_existing_titles_and_hashes,
    render_grid_sheet,
    sha256_file,
)

HEADERS = {
    "User-Agent": "WasteDetectionAcademicBot/2.3 (https://commons.wikimedia.org/; contact: waste_research_klcn155@university.edu)"
}

CATEGORY_SOURCES = [
    # --- 1: biological (need ~14 more QA_PASS -> fetch ~18 from curated categories) ---
    ("bio_banana_peel_cat", "1: biological", "Category:Banana peel", 5),
    ("bio_apple_cores_cat", "1: biological", "Category:Apple cores", 4),
    ("bio_orange_peels_cat", "1: biological", "Category:Orange peels", 4),
    ("bio_eggshells_cat", "1: biological", "Category:Egg shells", 4),
    ("bio_food_waste_cat", "1: biological", "Category:Food waste", 4),

    # --- 6: paper (need ~5 more QA_PASS -> fetch ~9 from curated categories) ---
    ("paper_crumpled_cat", "6: paper", "Category:Crumpled paper", 4),
    ("paper_coffee_cups_cat", "6: paper", "Category:Paper coffee cups", 4),
    ("paper_cups_cat", "6: paper", "Category:Paper cups", 2),

    # --- 2: cardboard (need ~5 more QA_PASS -> fetch ~9 from curated categories) ---
    ("cardboard_pizza_boxes_cat", "2: cardboard", "Category:Pizza boxes", 4),
    ("cardboard_pizza_trash_cat", "2: cardboard", "Category:Pizza trash cans", 3),
    ("cardboard_banana_boxes_cat", "2: cardboard", "Category:Banana boxes", 2),
    ("cardboard_boxes_cat", "2: cardboard", "Category:Cardboard boxes", 2),

    # --- 4: glass (need ~8 more QA_PASS -> fetch ~12 from curated categories) ---
    ("glass_broken_bottles_cat", "4: glass", "Category:Broken glass bottles", 4),
    ("glass_broken_beach_cat", "4: glass", "Category:Broken glass bottles on beaches", 3),
    ("glass_broken_cat", "4: glass", "Category:Broken bottles", 3),
    ("glass_empty_bottles_cat", "4: glass", "Category:Empty bottles", 4),

    # --- 8: shoes (need ~6 more QA_PASS -> fetch ~10 from curated categories) ---
    ("shoes_abandoned_cat", "8: shoes", "Category:Abandoned shoes", 4),
    ("shoes_abandoned_london_cat", "8: shoes", "Category:Abandoned shoes in London", 3),
    ("shoes_abandoned_boots_cat", "8: shoes", "Category:Abandoned boots", 2),
    ("shoes_used_cat", "8: shoes", "Category:Used shoes", 2),
    ("shoes_dirty_cat", "8: shoes", "Category:Dirty shoes", 2),
]

EXTRA_EXCLUDE = EXCLUDE_TITLE_KEYWORDS + [
    "dpla", "hollar", "smock", "raree", "psm", "cartoon", "caricature", "18th", "19th",
    "lithophane", "moth", "insect", "beetle", "microscope", "peelmold", "x-ray", "xray",
    "radiograph", "monument", "church", "ruins", "sculpture", "oldenburg", "paracommons",
]


def api_get_category(cmtitle: str) -> Dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": cmtitle,
        "gcmtype": "file",
        "gcmlimit": 25,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1024,
    }
    qs = urllib.parse.urlencode(params)
    url = f"https://commons.wikimedia.org/w/api.php?{qs}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    existing_data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    existing_candidates: List[Dict[str, Any]] = existing_data["candidates"]
    start_count = len(existing_candidates)

    seen_titles, seen_sha256 = load_existing_titles_and_hashes()
    for c in existing_candidates:
        seen_titles.add(c["source_file_title"])
        seen_sha256.add(c["sha256"])
    seen_urls: Set[str] = {c["download_url"] for c in existing_candidates}

    newly_added: List[Dict[str, Any]] = []

    for subgroup, target_cls, cmtitle, max_for_group in CATEGORY_SOURCES:
        time.sleep(0.35)
        try:
            res = api_get_category(cmtitle)
        except Exception as e:
            print(f"Category query error on {cmtitle}: {e}")
            time.sleep(1.5)
            continue

        pages = res.get("query", {}).get("pages", {})
        sorted_pages = sorted(pages.values(), key=lambda p: p.get("title", ""))
        group_count = 0

        for page in sorted_pages:
            if group_count >= max_for_group:
                break
            pid = int(page.get("pageid", -1))
            if pid < 0:
                continue
            title = page.get("title", "")
            if not title or title in seen_titles:
                continue
            lower_title = title.lower()
            if any(k.lower() in lower_title for k in EXTRA_EXCLUDE):
                continue
            if not (lower_title.endswith(".jpg") or lower_title.endswith(".jpeg") or lower_title.endswith(".png")):
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
            if orig_w < 450 or orig_h < 350:
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

            idx = len(existing_candidates) + 1
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.replace("File:", "")).strip("_").lower()[:32]
            ext = ".png" if dl_url.lower().endswith(".png") else ".jpg"
            cand_id = f"task8a_{idx:03d}_{subgroup}_{slug}"
            filename = f"{cand_id}{ext}"
            dest_path = IMAGES_DIR / filename

            try:
                req = urllib.request.Request(dl_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                dest_path.write_bytes(data)
                time.sleep(0.25)

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
                    "source_name": f"Wikimedia Commons ({cmtitle})",
                    "wikimedia_page_id": pid,
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
                existing_candidates.append(rec)
                newly_added.append(rec)
                print(f"  [ADDED #{idx:03d}] {filename} ({w}x{h}, {fsize:,} B, {target_cls}, lic={rec['license']})")
            except Exception as e:
                print(f"  [SKIP] {filename}: {e}")
                dest_path.unlink(missing_ok=True)
                time.sleep(0.8)

    total_bytes = sum(c["downloaded_bytes"] for c in existing_candidates)
    by_class_counts: Dict[str, int] = {}
    for r in existing_candidates:
        tc = r["target_class"]
        by_class_counts[tc] = by_class_counts.get(tc, 0) + 1

    existing_data["total_downloaded"] = len(existing_candidates)
    existing_data["total_downloaded_bytes"] = total_bytes
    existing_data["total_downloaded_mb"] = round(total_bytes / (1024 * 1024), 2)
    existing_data["raw_counts_by_target_class"] = by_class_counts
    existing_data["candidates"] = existing_candidates
    RAW_MANIFEST_P.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAdded {len(newly_added)} category candidates. Total = {len(existing_candidates)} ({existing_data['total_downloaded_mb']} MB)")
    print("Total raw counts by target class:", json.dumps(by_class_counts, indent=2))

    chunk_size = 18
    sheet_num = 6
    for i in range(start_count, len(existing_candidates), chunk_size):
        chunk = existing_candidates[i : i + chunk_size]
        start_num = i + 1
        end_num = i + len(chunk)
        sheet_p = GRIDS_DIR / f"raw_grid_sheet_{sheet_num:02d}_{start_num:03d}_to_{end_num:03d}.jpg"
        render_grid_sheet(chunk, sheet_p, cols=3)
        print(f"Rendered inspection grid: {sheet_p.name}")
        sheet_num += 1


if __name__ == "__main__":
    main()
