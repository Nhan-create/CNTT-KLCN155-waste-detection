# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Tìm kiếm, tải về và mở rộng mẫu ảnh thực tế bổ sung cho `fetch_additional_valid_task7_batteries`.
"""
scripts/fetch_additional_valid_task7_batteries.py
-------------------------------------------------
Fetches ~14 additional real-world multi-battery, 9V, NiMH, Li-ion, and corroded battery
photographs from Wikimedia Commons and renders `raw_grid_sheet_43_to_56.jpg` for visual inspection.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
IMAGES_DIR = OUT_DIR / "images"
RAW_MANIFEST_P = OUT_DIR / "raw_download_manifest.json"
GRIDS_DIR = OUT_DIR / "inspection_grids"

HEADERS = {"User-Agent": "WasteDetectionResearchBot/2.0 (academic research on waste sorting; contact: student_research@university.edu)"}

EXTRA_QUERIES = [
    ("multi_battery_assorted", "0: battery", "Duracell batteries", 3),
    ("multi_battery_assorted", "0: battery", "Energizer batteries", 3),
    ("multi_battery_assorted", "0: battery", "Varta batteries", 3),
    ("multi_battery_assorted", "0: battery", "NiMH rechargeable batteries", 3),
    ("phone_liion_swollen", "0: battery", "Nokia battery BL-5C", 2),
    ("phone_liion_swollen", "0: battery", "swollen battery", 3),
]

ALLOWED_LICENSE_KEYWORDS = ["cc", "public domain", "pd", "attribution", "free", "zero"]
EXCLUDE_KEYWORDS = [
    "diagram", "schema", "symbol", "icon", "logo", "chart", "graph", "map",
    "drawing", "vector", "svg", "patent", "illustration", "poster", "sign",
    "artillery", "cannon", "missile", "fort", "bunker", "castle", "ship",
    "truck", "vehicle", "car", "automobile", "bus", "train", "locomotive",
    "factory", "building", "store", "shop", "charger", "solar",
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


def main() -> None:
    from scripts.generate_task7_raw_inspection_grids import render_grid_sheet

    existing_data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    existing_candidates: List[Dict[str, Any]] = existing_data["candidates"]
    seen_titles = {c["source_file_title"] for c in existing_candidates}
    seen_urls = {c["download_url"] for c in existing_candidates}
    next_idx = len(existing_candidates) + 1
    newly_added: List[Dict[str, Any]] = []

    for subgroup, target_cls, query_str, max_for_group in EXTRA_QUERIES:
        if len(newly_added) >= 14:
            break
        srsearch = f"{query_str} filetype:bitmap"
        try:
            search_res = api_get({
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srlimit": 15,
                "srsearch": srsearch,
            })
        except Exception as e:
            print(f"Search error on {query_str}: {e}")
            continue

        hits = search_res.get("query", {}).get("search", [])
        added = 0
        for hit in hits:
            if added >= max_for_group or len(newly_added) >= 14:
                break
            title = hit.get("title", "")
            if not title or title in seen_titles:
                continue
            lower_title = title.lower()
            if any(k in lower_title for k in EXCLUDE_KEYWORDS):
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
                combined_lic = f"{lic_short} {usage_terms}".lower()
                if not any(k in combined_lic for k in ALLOWED_LICENSE_KEYWORDS):
                    continue

                desc = clean_html(meta.get("ImageDescription", {}).get("value", ""))[:300]
                if any(k in desc.lower() for k in EXCLUDE_KEYWORDS):
                    continue

                author = clean_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Contributor"

                if orig_size <= 2_500_000:
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

                slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.replace("File:", "")).strip("_").lower()[:34]
                ext = ".png" if dl_url.lower().endswith(".png") else ".jpg"
                cand_id = f"task7_{next_idx:02d}_{subgroup}_{slug}"
                filename = f"{cand_id}{ext}"
                dest_path = IMAGES_DIR / filename

                try:
                    req = urllib.request.Request(dl_url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = resp.read()
                    dest_path.write_bytes(data)
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
                    rec = {
                        "candidate_id": cand_id,
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
                        "author": author[:120],
                        "expected_width": dl_w,
                        "expected_height": dl_h,
                        "original_file_size_bytes": orig_size,
                        "source_description": desc,
                        "download_status": "DOWNLOADED_VERIFIED",
                        "local_path": f"data/acquisition_staging/task7_diverse_battery_batch/images/{filename}",
                        "actual_width": w,
                        "actual_height": h,
                        "downloaded_bytes": fsize,
                        "sha256": sha256_file(dest_path),
                        "phash_0": ph0,
                        "phash_rotations": [ph0, ph90, ph180, ph270],
                    }
                    existing_candidates.append(rec)
                    newly_added.append(rec)
                    seen_titles.add(title)
                    seen_urls.add(dl_url)
                    next_idx += 1
                    added += 1
                    print(f"  [ADDED #{len(existing_candidates)}] {filename} ({w}x{h}, {fsize:,} bytes)")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  [SKIP] {filename}: {e}")
                    if dest_path.exists():
                        dest_path.unlink()

    total_bytes = sum(c["downloaded_bytes"] for c in existing_candidates)
    existing_data["total_downloaded"] = len(existing_candidates)
    existing_data["total_downloaded_bytes"] = total_bytes
    existing_data["total_downloaded_mb"] = round(total_bytes / (1024 * 1024), 2)
    existing_data["candidates"] = existing_candidates
    RAW_MANIFEST_P.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")

    s3 = GRIDS_DIR / "raw_grid_sheet_43_to_56.jpg"
    render_grid_sheet(newly_added, s3, cols=3)
    print(f"Generated {s3} with {len(newly_added)} new candidates (total={len(existing_candidates)})")


if __name__ == "__main__":
    main()
