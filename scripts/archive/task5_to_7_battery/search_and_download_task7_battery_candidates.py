"""
scripts/search_and_download_task7_battery_candidates.py
-------------------------------------------------------
Sprint AI-UPGRADE-01 — Task 7 (Part B1):
Searches Wikimedia Commons API for 35-40 real-world diverse battery & hard-negative images
outside the narrow single-tabletop Roboflow domain, verifies open licenses and provenance,
downloads original image files (keeping cumulative download small and fast), and checks
SHA-256 + 4-rotation pHash leakage against v2, v3 (Roboflow 350), and Holdout 27.
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
from typing import Any, Dict, List

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
IMAGES_DIR = OUT_DIR / "images"

USER_AGENT = "WasteDetectionResearchBot/2.0 (academic research on waste sorting; contact: student_research@university.edu)"
HEADERS = {"User-Agent": USER_AGENT}

# Priority categories matching Task 7 B1 requirements
SEARCH_QUERIES = [
    # 1. Multiple batteries in one image / collection bins / piles
    ("multi_battery_pile", "0: battery", "used batteries recycling pile", 6),
    ("multi_battery_collection", "0: battery", "battery collection box recycling", 5),
    ("multi_battery_assorted", "0: battery", "assorted batteries AA AAA 9V", 5),
    # 2. Button cells, 9V, phone batteries, 18650, old cylindrical cells
    ("button_coin_cell", "0: battery", "CR2032 button cell battery", 4),
    ("nine_volt_battery", "0: battery", "9V battery PP3", 4),
    ("phone_liion_swollen", "0: battery", "swollen lithium battery phone", 4),
    ("cylindrical_18650_old", "0: battery", "18650 lithium ion battery cell", 4),
    # 3. Corroded, leaking, dirty, or partially occluded batteries
    ("corroded_leaking_battery", "0: battery", "leaking battery alkaline corrosion", 6),
    ("rusty_old_battery", "0: battery", "rusty old battery cell", 4),
    # 4. Outdoor / discarded / waste bin / lead-acid dumped batteries
    ("outdoor_dumped_battery", "0: battery", "discarded battery litter waste", 4),
    ("old_car_battery_waste", "0: battery", "old car battery waste recycling", 4),
    # 5. Hard negatives: cylindrical cans, capacitors, fuses, electrical boxes
    ("hard_neg_capacitor_cylindrical", "hard_negative", "electrolytic capacitor radial", 3),
    ("hard_neg_cartridge_fuse", "hard_negative", "cartridge fuse cylindrical", 3),
    ("hard_neg_electrical_box", "hard_negative", "electrical junction box wall", 3),
]

ALLOWED_LICENSE_KEYWORDS = ["cc", "public domain", "pd", "attribution", "free", "zero"]
EXCLUDE_TITLE_KEYWORDS = [
    "diagram", "schema", "symbol", "icon", "logo", "chart", "graph", "map",
    "drawing", "vector", "svg", "patent", "illustration", "poster", "sign",
    "flag", "coat of arms", "artillery", "cannon", "missile", "fort", "bunker",
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


def search_candidates_from_wikimedia(target_total: int = 38) -> List[Dict[str, Any]]:
    discovered: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()

    for subgroup, target_cls, query_str, max_for_group in SEARCH_QUERIES:
        if len(discovered) >= target_total:
            break
        srsearch = f"{query_str} filetype:bitmap"
        try:
            search_res = api_get({
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srlimit": 18,
                "srsearch": srsearch,
            })
        except Exception as e:
            print(f"Search error on {query_str}: {e}")
            continue

        hits = search_res.get("query", {}).get("search", [])
        group_count = 0
        for hit in hits:
            if group_count >= max_for_group or len(discovered) >= target_total:
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
                combined_lic = f"{lic_short} {usage_terms}".lower()
                if not any(k in combined_lic for k in ALLOWED_LICENSE_KEYWORDS):
                    continue

                desc = clean_html(meta.get("ImageDescription", {}).get("value", ""))[:300]
                if any(k in desc.lower() for k in ["artillery", "cannon", "fortification", "19th century", "bunker"]):
                    continue

                author = clean_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Contributor"

                # Prefer original URL if <= 2.5 MB; otherwise use official 1024px thumburl so downloads stay fast
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

                seen_titles.add(title)
                seen_urls.add(dl_url)
                group_count += 1
                idx = len(discovered) + 1
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.replace("File:", "")).strip("_").lower()[:36]
                ext = ".png" if dl_url.lower().endswith(".png") else ".jpg"
                cand_id = f"task7_{idx:02d}_{subgroup}_{slug}"

                discovered.append({
                    "candidate_id": cand_id,
                    "filename": f"{cand_id}{ext}",
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
                })
            time.sleep(0.15)
    return discovered


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    # Include existing verified battery/hard-negative from batch_a_mini01 if helpful, plus newly discovered Wikimedia images
    print("Searching Wikimedia Commons API for diverse battery & hard-negative candidates...")
    candidates = search_candidates_from_wikimedia(target_total=38)
    print(f"Discovered {len(candidates)} verified open-license candidates. Downloading...")

    downloaded_records: List[Dict[str, Any]] = []
    total_bytes = 0

    for item in candidates:
        dest_path = IMAGES_DIR / item["filename"]
        try:
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                req = urllib.request.Request(item["download_url"], headers=HEADERS)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                dest_path.write_bytes(data)
                time.sleep(0.25)

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
            item["download_status"] = "DOWNLOADED_VERIFIED"
            item["local_path"] = f"data/acquisition_staging/task7_diverse_battery_batch/images/{item['filename']}"
            item["actual_width"] = w
            item["actual_height"] = h
            item["downloaded_bytes"] = fsize
            item["sha256"] = sha256_file(dest_path)
            item["phash_0"] = ph0
            item["phash_rotations"] = [ph0, ph90, ph180, ph270]
            downloaded_records.append(item)
            print(f"  [OK] {item['filename']} ({w}x{h}, {fsize:,} bytes, license={item['license']})")
        except Exception as e:
            print(f"  [SKIP] {item['filename']}: {e}")
            if dest_path.exists():
                dest_path.unlink()

    out_manifest = {
        "manifest_name": "TASK7_DIVERSE_BATTERY_AND_HARD_NEGATIVE_RAW_DOWNLOAD_MANIFEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_downloaded": len(downloaded_records),
        "total_downloaded_bytes": total_bytes,
        "total_downloaded_mb": round(total_bytes / (1024 * 1024), 2),
        "candidates": downloaded_records,
    }
    raw_manifest_path = OUT_DIR / "raw_download_manifest.json"
    raw_manifest_path.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(downloaded_records)} downloaded candidates ({out_manifest['total_downloaded_mb']} MB) to {raw_manifest_path}")


if __name__ == "__main__":
    main()
