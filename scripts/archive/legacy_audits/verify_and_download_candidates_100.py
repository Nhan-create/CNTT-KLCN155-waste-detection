"""Verify and download 100 candidate images from Wikimedia Commons manifest.

Handles caching, individual title fallbacks for non-ASCII titles,
SHA-256 and pHash computation, and duplicate checks across:
- local 100 candidates
- 27 holdout images
- v2/v3 authoritative datasets
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image
import requests
import urllib3

sys.stdout.reconfigure(encoding="utf-8")
urllib3.disable_warnings()

BASE_DIR = (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout"))) / "demo_multiobject_100_20261012")
ORIGINALS_DIR = BASE_DIR / "originals"
MANIFESTS_DIR = BASE_DIR / "manifests"
REPORTS_DIR = BASE_DIR / "reports"

ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_JSON = MANIFESTS_DIR / "danh_sach_100_anh_test_10_nhan_candidates.json"

USER_AGENT = "WasteDetectionResearchBot/2.0 (academic study on environmental waste detection; contact@university.edu)"
HEADERS = {"User-Agent": USER_AGENT}


def get_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_authoritative_hashes() -> set[str]:
    auth_cache_file = Path("data/audit/authoritative_hashes_cache.json")
    if auth_cache_file.exists():
        with open(auth_cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.keys())
    return set()


def load_holdout_hashes() -> dict[str, str]:
    holdout_manifest = Path("reports/qa_acceptance/manifest_hinh_anh_rac.json")
    hashes = {}
    if holdout_manifest.exists():
        with open(holdout_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("files", []):
                h = item.get("sha256")
                fname = item.get("filename")
                if h and fname:
                    hashes[h] = fname
    return hashes


def fetch_single_title_metadata(session: requests.Session, title: str) -> dict[str, Any] | None:
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata|sha1",
        "format": "json",
    }
    try:
        r = session.get(api_url, params=params, verify=False, timeout=15)
        pages = r.json().get("query", {}).get("pages", {})
        for pid, pdata in pages.items():
            if int(pid) > 0 and "imageinfo" in pdata and pdata["imageinfo"]:
                return pdata
    except Exception as e:
        print(f"Error fetching metadata for {title}: {e}")
    return None


def main() -> None:
    print(f"Loading candidates manifest from: {CANDIDATES_JSON}")
    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    items = manifest_data["items"]
    print(f"Total candidate items: {len(items)}")

    authoritative_hashes = load_authoritative_hashes()
    holdout_hashes = load_holdout_hashes()
    print(f"Loaded {len(authoritative_hashes)} authoritative dataset hashes.")
    print(f"Loaded {len(holdout_hashes)} holdout hashes.")

    session = requests.Session()
    session.headers.update(HEADERS)

    # Check existing audit json if available
    audit_file = MANIFESTS_DIR / "candidate_100_download_audit.json"
    existing_records: dict[str, dict[str, Any]] = {}
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for rec in old_data.get("items", []):
                    if rec.get("download_success"):
                        dest = Path(rec.get("local_path", ""))
                        if dest.exists() and dest.stat().st_size > 0:
                            existing_records[rec["candidate_id"]] = rec
        except Exception:
            pass

    print(f"Already successfully downloaded and verified: {len(existing_records)} items.")

    download_records = []
    seen_sha256: dict[str, str] = {}
    seen_phash: dict[str, str] = {}

    success_count = 0
    fail_count = 0

    for idx, item in enumerate(items, 1):
        cid = item["candidate_id"]
        class_id = item["target_class_id"]
        class_name = item["target_class_name"]
        title = item["commons_file_title"]
        dest_filename = f"{cid}.jpg"
        dest_path = ORIGINALS_DIR / dest_filename

        # If already verified
        if cid in existing_records:
            rec = existing_records[cid]
            seen_sha256[rec["sha256"]] = cid
            seen_phash[rec["phash"]] = cid
            download_records.append(rec)
            success_count += 1
            print(f"[{idx}/100] CACHED {cid} ({class_name}) - sha256: {rec['sha256'][:8]}...")
            continue

        print(f"[{idx}/100] Processing {cid} ({class_name}): {title[:40]}...", end=" ", flush=True)

        pdata = fetch_single_title_metadata(session, title)
        if not pdata or not pdata.get("imageinfo"):
            print("FAILED (no imageinfo from API)")
            fail_count += 1
            download_records.append({
                "candidate_id": cid,
                "target_class_id": class_id,
                "target_class_name": class_name,
                "commons_file_title": title,
                "download_success": False,
                "error": "Failed to retrieve imageinfo from Wikimedia API",
            })
            continue

        ii = pdata["imageinfo"][0]
        direct_url = ii.get("url")
        ext = ii.get("extmetadata", {})
        artist = ext.get("Artist", {}).get("value", "Unknown")
        license_name = ext.get("LicenseShortName", {}).get("value", "Unknown")
        license_url = ext.get("LicenseUrl", {}).get("value", "")
        file_mime = ii.get("mime", "")
        wiki_sha1 = ii.get("sha1", "")

        download_ok = False
        img_bytes = b""
        for attempt in range(3):
            try:
                r = session.get(direct_url, verify=False, timeout=30)
                if r.status_code == 200:
                    img_bytes = r.content
                    download_ok = True
                    break
                time.sleep(1)
            except Exception:
                time.sleep(2)

        if not download_ok or not img_bytes:
            print("FAILED (download error)")
            fail_count += 1
            download_records.append({
                "candidate_id": cid,
                "target_class_id": class_id,
                "target_class_name": class_name,
                "commons_file_title": title,
                "download_success": False,
                "error": "Failed to download image bytes after 3 attempts",
            })
            continue

        sha256 = get_sha256(img_bytes)

        # Verify image using PIL
        try:
            with Image.open(BytesIO(img_bytes)) as pil_img:
                pil_img.verify()
            with Image.open(BytesIO(img_bytes)) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                width, height = pil_rgb.size
                phash_val = str(imagehash.phash(pil_rgb))
                pil_rgb.save(dest_path, "JPEG", quality=95)
        except Exception as e:
            print(f"FAILED (corrupt image: {e})")
            fail_count += 1
            download_records.append({
                "candidate_id": cid,
                "target_class_id": class_id,
                "target_class_name": class_name,
                "commons_file_title": title,
                "download_success": False,
                "error": f"Image verification failed: {e}",
            })
            continue

        dup_local = seen_sha256.get(sha256)
        dup_holdout = holdout_hashes.get(sha256)
        dup_train = sha256 in authoritative_hashes
        phash_dup_local = seen_phash.get(phash_val)

        seen_sha256[sha256] = cid
        seen_phash[phash_val] = cid

        record = {
            "candidate_id": cid,
            "target_class_id": class_id,
            "target_class_name": class_name,
            "commons_file_title": title,
            "individual_file_page_url": item.get("individual_file_page_url"),
            "file_download_redirect_url": item.get("file_download_redirect_url"),
            "direct_image_url": direct_url,
            "artist": artist,
            "license": license_name,
            "license_url": license_url,
            "file_mime": file_mime,
            "wikimedia_sha1": wiki_sha1,
            "local_filename": dest_filename,
            "local_path": str(dest_path),
            "width": width,
            "height": height,
            "file_size_bytes": len(img_bytes),
            "sha256": sha256,
            "phash": phash_val,
            "download_success": True,
            "duplicate_in_candidates": dup_local,
            "duplicate_in_holdout": dup_holdout,
            "duplicate_in_training_v2": dup_train,
            "phash_duplicate_in_candidates": phash_dup_local,
        }
        download_records.append(record)
        success_count += 1
        print(f"OK ({width}x{height}, {len(img_bytes)/1024:.1f} KB, sha256: {sha256[:8]}...)")

    print(f"\nFinal status: {success_count}/100 succeeded, {fail_count} failed.")

    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_candidates": len(items),
                    "download_success": success_count,
                    "download_failure": fail_count,
                    "destination_directory": str(ORIGINALS_DIR),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "items": download_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved complete audit manifest to: {audit_file}")


if __name__ == "__main__":
    main()
