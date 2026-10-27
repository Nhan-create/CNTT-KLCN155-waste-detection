# Mô tả: [Lưu trữ — Kịch bản mở rộng & kiểm định dữ liệu đa lớp (Task 8)] Tìm kiếm, tải về và mở rộng mẫu ảnh thực tế bổ sung cho `acquire_task8_wave2_large_batch`.
r"""
Task 8 Part B (B1 & B2) — Resumable Large-Scale Wave 2 Acquisition (1,000+ Candidate Images).

Sources:
1. Open Images V7 Train (`data/external/waste-training/openimages-train-slice-boxes.csv` + `openimages-train-slice-meta.csv`)
   supplemented by unused Open Images V7 Test/Val (`test-annotations-bbox.csv` / `openimages-validation-boxes.csv`)
   for rare classes (`/m/01kb5b` Flashlight/Battery, `/m/02w3r3` Paper towel, `/m/09gtd` Toilet paper, `/m/02jnhm` Tin can).
   - 100% disjoint by `ImageID` AND `AuthorProfileURL` from `real_v2` (including frozen 299 test & 274 v3_candidate test).
   - Filtered for `Rotation in ('', '0', '0.0')`, `IsGroupOf == '0'`, `IsDepiction == '0'`.
   - Zero intentional allocation to `7: plastic` per B2 requirement.
2. Wikimedia Commons Real-World Waste & Battery Supplement (~120 candidate images across `0: battery`,
   `1: biological`, `9: trash`, `2: cardboard`, `4: glass`, `5: metal`, `6: paper`, `8: shoes`).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WASTE_TRAINING_DIR = pathlib.Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(pathlib.Path(__file__).resolve().parents[3] / "data" / "external" / "waste-training")))
WAVE2_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch"
IMAGES_DIR = WAVE2_DIR / "images"
CHECKPOINT_PATH = WAVE2_DIR / "raw_download_checkpoint.json"
RAW_MANIFEST_PATH = WAVE2_DIR / "task8_wave2_raw_manifest.json"

USER_AGENT = "CNTT-KLCN155-WasteDetection-Research/2.0 (Academic Research; mailto:research@klcn155.edu.vn)"

# Open Images V7 MID -> (target_class_id, target_class_name, mid_human_name, default_domain_hint)
OI_MID_MAP: dict[str, tuple[int, str, str, str]] = {
    # 0: battery (proxy / battery-powered portable item requiring QA review)
    "/m/01kb5b": (0, "battery", "Flashlight", "AMBIGUOUS_CONTEXT"),
    # 1: biological
    "/m/09qck": (1, "biological", "Banana", "AMBIGUOUS_CONTEXT"),
    "/m/014j1m": (1, "biological", "Apple", "AMBIGUOUS_CONTEXT"),
    "/m/0cyhj_": (1, "biological", "Orange", "AMBIGUOUS_CONTEXT"),
    "/m/09k_b": (1, "biological", "Lemon", "AMBIGUOUS_CONTEXT"),
    "/m/07j87": (1, "biological", "Tomato", "AMBIGUOUS_CONTEXT"),
    "/m/09728": (1, "biological", "Bread", "AMBIGUOUS_CONTEXT"),
    "/m/033cnk": (1, "biological", "Egg", "AMBIGUOUS_CONTEXT"),
    # 2: cardboard
    "/m/025dyy": (2, "cardboard", "Box", "AMBIGUOUS_CONTEXT"),
    # 3: clothes
    "/m/0174n1": (3, "clothes", "Glove", "AMBIGUOUS_CONTEXT"),
    "/m/0162_1": (3, "clothes", "Towel", "AMBIGUOUS_CONTEXT"),
    "/m/01nq26": (3, "clothes", "Sock", "AMBIGUOUS_CONTEXT"),
    "/m/01n4qj": (3, "clothes", "Shirt", "AMBIGUOUS_CONTEXT"),
    "/m/07mhn": (3, "clothes", "Trousers", "AMBIGUOUS_CONTEXT"),
    "/m/032b3c": (3, "clothes", "Jacket", "AMBIGUOUS_CONTEXT"),
    "/m/09j2d": (3, "clothes", "Clothing", "AMBIGUOUS_CONTEXT"),
    # 4: glass
    "/m/04dr76w": (4, "glass", "Bottle", "AMBIGUOUS_CONTEXT"),
    "/m/09tvcd": (4, "glass", "Wine glass", "AMBIGUOUS_CONTEXT"),
    "/m/08hvt4": (4, "glass", "Jug", "AMBIGUOUS_CONTEXT"),
    # 5: metal
    "/m/02jnhm": (5, "metal", "Tin can", "AMBIGUOUS_CONTEXT"),
    # 6: paper
    "/m/02w3r3": (6, "paper", "Paper towel", "AMBIGUOUS_CONTEXT"),
    "/m/09gtd": (6, "paper", "Toilet paper", "AMBIGUOUS_CONTEXT"),
    "/m/0frqm": (6, "paper", "Envelope", "AMBIGUOUS_CONTEXT"),
    "/m/0bt_c3": (6, "paper", "Book", "AMBIGUOUS_CONTEXT"),
    # 8: shoes
    "/m/01b638": (8, "shoes", "Boot", "AMBIGUOUS_CONTEXT"),
    "/m/03nfch": (8, "shoes", "Sandal", "AMBIGUOUS_CONTEXT"),
    "/m/06k2mb": (8, "shoes", "High heels", "AMBIGUOUS_CONTEXT"),
    "/m/09j5n": (8, "shoes", "Footwear", "AMBIGUOUS_CONTEXT"),
    # 9: trash
    "/m/0bjyj5": (9, "trash", "Waste container", "ACTUAL_WASTE"),
}

# Quotas per class from Open Images V7 (~975 total) + Commons (~75 total) = ~1,050 candidates
CLASS_OI_QUOTAS: dict[int, int] = {
    0: 25,   # battery proxy candidates from Open Images (plus ~35 real battery images from Commons)
    1: 150,  # biological
    2: 145,  # cardboard
    3: 105,  # clothes
    4: 130,  # glass
    5: 130,  # metal
    6: 130,  # paper
    8: 125,  # shoes
    9: 45,   # trash / waste container
}

COMMONS_CATEGORY_QUERIES: list[tuple[str, int, str, int]] = [
    ("Category:Electric_batteries", 0, "battery", 14),
    ("Category:AA_batteries", 0, "battery", 10),
    ("Category:AAA_batteries", 0, "battery", 8),
    ("Category:Nine-volt_batteries", 0, "battery", 8),
    ("Category:Button_cells", 0, "battery", 8),
    ("Category:Rechargeable_batteries", 0, "battery", 8),
    ("Category:Alkaline_batteries", 0, "battery", 10),
    ("Category:Food_waste", 1, "biological", 12),
    ("Category:Compost", 1, "biological", 8),
    ("Category:Cardboard_boxes", 2, "cardboard", 8),
    ("Category:Corrugated_fiberboard", 2, "cardboard", 6),
    ("Category:Broken_glass", 4, "glass", 8),
    ("Category:Glass_recycling", 4, "glass", 6),
    ("Category:Aluminium_cans", 5, "metal", 8),
    ("Category:Paper_recycling", 6, "paper", 8),
    ("Category:Waste_paper", 6, "paper", 6),
    ("Category:Old_shoes", 8, "shoes", 8),
    ("Category:Litter", 9, "trash", 10),
    ("Category:Cigarette_butts", 9, "trash", 8),
]


def load_existing_exclusions() -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (existing_sha256, existing_oi_ids, existing_authors, existing_commons_titles)."""
    existing_sha256: set[str] = set()
    existing_oi_ids: set[str] = set()
    existing_authors: set[str] = set()
    existing_commons_titles: set[str] = set()

    # 1. Authoritative hashes cache
    auth_path = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if auth_path.exists():
        data = json.loads(auth_path.read_text(encoding="utf-8"))
        for v in data.values():
            if isinstance(v, dict) and v.get("sha256"):
                existing_sha256.add(v["sha256"])

    # 2. real_v2_source_manifest.csv
    val_meta_path = WASTE_TRAINING_DIR / "validation-images-with-rotation.csv"
    val_meta = {}
    if val_meta_path.exists():
        with open(val_meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                val_meta[r["ImageID"]] = r

    r2_manifest = REPO_ROOT / "data" / "detection" / "real_v2_source_manifest.csv"
    if r2_manifest.exists():
        with open(r2_manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("source") == "openimages_v7":
                    iid = r.get("source_image_id", "").strip()
                    if iid:
                        existing_oi_ids.add(iid)
                        m = val_meta.get(iid)
                        if m and m.get("AuthorProfileURL"):
                            existing_authors.add(m["AuthorProfileURL"].strip())

    # 3. Wave 1 manifest
    w1_path = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json"
    if w1_path.exists():
        w1 = json.loads(w1_path.read_text(encoding="utf-8"))
        for rec in w1.get("records", []):
            if rec.get("sha256"):
                existing_sha256.add(rec["sha256"])
            sid = rec.get("source_id", "")
            if sid.startswith("openimages_v7_val_"):
                existing_oi_ids.add(sid.replace("openimages_v7_val_", ""))
            if rec.get("original_title"):
                existing_commons_titles.add(rec["original_title"].strip())

    # 4. Scan staging folders for SHA256
    staging_root = REPO_ROOT / "data" / "acquisition_staging"
    for p in staging_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and "task8_wave2_large_batch" not in str(p):
            try:
                existing_sha256.add(hashlib.sha256(p.read_bytes()).hexdigest())
            except Exception:
                pass

    return existing_sha256, existing_oi_ids, existing_authors, existing_commons_titles


def collect_openimages_candidates(
    existing_oi_ids: set[str],
    existing_authors: set[str],
) -> list[dict[str, Any]]:
    """Select balanced, author-disjoint, non-rotated, non-depiction candidates from Open Images V7."""
    # Load Train slice first (Priority #1 in B1)
    splits_to_load = [
        (
            "train",
            WASTE_TRAINING_DIR / "openimages-train-slice-boxes.csv",
            WASTE_TRAINING_DIR / "openimages-train-slice-meta.csv",
        ),
        (
            "test",
            WASTE_TRAINING_DIR / "test-annotations-bbox.csv",
            WASTE_TRAINING_DIR / "test-images-with-rotation.csv",
        ),
        (
            "validation",
            WASTE_TRAINING_DIR / "openimages-validation-boxes.csv",
            WASTE_TRAINING_DIR / "validation-images-with-rotation.csv",
        ),
    ]

    # Human MIDs to exclude so we don't accidentally pull portraits when looking for clothes/shoes
    HUMAN_MIDS = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/05r655", "/m/01bl7v", "/m/0dzct"}

    # Group valid candidate images per primary target class
    candidates_by_class: dict[int, list[dict[str, Any]]] = {cid: [] for cid in CLASS_OI_QUOTAS}
    seen_image_ids: set[str] = set(existing_oi_ids)

    for split_name, box_csv, meta_csv in splits_to_load:
        if not box_csv.exists() or not meta_csv.exists():
            continue
        print(f"[Wave2-OI] Loading metadata from {meta_csv.name} ({split_name})...")
        meta_map: dict[str, dict[str, str]] = {}
        with open(meta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = row["ImageID"]
                if iid in seen_image_ids:
                    continue
                rot = row.get("Rotation", "").strip()
                if rot not in ("", "0", "0.0"):
                    continue
                author_url = row.get("AuthorProfileURL", "").strip()
                if author_url and author_url in existing_authors:
                    continue
                # Require valid author or landing URL for provenance compliance (A3)
                if not row.get("Author", "").strip() and not row.get("OriginalLandingURL", "").strip():
                    continue
                meta_map[iid] = row

        print(f"[Wave2-OI] Scanning boxes from {box_csv.name} ({len(meta_map)} eligible images)...")
        img_boxes: dict[str, list[dict[str, Any]]] = {}
        img_has_human: set[str] = set()
        with open(box_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = row["ImageID"]
                if iid not in meta_map:
                    continue
                lbl = row["LabelName"]
                if lbl in HUMAN_MIDS:
                    img_has_human.add(iid)
                    continue
                if lbl not in OI_MID_MAP:
                    continue
                if row.get("IsDepiction", "0") == "1" or row.get("IsGroupOf", "0") == "1":
                    continue
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
                bw, bh = xmax - xmin, ymax - ymin
                if bw <= 0.03 or bh <= 0.03 or (bw * bh) <= 0.0025:
                    continue
                cid, cname, mid_name, dom_hint = OI_MID_MAP[lbl]
                img_boxes.setdefault(iid, []).append(
                    {
                        "class_id": cid,
                        "class_name": cname,
                        "mid": lbl,
                        "mid_name": mid_name,
                        "xmin": round(xmin, 6),
                        "ymin": round(ymin, 6),
                        "xmax": round(xmax, 6),
                        "ymax": round(ymax, 6),
                        "is_occluded": int(row.get("IsOccluded", "0") or 0),
                        "is_truncated": int(row.get("IsTruncated", "0") or 0),
                        "is_inside": int(row.get("IsInside", "0") or 0),
                        "domain_hint": dom_hint,
                    }
                )

        # Prioritize specific rarer MIDs first within each class so we don't just get 150 'Book' or 'Clothing'
        MID_PRIORITY_ORDER = [
            "/m/01kb5b",  # Flashlight (battery proxy)
            "/m/09qck", "/m/014j1m", "/m/0cyhj_", "/m/09k_b", "/m/07j87", "/m/033cnk", "/m/09728",  # biological
            "/m/025dyy",  # cardboard Box
            "/m/0174n1", "/m/0162_1", "/m/01nq26", "/m/01n4qj", "/m/07mhn", "/m/032b3c", "/m/09j2d",  # clothes
            "/m/08hvt4", "/m/09tvcd", "/m/04dr76w",  # glass
            "/m/02jnhm",  # metal Tin can
            "/m/02w3r3", "/m/09gtd", "/m/0frqm", "/m/0bt_c3",  # paper
            "/m/01b638", "/m/03nfch", "/m/06k2mb", "/m/09j5n",  # shoes
            "/m/0bjyj5",  # trash Waste container
        ]
        mid_rank = {m: idx for idx, m in enumerate(MID_PRIORITY_ORDER)}

        # Sort eligible images by (has_multi_class DESC, best_mid_rank ASC, ImageID ASC)
        sorted_iids = sorted(
            [iid for iid in img_boxes if iid not in img_has_human and 1 <= len(img_boxes[iid]) <= 12],
            key=lambda x: (
                -len({b["class_id"] for b in img_boxes[x]}),
                min(mid_rank.get(b["mid"], 999) for b in img_boxes[x]),
                x,
            ),
        )

        # Limit max images per MID so classes with multiple MIDs stay diverse
        mid_counts: dict[str, int] = {}
        MID_CAP = {
            "/m/09j2d": 25,   # generic Clothing cap
            "/m/0bt_c3": 55,  # Book cap
            "/m/04dr76w": 65, # Bottle cap
            "/m/09j5n": 45,   # generic Footwear cap
        }

        for iid in sorted_iids:
            if iid in seen_image_ids:
                continue
            b_list = img_boxes[iid]
            # Pick primary class among classes that still need quota
            cids_in_img = sorted({b["class_id"] for b in b_list}, key=lambda c: len(candidates_by_class.get(c, [])))
            chosen_cid = None
            for cid in cids_in_img:
                if cid in CLASS_OI_QUOTAS and len(candidates_by_class[cid]) < CLASS_OI_QUOTAS[cid]:
                    # Check MID cap
                    primary_mids = [b["mid"] for b in b_list if b["class_id"] == cid]
                    if any(mid_counts.get(m, 0) >= MID_CAP.get(m, 9999) for m in primary_mids):
                        continue
                    chosen_cid = cid
                    break
            if chosen_cid is None:
                continue

            for m in {b["mid"] for b in b_list}:
                mid_counts[m] = mid_counts.get(m, 0) + 1

            m_row = meta_map[iid]
            seen_image_ids.add(iid)
            candidates_by_class[chosen_cid].append(
                {
                    "source_family": f"openimages_v7_{split_name}",
                    "split_origin": split_name,
                    "source_image_id": iid,
                    "primary_class_id": chosen_cid,
                    "primary_class_name": b_list[0]["class_name"],
                    "boxes": b_list,
                    "s3_url": f"https://open-images-dataset.s3.amazonaws.com/{split_name}/{iid}.jpg",
                    "thumbnail_url": m_row.get("Thumbnail300KURL", "").strip(),
                    "original_url": m_row.get("OriginalURL", "").strip(),
                    "original_landing_url": m_row.get("OriginalLandingURL", "").strip(),
                    "license": m_row.get("License", "https://creativecommons.org/licenses/by/2.0/").strip(),
                    "author": m_row.get("Author", "").strip(),
                    "author_profile_url": m_row.get("AuthorProfileURL", "").strip(),
                    "title": m_row.get("Title", "").strip(),
                    "original_md5": m_row.get("OriginalMD5", "").strip(),
                    "rotation": m_row.get("Rotation", "0").strip() or "0",
                }
            )

        # Check if all quotas are met
        if all(len(candidates_by_class[c]) >= q for c, q in CLASS_OI_QUOTAS.items()):
            break

    all_selected: list[dict[str, Any]] = []
    for cid, items in sorted(candidates_by_class.items()):
        print(f"  Class {cid} OI selected: {len(items)} / {CLASS_OI_QUOTAS[cid]}")
        all_selected.extend(items)
    return all_selected


def fetch_commons_candidates(existing_titles: set[str]) -> list[dict[str, Any]]:
    """Query Wikimedia Commons API for real-world waste & battery images."""
    selected: list[dict[str, Any]] = []
    seen_titles = set(existing_titles)

    for cat_title, cid, cname, quota in COMMONS_CATEGORY_QUERIES:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": cat_title,
                "gcmtype": "file",
                "gcmlimit": str(quota * 3),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": "1024",
                "format": "json",
            }
        )
        url = f"https://commons.wikimedia.org/w/api.php?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[Commons] Warning querying {cat_title}: {e}")
            continue

        pages = data.get("query", {}).get("pages", {})
        added = 0
        for _, pinfo in pages.items():
            if added >= quota:
                break
            title = pinfo.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            ii_list = pinfo.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png", "image/webp"):
                continue
            w, h = int(ii.get("width", 0)), int(ii.get("height", 0))
            if w < 300 or h < 300:
                continue
            ext = ii.get("extmetadata", {})
            lic_short = ext.get("LicenseShortName", {}).get("value", "CC-BY-SA-4.0")
            lic_url = ext.get("LicenseUrl", {}).get("value", "https://creativecommons.org/licenses/by-sa/4.0/")
            artist = ext.get("Artist", {}).get("value", "Wikimedia Commons Contributor")
            img_url = ii.get("thumburl") or ii.get("url")
            desc_url = ii.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}")
            if not img_url:
                continue
            seen_titles.add(title)
            selected.append(
                {
                    "source_family": "wikimedia_commons_wave2",
                    "split_origin": "commons_category",
                    "source_image_id": title,
                    "category_source": cat_title,
                    "primary_class_id": cid,
                    "primary_class_name": cname,
                    "boxes": [],  # Will be boxed & reviewed in B3 QA
                    "download_url": img_url,
                    "original_url": ii.get("url", img_url),
                    "original_landing_url": desc_url,
                    "license": f"{lic_short} ({lic_url})".strip(),
                    "author": artist,
                    "author_profile_url": desc_url,
                    "title": title,
                    "original_md5": "",
                    "rotation": "0",
                }
            )
            added += 1
        print(f"  [Commons] {cat_title} -> {added} candidates")
    return selected


def download_single_candidate(
    idx: int,
    cand: dict[str, Any],
    existing_sha256: set[str],
) -> dict[str, Any] | None:
    """Download candidate image with automatic fallback and verify integrity."""
    item_id = f"task8w2_{idx:04d}"
    out_file = IMAGES_DIR / f"{item_id}.jpg"

    urls_to_try = []
    if cand.get("thumbnail_url"):
        urls_to_try.append(("flickr_thumb_640", cand["thumbnail_url"]))
    if cand.get("s3_url"):
        urls_to_try.append(("aws_s3_openimages", cand["s3_url"]))
    if cand.get("download_url"):
        urls_to_try.append(("wikimedia_thumb", cand["download_url"]))
    if cand.get("original_url"):
        urls_to_try.append(("original_url", cand["original_url"]))

    raw_bytes = None
    used_endpoint = ""
    for ep_name, u in urls_to_try:
        if not u:
            continue
        for attempt in range(2):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                if len(data) >= 2048:
                    raw_bytes = data
                    used_endpoint = ep_name
                    break
            except Exception:
                time.sleep(0.25 * (attempt + 1))
        if raw_bytes:
            break

    if not raw_bytes:
        return None

    # Verify valid image and normalize to RGB JPEG if needed
    try:
        im = Image.open(io.BytesIO(raw_bytes))
        im.verify()
        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = im.size
        if w < 160 or h < 160:
            return None
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        final_bytes = buf.getvalue()
    except Exception:
        return None

    sha256 = hashlib.sha256(final_bytes).hexdigest()
    if sha256 in existing_sha256:
        return None

    out_file.write_bytes(final_bytes)
    rec = dict(cand)
    rec.update(
        {
            "item_id": item_id,
            "filename": out_file.name,
            "rel_path": f"data/acquisition_staging/task8_wave2_large_batch/images/{out_file.name}",
            "width": w,
            "height": h,
            "bytes": len(final_bytes),
            "sha256": sha256,
            "downloaded_from_endpoint": used_endpoint,
        }
    )
    return rec


def main() -> None:
    WAVE2_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing_sha256, existing_oi_ids, existing_authors, existing_commons_titles = load_existing_exclusions()
    print(
        f"[Wave2] Loaded exclusions: {len(existing_sha256)} SHA256s, "
        f"{len(existing_oi_ids)} OI IDs, {len(existing_authors)} OI Authors"
    )

    # Load checkpoint if resuming
    downloaded_records: list[dict[str, Any]] = []
    completed_source_ids: set[str] = set()
    if CHECKPOINT_PATH.exists():
        ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        downloaded_records = ckpt.get("records", [])
        for r in downloaded_records:
            completed_source_ids.add(r["source_image_id"])
            existing_sha256.add(r["sha256"])
        print(f"[Wave2] Resuming from checkpoint with {len(downloaded_records)} already downloaded images.")

    CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
    if len(downloaded_records) >= 970:
        oi_candidates = []
    else:
        oi_candidates = collect_openimages_candidates(existing_oi_ids | completed_source_ids, existing_authors)
    commons_candidates = fetch_commons_candidates(existing_commons_titles | completed_source_ids)

    all_candidates = oi_candidates + commons_candidates
    print(
        f"[Wave2] Total new candidates queued: {len(all_candidates)} "
        f"(OpenImages={len(oi_candidates)}, Commons={len(commons_candidates)})"
    )

    max_existing_idx = 0
    for r in downloaded_records:
        try:
            max_existing_idx = max(max_existing_idx, int(r["item_id"].replace("task8w2_", "")))
        except Exception:
            pass
    start_idx = max_existing_idx + 1
    oi_batch = [c for c in all_candidates if c["source_family"].startswith("openimages_v7")]
    cm_batch = [c for c in all_candidates if not c["source_family"].startswith("openimages_v7")]

    # 1. Fast concurrent download for Open Images V7 (S3 / Flickr CDN)
    if oi_batch:
        print(f"[Wave2] Downloading {len(oi_batch)} Open Images V7 candidates with 20 workers...")
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(download_single_candidate, start_idx + i, cand, existing_sha256): cand
                for i, cand in enumerate(oi_batch)
            }
            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                res = fut.result()
                if res is not None:
                    if res["sha256"] not in existing_sha256:
                        existing_sha256.add(res["sha256"])
                        downloaded_records.append(res)
                if done_count % 100 == 0 or done_count == len(oi_batch):
                    print(f"  [OI Progress] {done_count}/{len(oi_batch)} processed -> {len(downloaded_records)} saved")
                    CHECKPOINT_PATH.write_text(
                        json.dumps({"records": downloaded_records}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

    # 2. Polite download for Wikimedia Commons candidates (4 workers)
    next_idx = max_existing_idx + len(oi_batch) + 1
    if cm_batch:
        print(f"[Wave2] Downloading {len(cm_batch)} Wikimedia Commons candidates with 4 workers...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(download_single_candidate, next_idx + i, cand, existing_sha256): cand
                for i, cand in enumerate(cm_batch)
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None and res["sha256"] not in existing_sha256:
                    existing_sha256.add(res["sha256"])
                    downloaded_records.append(res)

    for r in downloaded_records:
        r["primary_class_name"] = CLASS_NAMES[r["primary_class_id"]]

    # Sort deterministically by item_id
    downloaded_records.sort(key=lambda r: r["item_id"])

    # Re-index item_ids cleanly or preserve exact filename mapping
    CHECKPOINT_PATH.write_text(
        json.dumps({"records": downloaded_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "wave": "TASK8_WAVE2_LARGE_BATCH",
        "generated_at": "2026-10-14",
        "total_candidates_attempted": len(downloaded_records) + len(all_candidates),
        "total_downloaded_unique_images": len(downloaded_records),
        "total_bytes_on_disk": sum(r["bytes"] for r in downloaded_records),
        "by_source_family": {},
        "by_primary_class": {},
        "records": downloaded_records,
    }
    for r in downloaded_records:
        sf = r["source_family"]
        summary["by_source_family"][sf] = summary["by_source_family"].get(sf, 0) + 1
        cn = f"{r['primary_class_id']}_{r['primary_class_name']}"
        summary["by_primary_class"][cn] = summary["by_primary_class"].get(cn, 0) + 1

    RAW_MANIFEST_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Wave2] Done! Total downloaded unique images: {len(downloaded_records)}")
    print("  By source family:", summary["by_source_family"])
    print("  By primary class:", summary["by_primary_class"])


if __name__ == "__main__":
    main()
