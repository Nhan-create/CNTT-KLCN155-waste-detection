# Mô tả: [Lưu trữ — Kịch bản mở rộng & kiểm định dữ liệu đa lớp (Task 8)] Tìm kiếm, tải về và mở rộng mẫu ảnh thực tế bổ sung cho `acquire_task8d_pilot_and_candidates`.
"""
Task 8D — Step 1: Execute 30-Image Source Yield Pilots (7 Sources) & Download High-Yield Real-Waste Candidate Pool.

Enforces:
- Task 2 & Task 3:
  - Evaluates 7 candidate sources on 30-image pilots (`pilot_30_per_source/`).
  - Immediately HALTS (`STOPPED_AFTER_PILOT`) the 4 sources that return clean product/household photos,
    same-specimen multi-angle repeats, or TrashNet/watermarked stock duplicates:
    1) `SRC_8D_01_KEREMBERKE_GARBAGE_OD` (20.0% real-waste yield; TrashNet + Shutterstock/Getty watermarks -> STOPPED)
    2) `SRC_8D_02_ALEXNOVA_MRS_TRASH_OD` (6.7% real-waste yield; 93% clean kitchen utensils/books/toys -> STOPPED)
    3) `SRC_8D_03_DMEDHI_GARBAGE_OD` (23.3% real-waste yield; TrashNet + glass labeled as Metal -> STOPPED)
    4) `SRC_8D_05_VOXEL51_FOOD_WASTE` (6.7% unique specimen yield; 18x repeats of same plate bonid -> STOPPED)
  - Expands ONLY the 3 verified high-yield real-waste sources:
    5) `SRC_8D_04_SHYNUAA_STREET_GARBAGE` (MIT License; urban street litter with CVAT bboxes, EXIF-normalized)
    6) `SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED` (CC BY 2.0; Open Images V7 strictly filtered by explicit waste/litter/decay Flickr titles or Waste container co-occurrence)
    7) `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` (CC0 / CC BY / CC BY-SA; explicit waste/decay/corroded/discarded queries across all 10 classes)
- Deduplicates every candidate by SHA-256 and 64-bit pHash (Hamming <= 8) against all existing project datasets
  (`real_v2`, `v3_candidate`, `processed_v2`, `Task 7`, `Wave 1`, `Wave 2`).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import re
import shutil
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import imagehash
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WT_DIR = pathlib.Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(pathlib.Path(__file__).resolve().parents[3] / "data" / "external" / "waste-training")))
W8D_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste"
PILOT_DIR = W8D_DIR / "pilot_30_per_source"
IMAGES_DIR = W8D_DIR / "images"
INSPECT_DIR = W8D_DIR / "inspection_sheets"
PILOT_CACHE = REPO_ROOT / "data" / "acquisition_staging" / "task8d_pilot_cache"

UA = "CNTT-KLCN155-WasteDetection-Research/2.0 (Academic Research; mailto:research@klcn155.edu.vn)"

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


def compute_phash_int(img: Image.Image) -> int:
    return int(str(imagehash.phash(img)), 16)


def hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def load_all_existing_hashes_and_ids() -> tuple[set[str], list[tuple[str, int]], set[str], set[str]]:
    sha_set: set[str] = set()
    phash_list: list[tuple[str, int]] = []
    used_oi_ids: set[str] = set()
    used_commons_titles: set[str] = set()

    auth_p = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if auth_p.exists():
        data = json.loads(auth_p.read_text(encoding="utf-8"))
        for k, v in data.items():
            if isinstance(v, dict):
                if v.get("sha256"):
                    sha_set.add(v["sha256"])
                if v.get("phash"):
                    try:
                        phash_list.append((k, int(v["phash"], 16)))
                    except Exception:
                        pass

    for p in (REPO_ROOT / "data" / "detection" / "images" / "real").glob("*"):
        used_oi_ids.add(p.stem.split("_")[0])

    for mp in [
        REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json",
        REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json",
        REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json",
    ]:
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            for r in m.get("records", m.get("items", [])):
                if r.get("sha256"):
                    sha_set.add(r["sha256"])
                ph = r.get("phash_hex") or r.get("phash_0") or r.get("phash")
                if ph:
                    try:
                        phash_list.append((r.get("item_id", r.get("candidate_id", "")), int(str(ph), 16)))
                    except Exception:
                        pass
                if r.get("source_image_id"):
                    used_oi_ids.add(str(r["source_image_id"]))
                if r.get("source_id"):
                    used_oi_ids.add(str(r["source_id"]))
                if r.get("original_title"):
                    used_commons_titles.add(str(r["original_title"]))
                if r.get("title"):
                    used_commons_titles.add(str(r["title"]))

    return sha_set, phash_list, used_oi_ids, used_commons_titles


def fetch_bytes(url: str, timeout: int = 12) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def gather_shynuaa_candidates(limit: int = 95) -> list[dict[str, Any]]:
    """Gather real urban street garbage images + CVAT YOLO boxes from Shynuaa/Street-garbage (MIT)."""
    candidates: list[dict[str, Any]] = []
    for split_name in ("valid", "test", "train"):
        if len(candidates) >= limit:
            break
        itree_b = fetch_bytes(
            f"https://huggingface.co/api/datasets/Shynuaa/Street-garbage/tree/main/yolo/{split_name}/images"
        )
        if not itree_b:
            continue
        itree = json.loads(itree_b.decode("utf-8"))
        for item in itree:
            if len(candidates) >= limit:
                break
            fn = item["path"].split("/")[-1]
            # Prioritize high-res 20251220_* urban captures with 1..4 distinct litter instances
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = fn.rsplit(".", 1)[0]
            lurl = f"https://huggingface.co/datasets/Shynuaa/Street-garbage/raw/main/yolo/{split_name}/labels/{ urllib.parse.quote(stem) }.txt"
            iurl = f"https://huggingface.co/datasets/Shynuaa/Street-garbage/resolve/main/yolo/{split_name}/images/{ urllib.parse.quote(fn) }"
            candidates.append(
                {
                    "source_family": "shynuaa_street_garbage",
                    "source_split": split_name,
                    "source_image_id": f"shynuaa_{split_name}_{stem}",
                    "original_title": fn,
                    "image_url": iurl,
                    "label_url": lurl,
                    "license": "MIT License",
                    "author": "Shynuaa (Street-garbage CVAT Urban Litter Dataset)",
                    "author_profile_url": "https://huggingface.co/datasets/Shynuaa/Street-garbage",
                    "landing_url": f"https://huggingface.co/datasets/Shynuaa/Street-garbage/blob/main/yolo/{split_name}/images/{fn}",
                    "attribution_required": False,
                    "session_hint": f"SHYNUAA_{split_name.upper()}_{fn[:13]}",
                }
            )
    return candidates


def gather_oi_waste_context_candidates(used_oi_ids: set[str], limit: int = 95) -> list[dict[str, Any]]:
    """Gather Open Images V7 candidates strictly filtered by explicit waste/decay/litter titles or Waste container co-occurrence."""
    cls_map = {}
    with (WT_DIR / "openimages-boxable-classes.csv").open("r", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                cls_map[r[0]] = r[1]

    oi_to_10 = {
        "/m/01kb5b": 0,
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
        "/m/02jnhm": 5,
        "/m/02w3r3": 6,
        "/m/09gtd": 6,
        "/m/0frqm": 6,
        "/m/0bt_c3": 6,
        "/m/057cc": 7,
        "/m/01b638": 8,
        "/m/03nfch": 8,
        "/m/06k2mb": 8,
        "/m/09j5n": 8,
        "/m/0bjyj5": 9,
    }

    waste_re = re.compile(
        r"\b(trash|garbage|litter|rubbish|waste|dump|dumpster|bin|recycl|discard|abandon|rotten|mold|mould|spoil|compost|peel|scrap|leftover|crush|dent|rust|corrod|broken|shatter|smash|torn|ripped|dirty|muddy|worn|old\s+shoe|old\s+boot|used\s+batter|dead\s+batter|cigarette|butt|fly.?tip|landfill)\b",
        re.I,
    )
    studio_re = re.compile(
        r"\b(white\s+background|isolated|product|catalog|clipart|illustration|vector|3d\s+render|museum|statue|painting)\b",
        re.I,
    )

    candidates: list[dict[str, Any]] = []
    for split_tag, mfile, bfile in [
        ("train", "openimages-train-slice-meta.csv", "openimages-train-slice-boxes.csv"),
        ("validation", "validation-images-with-rotation.csv", "openimages-validation-boxes.csv"),
        ("test", "test-images-with-rotation.csv", "test-annotations-bbox.csv"),
    ]:
        mp = WT_DIR / mfile
        bp = WT_DIR / bfile
        img_boxes = defaultdict(list)
        with bp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mid = r["LabelName"]
                if mid in oi_to_10 and r.get("IsDepiction", "0") == "0" and r.get("IsGroupOf", "0") == "0":
                    x1, y1, x2, y2 = float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 0.015 and bh > 0.015:
                        img_boxes[r["ImageID"]].append(
                            {
                                "class_id": oi_to_10[mid],
                                "class_name": CLASS_NAMES[oi_to_10[mid]],
                                "mid": mid,
                                "mid_name": cls_map.get(mid, mid),
                                "cx": round((x1 + x2) / 2.0, 6),
                                "cy": round((y1 + y2) / 2.0, 6),
                                "w": round(bw, 6),
                                "h": round(bh, 6),
                            }
                        )
        with mp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                iid = r["ImageID"]
                if iid in used_oi_ids or iid not in img_boxes:
                    continue
                if r.get("Rotation", "") not in ("", "0", "0.0"):
                    continue
                title = (r.get("Title", "") or "").strip()
                if studio_re.search(title):
                    continue
                bxs = img_boxes[iid]
                cids = sorted({b["class_id"] for b in bxs})
                has_waste_title = bool(waste_re.search(title))
                has_waste_cooccur = 9 in cids and len(cids) >= 2
                if not (has_waste_title or has_waste_cooccur):
                    continue
                url = r.get("Thumbnail300KURL", "") or r.get("OriginalURL", "")
                if not url:
                    continue
                author = (r.get("Author", "") or "").strip()
                author_url = (r.get("AuthorProfileURL", "") or "").strip()
                landing_url = (r.get("OriginalLandingURL", "") or "").strip()
                candidates.append(
                    {
                        "source_family": f"openimages_v7_{split_tag}_waste_filtered",
                        "source_split": split_tag,
                        "source_image_id": iid,
                        "original_title": title or f"OpenImages_{iid}",
                        "image_url": url,
                        "fallback_url": r.get("OriginalURL", ""),
                        "upstream_boxes": bxs,
                        "primary_class_id": cids[0],
                        "license": r.get("License", "https://creativecommons.org/licenses/by/2.0/"),
                        "author": author if author else f"Flickr_OriginalLanding_{iid}",
                        "author_profile_url": author_url,
                        "landing_url": landing_url,
                        "attribution_required": True,
                        "session_hint": f"OI_AUTHOR_{hashlib.md5((author_url or iid).encode()).hexdigest()[:10]}",
                    }
                )
    return candidates[:limit]


def gather_wikimedia_waste_candidates(used_commons_titles: set[str], limit: int = 180) -> list[dict[str, Any]]:
    """Gather explicit waste/decay/corroded/discarded images from Wikimedia Commons across all 10 classes."""
    search_plan = [
        # 0: battery
        (0, "corroded battery OR leaking battery OR rusty battery filetype:bitmap", 16),
        (0, "battery recycling station OR used batteries bin OR discarded battery filetype:bitmap", 16),
        # 1: biological
        (1, "rotten fruit OR moldy fruit OR decomposing fruit filetype:bitmap", 14),
        (1, "moldy bread OR spoiled food OR food scraps waste filetype:bitmap", 14),
        (1, "banana peel litter OR orange peel waste OR kitchen food waste filetype:bitmap", 12),
        # 2: cardboard
        (2, "discarded pizza box OR crushed cardboard box OR wet cardboard box filetype:bitmap", 18),
        (2, "discarded cardboard box street OR flattened cardboard box trash filetype:bitmap", 14),
        # 3: clothes
        (3, "discarded clothes OR torn clothing waste OR abandoned clothes ground filetype:bitmap", 18),
        (3, "dirty rag cloth litter OR old shirt trash OR discarded glove street filetype:bitmap", 14),
        # 4: glass
        (4, "broken glass bottle ground OR shattered bottle street OR discarded beer bottle filetype:bitmap", 18),
        (4, "empty wine bottle litter OR glass bottle trash grass filetype:bitmap", 14),
        # 5: metal
        (5, "crushed beer can OR crushed soda can OR crushed aluminium can filetype:bitmap", 18),
        (5, "rusty tin can ground OR discarded beverage can litter filetype:bitmap", 14),
        # 6: paper
        (6, "crumpled paper litter OR wet newspaper trash OR torn paper street filetype:bitmap", 16),
        (6, "discarded paper cup ground OR discarded paper bag litter filetype:bitmap", 14),
        # 8: shoes
        (8, "discarded shoes OR abandoned shoes ground OR old worn out shoes filetype:bitmap", 18),
        (8, "muddy old boots discarded OR single old shoe street litter filetype:bitmap", 14),
        # 9: trash & 7: plastic mixed
        (9, "overflowing trash bin plastic bag OR street litter bottle can filetype:bitmap", 16),
        (9, "cigarette butts litter ground OR garbage bags street dumping filetype:bitmap", 14),
    ]

    disallow_title_re = re.compile(
        r"\b(diagram|chart|map|logo|icon|symbol|sign|poster|drawing|illustration|svg|pdf|coat_of_arms|flag|stamp|painting|statue|sculpture|museum)\b",
        re.I,
    )

    found_titles: list[tuple[int, str]] = []
    seen_t = set(used_commons_titles)

    for cid, q, q_limit in search_plan:
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srnamespace": 6,
                "srsearch": q,
                "srlimit": 45,
                "format": "json",
            }
        )
        b = fetch_bytes(url)
        if not b:
            continue
        data = json.loads(b.decode("utf-8"))
        cnt = 0
        for h in data.get("query", {}).get("search", []):
            t = h["title"]
            if t in seen_t:
                continue
            if not t.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if disallow_title_re.search(t):
                continue
            seen_t.add(t)
            found_titles.append((cid, t))
            cnt += 1
            if cnt >= q_limit:
                break

    # Resolve imageinfo & extmetadata in batches of 25
    candidates: list[dict[str, Any]] = []
    for i in range(0, len(found_titles), 25):
        batch = found_titles[i : i + 25]
        t_map = {t: cid for cid, t in batch}
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "query",
                "titles": "|".join(t for _, t in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 1024,
                "format": "json",
            }
        )
        b = fetch_bytes(url)
        if not b:
            continue
        pages = json.loads(b.decode("utf-8")).get("query", {}).get("pages", {})
        for p in pages.values():
            t = p.get("title", "")
            if t not in t_map:
                continue
            ii_list = p.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            img_url = ii.get("thumburl") or ii.get("url")
            if not img_url:
                continue
            ext = ii.get("extmetadata", {})
            lic = ext.get("LicenseShortName", {}).get("value", "CC BY-SA")
            artist_raw = ext.get("Artist", {}).get("value", "Wikimedia Commons Contributor")
            artist = re.sub(r"<[^>]+>", "", artist_raw).strip() or "Wikimedia Commons Contributor"
            cid = t_map[t]
            candidates.append(
                {
                    "source_family": "wikimedia_commons_waste_context",
                    "source_split": "commons_waste_query",
                    "source_image_id": f"commons8d_{ p.get('pageid', hashlib.md5(t.encode()).hexdigest()[:8]) }",
                    "original_title": t,
                    "image_url": img_url,
                    "primary_class_id": cid,
                    "license": lic,
                    "author": artist[:120],
                    "author_profile_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(t)}",
                    "landing_url": ii.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(t)}"),
                    "attribution_required": not any(x in lic.upper() for x in ("CC0", "PUBLIC DOMAIN", "PD")),
                    "session_hint": f"COMMONS_ARTIST_{hashlib.md5(artist.encode()).hexdigest()[:10]}",
                }
            )
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return candidates


def build_contact_sheet(
    items: list[tuple[Image.Image, str]],
    out_path: pathlib.Path,
    title: str,
    cols: int = 6,
    tw: int = 260,
    th: int = 200,
) -> None:
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    hh = 40
    ch = th + 26
    sheet = Image.new("RGB", (cols * tw, hh + rows * ch), (24, 26, 30))
    dr = ImageDraw.Draw(sheet)
    dr.rectangle([0, 0, cols * tw, hh], fill=(40, 46, 58))
    dr.text((12, 12), title, fill=(255, 255, 255))
    for idx, (im, cap) in enumerate(items):
        r, c = idx // cols, idx % cols
        x0, y0 = c * tw, hh + r * ch
        cp = im.copy()
        cp.thumbnail((tw - 8, th - 8), Image.Resampling.BILINEAR)
        ox = x0 + (tw - cp.width) // 2
        oy = y0 + (th - cp.height) // 2
        sheet.paste(cp, (ox, oy))
        dr.text((x0 + 5, y0 + th + 4), cap[:38], fill=(215, 220, 230))
    sheet.save(out_path, format="JPEG", quality=88)


def main() -> None:
    for d in (W8D_DIR, PILOT_DIR, IMAGES_DIR, INSPECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Copy already-rendered pilot contact sheets into PILOT_DIR
    for src_name, dst_name in [
        ("pilot_src01_keremberke_30.jpg", "pilot_src01_keremberke_garbage_od_30.jpg"),
        ("pilot_src02_mrs_trash.jpg", "pilot_src02_alexnova_mrs_trash_od_30.jpg"),
        ("pilot_src03_dmedhi_30.jpg", "pilot_src03_dmedhi_garbage_od_30.jpg"),
        ("pilot_src04_shynuaa_30.jpg", "pilot_src04_shynuaa_street_garbage_30.jpg"),
        ("pilot_src05_voxel51.jpg", "pilot_src05_voxel51_food_waste_30.jpg"),
    ]:
        sp = PILOT_CACHE / src_name
        if sp.exists():
            shutil.copy2(sp, PILOT_DIR / dst_name)

    sha_set, phash_list, used_oi_ids, used_commons_titles = load_all_existing_hashes_and_ids()
    print(f"[Task8D] Loaded {len(sha_set)} existing SHA-256 hashes and {len(phash_list)} 64-bit pHashes.")

    c_shyn = gather_shynuaa_candidates(limit=95)
    c_oi = gather_oi_waste_context_candidates(used_oi_ids, limit=95)
    c_wiki = gather_wikimedia_waste_candidates(used_commons_titles, limit=180)
    print(
        f"[Task8D] Candidate pool gathered: Shynuaa={len(c_shyn)}, OI_WasteFiltered={len(c_oi)}, Wikimedia_WasteQueries={len(c_wiki)}"
    )

    all_cands = c_shyn + c_oi + c_wiki

    def download_one(idx_cand: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
        idx, cand = idx_cand
        raw_b = fetch_bytes(cand["image_url"], timeout=14)
        if not raw_b and cand.get("fallback_url"):
            raw_b = fetch_bytes(cand["fallback_url"], timeout=14)
        if not raw_b:
            return None
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_b))).convert("RGB")
        except Exception:
            return None
        w, h = im.size
        if w < 160 or h < 160:
            return None

        # Parse Shynuaa YOLO labels if present
        if cand["source_family"] == "shynuaa_street_garbage":
            lbl_b = fetch_bytes(cand["label_url"], timeout=10)
            bxs = []
            if lbl_b:
                for ln in lbl_b.decode("utf-8", errors="ignore").splitlines():
                    pts = ln.split()
                    if len(pts) >= 5:
                        cx, cy, bw, bh = map(float, pts[1:5])
                        if bw > 0.01 and bh > 0.01:
                            bxs.append({"class_id": 9, "class_name": "trash", "cx": cx, "cy": cy, "w": bw, "h": bh})
            if not bxs:
                return None
            cand["upstream_boxes"] = bxs
            cand["primary_class_id"] = 9

        out_buf = io.BytesIO()
        im.save(out_buf, format="JPEG", quality=92)
        norm_bytes = out_buf.getvalue()
        sha = hashlib.sha256(norm_bytes).hexdigest()
        ph_int = compute_phash_int(im)
        return {
            "order_idx": idx,
            "cand": cand,
            "norm_bytes": norm_bytes,
            "sha256": sha,
            "phash_int": ph_int,
            "phash_hex": f"{ph_int:016x}",
            "width": w,
            "height": h,
        }

    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(download_one, (i, c)) for i, c in enumerate(all_cands)]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                downloaded.append(res)

    downloaded.sort(key=lambda x: x["order_idx"])
    print(f"[Task8D] Successfully fetched {len(downloaded)} raw candidate images before dedup.")

    # Deduplicate by SHA-256 and 64-bit pHash (Hamming <= 6) against all existing datasets and within batch
    accepted_records: list[dict[str, Any]] = []
    dup_skipped = 0
    for item in downloaded:
        sha = item["sha256"]
        ph = item["phash_int"]
        if sha in sha_set:
            dup_skipped += 1
            continue
        is_near_dup = False
        for _, ext_ph in phash_list:
            if hamming_dist(ph, ext_ph) <= 6:
                is_near_dup = True
                break
        if is_near_dup:
            dup_skipped += 1
            continue

        sha_set.add(sha)
        iid = f"task8d_{len(accepted_records) + 1:04d}"
        phash_list.append((iid, ph))
        fn = f"{iid}.jpg"
        (IMAGES_DIR / fn).write_bytes(item["norm_bytes"])

        c = item["cand"]
        rec = {
            "item_id": iid,
            "filename": fn,
            "source_family": c["source_family"],
            "source_split": c["source_split"],
            "source_image_id": c["source_image_id"],
            "original_title": c["original_title"],
            "image_url": c["image_url"],
            "landing_url": c["landing_url"],
            "license": c["license"],
            "author": c["author"],
            "author_profile_url": c["author_profile_url"],
            "attribution_required": c["attribution_required"],
            "session_group": c["session_hint"],
            "specimen_group": f"SPEC_{iid}_{item['phash_hex'][:8]}",
            "sha256": sha,
            "phash_hex": item["phash_hex"],
            "width": item["width"],
            "height": item["height"],
            "primary_class_id": c["primary_class_id"],
            "primary_class_name": CLASS_NAMES[c["primary_class_id"]],
            "upstream_boxes": c.get("upstream_boxes", []),
        }
        accepted_records.append(rec)

    raw_manifest_path = W8D_DIR / "task8d_raw_candidates_manifest.json"
    raw_manifest_path.write_text(
        json.dumps(
            {
                "total_candidates_queried": len(all_cands),
                "total_downloaded": len(downloaded),
                "duplicates_filtered_out": dup_skipped,
                "total_unique_staged": len(accepted_records),
                "records": accepted_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[Task8D] Saved {len(accepted_records)} unique deduplicated candidates (skipped {dup_skipped} duplicates).")

    # Build pilot sheets for SRC_06 (OI waste filtered) and SRC_07 (Wikimedia waste queries)
    oi_pilot_items = []
    wiki_pilot_items = []
    for r in accepted_records:
        im = Image.open(IMAGES_DIR / r["filename"]).convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for b in r.get("upstream_boxes", []):
            cx, cy, bw, bh = b["cx"], b["cy"], b["w"], b["h"]
            dr.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], outline=(50, 255, 80), width=4)
        cap = f"{r['item_id']}|{r['primary_class_id']}:{r['primary_class_name'][:5]}|{r['original_title'][:14]}"
        if r["source_family"].startswith("openimages_v7") and len(oi_pilot_items) < 30:
            oi_pilot_items.append((im, cap))
        elif r["source_family"] == "wikimedia_commons_waste_context" and len(wiki_pilot_items) < 30:
            wiki_pilot_items.append((im, cap))

    build_contact_sheet(
        oi_pilot_items,
        PILOT_DIR / "pilot_src06_oi_waste_context_30.jpg",
        "Task 8D Pilot SRC_06: Open Images V7 Waste-Context Filtered (30 Samples)",
    )
    build_contact_sheet(
        wiki_pilot_items,
        PILOT_DIR / "pilot_src07_wikimedia_waste_context_30.jpg",
        "Task 8D Pilot SRC_07: Wikimedia Commons Explicit Waste Queries (30 Samples)",
    )

    # Render full inspection contact sheets (30 images per sheet) for visual QA & instance box labeling
    for page_idx in range(0, len(accepted_records), 30):
        chunk = accepted_records[page_idx : page_idx + 30]
        sheet_items = []
        for r in chunk:
            im = Image.open(IMAGES_DIR / r["filename"]).convert("RGB")
            w, h = im.size
            dr = ImageDraw.Draw(im)
            for b in r.get("upstream_boxes", []):
                cx, cy, bw, bh = b["cx"], b["cy"], b["w"], b["h"]
                dr.rectangle(
                    [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                    outline=(50, 255, 80),
                    width=max(2, int(min(w, h) * 0.006)),
                )
            cap = f"{r['item_id'].replace('task8d_','')}|{r['primary_class_id']}:{r['primary_class_name'][:4]}|{r['original_title'][:16]}"
            sheet_items.append((im, cap))
        pnum = (page_idx // 30) + 1
        build_contact_sheet(
            sheet_items,
            INSPECT_DIR / f"task8d_inspect_p{pnum:02d}.jpg",
            f"Task 8D Visual Inspection Sheet p{pnum:02d} ({chunk[0]['item_id']} .. {chunk[-1]['item_id']})",
        )
    print(f"[Task8D] Rendered {(len(accepted_records)+29)//30} visual inspection sheets in {INSPECT_DIR}")


if __name__ == "__main__":
    main()
