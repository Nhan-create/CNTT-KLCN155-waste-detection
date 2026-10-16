"""
scripts/fetch_task8_multi_source_supplement.py
----------------------------------------------
Expands the Task 8 Wave 1 acquisition batch (`data/acquisition_staging/task8a_multiclass_batch`)
with:
  1. Unused Open Images V7 validation images + official bounding boxes from
     `D:\\waste-training\\openimages-validation-boxes.csv` (AWS S3 `open-images-dataset.s3.amazonaws.com`,
     CC BY 2.0 / CC BY 4.0) across cardboard (`Box`), paper/plastic (`Coffee cup`, `Book`),
     glass/plastic (`Bottle`, `Wine glass`), shoes (`Boot`, `Sandal`), clothes (`Clothing`, `Glove`, `Towel`),
     biological (`Banana`, `Egg`), and trash/context (`Waste container`).
  2. Wikimedia Commons real-world waste candidates for the remaining classes (`0: battery`,
     `3: clothes`, `5: metal`, `7: plastic`, `9: trash`) so all 10 classes are covered in Wave 1
     from multiple distinct sources (not only Wikimedia Commons).
"""

from __future__ import annotations

import csv
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
import pandas as pd
from PIL import Image, ImageDraw

from scripts.search_and_download_task8a_candidates import (
    ALLOWED_LICENSE_KEYWORDS,
    EXCLUDE_TITLE_KEYWORDS,
    GRIDS_DIR,
    IMAGES_DIR,
    RAW_MANIFEST_P,
    REPO_ROOT,
    clean_html,
    load_existing_titles_and_hashes,
    sha256_file,
)

HEADERS = {
    "User-Agent": "WasteDetectionAcademicBot/2.4 (https://commons.wikimedia.org/; contact: waste_research_klcn155@university.edu)"
}

OI_BOXES_CSV = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training"))) / "openimages-validation-boxes.csv")
OI_CLASSES_CSV = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training"))) / "openimages-boxable-classes.csv")

HUMAN_LABELS = {
    "/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655", "/m/02p0tk3", "/m/035r7c",
    "/m/031n1", "/m/0dzf4", "/m/0k65p", "/m/04hgtk", "/m/0dzct", "/m/03q69",
}

# (label_mid, subgroup, proposed_target_class, max_images)
OI_TARGETS = [
    ("/m/025dyy", "oi_box_cardboard", "2: cardboard", 6),
    ("/m/02p5f1q", "oi_coffee_cup", "6: paper", 6),
    ("/m/0bt_c3", "oi_book_paper", "6: paper", 2),
    ("/m/04dr76w", "oi_bottle_container", "4: glass", 8),
    ("/m/09tvcd", "oi_wine_glass", "4: glass", 1),
    ("/m/01b638", "oi_boot_shoes", "8: shoes", 4),
    ("/m/03nfch", "oi_sandal_shoes", "8: shoes", 4),
    ("/m/09j2d", "oi_clothing", "3: clothes", 1),
    ("/m/0174n1", "oi_glove_clothes", "3: clothes", 3),
    ("/m/0162_1", "oi_towel_clothes", "3: clothes", 2),
    ("/m/03v5tg", "oi_straw_plastic", "7: plastic", 2),
    ("/m/0bjyj5", "oi_waste_container", "9: trash", 3),
]

WIKI_SUPPLEMENT_CATEGORIES = [
    ("metal_crushed_cans_cat", "5: metal", "Category:Crushed beverage cans", 4),
    ("metal_rusted_cans_cat", "5: metal", "Category:Rusted tin cans", 3),
    ("plastic_crushed_bottles_cat", "7: plastic", "Category:Crushed plastic bottles", 4),
    ("plastic_bags_litter_cat", "7: plastic", "Category:Plastic bags as litter", 3),
    ("clothes_discarded_cat", "3: clothes", "Category:Abandoned clothing", 4),
    ("trash_cigarette_butts_cat", "9: trash", "Category:Cigarette butts", 4),
    ("battery_used_cat", "0: battery", "Category:Used batteries", 3),
]


def render_grid_sheet_with_oi_boxes(items: List[Dict[str, Any]], out_path: Path, cols: int = 3) -> None:
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
            f"{item['actual_width']}x{item['actual_height']} | {item['target_class']} | {item['source_name'][:24]}",
            fill=(200, 210, 220),
        )

        img_path = REPO_ROOT / item["local_path"]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB").resize((cell_w - 40, cell_h - 40), Image.Resampling.BILINEAR)

        ix0 = x0 + 28
        iy0 = y0 + header_h + 20
        iw, ih = cell_w - 40, cell_h - 40
        canvas.paste(rgb, (ix0, iy0))

        # Draw coordinate grid
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

        # If pre-existing source boxes exist (Open Images V7), draw them in bright magenta
        for b_idx, sb in enumerate(item.get("preexisting_source_boxes", []), 1):
            xmin, ymin, xmax, ymax = sb["xyxy_norm"]
            bx0 = int(ix0 + xmin * iw)
            by0 = int(iy0 + ymin * ih)
            bx1 = int(ix0 + xmax * iw)
            by1 = int(iy0 + ymax * ih)
            draw.rectangle([bx0, by0, bx1, by1], outline=(255, 0, 255), width=2)
            draw.text((bx0 + 3, max(iy0 + 2, by0 + 2)), f"OI#{b_idx}:{sb['source_label_name']}", fill=(255, 120, 255))

        draw.rectangle([ix0, iy0, ix0 + iw, iy0 + ih], outline=(200, 200, 200), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def fetch_wiki_category(cmtitle: str) -> Dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": cmtitle,
        "gcmtype": "file",
        "gcmlimit": 20,
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
        seen_titles.add(c.get("source_file_title", ""))
        seen_sha256.add(c["sha256"])
    seen_urls: Set[str] = {c["download_url"] for c in existing_candidates}

    existing_oi_ids = {
        p.stem[3:] for p in (REPO_ROOT / "data" / "detection" / "images" / "real").glob("oi_*.jpg")
    }
    for c in existing_candidates:
        if c.get("openimages_id"):
            existing_oi_ids.add(c["openimages_id"])

    with open(OI_CLASSES_CSV, encoding="utf-8") as f:
        oi_class_names = {r[0]: r[1] for r in csv.reader(f)}

    df = pd.read_csv(OI_BOXES_CSV)
    human_imgs = set(df.loc[df.LabelName.isin(HUMAN_LABELS), "ImageID"])

    all_target_mids = {m for m, _, _, _ in OI_TARGETS}
    sub = df[
        df.LabelName.isin(all_target_mids)
        & ~df.ImageID.isin(existing_oi_ids | human_imgs)
        & (df.IsDepiction == 0)
        & (df.IsGroupOf == 0)
    ].copy()
    sub["area"] = (sub.XMax - sub.XMin) * (sub.YMax - sub.YMin)

    newly_added: List[Dict[str, Any]] = []

    print("=== STEP 1: Fetching Open Images V7 Official Bounding-Box Candidates ===")
    for mid, subgroup, target_cls, max_imgs in OI_TARGETS:
        g = sub[sub.LabelName == mid]
        valid_iids: List[str] = []
        for iid, ig in g.groupby("ImageID"):
            if iid in existing_oi_ids:
                continue
            if 1 <= len(ig) <= 4 and ig["area"].min() >= 0.02 and ig["area"].max() <= 0.85:
                valid_iids.append(str(iid))
        valid_iids.sort()

        group_added = 0
        for iid in valid_iids:
            if group_added >= max_imgs:
                break
            s3_url = f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"
            if s3_url in seen_urls:
                continue

            idx = len(existing_candidates) + 1
            cand_id = f"task8a_{idx:03d}_{subgroup}_{iid}"
            filename = f"{cand_id}.jpg"
            dest_path = IMAGES_DIR / filename

            try:
                req = urllib.request.Request(s3_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                if len(data) < 2048:
                    continue
                dest_path.write_bytes(data)

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

                # Extract all official OpenImages boxes on this image across target MIDs
                img_rows = sub[sub.ImageID == iid]
                pre_boxes = []
                for _, r in img_rows.iterrows():
                    xmin = round(max(0.0, float(r.XMin)), 4)
                    ymin = round(max(0.0, float(r.YMin)), 4)
                    xmax = round(min(1.0, float(r.XMax)), 4)
                    ymax = round(min(1.0, float(r.YMax)), 4)
                    if xmax - xmin > 0.005 and ymax - ymin > 0.005:
                        pre_boxes.append({
                            "source_mid": str(r.LabelName),
                            "source_label_name": oi_class_names.get(str(r.LabelName), str(r.LabelName)),
                            "xyxy_norm": [xmin, ymin, xmax, ymax],
                            "is_occluded": int(r.IsOccluded),
                            "is_truncated": int(r.IsTruncated),
                        })

                fsize = dest_path.stat().st_size
                existing_oi_ids.add(iid)
                seen_urls.add(s3_url)
                seen_sha256.add(file_sha)
                group_added += 1

                acquired_ts = datetime.now(timezone.utc).isoformat()
                rec = {
                    "candidate_id": cand_id,
                    "short_key": f"task8a_{idx:03d}",
                    "filename": filename,
                    "subgroup": subgroup,
                    "target_class": target_cls,
                    "source_name": "Open Images V7 Validation (Official Bounding Boxes)",
                    "openimages_id": iid,
                    "source_file_title": f"OpenImages_V7:validation/{iid}.jpg",
                    "source_page_url": f"https://storage.googleapis.com/openimages/web/visualizer/index.html?set=val&type=detection&id={iid}",
                    "download_url": s3_url,
                    "original_url": s3_url,
                    "is_1024px_wikimedia_rendition": False,
                    "has_preexisting_boxes": True,
                    "preexisting_source_boxes": pre_boxes,
                    "license": "CC BY 2.0 (Image) / CC BY 4.0 (Open Images V7 Annotations)",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "usage_terms": "Open Images Dataset V7 (CC BY 4.0 annotations, CC BY 2.0 Flickr image)",
                    "attribution_required": "true",
                    "credit_line": f"Open Images V7 validation/{iid}.jpg (CC BY 2.0 / CC BY 4.0)",
                    "author": f"Open Images V7 Contributor ({iid})",
                    "acquired_at_utc": acquired_ts,
                    "expected_width": w,
                    "expected_height": h,
                    "original_file_size_bytes": fsize,
                    "source_description": f"Open Images V7 validation image with {len(pre_boxes)} official box(es) for {oi_class_names.get(mid, mid)}",
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
                print(f"  [OI #{idx:03d}] {filename} ({w}x{h}, {fsize:,} B, {target_cls}, boxes={len(pre_boxes)})")
            except Exception as e:
                print(f"  [SKIP OI {iid}] {e}")
                dest_path.unlink(missing_ok=True)

    print("\n=== STEP 2: Fetching Wikimedia Commons Supplement for Remaining Classes (0, 3, 5, 7, 9) ===")
    for subgroup, target_cls, cmtitle, max_for_group in WIKI_SUPPLEMENT_CATEGORIES:
        time.sleep(0.35)
        try:
            res = fetch_wiki_category(cmtitle)
        except Exception as e:
            print(f"Wiki category error {cmtitle}: {e}")
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
            if any(k.lower() in lower_title for k in EXCLUDE_TITLE_KEYWORDS):
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
                time.sleep(0.22)

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
                    "has_preexisting_boxes": False,
                    "preexisting_source_boxes": [],
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
                print(f"  [WIKI #{idx:03d}] {filename} ({w}x{h}, {fsize:,} B, {target_cls})")
            except Exception as e:
                print(f"  [SKIP WIKI] {filename}: {e}")
                dest_path.unlink(missing_ok=True)

    total_bytes = sum(c["downloaded_bytes"] for c in existing_candidates)
    by_class_counts: Dict[str, int] = {}
    by_source_counts: Dict[str, int] = {}
    for r in existing_candidates:
        tc = r["target_class"]
        by_class_counts[tc] = by_class_counts.get(tc, 0) + 1
        sn = "Open Images V7" if "Open Images" in r.get("source_name", "") else "Wikimedia Commons"
        by_source_counts[sn] = by_source_counts.get(sn, 0) + 1

    existing_data["total_downloaded"] = len(existing_candidates)
    existing_data["total_downloaded_bytes"] = total_bytes
    existing_data["total_downloaded_mb"] = round(total_bytes / (1024 * 1024), 2)
    existing_data["raw_counts_by_target_class"] = by_class_counts
    existing_data["raw_counts_by_source_family"] = by_source_counts
    existing_data["candidates"] = existing_candidates
    RAW_MANIFEST_P.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAdded {len(newly_added)} multi-source candidates. Total = {len(existing_candidates)} ({existing_data['total_downloaded_mb']} MB)")
    print("By class:", json.dumps(by_class_counts, indent=2))
    print("By source:", json.dumps(by_source_counts, indent=2))

    chunk_size = 18
    sheet_num = 11
    for i in range(start_count, len(existing_candidates), chunk_size):
        chunk = existing_candidates[i : i + chunk_size]
        start_num = i + 1
        end_num = i + len(chunk)
        sheet_p = GRIDS_DIR / f"raw_grid_sheet_{sheet_num:02d}_{start_num:03d}_to_{end_num:03d}.jpg"
        render_grid_sheet_with_oi_boxes(chunk, sheet_p, cols=3)
        print(f"Rendered inspection grid: {sheet_p.name}")
        sheet_num += 1


if __name__ == "__main__":
    main()
