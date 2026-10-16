"""
scripts/stage_phase4e_whitelist_candidates.py
---------------------------------------------
Manages the safe, verified staging of Phase 4E.1 prioritized mini-pilot candidates
from Wikimedia Commons (2 biological waste, 2 urban utility cabinet hard-negatives).

P0 Safety Rules (Phase 4E.1):
1. Zero 0-byte label files generated on disk before full-frame physical negative inspection.
2. Zero placeholder or comment lines in YOLO .txt files (would crash float parsers).
3. Metadata JSON strictly tracks unannotated / unreviewed state.
4. YOLO .txt label files are ONLY generated after explicit visual QA and bounding box completion.
5. Training approval status is strictly 'PENDING_USER_APPROVAL' (NEVER 'APPROVED_FOR_TRAIN').
6. Pre-download gating: Displays exact files, URLs, sizes, licenses, and halts for user approval.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import imagehash

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGING_DIR = ROOT / "data" / "acquisition_staging" / "phase4e_staging"
IMAGES_DIR = STAGING_DIR / "images"
LABELS_DIR = STAGING_DIR / "labels"
OVERLAYS_DIR = STAGING_DIR / "overlays"
CONTACT_SHEET_DIR = STAGING_DIR / "contact_sheet"

MAX_ALLOWED_TOTAL_BYTES = 20 * 1024 * 1024  # 20 MB total cap for mini-pilot
CHUNK_SIZE = 16384  # 16 KB chunks

# -------------------------------------------------------------
# PRIORITIZED MINI-PILOT SHORTLIST (4 VERIFIED CANDIDATES)
# -------------------------------------------------------------
MINI_PILOT_SHORTLIST: Dict[str, Dict[str, Any]] = {
    "wiki_bio_banana_peel_ground": {
        "candidate_id": "wiki_bio_banana_peel_ground",
        "category": "biological",
        "filename": "wiki_bio_banana_peel_ground.jpg",
        "source_title": "File:Banana_peel_on_the_ground.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cf/Banana_peel_on_the_ground.jpg",
        "license": "CC BY-SA 4.0",
        "author": "Vladiator360",
        "resolution": "5184x3456",
        "expected_bytes": 4573954,
        "qa_criteria": "Verify authentic discarded banana skin lying on outdoor ground/floor (biological waste)."
    },
    "wiki_bio_street_food_remains": {
        "candidate_id": "wiki_bio_street_food_remains",
        "category": "biological",
        "filename": "wiki_bio_street_food_remains.jpg",
        "source_title": "File:Street_Food_Remains_(188107205).jpeg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Street_Food_Remains_%28188107205%29.jpeg",
        "license": "CC BY-SA 3.0",
        "author": "Andrew Currie",
        "resolution": "2048x1365",
        "expected_bytes": 294623,
        "qa_criteria": "Verify discarded street food leftovers on sidewalk, distinct from fresh dining dishes."
    },
    "wiki_neg_junction_boxes": {
        "candidate_id": "wiki_neg_junction_boxes",
        "category": "hard_negatives",
        "filename": "wiki_neg_junction_boxes.jpg",
        "source_title": "File:Junction_boxes_-_geograph.org.uk_-_580156.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Junction_boxes_-_geograph.org.uk_-_580156.jpg",
        "license": "CC BY-SA 2.0",
        "author": "Thomas Nugent",
        "resolution": "480x640",
        "expected_bytes": 110051,
        "qa_criteria": "Inspect 100% of frame to guarantee zero waste objects; utility boxes and wall background."
    },
    "wiki_neg_utility_box_leeds": {
        "candidate_id": "wiki_neg_utility_box_leeds",
        "category": "hard_negatives",
        "filename": "wiki_neg_utility_box_leeds.jpg",
        "source_title": "File:Utility_box,_Wesley_Street,_Beeston,_Leeds_(11th_August_2023)_001.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/13/Utility_box%2C_Wesley_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29_001.jpg",
        "license": "CC BY-SA 4.0",
        "author": "Mtaylor848",
        "resolution": "5152x3864",
        "expected_bytes": 5736159,
        "qa_criteria": "Inspect green sidewalk utility box against brick wall (direct countermeasure for rac_online_012 FP error)."
    }
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_phash_from_bytes(b: bytes) -> str:
    with Image.open(io.BytesIO(b)) as img:
        return str(imagehash.phash(img))


def main():
    parser = argparse.ArgumentParser(description="Stage Phase 4E.1 mini-pilot real waste candidates.")
    parser.add_argument("--approve-download", action="store_true", help="Explicit user approval required to download.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect candidate metadata without network traffic.")
    args = parser.parse_args()

    total_expected_bytes = sum(c["expected_bytes"] for c in MINI_PILOT_SHORTLIST.values())

    print("=" * 75)
    print("PHASE 4E.1: REAL WASTE MINI-PILOT STAGING MANAGER")
    print("=" * 75)
    print(f"Staging Directory:         {STAGING_DIR}")
    print(f"Prioritized Shortlist:     {len(MINI_PILOT_SHORTLIST)} verified images")
    print(f"Total Expected Download:   {total_expected_bytes:,} bytes (~{total_expected_bytes / (1024*1024):.2f} MB)")
    print(f"Hard Cap Limit:            {MAX_ALLOWED_TOTAL_BYTES / (1024*1024):.1f} MB")

    for cid, info in MINI_PILOT_SHORTLIST.items():
        print(f"\n- Candidate ID: {cid} ({info['category'].upper()})")
        print(f"  Filename:     {info['filename']}")
        print(f"  Source Title: {info['source_title']}")
        print(f"  URL:          {info['url']}")
        print(f"  License:      {info['license']} (Author: {info['author']})")
        print(f"  Resolution:   {info['resolution']}, Expected Size: {info['expected_bytes']:,} B")
        print(f"  QA Criteria:  {info['qa_criteria']}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETED] Metadata inspection finished. Zero network requests made.")
        sys.exit(0)

    # STRICT FAIL-CLOSED CHECK: Halt execution until user explicitly passes --approve-download
    if not args.approve_download:
        print(
            "\n[ACTION BLOCKED: USER APPROVAL REQUIRED]\n"
            "Pre-download safety gate engaged. All real image downloads require explicit user authorization.\n"
            "To proceed, the user must provide explicit instruction, then run with: --approve-download\n",
            file=sys.stderr
        )
        sys.exit(1)

    print("\n[APPROVED BY USER] Downloading prioritized mini-pilot candidates...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-split duplicate and near-duplicate leakage verification
    from scripts.ingest_roboflow_battery_export import load_authoritative_split_hashes
    split_hashes, split_phashes = load_authoritative_split_hashes()
    total_existing = sum(len(s) for s in split_hashes.values())
    print(f"Loaded {total_existing} existing authoritative split hashes across v2, v3, and holdout.")

    downloaded = []
    total_bytes = 0
    seen_batch_shas = set()
    seen_batch_phashes: Dict[str, imagehash.ImageHash] = {}

    for cid, info in MINI_PILOT_SHORTLIST.items():
        print(f"\nFetching {cid} ({info['filename']})...")
        req = urllib.request.Request(info["url"], headers={"User-Agent": "WasteDetectionResearch/1.0 (contact@waste-detection.org)"})

        buf = bytearray()
        with urllib.request.urlopen(req, timeout=30) as resp:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                buf.extend(chunk)
                if total_bytes + len(buf) > MAX_ALLOWED_TOTAL_BYTES:
                    raise ValueError(f"Exceeded hard size limit ({MAX_ALLOWED_TOTAL_BYTES} bytes)")

        data = bytes(buf)
        total_bytes += len(data)

        # Validate decode with PIL
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.size
            img_format = im.format
            cand_phash = imagehash.phash(im)

        dest_img = IMAGES_DIR / info["filename"]
        dest_img.write_bytes(data)
        sha = sha256_bytes(data)
        phash_str = str(cand_phash)

        # 1. Exact collision check against authoritative splits (v2, v3, holdout)
        exact_split_collisions = [sname for sname, shas in split_hashes.items() if sha in shas]

        # 2. Near-duplicate collision check against authoritative splits (pHash Hamming distance <= 8)
        near_dup_collision = None
        for sname, p_dict in split_phashes.items():
            for ref_name, ref_ph in p_dict.items():
                dist = cand_phash - ref_ph
                if dist <= 8:
                    near_dup_collision = f"NEAR_DUP_{sname.upper()}_{ref_name}_DIST_{dist}"
                    break
            if near_dup_collision:
                break

        # 3. Batch collision check
        batch_dup = sha in seen_batch_shas
        seen_batch_shas.add(sha)
        seen_batch_phashes[cid] = cand_phash

        is_leakage_clean = (len(exact_split_collisions) == 0 and near_dup_collision is None and not batch_dup)

        # P0 REQUIREMENT:
        # 1. Do NOT write 0-byte labels for hard-negatives before physical inspection.
        # 2. Do NOT write placeholder comment lines into YOLO .txt files.
        # 3. Keep labels_dir empty; store pending state strictly in metadata JSON.
        record = {
            "candidate_id": cid,
            "filename": info["filename"],
            "category": info["category"],
            "sha256": sha,
            "phash": phash_str,
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "size_bytes": len(data),
            "format": img_format,
            "license": info["license"],
            "author": info["author"],
            "source_url": info["url"],
            "qa_criteria": info["qa_criteria"],
            "exact_split_collisions": exact_split_collisions,
            "near_dup_collision": near_dup_collision,
            "split_leakage_clean": is_leakage_clean,
            "has_disk_label_file": False,  # Strict: no premature .txt files on disk
            "annotation_status": "PENDING_HUMAN_ANNOTATION" if info["category"] != "hard_negatives" else "PENDING_NEGATIVE_INSPECTION",
            "qa_status": "DOWNLOADED_AWAITING_VISUAL_REVIEW",
            "training_approval_status": "PENDING_USER_APPROVAL"  # Never auto-approved
        }
        downloaded.append(record)
        print(f"  -> Saved {info['filename']} ({w}x{h}, {len(data):,} B, sha: {sha[:10]}..., phash: {phash_str})")
        print(f"  -> Cross-split leakage check: {'CLEAN (Zero collision)' if is_leakage_clean else 'FLAGGED: ' + str(exact_split_collisions or near_dup_collision)}")
        print(f"  -> Disk label generation: DEFERRED (tracked in metadata JSON only)")

    if downloaded:
        from PIL import ImageDraw
        # Create 2x2 contact sheet of downloaded original images with text metadata
        thumb_w, thumb_h = 640, 480
        sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 2), color=(20, 24, 30))
        draw = ImageDraw.Draw(sheet)

        for idx, rec in enumerate(downloaded):
            col = idx % 2
            row = idx // 2
            img_path = IMAGES_DIR / rec["filename"]
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                im_rgb.thumbnail((thumb_w - 30, thumb_h - 60))
                x_off = col * thumb_w + (thumb_w - im_rgb.width) // 2
                y_off = row * thumb_h + 35 + (thumb_h - 60 - im_rgb.height) // 2
                sheet.paste(im_rgb, (x_off, y_off))

            # Header bar for each cell
            cell_x = col * thumb_w
            cell_y = row * thumb_h
            draw.rectangle([cell_x, cell_y, cell_x + thumb_w - 1, cell_y + 30], fill=(35, 40, 50))
            tag = f"[{rec['category'].upper()}] {rec['candidate_id']} ({rec['resolution']})"
            draw.text((cell_x + 10, cell_y + 8), tag, fill=(240, 240, 240))

            # Footer bar for each cell
            draw.rectangle([cell_x, cell_y + thumb_h - 25, cell_x + thumb_w - 1, cell_y + thumb_h - 1], fill=(25, 30, 38))
            meta_tag = f"SHA: {rec['sha256'][:12]}... | pHash: {rec['phash']} | Size: {rec['size_bytes']:,} B"
            draw.text((cell_x + 10, cell_y + thumb_h - 20), meta_tag, fill=(180, 190, 200))

        sheet_path = CONTACT_SHEET_DIR / "mini_pilot_contact_sheet.jpg"
        sheet.save(sheet_path, "JPEG", quality=92)
        print(f"Contact sheet saved: {sheet_path}")

    summary_file = STAGING_DIR / "mini_pilot_staging_summary.json"
    summary_file.write_text(json.dumps({
        "status": "STAGED_AWAITING_HUMAN_QA",
        "total_images": len(downloaded),
        "total_bytes": total_bytes,
        "disk_label_files_count": 0,  # Strict guarantee
        "records": downloaded
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"PHASE 4E.2 MINI-PILOT STAGING COMPLETED: {len(downloaded)} images ({total_bytes:,} bytes)")
    print(f"Premature .txt label files on disk: 0 (Strict compliance)")
    print(f"Summary saved: {summary_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
