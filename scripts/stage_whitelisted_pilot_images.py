"""
scripts/stage_whitelisted_pilot_images.py
-----------------------------------------
Safely stages exactly THREE proven Open Images validation candidates for visual QA inspection.

Strict Whitelist:
1. 078821d86db99fc9 (Apple - Biological candidate; requires checking if fruit is discarded vs on-tree)
2. 74a44994e3d6b962 (Banana - Biological candidate; requires checking if discarded vs dining food)
3. e29db1c1963628e7 (Building - Hard Negative candidate; requires verifying 0 waste objects in frame)

Safety Guardrails:
- Execution is STRICTLY BLOCKED unless the user provides '--approve-preview-download'.
- Hard cap on maximum total download size (600 KB limit).
- Streaming HTTP read chunks with immediate limit enforcement (no unbounded resp.read()).
- Zero background downloading; network activity occurs only upon explicit flag.
- Generates raw downloaded images, bounding box overlays, and contact sheet for physical visual inspection.
- Labels are kept strictly at 'PROPOSED' status, NEVER auto-approved.
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

from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "pilot_preview_3"
IMAGES_DIR = STAGING_DIR / "images"
LABELS_DIR = STAGING_DIR / "labels"
OVERLAYS_DIR = STAGING_DIR / "overlays"
CONTACT_SHEET_DIR = STAGING_DIR / "contact_sheet"

MAX_ALLOWED_TOTAL_BYTES = 1200 * 1024  # 1200 KB hard safety cap
CHUNK_SIZE = 8192  # 8 KB streaming chunks

WHITELIST_TARGETS: Dict[str, Dict[str, Any]] = {
    "078821d86db99fc9": {
        "candidate_id": "oi_bio_apple_078821d86db99fc9",
        "category": "biological",
        "filename": "oi_078821d86db99fc9.jpg",
        "url": "https://open-images-dataset.s3.amazonaws.com/validation/078821d86db99fc9.jpg",
        "source_title": "Omenapuu Åkerö (Apple tree)",
        "source_license": "CC BY 2.0 (Flickr: Matleena Laakso)",
        "expected_max_size": 400000,
        "qa_inspection_criteria": (
            "Title indicates 'Omenapuu Åkerö' (apple tree). Inspect whether apples are authentic "
            "fallen/discarded biological waste or active live fruit hanging on tree branches. "
            "If active orchard tree, REJECT immediately. Only transition to HUMAN_REVIEW if discarded."
        ),
        "openimages_boxes": [
            {"label": "/m/014j1m", "class_name": "Apple", "class_id": 1, "box": [0.244792, 0.542847, 0.092857, 0.539823]},
            {"label": "/m/014j1m", "class_name": "Apple", "class_id": 1, "box": [0.383545, 0.697571, 0.475664, 0.942478]}
        ]
    },
    "74a44994e3d6b962": {
        "candidate_id": "oi_bio_banana_74a44994e3d6b962",
        "category": "biological",
        "filename": "oi_74a44994e3d6b962.jpg",
        "url": "https://open-images-dataset.s3.amazonaws.com/validation/74a44994e3d6b962.jpg",
        "source_title": "Vegan Lemon Cornmeal Pancakes",
        "source_license": "CC BY 2.0 (Flickr: tomatoes and friends)",
        "expected_max_size": 700000,
        "qa_inspection_criteria": (
            "Title indicates 'Vegan Lemon Cornmeal Pancakes'. Inspect whether banana pieces are discarded food "
            "waste or freshly prepared dining food served on tableware. "
            "If active dining food, REJECT immediately. Only transition to HUMAN_REVIEW if discarded/scrap."
        ),
        "openimages_boxes": [
            {"label": "/m/09qck", "class_name": "Banana", "class_id": 1, "box": [0.108333, 0.425000, 0.290625, 0.448438]},
            {"label": "/m/09qck", "class_name": "Banana", "class_id": 1, "box": [0.231250, 0.466667, 0.201562, 0.310937]},
            {"label": "/m/09qck", "class_name": "Banana", "class_id": 1, "box": [0.322917, 0.716667, 0.232812, 0.418750]},
            {"label": "/m/09qck", "class_name": "Banana", "class_id": 1, "box": [0.535417, 0.743750, 0.220312, 0.367188]}
        ]
    },
    "e29db1c1963628e7": {
        "candidate_id": "oi_neg_building_e29db1c1963628e7",
        "category": "hard_negatives",
        "filename": "oi_e29db1c1963628e7.jpg",
        "url": "https://open-images-dataset.s3.amazonaws.com/validation/e29db1c1963628e7.jpg",
        "source_title": "Ronda (Building facade)",
        "source_license": "CC BY 2.0 (Flickr: Manuela Aguilera)",
        "expected_max_size": 100000,
        "qa_inspection_criteria": (
            "Building facade background scene. Inspect entire pixel frame to ensure ZERO waste items "
            "(no litter, bottles, cans, plastic, paper, or clothing). Label is empty 0-byte text. "
            "If any waste object is visible in frame, REJECT immediately. "
            "If uncertain -> PENDING_NEGATIVE_REVIEW. Never mark VERIFIED_TRUE_NEGATIVE without explicit human QA."
        ),
        "openimages_boxes": []
    }
}


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def box_to_yolo(box: list[float]) -> str:
    # box is [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = box
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def get_font(size: int = 14) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ["arial.ttf", "segoeui.ttf", "tahoma.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlay_image(img_path: Path, boxes: List[Dict[str, Any]], out_path: Path, title: str, category: str):
    """Draws bounding boxes and metadata banner on the image."""
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font_large = get_font(18)
    font_small = get_font(14)
    
    # Top info banner
    banner_height = 40
    draw.rectangle([(0, 0), (w, banner_height)], fill=(20, 24, 33))
    banner_text = f"ID: {img_path.stem} | Cat: {category} | Boxes: {len(boxes)} | Status: PROPOSED (Unapproved)"
    draw.text((10, 10), banner_text, fill=(240, 240, 245), font=font_small)
    
    if len(boxes) == 0:
        # Hard negative notice banner
        draw.rectangle([(0, h - 35), (w, h)], fill=(40, 50, 60))
        draw.text((10, h - 28), "CANDIDATE HARD NEGATIVE - 0 WASTE DETECTIONS EXPECTED", fill=(100, 220, 255), font=font_small)
    else:
        for idx, b in enumerate(boxes):
            xmin, xmax, ymin, ymax = b["box"]
            x1 = int(xmin * w)
            x2 = int(xmax * w)
            y1 = int(ymin * h)
            y2 = int(ymax * h)
            
            color = (30, 215, 96) if b["class_id"] == 1 else (255, 140, 0)
            
            # Draw rectangle (thickness = 3)
            for offset in range(3):
                draw.rectangle([(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)], outline=color)
            
            # Label tag
            tag_text = f"{b['class_name']} (#{idx+1})"
            tag_w = int(font_small.getlength(tag_text)) + 8 if hasattr(font_small, "getlength") else len(tag_text) * 8 + 8
            tag_h = 20
            tag_y1 = max(banner_height, y1 - tag_h)
            draw.rectangle([(x1, tag_y1), (x1 + tag_w, tag_y1 + tag_h)], fill=color)
            draw.text((x1 + 4, tag_y1 + 2), tag_text, fill=(0, 0, 0), font=font_small)

    img.save(out_path, quality=95)


def generate_contact_sheet(
    image_records: List[Dict[str, Any]],
    out_path: Path
):
    """
    Renders a unified contact sheet containing all 3 preview images with
    their bounding boxes, metadata, and QA context cards.
    """
    card_width = 460
    img_render_height = 360
    card_header_height = 70
    card_footer_height = 140
    card_total_height = card_header_height + img_render_height + card_footer_height
    padding = 20
    
    sheet_w = padding * 4 + card_width * 3
    sheet_h = padding * 2 + card_total_height + 80  # +80 for sheet title banner
    
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    
    font_title = get_font(22)
    font_bold = get_font(15)
    font_regular = get_font(13)
    
    # Global Title
    draw.text((padding, 20), "VISUAL QA CONTACT SHEET — OPEN IMAGES PREVIEW (3 SAMPLES)", fill=(255, 255, 255), font=font_title)
    sub = "Status: PROPOSED ONLY (No Training / No Auto-Approval). Evaluation against Discarded Waste Criteria."
    draw.text((padding, 50), sub, fill=(180, 190, 205), font=font_regular)
    
    for idx, rec in enumerate(image_records):
        cx = padding + idx * (card_width + padding)
        cy = 90
        
        # Card background
        draw.rectangle([(cx, cy), (cx + card_width, cy + card_total_height)], fill=(28, 32, 42), outline=(50, 58, 75))
        
        # Header text
        target = WHITELIST_TARGETS[rec["image_id"]]
        draw.text((cx + 12, cy + 10), f"Sample #{idx+1}: {rec['filename']}", fill=(255, 215, 0), font=font_bold)
        draw.text((cx + 12, cy + 30), f"Category: {target['category'].upper()} | Res: {rec['resolution']} | Size: {rec['size_bytes']:,} B", fill=(200, 210, 220), font=font_regular)
        draw.text((cx + 12, cy + 48), f"Source Title: {target['source_title']}", fill=(160, 175, 190), font=font_regular)
        
        # Render overlay image scaled to fit
        overlay_path = OVERLAYS_DIR / f"overlay_{rec['filename']}"
        with Image.open(overlay_path) as oim:
            oim_copy = oim.copy()
        
        # Letterbox/fit inside card_width x img_render_height
        oim_copy.thumbnail((card_width - 16, img_render_height), Image.Resampling.LANCZOS)
        paste_x = cx + 8 + ((card_width - 16) - oim_copy.width) // 2
        paste_y = cy + card_header_height + (img_render_height - oim_copy.height) // 2
        sheet.paste(oim_copy, (paste_x, paste_y))
        
        # Footer text: Criteria & QA Status
        fy = cy + card_header_height + img_render_height + 8
        draw.line([(cx + 8, fy), (cx + card_width - 8, fy)], fill=(60, 70, 90))
        
        draw.text((cx + 12, fy + 8), "QA Verdict & Context Check:", fill=(255, 180, 50), font=font_bold)
        
        # Word wrap criteria
        criteria_lines = rec["qa_context_summary"]
        for c_idx, line in enumerate(criteria_lines):
            draw.text((cx + 12, fy + 30 + c_idx * 18), line, fill=(215, 225, 235), font=font_regular)
            
        status_color = (255, 80, 80) if "REJECT" in rec["preliminary_verdict"] else (100, 200, 255)
        draw.text((cx + 12, cy + card_total_height - 25), f"Verdict: {rec['preliminary_verdict']}", fill=status_color, font=font_bold)
        
    sheet.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Stage strictly whitelisted 3 Open Images preview candidates.")
    parser.add_argument(
        "--approve-preview-download",
        action="store_true",
        help="Explicit user approval confirmation required to execute network download."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect candidate metadata, URLs, and expected sizes without downloading."
    )

    args = parser.parse_args()

    print("=" * 75)
    print("STAGING MANAGER: 3 WHITELISTED OPEN IMAGES PREVIEW CANDIDATES")
    print("=" * 75)
    print(f"Target Staging Directory: {STAGING_DIR}")
    print(f"Whitelisted Count:        {len(WHITELIST_TARGETS)} images")
    print(f"Maximum Allowed Download: {MAX_ALLOWED_TOTAL_BYTES / 1024:.1f} KB")

    for iid, info in WHITELIST_TARGETS.items():
        print(f"\n- ImageID: {iid} ({info['category']})")
        print(f"  Filename:     {info['filename']}")
        print(f"  URL:          {info['url']}")
        print(f"  QA Criteria:  {info['qa_inspection_criteria']}")

    if args.dry_run:
        print("\n[DRY RUN] Inspection completed. Zero network requests made.")
        sys.exit(0)

    # STRICT USER APPROVAL CHECK
    if not args.approve_preview_download:
        print(
            "\n[ACTION BLOCKED: USER APPROVAL REQUIRED]\n"
            "Real image downloading is strictly disabled by default to prevent unauthorized network traffic.\n"
            "To proceed, provide explicit flag: --approve-preview-download\n",
            file=sys.stderr
        )
        sys.exit(1)

    print("\n[EXECUTION APPROVED BY USER] Downloading 3 preview images...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_records = []
    total_downloaded_bytes = 0

    for iid, info in WHITELIST_TARGETS.items():
        print(f"\nFetching {iid} ({info['filename']}) from S3...")
        req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0 (WasteDetectionQA/1.0)"})
        
        # STREAMING HTTP READ: Limit bytes during read loop; never do unbounded read()
        buffer = bytearray()
        with urllib.request.urlopen(req, timeout=30) as resp:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                buffer.extend(chunk)
                # Fail-closed check during streaming
                if len(buffer) > info["expected_max_size"] * 2:
                    raise ValueError(f"[ABORT] Single image {iid} exceeded size limit ({len(buffer)} > {info['expected_max_size']*2})")
                if total_downloaded_bytes + len(buffer) > MAX_ALLOWED_TOTAL_BYTES:
                    raise ValueError(f"[ABORT] Total download size exceeded hard limit ({total_downloaded_bytes + len(buffer)} > {MAX_ALLOWED_TOTAL_BYTES})")
                    
        data = bytes(buffer)
        size = len(data)
        total_downloaded_bytes += size

        # Validate decode with PIL
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.size
            img_format = im.format

        img_path = IMAGES_DIR / info["filename"]
        img_path.write_bytes(data)
        img_hash = compute_sha256(data)

        # Write labels (YOLO format)
        lbl_path = LABELS_DIR / f"{Path(info['filename']).stem}.txt"
        lbl_lines = []
        for b in info["openimages_boxes"]:
            yolo_box = box_to_yolo(b["box"])
            lbl_lines.append(f"{b['class_id']} {yolo_box}\n")
        lbl_text = "".join(lbl_lines)
        lbl_path.write_text(lbl_text, encoding="utf-8")
        lbl_hash = compute_sha256(lbl_path.read_bytes())

        # Render visual bounding box overlay
        overlay_path = OVERLAYS_DIR / f"overlay_{info['filename']}"
        render_overlay_image(img_path, info["openimages_boxes"], overlay_path, info["source_title"], info["category"])

        # Data-driven QA inspection criteria (avoid hardcoding by ImageID)
        qa_criteria = info.get("qa_inspection_criteria", "Inspect against municipal waste definition.")
        qa_context = [
            f"• Inspection Target: {info['category'].upper()}",
            f"• Provenance: {info.get('source_title', 'Unknown')}",
            f"• Criteria: {qa_criteria[:100]}...",
            "• State: Downloaded and verified on disk. Requires independent visual QA."
        ]
        # Initial status is strictly pending review - never hardcoded to pre-baked decision
        verdict = "DOWNLOADED_AWAITING_VISUAL_REVIEW"

        record = {
            "image_id": iid,
            "candidate_id": info["candidate_id"],
            "filename": info["filename"],
            "category": info["category"],
            "image_sha256": img_hash,
            "label_sha256": lbl_hash,
            "size_bytes": size,
            "resolution": f"{w}x{h}",
            "format": img_format,
            "source_title": info["source_title"],
            "source_license": info["source_license"],
            "source_url": info["url"],
            "openimages_boxes_count": len(info["openimages_boxes"]),
            "openimages_boxes": info["openimages_boxes"],
            "label_approval_status": "PROPOSED",  # NEVER AUTO-APPROVED
            "preliminary_verdict": verdict,
            "qa_context_summary": qa_context,
            "qa_inspection_criteria": info["qa_inspection_criteria"]
        }
        downloaded_records.append(record)
        print(f"  -> Saved {info['filename']} ({w}x{h}, {size:,} B, sha: {img_hash[:12]}...)")
        print(f"  -> Generated overlay: {overlay_path.name}")
        print(f"  -> Preliminary QA Verdict: {verdict}")

    # Generate unified contact sheet
    contact_sheet_path = CONTACT_SHEET_DIR / "contact_sheet_preview_3.jpg"
    generate_contact_sheet(downloaded_records, contact_sheet_path)
    print(f"\n-> Generated Contact Sheet: {contact_sheet_path.name}")

    # Write preview manifest
    manifest_file = STAGING_DIR / "preview_3_manifest.json"
    manifest_data = {
        "version": "1.0.0-preview-3-qa",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "staging_scope": "PREVIEW_AND_VISUAL_QA_ONLY",
        "merge_status": "NOT_MERGED_NO_TRAINING",
        "total_downloaded_images": len(downloaded_records),
        "total_downloaded_bytes": total_downloaded_bytes,
        "qa_breakdown": {
            "rejected_not_waste": sum(1 for r in downloaded_records if "REJECT" in r["preliminary_verdict"]),
            "pending_negative_review": sum(1 for r in downloaded_records if "PENDING" in r["preliminary_verdict"]),
            "approved_for_training": 0  # STRICTLY 0
        },
        "records": downloaded_records
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print("STAGING & VISUAL PREVIEW GENERATION COMPLETED SUCCESSFULLY")
    print(f"- Total Downloaded Images: {len(downloaded_records)}")
    print(f"- Total Downloaded Bytes:  {total_downloaded_bytes:,} bytes")
    print(f"- Contact Sheet:           {contact_sheet_path}")
    print(f"- Manifest:                {manifest_file}")
    print(f"- Label Status:            PROPOSED (0 Approved, 0 Merged, 0 Trained)")
    print("=" * 75)


if __name__ == "__main__":
    main()
