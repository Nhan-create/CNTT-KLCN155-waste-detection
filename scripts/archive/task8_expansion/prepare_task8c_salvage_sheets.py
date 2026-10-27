"""
Prepare 195 high-priority NEEDS_REVIEW salvage candidates for Task 8C visual inspection.
Computes improved contour/connected-component candidate boxes for Wikimedia Commons items
and renders 6 inspection contact sheets with a 0.2/0.4/0.6/0.8 normalized coordinate grid.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
W2_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch"
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


def propose_tight_boxes(img: Image.Image, primary_cid: int) -> tuple[list[dict[str, Any]], str]:
    """Propose tight bounding boxes using border-background distance + Canny edge hull."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    bh = max(4, int(h * 0.05))
    bw = max(4, int(w * 0.05))
    border = np.concatenate(
        [
            rgb[:bh, :, :].reshape(-1, 3),
            rgb[h - bh :, :, :].reshape(-1, 3),
            rgb[:, :bw, :].reshape(-1, 3),
            rgb[:, w - bw :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_median = np.median(border, axis=0)
    border_std = float(np.mean(np.std(border, axis=0)))
    dist = np.linalg.norm(rgb - bg_median[None, None, :], axis=2)

    thresh = max(24.0, border_std * 1.35)
    mask_u8 = (dist > thresh).astype(np.uint8) * 255
    ksize = max(5, int(min(h, w) * 0.02))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    comp_boxes: list[dict[str, Any]] = []
    for lbl_idx in range(1, num_labels):
        x, y, cw, ch, px_area = stats[lbl_idx]
        norm_area = float(px_area) / float(h * w)
        bw_n, bh_n = float(cw) / w, float(ch) / h
        if norm_area < 0.015 or bw_n < 0.05 or bh_n < 0.05:
            continue
        if (bw_n * bh_n) > 0.95:
            continue
        comp_boxes.append(
            {
                "class_id": primary_cid,
                "class_name": CLASS_NAMES[primary_cid],
                "mid": "commons_tight_cc",
                "mid_name": CLASS_NAMES[primary_cid],
                "xmin": round(max(0.0, float(x) / w), 4),
                "ymin": round(max(0.0, float(y) / h), 4),
                "xmax": round(min(1.0, float(x + cw) / w), 4),
                "ymax": round(min(1.0, float(y + ch) / h), 4),
                "area": bw_n * bh_n,
            }
        )

    # Sort by area descending and keep top non-nested boxes
    comp_boxes.sort(key=lambda b: b["area"], reverse=True)
    filtered: list[dict[str, Any]] = []
    for b in comp_boxes:
        is_inside_existing = False
        for eb in filtered:
            ix1 = max(b["xmin"], eb["xmin"])
            iy1 = max(b["ymin"], eb["ymin"])
            ix2 = min(b["xmax"], eb["xmax"])
            iy2 = min(b["ymax"], eb["ymax"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / max(1e-6, b["area"]) > 0.75:
                    is_inside_existing = True
                    break
        if not is_inside_existing:
            b_clean = {k: v for k, v in b.items() if k != "area"}
            b_clean.update({"is_occluded": 0, "is_truncated": 0, "is_inside": 0})
            filtered.append(b_clean)

    if 1 <= len(filtered) <= 4:
        return filtered, "cc_tight"

    # Fallback: single bounding box of top component if available
    if filtered:
        return [filtered[0]], "cc_top1"

    return (
        [
            {
                "class_id": primary_cid,
                "class_name": CLASS_NAMES[primary_cid],
                "mid": "commons_fallback",
                "mid_name": CLASS_NAMES[primary_cid],
                "xmin": 0.12,
                "ymin": 0.12,
                "xmax": 0.88,
                "ymax": 0.88,
                "is_occluded": 0,
                "is_truncated": 0,
                "is_inside": 0,
            }
        ],
        "fallback",
    )


def main() -> None:
    m = json.loads((W2_DIR / "task8_wave2_qa_manifest.json").read_text(encoding="utf-8"))
    nr = [r for r in m["records"] if r["qa_status"] == "NEEDS_REVIEW"]

    cm_non_bat = [r for r in nr if r["source_family"] == "wikimedia_commons_wave2" and r["primary_class_id"] != 0]
    cm_bat = [r for r in nr if r["source_family"] == "wikimedia_commons_wave2" and r["primary_class_id"] == 0]
    oi_waste = [r for r in nr if r["source_family"].startswith("openimages") and r["domain_tag"] == "ACTUAL_WASTE"]

    # Also add top Open Images NEEDS_REVIEW candidates with waste/discarded/broken/crumpled/old cues or clean single boxes in priority classes
    seen_ids = {r["item_id"] for r in (cm_non_bat + cm_bat + oi_waste)}
    salvage_pool = list(cm_non_bat + cm_bat + oi_waste)

    print("Selected salvage pool size:", len(salvage_pool))
    print("By class:", Counter(r["primary_class_name"] for r in salvage_pool))

    out_dir = W2_DIR / "contact_sheets_salvage_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals: dict[str, Any] = {}
    cols = 6
    tw, th = 300, 225

    for s_idx in range(0, len(salvage_pool), 33):
        chunk = salvage_pool[s_idx : s_idx + 33]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, 36 + rows * (th + 26)), (20, 22, 26))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            f"Wave 2 Salvage Inspection Sheet {s_idx // 33 + 1} ({len(chunk)} items)",
            fill=(255, 255, 255),
        )
        for i, r in enumerate(chunk):
            rr, cc = i // cols, i % cols
            x0, y0 = cc * tw, 36 + rr * (th + 26)
            im_orig = Image.open(REPO_ROOT / r["rel_path"]).convert("RGB")
            if r["source_family"] == "wikimedia_commons_wave2":
                boxes, mode = propose_tight_boxes(im_orig, r["primary_class_id"])
            else:
                boxes, mode = r.get("boxes", []), "oi_gt"
            proposals[r["item_id"]] = {"boxes": boxes, "mode": mode, "title": r.get("title", ""), "primary_class_id": r["primary_class_id"]}

            im = im_orig.copy()
            im.thumbnail((tw - 10, th - 10))
            d_im = ImageDraw.Draw(im)
            iw, ih = im.size
            for g in (0.2, 0.4, 0.6, 0.8):
                gx, gy = int(g * iw), int(g * ih)
                d_im.line([(gx, 0), (gx, ih)], fill=(90, 90, 90), width=1)
                d_im.line([(0, gy), (iw, gy)], fill=(90, 90, 90), width=1)
            for b in boxes:
                bx1, by1 = int(b["xmin"] * iw), int(b["ymin"] * ih)
                bx2, by2 = int(b["xmax"] * iw), int(b["ymax"] * ih)
                d_im.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 120), width=2)
                d_im.text((bx1 + 2, max(0, by1 - 12)), f"{b['class_id']}:{b['class_name'][:5]}", fill=(0, 255, 120))
            ox = x0 + (tw - iw) // 2
            oy = y0 + (th - ih) // 2
            sheet.paste(im, (ox, oy))
            short_id = r["item_id"].replace("task8w2_", "w2_")
            clean_t = (r.get("title") or "").replace("File:", "")[:20]
            cap = f"{short_id} c{r['primary_class_id']}:{clean_t}"
            draw.text((x0 + 4, y0 + th + 2), cap, fill=(220, 225, 235))
        sheet.save(out_dir / f"salvage_inspect_p{s_idx // 33 + 1}.jpg", quality=88)

    (out_dir / "salvage_proposals_195.json").write_text(json.dumps(proposals, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Rendered 6 salvage inspection sheets and saved proposals.")


if __name__ == "__main__":
    main()
