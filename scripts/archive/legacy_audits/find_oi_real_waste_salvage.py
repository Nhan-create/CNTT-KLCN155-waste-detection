"""
Render an inspection sheet of 36 Open Images NEEDS_REVIEW candidates with 1..3 boxes
across biological, cardboard, clothes, glass, metal, paper, and shoes for Task 8C Task 4 salvage.
"""

from __future__ import annotations

import json
import pathlib
from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
W2_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch"


def main() -> None:
    m = json.loads((W2_DIR / "task8_wave2_qa_manifest.json").read_text(encoding="utf-8"))
    nr = [
        r
        for r in m["records"]
        if r["qa_status"] == "NEEDS_REVIEW" and r["source_family"].startswith("openimages")
    ]

    # Prioritize visual_contact_sheet_audit items and waste-keyword items with 1..3 boxes
    waste_kws = (
        "trash", "waste", "litter", "garbage", "recycle", "discard", "broken", "crush",
        "crumple", "rotten", "peel", "mold", "worn", "old", "dirty", "used", "empty",
        "street", "ground", "beach", "bin", "carton", "cardboard",
    )

    scored = []
    for r in nr:
        t = (r.get("title") or "").lower()
        n_b = len(r.get("boxes", []))
        if not (1 <= n_b <= 3):
            continue
        score = 0
        if any(k in t for k in waste_kws):
            score += 3
        if "visual_contact_sheet" in r.get("qa_reason", ""):
            score += 2
        if r["primary_class_id"] in (1, 2, 4, 6):
            score += 1
        if score >= 2:
            scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], x[1]["item_id"]))
    selected = [x[1] for x in scored[:36]]

    cols, tw, th = 6, 300, 225
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, 36 + rows * (th + 26)), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 10),
        f"Wave 2 Salvage Inspection Sheet 8 - Open Images Candidates ({len(selected)} items)",
        fill=(255, 255, 255),
    )
    for i, r in enumerate(selected):
        rr, cc = i // cols, i % cols
        x0, y0 = cc * tw, 36 + rr * (th + 26)
        im = Image.open(REPO_ROOT / r["rel_path"]).convert("RGB")
        im.thumbnail((tw - 10, th - 10))
        d_im = ImageDraw.Draw(im)
        iw, ih = im.size
        for g in (0.2, 0.4, 0.6, 0.8):
            gx, gy = int(g * iw), int(g * ih)
            d_im.line([(gx, 0), (gx, ih)], fill=(90, 90, 90), width=1)
            d_im.line([(0, gy), (iw, gy)], fill=(90, 90, 90), width=1)
        for b in r.get("boxes", []):
            bx1, by1 = int(b["xmin"] * iw), int(b["ymin"] * ih)
            bx2, by2 = int(b["xmax"] * iw), int(b["ymax"] * ih)
            d_im.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 120), width=2)
            d_im.text((bx1 + 2, max(0, by1 - 12)), f"{b['class_id']}:{b['class_name'][:5]}", fill=(0, 255, 120))
        sheet.paste(im, (x0 + (tw - iw) // 2, y0 + (th - ih) // 2))
        short_id = r["item_id"].replace("task8w2_", "w2_")
        clean_t = (r.get("title") or "")[:20]
        draw.text((x0 + 4, y0 + th + 2), f"{short_id} c{r['primary_class_id']}:{clean_t}", fill=(220, 225, 235))

    out_p = W2_DIR / "contact_sheets_salvage_inspect" / "salvage_inspect_p8.jpg"
    sheet.save(out_p, quality=88)
    print("Saved p8 with", len(selected), "items:", [r["item_id"] for r in selected])


if __name__ == "__main__":
    main()
