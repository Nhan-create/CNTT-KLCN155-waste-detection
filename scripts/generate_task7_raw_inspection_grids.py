"""
scripts/generate_task7_raw_inspection_grids.py
----------------------------------------------
Renders 2 high-clarity inspection contact sheets of all 42 downloaded Task 7 images
with normalized 0.1..0.9 coordinate grid overlays so we can visually inspect every
image and determine exact per-object full-battery bounding boxes and QA verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
RAW_MANIFEST_P = OUT_DIR / "raw_download_manifest.json"
GRIDS_DIR = OUT_DIR / "inspection_grids"


def render_grid_sheet(items: list[dict], out_path: Path, cols: int = 3) -> None:
    cell_w, cell_h = 560, 460
    header_h = 44
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + header_h)), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * (cell_h + header_h)

        # Header
        short_id = item["candidate_id"][:42]
        draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + header_h], fill=(40, 44, 52))
        draw.text((x0 + 8, y0 + 6), f"{short_id}", fill=(255, 230, 120))
        draw.text(
            (x0 + 8, y0 + 24),
            f"{item['actual_width']}x{item['actual_height']} | {item['subgroup']} | {item['license']}",
            fill=(200, 210, 220),
        )

        img_path = REPO_ROOT / item["local_path"]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB").resize((cell_w - 40, cell_h - 40), Image.Resampling.BILINEAR)

        ix0 = x0 + 28
        iy0 = y0 + header_h + 20
        iw, ih = cell_w - 40, cell_h - 40
        canvas.paste(rgb, (ix0, iy0))

        # Draw 0.1 normalized grid lines
        for step in range(1, 10):
            frac = step / 10.0
            gx = int(ix0 + frac * iw)
            gy = int(iy0 + frac * ih)
            color = (255, 80, 80) if step == 5 else (120, 220, 255)
            # Dashed/light grid
            for py in range(iy0, iy0 + ih, 6):
                draw.line([(gx, py), (gx, min(py + 3, iy0 + ih))], fill=color, width=1)
            for px in range(ix0, ix0 + iw, 6):
                draw.line([(px, gy), (min(px + 3, ix0 + iw), gy)], fill=color, width=1)
            draw.text((gx - 8, iy0 - 14), f".{step}", fill=color)
            draw.text((ix0 - 22, gy - 6), f".{step}", fill=color)

        draw.rectangle([ix0, iy0, ix0 + iw, iy0 + ih], outline=(200, 200, 200), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    candidates = data["candidates"]
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    s1 = GRIDS_DIR / "raw_grid_sheet_01_to_21.jpg"
    s2 = GRIDS_DIR / "raw_grid_sheet_22_to_42.jpg"
    render_grid_sheet(candidates[:21], s1, cols=3)
    render_grid_sheet(candidates[21:], s2, cols=3)
    print(f"Generated {s1} and {s2}")


if __name__ == "__main__":
    main()
