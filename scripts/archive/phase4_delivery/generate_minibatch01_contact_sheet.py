"""
Generate contact sheet and inspect Mini-Batch 01 images.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

def main():
    root = Path(__file__).resolve().parents[3]
    img_dir = root / "data/acquisition_staging/batch_a_mini01/images"
    out_dir = root / "reports/source_verification/contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(root / "reports/source_verification/minibatch01_download_log.json", encoding="utf-8") as f:
        log_data = json.load(f)
        
    results = log_data["results"]
    print(f"Creating contact sheet for {len(results)} images...")
    
    # 2 rows x 5 cols
    cell_w, cell_h = 400, 400
    rows, cols = 2, 5
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    
    for idx, r in enumerate(results):
        fn = r["filename"]
        p = img_dir / fn
        if not p.exists():
            continue
        with Image.open(p) as img:
            # Thumbnail preserving aspect ratio
            img_thumb = img.copy()
            img_thumb.thumbnail((cell_w - 20, cell_h - 60))
            
            c_x = (idx % cols) * cell_w
            c_y = (idx // cols) * cell_h
            
            # Paste centered
            offset_x = c_x + (cell_w - img_thumb.width) // 2
            offset_y = c_y + 40 + (cell_h - 60 - img_thumb.height) // 2
            sheet.paste(img_thumb, (offset_x, offset_y))
            
            # Header text
            title = f"{idx+1}. {r['candidate_id'][:25]}"
            meta = f"[{r['category']}] {r['resolution'][0]}x{r['resolution'][1]}"
            draw.text((c_x + 10, c_y + 5), title, fill=(0, 0, 0))
            draw.text((c_x + 10, c_y + 20), meta, fill=(60, 60, 60))
            
            # Border
            draw.rectangle([c_x, c_y, c_x + cell_w - 1, c_y + cell_h - 1], outline=(200, 200, 200))
            
    out_path = out_dir / "minibatch01_contact_sheet.jpg"
    sheet.save(out_path, quality=90)
    print(f"Saved contact sheet to: {out_path}")

if __name__ == "__main__":
    main()
