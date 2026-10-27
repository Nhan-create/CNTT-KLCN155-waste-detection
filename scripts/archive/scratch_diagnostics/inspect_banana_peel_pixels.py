# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Kiểm tra chi tiết dữ liệu, khung bao và thuộc tính của `banana peel pixels`.
"""
scripts/inspect_banana_peel_pixels.py
Find exact bounding box of the banana peel in wiki_bio_banana_peel_ground.jpg
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg"

im = Image.open(IMG_P)
w, h = im.size
print(f"Image size: {w}x{h}")

# The banana peel is in the center-lower area of the asphalt path.
# Let's inspect coordinates in [ymin, xmin, ymax, xmax]
# Let's crop a window around the center:
# x: from ~0.15 to ~0.95
# y: from ~0.55 to ~0.90
# We can save crops or inspect brightness/color
crop_box = (int(0.15 * w), int(0.55 * h), int(0.95 * w), int(0.90 * h))
crop = im.crop(crop_box)
crop.save(ROOT / "scratch_banana_crop.jpg", quality=90)
print(f"Saved scratch_banana_crop.jpg with crop: {crop_box}")
