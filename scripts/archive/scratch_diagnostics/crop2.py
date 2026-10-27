# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Đo đạc tọa độ điểm ảnh và cắt vùng mẫu kiểm tra cho kịch bản `crop2`.
"""
Find vertical and horizontal extent of banana peel
"""
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg"

im = Image.open(IMG_P)
# crop around banana peel
crop_box = (800, 600, 5000, 2200)
crop = im.crop(crop_box)
crop.save(ROOT / "scratch_banana_crop2.jpg", quality=90)
print(f"Saved scratch_banana_crop2.jpg with {crop_box}")
