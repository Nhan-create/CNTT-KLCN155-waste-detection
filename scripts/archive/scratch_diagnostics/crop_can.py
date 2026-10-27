# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Đo đạc tọa độ điểm ảnh và cắt vùng mẫu kiểm tra cho kịch bản `crop_can`.
"""
Zoom in on bottom right of utility cabinet in wiki_neg_utility_box_leeds.jpg
"""
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_neg_utility_box_leeds.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]
print(f"Image size: {w}x{h}")

# The painted box is centered around x ~ 0.35 to 0.70, y ~ 0.35 to 0.85
# Let's crop the bottom right of the box
crop = bgr[int(0.65*h):int(0.85*h), int(0.55*w):int(0.75*w)]
cv2.imwrite(str(ROOT / "scratch_utility_box_bottom_right.jpg"), crop)
print("Saved scratch_utility_box_bottom_right.jpg")
