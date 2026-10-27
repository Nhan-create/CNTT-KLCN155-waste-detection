# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Đo đạc tọa độ điểm ảnh và cắt vùng mẫu kiểm tra cho kịch bản `test_adj_banana`.
"""
Verify adjusted banana peel box
"""
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]

# Adjusted box
x1, y1, x2, y2 = 2150, 1300, 5010, 2110
bw, bh = x2 - x1, y2 - y1

xc = (x1 + x2) / (2.0 * w)
yc = (y1 + y2) / (2.0 * h)
wn = bw / w
hn = bh / h

print(f"Exact physical box: [{x1}, {y1}, {x2}, {y2}], size: {bw}x{bh}")
print(f"Normalized YOLO: 1 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

# Crop visual overlay
vis = bgr.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 6)
cv2.putText(vis, f"1: biological (banana_peel)", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)

crop_vis = vis[1100:2300, 1900:5150]
cv2.imwrite(str(ROOT / "scratch_banana_perfect_box.jpg"), crop_vis)
print("Saved scratch_banana_perfect_box.jpg")
