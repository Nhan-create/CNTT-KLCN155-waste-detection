# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Đo đạc tọa độ điểm ảnh và cắt vùng mẫu kiểm tra cho kịch bản `measure_banana_cc`.
"""
Find exact banana peel component using connectedComponentsWithStats
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CROP_P = ROOT / "scratch_banana_crop2.jpg"

bgr = cv2.imread(str(CROP_P))
h_crop, w_crop = bgr.shape[:2]
x0, y0 = 800, 600

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower = np.array([14, 80, 80])
upper = np.array([35, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
mask[:450, :] = 0  # ignore people/background above peel

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
# Sort by area (ignore background label 0)
areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
areas.sort(key=lambda x: x[1], reverse=True)

print(f"Top connected component areas: {areas[:5]}")
for rank, (lbl, area) in enumerate(areas[:3]):
    bx = stats[lbl, cv2.CC_STAT_LEFT]
    by = stats[lbl, cv2.CC_STAT_TOP]
    bw = stats[lbl, cv2.CC_STAT_WIDTH]
    bh = stats[lbl, cv2.CC_STAT_HEIGHT]
    print(f"Rank {rank}: label={lbl}, area={area}, bbox in crop: x={bx}, y={by}, w={bw}, h={bh}")
    print(f"  Full image: x1={x0+bx}, y1={y0+by}, x2={x0+bx+bw}, y2={y0+by+bh}")
    
# Draw top component mask
top_lbl = areas[0][0]
mask_top = (labels == top_lbl).astype(np.uint8) * 255
# Find contour of top component
contours, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bx, by, bw, bh = cv2.boundingRect(contours[0])
print(f"\nFinal exact peel bbox: x={bx}, y={by}, w={bw}, h={bh}")
x1_full = x0 + bx
y1_full = y0 + by
x2_full = x1_full + bw
y2_full = y1_full + bh
print(f"Full image: [{x1_full}, {y1_full}, {x2_full}, {y2_full}]")

# Draw and save
vis = bgr.copy()
cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
cv2.imwrite(str(ROOT / "scratch_banana_clean_component.jpg"), vis)
print("Saved scratch_banana_clean_component.jpg")
