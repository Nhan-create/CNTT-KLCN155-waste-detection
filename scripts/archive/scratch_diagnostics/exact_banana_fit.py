"""
Find exact x_min and x_max of the banana peel
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CROP_P = ROOT / "scratch_banana_crop2.jpg"

bgr = cv2.imread(str(CROP_P))
# Crop vertical slice around peel: y from 700 to 1550
peel_strip = bgr[700:1550, :]

# In this strip, banana peel has high BGR values with R > 150, G > 120, B < 100
b = peel_strip[:, :, 0].astype(float)
g = peel_strip[:, :, 1].astype(float)
r = peel_strip[:, :, 2].astype(float)

# Yellow banana peel detector: R is high, G is medium-high, B is distinctly lower (yellow hue)
# and saturation is much higher than grey asphalt
yellow_score = (r + g) / 2.0 - b
is_peel = (yellow_score > 60) & (r > 130) & (g > 100)

# Column-wise check: which columns have at least 15 peel pixels?
col_counts = np.sum(is_peel, axis=0)
peel_cols = np.where(col_counts > 10)[0]

# Row-wise check: which rows have at least 10 peel pixels?
row_counts = np.sum(is_peel, axis=1)
peel_rows = np.where(row_counts > 10)[0]

x0, y0 = 800, 600
cx1, cx2 = peel_cols[0], peel_cols[-1]
cy1, cy2 = peel_rows[0] + 700, peel_rows[-1] + 700

print(f"Crop-relative peel bbox: x1={cx1}, y1={cy1}, x2={cx2}, y2={cy2}")
gx1, gy1, gx2, gy2 = x0 + cx1, y0 + cy1, x0 + cx2, y0 + cy2
gw = gx2 - gx1
gh = gy2 - gy1
print(f"Full-image peel bbox: x1={gx1}, y1={gy1}, x2={gx2}, y2={gy2} (w={gw}, h={gh})")

img_w, img_h = 5184, 3456
xc = (gx1 + gx2) / (2.0 * img_w)
yc = (gy1 + gy2) / (2.0 * img_h)
wn = gw / img_w
hn = gh / img_h
print(f"Normalized YOLO: 1 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

# Draw exact rectangle and save
vis = bgr.copy()
cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 4)
cv2.putText(vis, f"Banana Peel: [{gx1}, {gy1}, {gx2}, {gy2}]", (cx1, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.imwrite(str(ROOT / "scratch_banana_exact_fitted.jpg"), vis)
print("Saved scratch_banana_exact_fitted.jpg")
