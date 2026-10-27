# Mô tả: Phân tích chi tiết kết quả dự đoán trên từng ảnh đơn lẻ trong tập kiểm định QA.
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_isolated_image(img_path, threshold=245):
    # For images on clean/white background, find exact bounding box
    img = Image.open(img_path)
    arr = np.array(img)
    
    if arr.shape[-1] == 4:
        # Check alpha channel
        alpha = arr[:, :, 3]
        mask = alpha > 10
    else:
        # Check non-white pixels
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask = gray < threshold

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return [int(x0), int(y0), int(x1), int(y1)]

# Let's test on battery-sample.png, pin_vuong, images (4), images (5), images (12)
isolated_candidates = [
    "data/holdout/battery-sample.png",
    "data/holdout/pin_vuong_4f15ab32849e43a8abdaef2bf000463f.webp",
    "data/holdout/images (4).jpg",
    "data/holdout/images (5).jpg",
    "data/holdout/images (8).jpg",
    "data/holdout/images (12).jpg",
    "data/holdout/images (18).jpg",
]

for p in isolated_candidates:
    bbox = analyze_isolated_image(p)
    im = Image.open(p)
    print(f"{Path(p).name} ({im.width}x{im.height}): {bbox}")
