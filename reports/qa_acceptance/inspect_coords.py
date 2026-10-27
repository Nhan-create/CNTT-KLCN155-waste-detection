# Mô tả: Kiểm tra và đối chiếu tọa độ khung bao giữa nhãn chuẩn và dự đoán của mô hình trên tập QA.
import cv2
import numpy as np
from PIL import Image

img = Image.open("data/holdout/battery-sample.png")
arr = np.array(img)
# Let's crop out the text header and find the battery
# text header is near the top (y < 40)
# Let's find rows with dark pixels below y=40
gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
mask = (gray < 240) & (np.arange(arr.shape[0])[:, None] > 40)
coords = np.argwhere(mask)
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0) + 1
print("battery-sample:", [int(x0), int(y0), int(x1), int(y1)])

# images (4).jpg
img4 = Image.open("data/holdout/images (4).jpg")
arr4 = np.array(img4)
gray4 = cv2.cvtColor(arr4, cv2.COLOR_RGB2GRAY)
mask4 = gray4 < 245
coords4 = np.argwhere(mask4)
print("images (4).jpg coords:", coords4.min(axis=0), coords4.max(axis=0))

# images (5).jpg
img5 = Image.open("data/holdout/images (5).jpg")
arr5 = np.array(img5)
gray5 = cv2.cvtColor(arr5, cv2.COLOR_RGB2GRAY)
mask5 = gray5 < 250
coords5 = np.argwhere(mask5)
print("images (5).jpg coords:", coords5.min(axis=0), coords5.max(axis=0))
