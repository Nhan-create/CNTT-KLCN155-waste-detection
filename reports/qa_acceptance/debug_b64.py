# Mô tả: Tiện ích giải mã và kiểm tra nhanh chuỗi ảnh mã hóa Base64 phục vụ gỡ lỗi giao diện QA.
import cv2

im = cv2.imread("data/holdout/base64-1717231151649602366964.jpeg")
h, w = im.shape[:2]

# In 2000x1334:
# Bottle 1 (Saigon beer, lying down on the left):
# approx x: 175 to 825, y: 650 to 1150
# Bottle 2 (Small jar near bottle 1):
# approx x: 310 to 570, y: 610 to 820
# Bottle 3 (Saigon beer lying down on right):
# approx x: 1350 to 1850, y: 350 to 710
# Bottle 4 (Clear bottle with red neck at bottom):
# approx x: 1110 to 1420, y: 1010 to 1334
# Face mask (white):
# approx x: 1190 to 1585, y: 810 to 1120

cv2.imwrite("reports/qa_acceptance/debug_b64_b1.jpg", im[650:1150, 175:825])
cv2.imwrite("reports/qa_acceptance/debug_b64_b2.jpg", im[610:820, 310:570])
cv2.imwrite("reports/qa_acceptance/debug_b64_b3.jpg", im[350:710, 1350:1850])
cv2.imwrite("reports/qa_acceptance/debug_b64_b4.jpg", im[1010:1334, 1110:1420])
cv2.imwrite("reports/qa_acceptance/debug_b64_mask.jpg", im[810:1120, 1190:1585])
print("Saved base64 crops")
