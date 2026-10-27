# Mô tả: Tiện ích chẩn đoán các trường hợp dự đoán thuộc lớp giày dép (shoes) trên tập kiểm định QA.
import cv2
import numpy as np

im = cv2.imread("data/holdout/images (15).jpg")
h, w = im.shape[:2]

# Let's define the 4 pairs / instances:
# 1. Yellow/beige perforated slippers at top:
#    approx y: 15 to 70, x: 110 to 200
# 2. Black & white sneakers (Vans):
#    approx y: 90 to 160, x: 95 to 195
# 3. Beige sandals (right of sneakers):
#    approx y: 80 to 145, x: 160 to 230
# 4. Grey slippers (bottom left):
#    approx y: 110 to 190, x: 15 to 115

# Let's save crops to inspect them
cv2.imwrite("reports/qa_acceptance/debug_shoe1.jpg", im[15:70, 110:200])
cv2.imwrite("reports/qa_acceptance/debug_shoe2.jpg", im[90:165, 95:195])
cv2.imwrite("reports/qa_acceptance/debug_shoe3.jpg", im[80:150, 160:230])
cv2.imwrite("reports/qa_acceptance/debug_shoe4.jpg", im[110:192, 15:115])
print("Saved debug shoe crops")
