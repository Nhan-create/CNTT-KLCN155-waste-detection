# BÁO CÁO KIỂM KÊ VÀ THIẾT LẬP GROUND TRUTH NGHIỆM THU ĐỘC LẬP
**Dự án:** CNTT-KLCN155 - Multi-Waste Detection  
**Đơn vị thực hiện:** Sub-Agent D (QA Independent Evaluator)  
**Thời gian hoàn thành:** 2026-10-09  
**Mục tiêu:** Kiểm kê toàn diện tập ảnh nghiệm thu `D:\Hinh_Anh_Rac`, thiết lập Ground Truth đóng băng độc lập từ ảnh gốc và module nghiệm thu tiêu chuẩn DoD nghiêm ngặt.

---

## 1. Kết quả kiểm kê toàn diện (Inventory) tập ảnh `D:\Hinh_Anh_Rac`

- **Tổng số tệp ảnh quét được (đệ quy):** **27 file**
- **Trạng thái giải mã (Decode Status):**
  * **27/27** file giải mã thành công bằng cả **PIL** và **OpenCV** (100% VALID).
  * **0** file bị lỗi (0 Corrupt).
- **Định dạng tập tin:** JPEG (18 file), WEBP (8 file), PNG (1 file).
- **Phát hiện quan trọng:** Tệp `images (13).jpg` và `images (7).jpg` là hai bản sao giống hệt nhau bit-for-bit với mã băm SHA-256 trùng khớp:  
  `88b331d53c554a6eabcd700291dc4e0c2418798a3c45ff1ca6b2b00dc186fee7`.
- **Tệp lưu trữ:** [`manifest_hinh_anh_rac.json`](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/manifest_hinh_anh_rac.json)

---

## 2. Thiết lập Ground Truth đóng băng độc lập (Frozen GT)

- **Nguyên tắc độc lập:** Ground Truth được quan sát và gán nhãn trực tiếp từ ảnh gốc đầy đủ độ phân giải, độc lập hoàn toàn với bất kỳ mô hình dự đoán nào.
- **Thống kê Ground Truth:**
  * **Tổng số ảnh:** 27 ảnh.
  * **Tổng số vật thể (instances):** **57 vật thể**.
  * **Người duyệt (Reviewer):** Ghi rõ `Agent_D_QA`.
  * **Hệ thống phân lớp:** Tuân thủ tuyệt đối 10 lớp chuẩn theo `src/detection/schema.py`:
    `[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`.
  * **Độ chính xác tọa độ:** Bounding box pixel-level snug `[x1, y1, x2, y2]`, đã qua kiểm tra hình học và tạo ảnh overlay trực quan tại `reports/qa_acceptance/gt_visualizations/`.
- **Tệp lưu trữ:** [`ground_truth_hinh_anh_rac.json`](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/ground_truth_hinh_anh_rac.json)

---

## 3. Module tính toán chỉ số nghiệm thu Definition of Done (DoD)

- **Mã nguồn:** [`acceptance_evaluator.py`](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/acceptance_evaluator.py)
- **Thuật toán ghép cặp:** Hungarian Algorithm (Linear Sum Assignment) với ma trận chi phí dựa trên `-IoU` kết hợp phạt lỗi sai lớp (class mismatch penalty), có cơ chế dự phòng Greedy IoU matching.
- **Tiêu chuẩn kiểm tra nghiệm thu DoD:**
  1. **Coverage 100%:** Toàn bộ 27 ảnh trong manifest phải được đánh giá đầy đủ.
  2. **False Negatives (FN) = 0:** Không bỏ sót bất kỳ vật thể nào trong Ground Truth.
  3. **False Positives (FP) = 0:** Không có dự đoán rác ảo không tồn tại.
  4. **Wrong Class = 0:** Lớp dự đoán phải khớp chính xác với lớp Ground Truth.
  5. **IoU >= 0.75:** Tỉ lệ giao nhau trên hợp giữa Pred và GT đạt tối thiểu 75%.
  6. **Intersection / Area(GT) >= 0.99:** Khung dự đoán phải bao trọn tối thiểu 99% diện tích vật thể thật.
  7. **Khung hợp lệ:** `0 <= x1 < x2 <= w` và `0 <= y1 < y2 <= h`.
- **Kiểm thử tự động:** Đã bổ sung 8 unit tests tại [`tests/qa/test_acceptance_evaluator.py`](file:///d:/CNTT-KLCN155-waste-detection/tests/qa/test_acceptance_evaluator.py) (kết quả chạy thực tế: **8/8 PASSED** trong 0.54s).

---

## 4. Trạng thái sẵn sàng cho Baseline Run

Sub-Agent D đã hoàn tất 100% hạ tầng kiểm thử và Ground Truth độc lập. Sẵn sàng tiếp nhận kết quả inference từ Sub-Agent B (Model/Inference) để chạy đối soát baseline và xuất bảng báo cáo nghiệm thu chi tiết 1-1 cho từng ảnh và từng vật thể.
