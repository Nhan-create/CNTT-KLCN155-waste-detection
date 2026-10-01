# 09 — EXPERIMENTS AND ERROR ANALYSIS: THỰC NGHIỆM ĐỐI CHỨNG VÀ PHÂN TÍCH SAI SỐ (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã cập nhật quy tắc WBF có điều kiện)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Thực hiện đầy đủ các cam kết học thuật trong đề tài:
1. **Nghiên cứu đối chứng (Comparative Study):** Đặt mô hình chính `YOLOv8n` đối chứng trực tiếp với mô hình siêu nhẹ `SSDLite320-MobileNetV3` trên chuẩn đánh giá MS COCO (mAP@0.5, mAP@0.5:0.95, AP từng lớp, Precision, Recall, F1).
2. **Nghiên cứu triệt tiêu (Ablation Study 4 nhánh):** Khảo sát độc lập ảnh hưởng của 4 chiến lược tăng cường dữ liệu: (1) Baseline, (2) Hình học, (3) Ánh sáng/màu sắc, (4) Kết hợp + Copy-Paste.
3. **Tối ưu hóa thuật toán hợp nhất Weighted Boxes Fusion (WBF):** Tìm kiếm lưới siêu tham số (Grid Search) hoàn toàn trên tập **Validation** để xác định bộ trọng số tối ưu $(w_{\text{YOLO}}, w_{\text{SSD}})$.
4. **Phân tích sai số định lượng (Comprehensive Error Analysis):** Bóc tách chi tiết các loại lỗi: bỏ sót, nhận nhầm, trùng lặp box, sai số đếm số lượng, và kiểm tra độ bền vững trong các điều kiện khắc nghiệt (Stress-testing).

### 1.2. Vấn đề cần giải quyết
1. **Nghịch lý "Vận động viên cõng người đi bộ":** Nếu YOLOv8n có độ chính xác vượt trội so với SSDLite320, việc kết hợp WBF cào bằng tỷ lệ 50-50 sẽ kéo tụt chất lượng của hệ thống. WBF phải được chứng minh có lợi ích thực sự trên tập Validation mới được đưa vào luồng chính.
2. **Tránh rò rỉ dữ liệu khi chọn tham số:** Tuyệt đối không tinh chỉnh trọng số WBF hoặc ngưỡng confidence trên tập Test.
3. **Đo độ trễ trung thực:** Đo thời gian suy luận toàn luồng (End-to-End Latency) sau khi warm-up, có đồng bộ GPU (`torch.cuda.synchronize()`).

---

## 2. Thiết kế Grid Search WBF trên Validation Set

Thực thi script [src/detection/tune.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/tune.py) quét trên tập Validation:
- **Trọng số kết hợp $(w_{\text{YOLO}}, w_{\text{SSD}})$:** Quét qua các cặp $[1.0, 0.0], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]$.
- **Ngưỡng IoU (`iou_thr`):** $[0.45, 0.50, 0.55, 0.60]$.
- **Ngưỡng lọc hộp bao (`skip_box_thr`):** $[0.20, 0.25, 0.30]$.
- **Hàm mục tiêu:** $\arg\max \text{mAP@0.5}_{\text{Validation}}$.

### Điều kiện áp dụng WBF vào hệ thống chính thức
> **QUY TẮC CÓ ĐIỀU KIỆN (CONDITIONAL GATE):**  
> WBF chỉ được cấu hình làm chế độ mặc định khi:
> 1. $\text{mAP@0.5}_{\text{WBF}} \ge \max\left(\text{mAP@0.5}_{\text{YOLO}}, \text{mAP@0.5}_{\text{SSD}}\right) + 0.015$ (tăng tối thiểu 1.5% mAP50).
> 2. Độ trễ suy luận toàn luồng $\le 60\text{ ms/ảnh}$ trên GPU RTX 2050.  
> Nếu không đạt cả 2 điều kiện, **YOLOv8n sẽ là mô hình mặc định duy nhất**, và WBF được giữ lại như một tùy chọn đối chứng học thuật trong giao diện.

---

## 3. Tiêu chí nghiệm thu đo được

1. **Hiệu năng COCO mAP trên tập Test thực tế:**
   - YOLOv8n: $\text{mAP@0.5} \ge 0.700$, $\text{mAP@0.5:0.95} \ge 0.450$.
   - SSDLite320: $\text{mAP@0.5} \ge 0.550$, $\text{mAP@0.5:0.95} \ge 0.350$.
2. **Sai số đếm số lượng vật thể:**
   - $\text{MAE}_{\text{count}} \le 0.35$ vật thể/ảnh trên toàn bộ tập Test.
3. **Phân tích ca lỗi:** Báo cáo chi tiết danh mục 50 ca lỗi điển hình kèm ảnh minh họa và nguyên nhân vật lý.
