# KẾ HOẠCH VÀ HỆ THỐNG CỔNG KIỂM SOÁT PHẦN 2 (TASK 2)
## ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 2 — Đánh giá Gate A & Chuẩn bị dữ liệu detection  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead ML:** Antigravity Engineering Team  
**Thời điểm phê duyệt:** 02/10/2026  
**Trạng thái kế hoạch:** **`FROZEN_AND_ACTIVE`**  

---

## 1. Mục tiêu và Phạm vi Ranh giới của Phần 2

### 1.1. Mục tiêu cốt lõi
1. **Đánh giá Cổng kiểm soát Gate A (Single-Object Gate):** Mở niêm phong tập Final Test độc lập (2.223 ảnh vật lý trong `data/processed_v2/test/`) lần đầu tiên. Đánh giá tính tổng quát hóa của checkpoint tối ưu `best_model.pt` (MobileNetV3-Large) trên dữ liệu chưa từng thấy.
2. **Chuẩn bị Dữ liệu Bounding Box Đa rác:** Kiểm kê toàn diện nguồn dữ liệu detection hiện có (`dataset-v1`), bóc tách và cô lập hoàn toàn 1.305 ảnh tổng hợp (Mendeley Synthetic), xây dựng công cụ thẩm định nội bộ ([`src/ui/review_tool.py`](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py)) thay thế CVAT, và xây dựng validator kiểm tra bounding box.

### 1.2. Giới hạn nghiêm ngặt
* **TUYỆT ĐỐI CHƯA HUẤN LUYỆN YOLOv8n hoặc SSDLite320 trong Phần 2.** Huấn luyện detector thuộc phạm vi độc quyền của **PHẦN 3**.
* Không sử dụng các ca đoán sai của tập Test để tinh chỉnh siêu tham số hay huấn luyện lại mô hình phân loại. Nếu Gate A thất bại, bảo toàn trung thực kết quả và báo cáo nguyên nhân.

---

## 2. Danh mục Nhiệm vụ Chi tiết (Work Breakdown Structure)

| Mã Task | Tên Nhiệm vụ | Nội dung Kỹ thuật | Sản phẩm Đầu ra | Trạng thái |
|:---|:---|:---|:---|:---:|
| **P2.1** | **Hoàn tất dọn dẹp Gói Phần 1** | Loại bỏ 45 tệp test cũ import `src.*`, chỉ đóng gói `tests/test_verification_gates.py`, xuất zip 49.41 MB và manifest. | `CNTT-KLCN155_PART01_VERIFIED_R2.zip` (SHA-256: `a8a75c0dca...`) | `COMPLETED` |
| **P2.2** | **Đóng băng Protocol & Đánh giá Gate A** | Đóng băng quy trình đánh giá, đo CPU latency (warmup 50, test 200 lượt forward bs=1), chạy suy luận 2.223 ảnh test, băm SHA-256 trực tiếp từ đĩa, xuất metrics, ma trận nhầm lẫn, phân tích lỗi. | `artifacts/part02/gate_a/*` (`test_predictions.csv`, `gate_a_metrics.json`, ...) | `IN_PROGRESS` |
| **P2.3** | **Kiểm kê & Chuẩn hóa Dữ liệu Bounding Box** | Kiểm kê 1.419 ảnh `dataset-v1`, cô lập 1.305 ảnh Mendeley Synthetic (`is_synthetic=True`), chuẩn hóa 114 ảnh OpenImages, tạo manifest đa rác `manifest_detection_v1.csv`. | `data/detection/manifest_detection_v1.csv` | `IN_PROGRESS` |
| **P2.4** | **Phát triển Review Tool & Validator BBox** | Viết công cụ thẩm định Streamlit [`src/ui/review_tool.py`](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) hỗ trợ thêm/sửa/xóa/duyệt box; viết script validator [`scripts/validate_detection_annotations.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/validate_detection_annotations.py) và bộ kiểm thử [`tests/test_detection_annotations.py`](file:///D:/CNTT-KLCN155-waste-detection/tests/test_detection_annotations.py). | `src/ui/review_tool.py`, `tests/test_detection_annotations.py` | `IN_PROGRESS` |
| **P2.5** | **Kiểm chứng MCP UI & Báo cáo Nghiệm thu** | Khởi chạy Review Tool, dùng MCP `chrome-devtools` tương tác thật trên giao diện web, chụp ảnh bằng chứng thao tác; lập biên bản nghiệm thu `PART_02_ACCEPTANCE.md`. | `artifacts/part02/ui_evidence/*.png`, `docs/plan/PART_02_ACCEPTANCE.md` | `PENDING` |

---

## 3. Hệ thống Cổng Kiểm soát Nghiệm thu (Acceptance Gates)

### 3.1. Cổng Gate A (Cổng Đánh giá Mô hình Đơn rác trên Final Test)

Mô hình MobileNetV3-Large (`best_model.pt`) bắt buộc phải vượt qua toàn bộ 6 tiêu chuẩn định lượng trên tập Final Test độc lập (2.223 ảnh):

| Chỉ số / Tiêu chí | Ngưỡng bắt buộc | Mục đích & Cơ sở khoa học |
|:---|:---:|:---|
| **Top-1 Accuracy** | $\ge \mathbf{88.0\%}$ | Đảm bảo khả năng phân loại vượt trội trên ảnh chưa từng thấy so với baseline ngẫu nhiên (10%). |
| **Macro-Averaged F1** | $\ge \mathbf{0.850}$ | Tránh thiên lệch lớp, đánh giá công bằng trên toàn bộ 10 lớp phân loại. |
| **Recall riêng lớp `battery`** | $\ge \mathbf{88.0\%}$ | Pin là rác thải nguy hại gây cháy nổ và độc hại môi trường, tuyệt đối không được bỏ sót. |
| **Recall tối thiểu từng lớp** | $\ge \mathbf{80.0\%}$ | Mọi lớp rác sinh hoạt đều phải đạt ngưỡng nhận diện chấp nhận được. |
| **CPU Forward Latency (BS=1)** | $< \mathbf{20.0\text{ ms/ảnh}}$ | Đảm bảo mô hình chạy thời gian thực trên CPU thiết bị biên ($>50\text{ FPS}$). |
| **Kiểm toán Rò rỉ Dữ liệu** | $\mathbf{Zero\ Leakage}$ | Không trùng lặp SHA-256 hay rò rỉ burst-shot giữa test và train/val. |

### 3.2. Cổng Gate Bbox (Cổng Chuẩn bị Dữ liệu Bounding Box Đa rác)

Tập dữ liệu và công cụ gán nhãn đa rác bắt buộc phải đáp ứng 5 tiêu chí:
1. **Cô lập ảnh tổng hợp:** 100% ảnh Mendeley Synthetic phải được gắn nhãn `is_synthetic = True`, chỉ dùng cho pretraining, tuyệt đối không được lọt vào tập Validation/Test của phát hiện đa rác.
2. **Định dạng nhãn chuẩn YOLO:** Toàn bộ bounding box phải tuân thủ chuẩn `<class_id> <x_center> <y_center> <w> <h>` với tọa độ chuẩn hóa trong khoảng $(0, 1]$, $x_{\min} \ge 0, y_{\min} \ge 0, x_{\max} \le 1, y_{\max} \le 1, w > 0, h > 0$.
3. **Chặn lỗi BBox:** Script validator tự động phải phát hiện và chặn đứng (exit code 1) các trường hợp tọa độ âm, vượt ngoài $[0, 1]$, class ID ngoài $[0, 9]$, và box trùng lặp.
4. **Công cụ Review Tool hoạt động thật:** Công cụ [`src/ui/review_tool.py`](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) chạy được trên giao diện web, có bằng chứng ảnh chụp MCP ghi nhận thao tác thêm/sửa/xóa/lưu/mở lại bounding box.
5. **Thống kê trung thực:** Báo cáo chi tiết số lượng ảnh thật và ảnh synthetic hiện có; không sinh ảnh giả hay nhãn giả tạo số lượng ảo.
