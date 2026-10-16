# Kế hoạch điều tra nguyên nhân mô hình nhận diện đa loại rác hoạt động kém
> Ngày: 2026-10-08 | Kỹ sư CV & QA độc lập: Antigravity | Dự án: CNTT-KLCN155

## 1. Mục tiêu và phạm vi
- Thực hiện một nhiệm vụ duy nhất: Điều tra vì sao mô hình nhận diện đa loại rác của dự án đang hoạt động kém.
- Xác định nguyên nhân bằng chứng thực tế từ mã nguồn, dữ liệu huấn luyện, checkpoint, quy trình tiền xử lý, thuật toán hậu xử lý và giao diện người dùng.
- Tái hiện chính xác hình ảnh lỗi của người dùng (`rac-1562828175-1562828212-3852-1562828313.webp`).
- Đề xuất giải pháp khắc phục có lộ trình ưu tiên, chỉ ra đánh đổi precision/recall.

## 2. Bảy giả thuyết cạnh tranh cần kiểm chứng
1. **Giả thuyết H1 (Mô hình/Checkpoint đang chạy thực tế):** Hệ thống trên UI không chạy SSDLite320 như mô hình chính đã chốt mà mặc định chuyển sang YOLOv8n hoặc WBF; checkpoint đang nạp là checkpoint nào, được train ra sao.
2. **Giả thuyết H2 (Lệch phân phối dữ liệu & Bản chất tập dữ liệu huấn luyện):** Dữ liệu dùng để huấn luyện detector thực chất chứa 99.1% (1.305/1.317 ảnh) ảnh ghép nhân tạo (Mendeley synthetic), chỉ có 12 ảnh thật; tập validation chỉ có 5 ảnh, test 5 ảnh; các bounding box nhân tạo bị gán nhãn sai/lệch kích thước.
3. **Giả thuyết H3 (Sai lệch kích thước / Aspect Ratio / Preprocessing):** Ảnh gốc kích thước 532x299 bị resize hoặc letterbox về 320x320 hoặc 640x640; kiểm tra việc chuẩn hóa, RGB/BGR, padding và việc chuyển đổi ngược tọa độ bounding box.
4. **Giả thuyết H4 (Lọc ngưỡng Confidence & NMS):** Ngưỡng confidence mặc định trên UI là 0.40 - 0.45 trong khi mô hình huấn luyện trên dữ liệu kém chỉ đạt mean confidence < 0.10 cho vật thể thật, khiến phần lớn rác bị bỏ sót.
5. **Giả thuyết H5 (Ensemble WBF làm giảm hiệu năng):** WBF kết hợp SSDLite và YOLOv8n có trọng số hoặc logic ghép box không tối ưu, khiến mAP của ensemble (2.66%) thấp hơn cả mô hình đơn lẻ (YOLO 9.11%).
6. **Giả thuyết H6 (Gán nhãn sai & hiện tượng hallucination):** YOLOv8n dự đoán "Quần áo 76%" ở vùng nền trời/ngôi nhà mờ ảo là do đâu (mẫu huấn luyện quần áo trong tập synthetic hay OpenImages/COCO)?
7. **Giả thuyết H7 (Kiến trúc mô hình nhẹ & Giới hạn dung lượng):** SSDLite320 kế thừa MobileNetV3 backbone nhưng đầu phát hiện (detection head) chưa hội tụ hoặc cấu hình anchors/priors không phù hợp cho rác đa vật thể dày đặc.

## 3. Các bước triển khai chi tiết
- [ ] Task 1: Thu thập Runtime Manifest & Xác minh môi trường thực tế (Commit, venv, packages, GPU, Checkpoint SHA-256).
- [ ] Task 2: Điều tra Data Audit & Training Logs (Nguồn gốc 1.305 ảnh synthetic, 22 ảnh thật, phân bổ nhãn, ma trận nhầm lẫn lúc huấn luyện).
- [ ] Task 3: Phân tích chi tiết kiến trúc và luồng dữ liệu (Preprocessing, Resize, Coordinate Mapping, Detection Heads).
- [ ] Task 4: Tái hiện chi tiết ảnh lỗi `rac-1562828175-1562828212-3852-1562828313.webp` qua script độc lập và Streamlit logic; ghi nhận từng bước biến đổi tensor, raw detections, sau NMS, sau WBF.
- [ ] Task 5: Viết bài kiểm thử hồi quy (TDD Regression Test) chứng minh lỗi pipeline hoặc giả định tọa độ/ngưỡng.
- [ ] Task 6: Đánh giá định lượng trên tập validation và tập audit ảnh thực tế; đo lường độ trễ (latency) và tài nguyên.
- [ ] Task 7: Phân tích giao diện UI (Streamlit), kiểm tra hiển thị box, màu sắc nhãn, bảng đếm.
- [ ] Task 8: Đóng gói báo cáo đầy đủ theo DoD vào `reports/multi_waste_detection_audit/<timestamp>/` kèm file checksums và ZIP evidence.
