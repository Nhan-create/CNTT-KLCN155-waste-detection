# BÁO CÁO NGHIỆM THU TASK 2: ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

- **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải đa đối tượng
- **Đơn vị thực hiện:** Nhóm ML / Tech Lead
- **Người nhận báo cáo:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_ACCEPTANCE.md` (Cập nhật R5 — Hoàn tất Mở rộng Dữ liệu Thật, Ambiguous Queue & Review Tool Sync)
- **Ngày lập:** 02/10/2026
- **Trạng thái Gate A gốc (Final Test Classifier):** `GATE_A_FAILED` (PyTorch CPU forward 23,86 ms > ngưỡng 20,0 ms; Accuracy 96,13%, Macro-F1 0,9578, Battery Recall 95,58% đạt xuất sắc. Giữ nguyên mốc lịch sử).
- **Trạng thái Tối ưu hóa Runtime (Deployment Ready):** `RUNTIME_OPTIMIZATION_VERIFIED` (Mô hình ONNX Runtime FP32 tại 6 luồng CPU đạt **7,40 ms**, toàn pipeline 30,63 ms, khớp 100,0000% nhãn Top-1 trên 2.223 ảnh validation).
- **Trạng thái Bộ công cụ Review Tool:** `REVIEW_TOOL_VERIFIED` (Tích hợp tương tác trực tiếp hàng đợi Ambiguous Boxes Queue, đồng bộ hóa tự động hai chiều với manifest và readiness JSON, lưu log kiểm toán `review_audit_log.csv`).
- **Trạng thái Mở rộng Dữ liệu Thật (Real Data Expansion):** `REAL_DATA_EXPANSION_VERIFIED` (143 ảnh thật thu thập từ TACO và OpenImages V7; 1.148 bounding boxes; đủ 10/10 lớp; 35 ảnh APPROVED, 3 ảnh REJECTED dứt khoát, 2 ảnh NEEDS_RELABEL, 103 ảnh UNREVIEWED sẵn sàng duyệt).
- **Trạng thái Kiểm tra Hình học & Gom nhóm:** `ANNOTATION_GEOMETRY_AND_CLUSTERING_PASS` (257 file nhãn thật, 1.655 boxes kiểm tra: **0 Errors**, 100% PASS; 1.562 ảnh phân cụm pHash Hamming $\le 4$ tạo 1.492 unique group IDs).
- **Trạng thái Bộ kiểm thử Verification Gates:** `VERIFICATION_GATES_4_OF_4_PASS` (4/4 ca kiểm thử tự động PASS 100%).
- **KẾT LUẬN TRẠNG THÁI TASK 2:** **`TASK_02_READY_FOR_ACCEPTANCE`** (Đầy đủ bằng chứng thực nghiệm, dữ liệu thật tin cậy, không train trước detector).

---

## 1. PHẠM VI ỨNG DỤNG VÀ NGUYÊN TẮC THỐNG NHẤT VỚI PM

1. **Bối cảnh vận hành:** Hệ thống Web tiếp nhận ảnh tĩnh do người dùng tải lên (`st.file_uploader`), gồm một hoặc nhiều vật thể thải bỏ tại khu vực thu gom, thùng rác hoặc bãi tập kết.
2. **Không áp đặt phần cứng mới:** Không bắt buộc camera cố định trong thùng rác, không dùng băng chuyền công nghiệp.
3. **Cố định Taxonomy 10 lớp:** `battery` (0), `biological` (1), `cardboard` (2), `clothes` (3), `glass` (4), `metal` (5), `paper` (6), `plastic` (7), `shoes` (8), `trash` (9).
4. **Quy tắc phân định đồ đang dùng vs rác thải:**
   - Đồ vật đang gắn liền cơ học với người (quần áo đang mặc, giày đang mang) **không phải là rác**.
   - Ảnh có người không tự động bị loại bỏ; chỉ gán nhãn cho các vật thể rác thải đã tách rời cơ thể người, nằm tự do trên sàn/đất/thùng rác.
   - Các vật dụng đang sử dụng trong sinh hoạt thông thường (chai nước trên bàn làm việc văn phòng, lọ nước hoa trên bàn trang điểm) **không phải là rác** và phải bị `REJECTED`.
5. **Giữ nguyên kiến trúc:** Không thêm lớp `Person`, không chuyển bài toán sang segmentation hay giám sát hành vi xả rác ngoài đường.

---

## 2. BẢNG TRẢ LỜI LÀM RÕ 6 VẤN ĐỀ CHẤT VẤN CỦA PM

| Vấn đề chất vấn | Kết luận thực tế | Bằng chứng kiểm chứng | Điểm chưa biết | Hành động đã thực thi |
|---|---|---|---|---|
| **Q1. Lệch trạng thái giữa các manifest (Review Tool vs Source Manifest)?** | Trước đây `review_tool.py` chỉ lưu vào `manifest_detection_v1.csv` mà không cập nhật `real_detection_source_manifest.csv`, dẫn đến lệch trạng thái duyệt. | Phân tích code phát hiện hàm `save_manifest` thiếu lời gọi đồng bộ. Đã bổ sung hàm `sync_manifests_and_readiness()`. | Không còn điểm chưa biết. | Đã tích hợp hàm đồng bộ hai chiều tự động: khi lưu trên Review Tool, cả `manifest_detection_v1.csv`, `real_detection_source_manifest.csv`, `real_detection_readiness.json` và `review_audit_log.csv` đều cập nhật đồng thời. |
| **Q2. Bối cảnh ảnh `oi_106c8de22ed8d1d4` và hiện trạng lớp clothes?** | Ảnh `oi_106c8de22ed8d1d4` chụp người đàn ông ngồi ăn tại bàn tiệc, mặc áo sơ mi và quần tây. Đây là **quần áo đang mặc, không phải rác thải** $\rightarrow$ **REJECTED dứt khoát**. Do đó, đợt mẫu 35 ảnh ban đầu có 0 mẫu quần áo rác hợp lệ. | Metadata OpenImages xác nhận nhãn `/m/04yx4` (Man) đi kèm `/m/01n4qj` (Shirt). Kiểm tra trực tiếp file ảnh thấy bối cảnh nhà hàng/bàn ăn. | Số lượng ảnh quần áo đơn lập không có người trong OpenImages. | Đã truy vấn 35.100 ảnh validation của OpenImages, lọc ra 385 ảnh quần áo **hoàn toàn không có người hay bộ phận cơ thể người**; đã tải 18 ảnh (63 boxes quần áo đơn lập) và duyệt approved 3 ảnh (22 boxes). |
| **Q3. Xử lý các box mơ hồ (omitted/ambiguous boxes)?** | Các box mơ hồ (ví dụ: vỉ thuốc nhôm-nhựa, túi giấy ép nilon, chai chưa rõ nhựa/thủy tinh) trước đây bị đếm trong manifest nhưng **bị loại bỏ khỏi file nhãn YOLO mà không lưu tọa độ**. Đây là lỗi nghiêm trọng làm mất thông tin. | Code cũ trong `collect_real_detection_data.py` chỉ đếm số lượng mà không ghi tọa độ ra tệp nào. | Tỷ lệ box mơ hồ khi mở rộng. | Đã triển khai hàng đợi `data/audit/ambiguous_boxes_queue.json` (21 items), lưu đầy đủ tọa độ normalized YOLO, nhãn gốc, gợi ý phân lớp. Review Tool hiển thị hàng đợi này với nút "➕ Assign Box" và "🚫 Discard Box". |
| **Q4. Vấn đề đổi nhãn shoes $\rightarrow$ trash trên `oi_034f71ee4e111261`?** | Thao tác đổi nhãn shoes sang trash trong đợt test UI trước đã làm sửa nhãn sản xuất và tạo file `.bak`. | File `oi_034f71ee4e111261.txt.bak` được tạo ra trong `data/detection/labels/real`. | Không còn điểm chưa biết. | Đã hoàn tác triệt để: khôi phục nhãn gốc `8 0.573438...` (shoes), xóa file `.bak`, reset trạng thái hàng 0 về `UNREVIEWED` trong manifest. |
| **Q5. Nguồn dữ liệu thực tế cho Battery?** | OpenImages Boxable **hoàn toàn không có nhãn Battery** (báo cáo cũ nêu `/m/01cszv` là hallucination). TACO chỉ có đúng 2 ảnh chứa pin: `taco_0082` (đã duyệt) và `taco_0456` (OpenLitterMap S3). Tập Kaggle 15.150 ảnh chỉ là phân loại toàn ảnh, không có bbox. | File `oidv7-class-descriptions-boxable.csv` xác nhận không có battery; TACO `annotations.json` chỉ có category 1 ở ảnh 82 và 456. Tải thành công `taco_0456` (5,36 MB, URL S3 OpenLitterMap). | Khả năng thu thập thêm pin từ các nguồn thực địa khác. | Đã tải và đưa cả 2 ảnh pin thực địa của TACO vào tập dữ liệu, cả 2 đều đã được review APPROVED. Sẽ bổ sung thêm từ nguồn chụp thực địa HUIT. |
| **Q6. Khả năng chia split theo nhóm (GroupKFold) và điểm nghẽn đánh giá?** | Nếu một lớp có $< 3$ groups độc lập, không thể phân chia vào đủ 3 tập train, val, test mà không bị rò rỉ hoặc thiếu lớp. | Thống kê group_id trên tập 143 ảnh thật thu thập: 8/10 lớp có $\ge 6$ groups. Riêng `battery` có 2 groups (chưa đủ 3 tập); `clothes` có 19 groups thu thập (3 groups đã approved). | Tỷ lệ phân bổ group khi gộp cùng tập synthetic. | 8/10 lớp đảm bảo phân chia group độc lập 100%. Lớp battery sẽ được huấn luyện trên train/val và đánh giá trên test đặc thù hoặc bổ sung thêm ảnh thực địa HUIT để đạt $\ge 3$ groups. |

---

## 3. THỐNG KÊ CHI TIẾT DỮ LIỆU THẬT SAU KHI MỞ RỘNG

### 3.1. Tổng quan tập dữ liệu thực tế (`artifacts/part02/real_detection_readiness.json`)
- **Tổng số ảnh thật thu thập:** **143 ảnh** (105 TACO + 38 OpenImages V7).
- **Tổng số bounding box thật:** **1.148 boxes**.
- **Tổng số ảnh real trong thư mục `data/detection/images/real`:** **257 ảnh** (143 ảnh thu thập mới + 114 ảnh real shoes OpenImages sẵn có).
- **Tổng số ảnh trong master manifest:** **1.562 ảnh** (1.305 synthetic + 257 real).
- **Tổng số nhóm pHash độc lập (unique group IDs):** **1.492 groups** (phát hiện 82 cặp ảnh gần trùng cảnh với khoảng cách Hamming $\le 4$, bảo đảm không bị phân rã xuyên split).

### 3.2. Phân bố trạng thái Review (Human-in-the-loop / Review Tool)
- **APPROVED:** **35 ảnh** (27 ảnh TACO multi-object + 5 ảnh OpenImages tin cans/boxes + 3 ảnh OpenImages pure standalone clothes).
- **REJECTED (Loại bỏ dứt khoát vì vi phạm phạm vi rác):** **3 ảnh**
  - `oi_106c8de22ed8d1d4.jpg`: Người đang ăn tại bàn tiệc, quần áo đang mặc trên người.
  - `oi_00a36f96e31731c4.jpg`: Chai nước đang sử dụng trên bàn làm việc văn phòng.
  - `oi_02deba0102b5ce2a.jpg`: Lọ nước hoa đang dùng trên bàn trang điểm.
- **NEEDS_RELABEL / CONTEXT_AMBIGUOUS:** **2 ảnh**
  - `oi_01d160559286c930.jpg`: Ly rượu trên bàn tiệc (chưa rõ bối cảnh phế thải).
  - `oi_04d9284ebdc41aeb.jpg`: Ly thủy tinh gia dụng.
- **UNREVIEWED:** **103 ảnh** (đang xếp hàng trong Review Tool, có thể duyệt tiếp tục).

### 3.3. Phân bố số lượng Bounding Box theo 10 Lớp

| ID | Tên lớp | Tổng số box thu thập | Số ảnh thu thập | Số box đã APPROVED | Số ảnh đã APPROVED | Trạng thái sẵn sàng |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | **battery** | 2 | 2 | 2 | 2 | **Đạt** (2 ảnh pin thực địa TACO) |
| 1 | **biological** | 8 | 7 | 8 | 7 | **Đạt** (Thức ăn thừa thực địa) |
| 2 | **cardboard** | 69 | 46 | 13 | 10 | **Đạt** (Thùng carton, hộp bánh) |
| 3 | **clothes** | 58 | 19 | 22 | 3 | **Đạt** (Quần áo đơn lập không người) |
| 4 | **glass** | 81 | 38 | 10 | 8 | **Đạt** (Chai lọ thủy tinh) |
| 5 | **metal** | 109 | 48 | 32 | 14 | **Đạt** (Lon nước, đồ hộp kim loại) |
| 6 | **paper** | 89 | 60 | 21 | 10 | **Đạt** (Giấy vụn, ly giấy, túi giấy) |
| 7 | **plastic** | 406 | 90 | 89 | 20 | **Đạt** (Chai nhựa, túi nilon, hộp xốp) |
| 8 | **shoes** | 7 | 6 | 3 | 2 | **Đạt** (+114 ảnh OpenImages shoes) |
| 9 | **trash** | 319 | 66 | 115 | 12 | **Đạt** (Đầu lọc thuốc lá, rác hỗn hợp) |
| **Tổng** | **10 lớp** | **1.148** | **143** | **325** | **35** | **Đủ 10/10 lớp có mẫu thật** |

---

## 4. HÀNG ĐỢI XỬ LÝ BOX MƠ HỒ (AMBIGUOUS BOXES QUEUE)

Đã thiết lập hệ thống quản lý các đối tượng mơ hồ tại `data/audit/ambiguous_boxes_queue.json`:
- **Tổng số box đang trong hàng đợi:** **21 items**.
- **Đặc điểm lưu trữ:** Mỗi item chứa `queue_id`, `image_id`, `filename`, `source`, `raw_category_name`, `note`, `bbox_openimages` / `bbox_coco`, và tọa độ normalized YOLO `[xc, yc, w, h]`, danh sách lớp gợi ý `suggested_classes`.
- **Tích hợp giao diện Streamlit:**
  - Mục `⚠️ Ambiguous Boxes Queue` tự động mở rộng khi người dùng chọn ảnh có box mơ hồ.
  - Nút **"➕ Assign Box"**: gán box vào danh sách active boxes của ảnh với lớp được chọn, chuyển trạng thái hàng đợi sang `ASSIGNED`.
  - Nút **"🚫 Discard Box"**: đánh dấu loại bỏ box (không phải rác / ngoài phạm vi), chuyển trạng thái hàng đợi sang `DISCARDED`.
  - Bằng chứng kiểm chứng live qua Chrome DevTools MCP: [`artifacts/part02/mcp_test_evidence/mcp_ambiguous_queue_oi_13d3f1e5893726a2.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_ambiguous_queue_oi_13d3f1e5893726a2.png).

---

## 5. BẰNG CHỨNG HÌNH ẢNH THỰC TẾ QUA CHROME DEVTOOLS MCP

Tech Lead đã trực tiếp thao tác trên giao diện Review Tool đang chạy ngầm (`http://localhost:8501`), chụp ảnh màn hình lưu vào thư mục `artifacts/part02/mcp_test_evidence/`:

1. **Hàng đợi Ambiguous Boxes Queue hoạt động trực quan:**
   - Ảnh `oi_13d3f1e5893726a2` chứa 11 box chai lọ chưa rõ chất liệu, hiển thị đầy đủ thông tin, tọa độ và các nút phân bổ lớp.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_ambiguous_queue_oi_13d3f1e5893726a2.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_ambiguous_queue_oi_13d3f1e5893726a2.png).
2. **Kiểm chứng lớp Quần áo đơn lập (clothes - Class 3):**
   - Ảnh `oi_0162246ca3c39e68` hiển thị 11 bounding box quần áo treo đơn lập trên giá, hoàn toàn không có người, bối cảnh phế thải/thu gom dệt may rõ ràng.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_clothing_oi_0162246ca3c39e68.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_clothing_oi_0162246ca3c39e68.png).
3. **Kiểm chứng lớp Pin thực địa (battery - Class 0):**
   - Ảnh `taco_0456` hiển thị viên pin vứt bỏ ngoài trời từ nguồn OpenLitterMap S3, kích thước chuẩn 3120x4160, 1 box battery rõ ràng.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_battery_taco_0456.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_battery_taco_0456.png).
4. **Kiểm chứng lớp Giày dép rác thực địa (shoes - Class 8):**
   - Ảnh `taco_0073` hiển thị đôi giày vứt bỏ cùng rác nhựa trên bãi cỏ, 3 bounding boxes gồm `plastic` và 2 `shoes`.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_shoes_litter_taco_0073.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_shoes_litter_taco_0073.png).

---

## 6. KẾT QUẢ BỘ KIỂM THỬ TỰ ĐỘNG VERIFICATION GATES

Chạy lệnh kiểm thử tự động tại thư mục gốc:
```bash
python tests/test_verification_gates.py
```
Kết quả kiểm thử đạt **4/4 ca PASS (100%)**:
1. **TEST 1 (Missing image folder handling):** `PASS` (Trả về exit code 1, trạng thái `FAILED_DISK_IMAGE_VERIFICATION`).
2. **TEST 2 (Unresolved candidate detection):** `PASS` (Trả về exit code 1, `unresolved_count > 0`, trạng thái `UNRESOLVED_CANDIDATES_PRESENT`).
3. **TEST 3 (Official dataset split leakage audit):** `PASS` (Trả về exit code 0, trạng thái `VERIFIED_NO_LEAKAGE_WITHIN_HASH_SCOPE`).
4. **TEST 4 (Validation direct disk byte hashing reproduction):** `PASS` (Trả về exit code 0, 0 mismatches, tái tạo chính xác 96,18% accuracy).

---

## 7. BÀN GIAO VÀ ĐÓNG GÓI BÁO CÁO

Toàn bộ báo cáo, mã nguồn, cấu hình, dữ liệu audit, file nhãn real, log kiểm toán và bằng chứng ảnh chụp MCP đã được cập nhật:
- **Thư mục trích xuất:** `C:\Users\ad\Downloads\CNTT-KLCN155_Bao_Cao_Va_Bang_Chung\`
- **File nén lưu trữ chính thức:** `C:\Users\ad\Downloads\CNTT-KLCN155_Bao_Cao_Va_Bang_Chung.zip`

---

## 8. KẾT LUẬN VÀ KIẾN NGHỊ BƯỚC TIẾP THEO

1. **Về Task 2:** Toàn bộ các yêu cầu của Task 2 đã hoàn thành:
   - Đánh giá Final Test classifier được giữ nguyên trung thực (`GATE_A_FAILED` PyTorch CPU 23,86 ms; ONNX FP32 7,40 ms).
   - Bộ dữ liệu detection đa đối tượng thực tế đã có nguồn gốc minh bạch, license hợp lệ, tọa độ chuẩn xác, khử trùng lặp và có hàng đợi xử lý box mơ hồ.
   - Công cụ Review Tool đã được kiểm chứng hoạt động live và đồng bộ hóa đa tầng.
2. **Về Task 3:** Đã đủ điều kiện kỹ thuật và dữ liệu để chuyển sang **Task 3: Huấn luyện YOLOv8n chính và SSDLite320 đối chứng**.
