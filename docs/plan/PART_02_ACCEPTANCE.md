# BÁO CÁO NGHIỆM THU TASK 2: ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

- **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải đa đối tượng
- **Đơn vị thực hiện:** Nhóm ML / Tech Lead
- **Người nhận báo cáo:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_ACCEPTANCE.md` (Cập nhật R4 — Bổ sung Đợt Mẫu Kiểm Chứng Dữ Liệu Thật)
- **Ngày lập:** 02/10/2026
- **Trạng thái Gate A gốc (Final Test Classifier):** `GATE_A_FAILED` (PyTorch CPU forward 23,86 ms > ngưỡng 20,0 ms; các chỉ tiêu Accuracy 96,13%, Macro-F1 0,9578, Battery Recall 95,58% đều đạt xuất sắc. Giữ nguyên không sửa lịch sử).
- **Trạng thái Tối ưu hóa Runtime (Deployment Ready):** `RUNTIME_OPTIMIZATION_VERIFIED` (Mô hình ONNX Runtime FP32 tại 6 luồng CPU đạt **7,40 ms**, toàn pipeline 30,63 ms, khớp 100,0000% nhãn Top-1 trên 2.223 ảnh validation).
- **Trạng thái Công cụ Review Tool:** `REVIEW_TOOL_VERIFIED` (Đạt 9/9 ca kiểm thử chức năng tự động Playwright trên sandbox cô lập; kiểm chứng thao tác live bằng Chrome DevTools MCP; bảo toàn 100% hash SHA-256 của toàn bộ dữ liệu gốc).
- **Trạng thái Đợt Mẫu Dữ Liệu Thật (Real Pilot Batch):** `PILOT_VERIFIED_SUCCESSFUL` (Đã tải, chuyển đổi và kiểm chứng 35 ảnh thật từ TACO và OpenImages V7; 300 boxes; 71,4% ảnh đa vật thể; đại diện đủ 10/10 lớp; 0 lỗi hình học / 0 lỗi ngoài biên; kiểm chứng live qua Chrome DevTools MCP).
- **KẾT LUẬN TRẠNG THÁI TASK 2:** **`TOOL_VERIFIED_DATA_PENDING`** (Công cụ review đã đạt chuẩn, đợt mẫu dữ liệu thật đã thông suốt pipeline; đang mở rộng thu thập đủ số lượng nhãn thực tế cho 9 lớp còn lại trước khi chuyển sang Phần 3).

---

## 1. PHẠM VI ỨNG DỤNG ĐÃ ĐƯỢC CHỐT

Theo chỉ đạo của PM Ngô Thanh Nhân:
1. **Bối cảnh vận hành:** Hệ thống Web nhận ảnh tĩnh do người dùng tải lên (`st.file_uploader`), gồm một hoặc nhiều vật thể thải bỏ tại khu vực thu gom, thùng rác hoặc bãi tập kết.
2. **Không yêu cầu phần cứng đặc biệt:** Không bắt buộc camera cố định trong thùng rác, không dùng băng chuyền.
3. **Cố định taxonomy:** Giữ nguyên 10 lớp đã thống nhất: `battery` (0), `biological` (1), `cardboard` (2), `clothes` (3), `glass` (4), `metal` (5), `paper` (6), `plastic` (7), `shoes` (8), `trash` (9).
4. **Quy tắc phân định đồ đang sử dụng vs rác:** Đồ vật đang gắn liền cơ học với cơ thể người (quần áo đang mặc, giày đang mang) không phải là rác. Ảnh có người không tự động bị loại, nhưng chỉ gán nhãn cho các vật thể rác đã tách rời cơ thể người, nằm tự do trên sàn/đất/thùng rác.
5. **Giữ nguyên kiến trúc:** Không thêm lớp `Person`, không đổi sang bài toán segmentation hay surveillance ngoài đường phố.

---

## 2. BẰNG CHỨNG KHẢO SÁT VÀ ĐỐI CHIẾU 7 CÂU HỎI TRƯỚC KHI TẢI HÀNG LOẠT

| Câu hỏi | Kết luận | Bằng chứng thực tế | Điểm chưa biết | Hành động tiếp theo |
|---|---|---|---|---|
| **1. Phiên bản, URL, license, loại annotation của từng nguồn?** | **OpenImages V7:** AWS S3 (`https://open-images-dataset.s3.amazonaws.com/validation/`), CC BY 2.0 / CC BY 4.0, normalized bbox [XMin, XMax, YMin, YMax].<br>**TACO:** GitHub Pedropro master v0.1, Flickr CDN, CC BY 4.0, COCO bbox [x, y, w, h] + Polygon Segmentation. | File metadata `openimages-boxable-classes.csv` (603 classes), `openimages-validation-boxes.csv` (23,6 MB); file TACO `annotations.json` (1.500 ảnh, 4.784 annotations). | Thời gian duy trì liên kết ảnh của các tài khoản Flickr cá nhân trên TACO. | Tải và lưu trữ ảnh gốc cục bộ tại `data/detection/images/real` cùng mã băm SHA-256 đối soát. |
| **2. Nguồn có bbox thật hay chỉ nhãn phân loại toàn ảnh?** | **OpenImages V7 và TACO đều có bounding box thực sự.**<br>TrashNet, Kaggle Garbage, VN-trash (Part 1) **chỉ có nhãn phân loại toàn ảnh (image-level)**, không có bounding box. | Kiểm tra schema file nhãn: TACO có trường `bbox` [x, y, w, h]; OpenImages có `XMin, XMax, YMin, YMax`. File Part 1 lưu theo cấu trúc thư mục lớp, không có file `.txt` tọa độ. | Liệu có thể tận dụng ảnh Part 1 để tạo cảnh ghép mosaic không. | Tuyệt đối không coi ảnh classification là dữ liệu detection; chỉ dùng TACO, OpenImages và ảnh chụp thực tế có bbox. |
| **3. Ánh xạ theo vật liệu hay tên vật thể? "Bottle" có đủ thông tin không?** | Tên vật thể đơn thuần **không đủ để xác định vật liệu**. Ví dụ: `Bottle` (`/m/04dr76w`) có thể là nhựa, thủy tinh hoặc kim loại; `Box` (`/m/025dyy`) có thể là bìa carton, nhựa hoặc gỗ.<br>TACO phân loại theo **vật thể + vật liệu kết hợp** (`Clear plastic bottle` vs `Glass bottle` vs `Drink can`) nên ánh xạ chính xác 100%. | Đã định nghĩa file `configs/detection_source_mapping.yaml`: các nhãn mơ hồ được gắn cờ `REQUIRES_REVIEW` để người duyệt kiểm tra bằng mắt, không tự động gán bừa. | Số lượng ảnh có nhãn mơ hồ trong tập mở rộng. | Đưa toàn bộ box mơ hồ vào hàng đợi `REQUIRES_REVIEW` trong Review Tool. |
| **4. Nhãn có đầy đủ cho mọi vật thể trong ảnh không? Xử lý nhãn thiếu?** | **OpenImages có nhãn thưa (sparse annotation)**, nhiều vật thể phụ không được gán nhãn.<br>**TACO có mật độ nhãn rác cao hơn** nhưng vẫn có thể sót rác nhỏ ở góc xa. | Quan sát thực tế: ảnh `oi_106c8de22ed8d1d4` có người và ly rượu nhưng quần áo người không được gán nhãn đầy đủ. | Tỷ lệ sót nhãn ở các ảnh góc rộng. | Quy trình review: người duyệt dùng nút "➕ Add New Bounding Box" trên Review Tool để bổ sung vật thể thuộc 10 lớp trước khi bấm `APPROVED`. |
| **5. Nguồn cho battery, biological, clothes, shoes? Bằng chứng mẫu?** | - `battery`: TACO có 2 box (ảnh `taco_0082`); OpenImages Boxable **không có class Battery** (báo cáo cũ nêu `/m/01cszv` là hallucination). Bổ sung thêm từ Roboflow Waste Battery open datasets.<br>- `biological`: TACO có 8 box Food waste (ảnh 68, 72, 81, 92, 704).<br>- `clothes`: OpenImages `/m/09j2d` (16.083 boxes, cần lọc ảnh quần áo đơn lập).<br>- `shoes`: Đã có 114 ảnh thật OpenImages (507 boxes) + 7 boxes TACO. | Đã tải thành công và kiểm chứng ảnh `taco_0082` (chứa pin + lon kim loại + nắp nhựa) và ảnh `taco_0068`, `taco_0081` (thức ăn thừa). | Tốc độ tải tập Roboflow Battery khi mở rộng. | Kết hợp TACO + Roboflow Waste Battery để đủ mẫu pin thực tế. |
| **6. Gom nhóm ảnh trùng cảnh/đối tượng trước khi chia split?** | Sử dụng thuật toán **Perceptual Hashing (pHash) với Disjoint Set Union (DSU)** tại `scripts/cluster_detection_groups.py`. Khoảng cách Hamming $\le 4$ sẽ gộp chung vào một `group_id`. Khi chia split, toàn bộ ảnh trong cùng `group_id` bắt buộc nằm trong cùng split. | Đã chạy trên toàn bộ 1.454 ảnh, phát hiện 79 cặp ảnh gần trùng, gán 1.387 group ID duy nhất. | Số lượng chuỗi ảnh chụp liên tiếp khi thu thập thêm dữ liệu mới. | Tự động chạy lại `cluster_detection_groups.py` sau mỗi đợt nhập dữ liệu. |
| **7. Phân định ai review thủ công và chụp ảnh tại HUIT?** | **AI / Tự động hóa 100%:** Tải ảnh, chuyển đổi tọa độ, ánh xạ lớp, lọc hình học, tính hash, gom group ID, tích hợp Review Tool.<br>**Con người tham gia:** Review thủ công các trường hợp vật liệu mơ hồ; chụp 50–100 ảnh thực tế tại khuôn viên HUIT (thùng rác, sân trường). | Script `scripts/collect_real_detection_data.py` tự động hóa hoàn toàn khâu thu thập và chuyển đổi. | Thời gian biểu của sinh viên/PM khi chụp ảnh thực địa HUIT. | Không để toàn bộ tiến độ chờ ảnh HUIT; ưu tiên hoàn thiện dữ liệu mở trước. |

---

## 3. KẾT QUẢ ĐỢT MẪU KIỂM CHỨNG DỮ LIỆU THẬT (REAL PILOT BATCH)

Tech Lead đã thực thi đợt mẫu 35 ảnh thật theo đúng yêu cầu kiểm chứng pipeline:

### 3.1. Thống kê đợt mẫu (Trích xuất từ `artifacts/part02/real_detection_readiness.json`)
- **Tổng số ảnh thật kiểm chứng:** 35 ảnh (25 ảnh TACO + 10 ảnh OpenImages V7).
- **Tổng số bounding box thật:** 300 boxes.
- **Tỷ lệ ảnh đa đối tượng (Multi-object):** 25 / 35 ảnh (**71,4%**).
- **Mức độ đại diện 10 lớp trong đợt mẫu:**
  - `battery` (Class 0): 1 ảnh (taco_0082) [REPRESENTED]
  - `biological` (Class 1): 7 ảnh [REPRESENTED]
  - `cardboard` (Class 2): 10 ảnh [REPRESENTED]
  - `clothes` (Class 3): 1 ảnh [REPRESENTED]
  - `glass` (Class 4): 12 ảnh [REPRESENTED]
  - `metal` (Class 5): 14 ảnh [REPRESENTED]
  - `paper` (Class 6): 10 ảnh [REPRESENTED]
  - `plastic` (Class 7): 20 ảnh [REPRESENTED]
  - `shoes` (Class 8): 1 ảnh [REPRESENTED] (cộng với 114 ảnh OpenImages sẵn có)
  - `trash` (Class 9): 12 ảnh [REPRESENTED]
- **Đại diện taxonomy:** **10 / 10 lớp đều có mẫu thực tế trong đợt pilot**.

### 3.2. Phát hiện và khắc phục lỗi chuyển đổi tọa độ COCO
- **Phát hiện quan trọng:** Khi tải ảnh resized từ CDN (`flickr_640_url`), tọa độ tuyệt đối `[x, y, w, h]` trong COCO vẫn dựa trên kích thước ảnh gốc (`meta["width"]`, `meta["height"]`). Nếu chia cho kích thước ảnh tải về sẽ gây hiện tượng box bị phóng đại vượt ra ngoài khung hình.
- **Khắc phục:** Đã chuẩn hóa chính xác bằng kích thước gốc metadata trong `scripts/collect_real_detection_data.py`.
- **Kết quả tái kiểm tra bằng `scripts/validate_detection_annotations.py`:**
  - Files checked: 149 tệp nhãn real
  - Boxes checked: 807 boxes
  - **Errors: 0 | Warnings: 0 | Status: PASS**.

---

## 4. BẰNG CHỨNG KIỂM CHỨNG TƯƠNG TÁC THỰC TẾ QUA CHROME DEVTOOLS MCP

Đã sử dụng MCP Chrome DevTools tương tác trực tiếp trên công cụ Review Tool (`http://localhost:8501`), chụp và lưu trữ 3 ảnh bằng chứng vật lý:

1. **Kiểm chứng nạp ảnh thật đa đối tượng thành công:**
   - Ảnh `taco_0082.jpg` được nạp với đầy đủ 7 bounding boxes gồm `battery`, `metal`, `plastic`.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_pilot_taco_0082_loaded.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_pilot_taco_0082_loaded.png).
2. **Kiểm chứng chuyển đổi trạng thái và ghi đĩa:**
   - Chuyển trạng thái `taco_0082` sang `APPROVED`, nhập ghi chú review, lưu thành công xuống đĩa và manifest, cập nhật chỉ số dashboard `Approved: 1 (0.1%)`.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_pilot_taco_0082_approved.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_pilot_taco_0082_approved.png).
3. **Kiểm chứng sửa đổi nhãn và xử lý trường hợp cần relabel:**
   - Thao tác đổi lớp từ `shoes` sang `trash`, đánh dấu `NEEDS_RELABEL`, thêm ghi chú "Class corrected from shoes to trash. Marked as NEEDS_RELABEL for secondary review", lưu thành công xuống file `.txt` và manifest.
   - Bằng chứng: [`artifacts/part02/mcp_test_evidence/mcp_real_pilot_relabel_edited.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_real_pilot_relabel_edited.png).

---

## 5. CÁC FILE ĐÃ TẠO VÀ SỬA ĐỔI TRONG REPO

| Đường dẫn file | Thao tác | Mô tả chức năng & Vị trí sửa |
|---|:---:|---|
| [`configs/detection_source_mapping.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_source_mapping.yaml) | **Tạo mới** | Cấu hình ánh xạ taxonomy 10 lớp, quy tắc tiếp nhận, ánh xạ 60 categories TACO và các mã lớp OpenImages V7. |
| [`scripts/collect_real_detection_data.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/collect_real_detection_data.py) | **Tạo mới** | Pipeline tải dữ liệu thật tự động, hỗ trợ chạy tiếp sau gián đoạn, chuẩn hóa tọa độ COCO/OpenImages sang YOLO, tính SHA-256, pHash, group_id và cập nhật manifest. |
| [`data/audit/real_detection_source_manifest.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/real_detection_source_manifest.csv) | **Tạo mới** | Bảng manifest nguồn gốc chi tiết từng ảnh (source, source_id, url, license, image_hash, label_hash, group_id, classes, boxes, status, split). |
| [`artifacts/part02/real_detection_readiness.json`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/real_detection_readiness.json) | **Tạo mới** | Báo cáo JSON thống kê độ sẵn sàng của dữ liệu thật (độ bao phủ lớp, tỷ lệ đa vật thể, số lượng box). |
| [`src/ui/review_tool.py`](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) | **Chỉnh sửa** | Dòng 179–190: Bổ sung các bộ lọc nguồn `Real (All)`, `Real (TACO)`, `Real (OpenImages)`, `Synthetic (Mendeley)` giúp phân tách rõ ràng khi review. |
| [`data/detection/manifest_detection_v1.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/manifest_detection_v1.csv) | **Cập nhật** | Tích hợp 35 ảnh thật mới vào master manifest, nâng tổng số ảnh quản lý lên 1.454 ảnh. |

---

## 6. TIẾN ĐỘ THỜI GIAN VÀ KẾ HOẠCH MỞ RỘNG TIẾP THEO

### 6.1. Đánh giá tốc độ thực tế
- Tốc độ tải và chuyển đổi ảnh đợt mẫu: ~35 ảnh hoàn tất trong **45 giây** (bao gồm tải ảnh, parse JSON, chuẩn hóa bbox, tính pHash và SHA-256).
- Tốc độ review thủ công: trung bình **30–45 giây / ảnh** trên Review Tool (chủ yếu kiểm tra vật liệu mơ hồ và rà soát vật thể sót).

### 6.2. Kế hoạch mở rộng dữ liệu thật (Giai đoạn tiếp theo của Task 2)
1. **Mở rộng TACO:** Tải thêm 200–300 ảnh rác thực tế đa vật thể từ TACO cho các lớp `plastic`, `glass`, `metal`, `cardboard`, `paper`, `trash`, `biological`: **Dự kiến 2–3 giờ làm việc**.
2. **Bổ sung lớp Battery & Clothes:**
   - Trích xuất 50 ảnh pin thực tế từ Roboflow Waste Battery Open Dataset.
   - Trích xuất 50 ảnh quần áo đơn lập (discarded clothing) từ OpenImages V7 (đã loại trừ người mặc).
   - **Dự kiến 2 giờ làm việc**.
3. **Review và dán nhãn chuẩn hóa:** Dùng Review Tool duyệt nhanh tập mở rộng (tập trung xác nhận vật liệu mơ hồ): **Dự kiến 4–6 giờ làm việc**.
4. **Phần chụp ảnh thực địa HUIT (Con người thực hiện):** Chụp 50–100 ảnh tại khuôn viên trường làm tập test thực địa (có thể thực hiện song song hoặc sau khi tập dữ liệu mở đã sẵn sàng).

---

## 7. BÀN GIAO GÓI HỒ SƠ VÀ ARCHIVE DOWNLOADS

Toàn bộ báo cáo, mã nguồn, test suite, bảng dữ liệu audit, log và hình ảnh bằng chứng MCP đã được cập nhật và đóng gói:

- **Thư mục trích xuất:** `C:\Users\ad\Downloads\CNTT-KLCN155_Bao_Cao_Va_Bang_Chung\`
- **File nén lưu trữ chính thức:** `C:\Users\ad\Downloads\CNTT-KLCN155_Bao_Cao_Va_Bang_Chung.zip`
- **Mã băm SHA-256 mới nhất:** Sẽ được cập nhật ngay sau khi chạy lệnh đóng gói bên dưới.
- **Báo cáo Word tổng hợp:** [`artifacts/part02/Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx)
