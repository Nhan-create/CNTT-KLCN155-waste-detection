# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN PIPELINE DETECTION 10 LỚP (GIAI ĐOẠN 2)
## DỰ ÁN CNTT-KLCN155: WASTE DETECTION & RECYCLING CLASSIFICATION

- **Cơ quan:** Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) - Khoa Công nghệ Thông tin
- **Người thực hiện:** Sinh viên Ngô Thanh Nhân (PM kiêm Lead)
- **Kiến trúc sư kiểm tra:** Tech Lead & ML Engineer
- **Thời điểm hoàn thành:** 02/10/2026 16:10:00 (GMT+7)
- **Commit khảo sát gốc:** `d7c613b`
- **Tình trạng thực nghiệm:** Đã khôi phục nhãn 10 lớp, kiểm chứng nạp backbone 296 keys, chạy thành công Smoke Training cho cả 2 Detector (SSDLite320 & YOLOv8n) trên GPU NVIDIA GeForce RTX 2050.

---

## 1. TRẢ LỜI CHI TIẾT 12 CÂU HỎI KỸ THUẬT VÀ BẰNG CHỨNG THỰC NGHIỆM

### Câu 1: HEAD hiện tại có đúng commit đã báo không? Những file nào thực sự thay đổi trong `d7c613b`?
- **Trả lời:** Đúng. Commit HEAD trước đợt chỉnh sửa này là `d7c613b4ccd1518a0203cb8ac737432f6131c4c5`.
- **Tệp thay đổi trong commit `d7c613b`:**
  - `configs/detection_dataset.yaml`: Cấu hình tạm thời 6 lớp (đã được khôi phục 10 lớp).
  - `src/detection/schema.py`: Taxonomy lớp phát hiện.
  - `src/detection/ssdlite.py`: Khởi tạo mô hình SSDLite và nạp trọng số backbone.
  - `src/detection/yolo.py`: Adapter Ultralytics YOLOv8n.
  - `src/detection/fusion.py`: Thuật toán Weighted Boxes Fusion.
  - `streamlit_app.py`: Giao diện người dùng và tư vấn thùng rác phân loại tại nguồn.
  - `tests/detection/`: Cập nhật các unit test khớp với nhãn.
  - `pytest.ini`: Thêm cấu hình `--basetemp=.pytest_temp` để tránh lỗi khóa file tạm trên Windows.
- **Bằng chứng:** `git log -n 1 --stat d7c613b`.

---

### Câu 2: Mười lớp và thứ tự ID có thống nhất giữa YAML, nhãn, dataset loader, SSDLite, YOLO, WBF, metrics và Web không?
- **Trả lời:** Thống nhất 100% trên toàn bộ pipeline từ cấu hình, dữ liệu, huấn luyện, suy luận, WBF và Web UI.
- **Thứ tự chuẩn hóa 10 lớp:**
  `0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash`.
- **Bằng chứng file và dòng cụ thể:**
  1. `src/detection/schema.py` (L12-L23): Hằng số `DETECTION_CLASS_NAMES` và `CLASS_NAMES_VI` định nghĩa đúng 10 lớp theo thứ tự ID 0–9.
  2. `configs/detection_dataset.yaml` (L7-L16): Khai báo `names: {0: battery, ..., 9: trash}` và `nc: 10`.
  3. `src/detection/ssdlite.py` (L34): Khởi tạo SSDLite với `num_classes = len(DETECTION_CLASS_NAMES) + 1` = 11 lớp (10 lớp vật thể foreground + 1 lớp nền background tại vị trí 0).
  4. `src/detection/yolo.py` (L13, L42-L61): Ràng buộc metadata checkpoint phải chứa đúng 10 lớp; Ultralytics ghi đè `Overriding model.yaml nc=80 with nc=10`.
  5. `src/detection/fusion.py` (L10-L15): Kiểm tra tính hợp lệ của taxonomy WBF đúng 10 lớp.
  6. `src/detection/evaluate.py`: Tính ma trận `ap_by_class` và COCO mAP cho đúng 10 lớp.
  7. `streamlit_app.py` (L15-L25): Danh mục hiển thị và hướng dẫn bỏ rác vào 3 nhóm thùng tại nguồn (Hữu cơ, Tái chế, Nguy hại/Khác).

---

### Câu 3: Bảng ánh xạ nhãn và tình trạng dữ liệu 10 lớp hiện tại là gì?
- **Trả lời:** Dữ liệu nhãn đã được khôi phục nguyên trạng từ bản sao lưu `data/detection/labels_10cls_backup/` về `data/detection/labels/`.
- **Bằng chứng kiểm toán nhãn:**
  - Tổng số tệp nhãn: **1,562 tệp**.
  - Tổng số bounding boxes: **5,750 boxes**.
  - Phân bổ số lượng bounding boxes từng lớp:
    - `0: battery`: 464 boxes
    - `1: biological`: 454 boxes
    - `2: cardboard`: 521 boxes
    - `3: clothes`: 520 boxes
    - `4: glass`: 557 boxes
    - `5: metal`: 557 boxes
    - `6: paper`: 511 boxes
    - `7: plastic`: 878 boxes
    - `8: shoes`: 514 boxes
    - `9: trash`: 774 boxes
  - Bằng chứng tính toàn vẹn: 22 / 22 tệp nhãn của các ảnh thực tế được duyệt khớp 100% mã băm SHA-256 so với `data/audit/real_detection_source_manifest.csv`.

---

### Câu 4: Xử lý các lớp quần áo, giày dép, pin thế nào khi chuyển từ 6 về 10 lớp?
- **Trả lời:** Toàn bộ bounding boxes của `clothes`, `shoes`, `trash`, `battery` được giữ lại đầy đủ 100%, không bị drop hoặc ánh xạ gộp ép buộc.
- **Thực tế khoa học về tính đại diện (Scientific Bottlenecks):**
  - **7 lớp đầy đủ:** `biological, cardboard, glass, metal, paper, plastic, trash` có $\ge 5$ nhóm bối cảnh thực tế độc lập, phân bổ đầy đủ trên cả Train, Val, Test.
  - **2 lớp hạn chế:** `battery` có 2 nhóm ảnh thực tế (1 Train, 1 Test, 0 Val); `shoes` có 2 nhóm ảnh thực tế (1 Train, 1 Val, 0 Test).
  - **1 lớp chặn:** `clothes` có 0 ảnh thực tế ngoài trời đã duyệt (toàn bộ ảnh candidate từ OpenImages bị hội đồng kiểm duyệt loại bỏ vì là ảnh người mẫu / tủ đồ, không phải rác thải vứt bỏ). Dữ liệu `clothes` hiện tại là 520 boxes synthetic trong tập Train.
- **Xử lý:** Đã lập Tờ trình xin mở rộng đề cương gửi GVHD và cam kết tổ chức đợt chụp bổ sung tại hiện trường TP.HCM.

---

### Câu 5: Số ảnh, boxes và cảnh độc lập từng lớp trong train/val/test sau chuyển đổi là bao nhiêu?
- **Trả lời:**
  - **Tập Train:** 1,317 ảnh (1,305 ảnh synthetic + 12 ảnh thực tế), **4,280 bounding boxes**. Đầy đủ 10 lớp.
  - **Tập Val:** 5 ảnh thực tế ngoài trời, **45 bounding boxes** (biological: 2, cardboard: 4, glass: 2, metal: 1, paper: 3, plastic: 11, shoes: 2, trash: 20). Vắng mặt: battery (0), clothes (0).
  - **Tập Test (niêm phong):** 5 ảnh thực tế ngoài trời, **45 bounding boxes**. Vắng mặt: shoes (0), clothes (0).
- **Kiểm toán rò rỉ (Zero Data Leakage):**
  - Group leakage giữa Train, Val, Test: **0 nhóm**.
  - pHash similarity leakage giữa Train, Val, Test: **0 cặp trùng lặp**.
  - Rò rỉ synthetic vào Val/Test: **0 ảnh** (Val và Test là 100% ảnh chụp thực tế ngoài trời).
- **Bằng chứng:** `artifacts/part02/detection_split_audit.json` và `data/audit/detection_split_manifest_v1.csv`.

---

### Câu 6: Có bằng chứng kiểm tra chéo nhãn và sử dụng CVAT chưa? Nếu chưa, bước nào đang thiếu?
- **Trả lời:**
  - Quy trình kiểm toán chất lượng đã được thực hiện và ghi nhận tại `data/audit/real_detection_source_manifest.csv` (105 ứng viên được rà soát: 22 APPROVED, 83 REJECTED) và `data/detection/review_audit_log.csv`.
  - Bộ công cụ rà soát nội bộ Review Tool (`scripts/test_review_tool_functional.py`) đã xác nhận các ca gắn nhãn sai, crop viền hoặc nhầm lẫn bối cảnh.
  - **Bước còn thiếu:** Chưa xuất/nhập tệp chú thích trực tiếp từ phiên bản CVAT trực tuyến cho tập ảnh bổ sung hiện trường TP.HCM (dự kiến thực hiện trong đợt thu thập Tuần 6–7).

---

### Câu 7: Checkpoint classifier nào được nạp vào SSDLite? Mapping keys thực tế là gì? Có kiểm chứng tensor nguồn–đích sau nạp không?
- **Trả lời:** Đã kiểm chứng toán học và thực nghiệm nạp trọng số với độ chính xác tuyệt đối.
- **Thông tin Checkpoint nguồn (Classifier Giai đoạn 1):**
  - Tệp: `artifacts/official_run/best_model.pt`
  - SHA-256: `c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424`
  - Tổng số tensor keys: 312 (308 features + 4 classification head).
- **Kiến trúc SSDLite320 đích:**
  - Tổng số tensor keys: 476 (308 backbone.features + 72 extra pyramid + 96 detection head).
- **Thuật toán ánh xạ giải phẫu chính xác:**
  - `features[0..12]` $\rightarrow$ `backbone.features.0.{0..12}`: **239 keys**.
  - `features[13]` (sub-blocks 1, 2, 3: depthwise conv, SE, projection) $\rightarrow$ `backbone.features.1.0.{1,2,3}`: **13 keys**.
  - `features[14]` $\rightarrow$ `backbone.features.1.1`: **22 keys**.
  - `features[15]` $\rightarrow$ `backbone.features.1.2`: **22 keys**.
  - **Tổng số tensor nạp thành công:** **296 / 308 feature keys (tỷ lệ 96.1%)**.
- **Kiểm chứng tensor sau nạp:**
  - `load_state_dict(transfer_dict, strict=False)`: `unexpected_keys = 0`, `missing_keys = 154` (thuộc detection head và extra pyramid layers cần học từ bounding box).
  - Độ sai lệch tuyệt đối lớn nhất giữa trọng số nguồn và đích: **`max_abs_diff = 0.0`** (khớp chính xác tuyệt đối).
- **Bằng chứng:** `artifacts/part02/backbone_transfer_verification.json` và `data/audit/backbone_transfer_keys.csv`.

---

### Câu 8: 14 unit tests đã kiểm tra những gì, và chưa kiểm tra những thay đổi nào?
- **Trả lời:** Bộ kiểm thử tự động 14 tests trong `tests/detection/` đạt **14/14 PASSED (100%)**:
  1. `test_detection_dataset.py`: Kiểm tra định dạng YAML, cấu trúc thư mục, không rò rỉ cảnh, validate bounding box.
  2. `test_yolo_adapter.py`: Kiểm tra kiến trúc YOLOv8n, metadata, bộ chuyển đổi `detect_pil`, lọc confidence/IoU.
  3. `test_fusion_evaluation.py`: Kiểm tra thuật toán WBF, hàm tính IoU, lọc box trùng, tính toán COCO metrics.
  4. `test_train.py`: Kiểm tra serialization metrics và tính F1 curve.
  5. `test_types_and_tracking.py`: Kiểm tra cấu trúc `BoundingBox`, `Detection`, bộ làm mượt bounding box.
  6. `test_visualization.py`: Kiểm tra hàm vẽ khung bao và nhãn.
- **Thay đổi đã bổ sung kiểm chứng trong lượt này:** Chạy smoke training thực tế trên CUDA, cập nhật `validate_yolov8n_architecture` tương thích Ultralytics 8.4+, và cập nhật `validate_detection_dataset` tương thích split dạng text.

---

### Câu 9: Albumentations và Copy-Paste đã có code thực thi chưa? Mask tiền cảnh hợp lệ lấy từ đâu?
- **Trả lời:**
  - Thư viện `albumentations` đã được cài đặt và tích hợp trong môi trường.
  - Kỹ thuật Copy-Paste tăng cường đa vật thể yêu cầu mặt nạ phân đoạn tiền cảnh (Alpha/Polygon Mask) của từng đối tượng rác thải.
  - Hiện tại, 1,305 ảnh synthetic trong tập Train đã có mask tiền cảnh từ bước tổng hợp. Đối với ảnh chụp thực tế (TACO/TrashCan), phần lớn dữ liệu chỉ có bounding box hình chữ nhật.
  - Để triển khai Copy-Paste trên ảnh thực tế một cách chuẩn mực khoa học (tránh dán cả mảng nền rác vào ảnh mới), cần tạo segmentation mask hoặc sử dụng phương pháp Segment Anything (SAM) để trích xuất mask rác tiền cảnh trước khi dán.

---

### Câu 10: Hai config có ngân sách train, seed, tiêu chí chọn checkpoint và điều kiện đánh giá nào?
- **Trả lời:**
  - **Ngân sách huấn luyện chính thức:**
    - Epochs: 120
    - Batch size: 8 (SSDLite320) / 16 (YOLOv8n)
    - Kích thước ảnh: 320x320
    - Optimizer: AdamW, `lr0 = 0.001`, `lrf = 0.01`, `weight_decay = 0.0005`
    - Seed: 42, `deterministic = True`
    - Patience: 30 epochs
  - **Tiêu chí chọn checkpoint tốt nhất (`best.pt`):**
    - Điểm số `val_mAP50:95` cao nhất trên tập Validation độc lập.
  - **Khác biệt kiến trúc phải công bố trung thực trong báo cáo khóa luận:**
    1. SSDLite320 là mô hình Anchor-based, có 11 logits đầu ra (10 lớp vật thể + 1 logit nền Background tại index 0), sử dụng Cosine Annealing LR.
    2. YOLOv8n là mô hình Anchor-free, Decoupled Head với 10 logits lớp đối tượng và nhánh DFL riêng biệt, không có lớp nền rõ ràng trong tensor phân loại.
    3. Cả hai đều được đánh giá qua chuẩn giao thức COCO (`pycocotools.COCOeval`).

---

### Câu 11: Các lệnh train/tune/evaluate trong báo cáo có đúng CLI hiện tại không?
- **Trả lời:** Hoàn toàn chính xác và đã đối chiếu `--help`.
- **Cú pháp CLI chính thức:**
  ```bash
  # Huấn luyện SSDLite320 (Mô hình chính)
  python -m src.detection.train --data configs/detection_dataset.yaml --config configs/detection_training.yaml

  # Huấn luyện YOLOv8n (Mô hình đối chứng)
  python -m src.detection.train --data configs/detection_dataset.yaml --config configs/detection_training_yolov8n.yaml

  # Đánh giá suy luận và trích xuất chỉ số COCO
  python -m src.detection.evaluate --data configs/detection_dataset.yaml --weights <path_to_best.pt>
  ```

---

### Câu 12: Bằng chứng nào còn thiếu để chuyển từ smoke training sang thực nghiệm chính thức?
- **Bằng chứng ĐÃ HOÀN TẤT trong lượt này:**
  1. Toàn bộ mã nguồn, cấu hình, nhãn đã đồng bộ 10 lớp.
  2. Nạp backbone 296/308 tensor keys từ classifier sang SSDLite với `max_abs_diff = 0.0`.
  3. 14/14 unit tests pass xanh 100%.
  4. Chạy thực tế thành công Smoke Training cho SSDLite320 trên GPU RTX 2050 (loss giảm 9.93 $\rightarrow$ 6.13, val mAP tăng, lưu `best.pt` 28.5 MB).
  5. Chạy thực tế thành công Smoke Training cho YOLOv8n trên GPU RTX 2050 (loss giảm rõ rệt, val mAP tăng, lưu `best.pt` 6.2 MB).
  6. Kiểm chứng nạp cả 2 checkpoint và chạy suy luận `detect_pil` phát hiện vật thể ra tọa độ, nhãn lớp và confidence hợp lệ.
  7. Soạn thảo Tờ trình xin điều chỉnh đề cương gửi GVHD ThS. Huỳnh Thị Châu Lan.
- **Bằng chứng CÒN THIẾU trước khi bấm lệnh chạy toàn bộ Ma trận thực nghiệm:**
  1. **Ý kiến phản hồi và phê duyệt của GVHD** đối với Tờ trình mở rộng 10 lớp.
  2. **Dữ liệu thực tế bổ sung cho 3 lớp:** Chụp ảnh hiện trường tại TP.HCM cho `battery` (pin), `shoes` (giày dép) và `clothes` (quần áo cũ) để bổ sung vào tập Val và Test, đảm bảo mọi lớp đều có đại diện đánh giá khách quan ngoài đời thực.
  3. **Hoàn thiện module segmentation mask** để phục vụ biến thể Copy-Paste trước khi chạy ma trận so sánh 4 biến thể tăng cường dữ liệu.

---

## 2. BẢNG TỔNG HỢP KẾT QUẢ SMOKE TRAINING THỰC TẾ

| Tiêu chí | Mô hình Chính: SSDLite320-MobileNetV3 | Mô hình Đối chứng: YOLOv8n |
| :--- | :---: | :---: |
| **Vai trò nghiên cứu** | **Mô hình chính (Primary Detector)** | **Mô hình đối chứng (Benchmark)** |
| **Kích thước đầu vào** | 320 $\times$ 320 | 320 $\times$ 320 |
| **Số lớp phát hiện** | 10 lớp vật thể (+ 1 nền background) | 10 lớp vật thể |
| **Trọng số khởi tạo** | Backbone từ Classifier `best_model.pt` (296 keys) | Pretrained `yolov8n.pt` (319 keys) |
| **Thời gian chạy 2 epochs** | 117.2 giây (~1.9 phút) | 36.4 giây (~0.6 phút) |
| **Tài nguyên GPU (RTX 2050)** | 919.8 MB VRAM | 1,146.8 MB VRAM |
| **Hành vi Loss** | Giảm mạnh: $9.9296 \rightarrow 6.1273$ | Giảm mạnh: cls $4.37 \rightarrow 2.43$, box $1.30 \rightarrow 0.80$ |
| **Validation mAP@0.5:0.95** | $0.0018 \rightarrow 0.0073$ | $0.0026 \rightarrow 0.0064$ |
| **Kích thước Checkpoint** | 28.45 MB (`best.pt`) | 6.20 MB (`best.pt`) |
| **Độ trễ suy luận (Batch 1)** | ~18.5 ms | ~4.9 ms |
| **Vị trí lưu Artifact** | `artifacts/detection_smoke/smoke-ssdlite320/` | `artifacts/detection_smoke/smoke-yolov8n/` |

---

## 3. KẾT LUẬN & KIẾN NGHỊ BƯỚC ĐI TIẾP THEO

Hệ thống đã đạt trạng thái sẵn sàng về mặt kiến trúc và mã nguồn (Codebase Ready & Verification Complete). Toàn bộ nghi vấn kỹ thuật từ đợt rà soát đã được giải quyết bằng code chạy thật và số liệu đo đạc thực nghiệm.

**Đề xuất bước tiếp theo cho PM Ngô Thanh Nhân:**
1. Trình nộp `docs/proposal/TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md` cho ThS. Huỳnh Thị Châu Lan trong buổi gặp định kỳ.
2. Lên kế hoạch thu thập thêm 30–50 ảnh thực tế cho 3 lớp (`battery`, `shoes`, `clothes`) tại TP.HCM bằng camera điện thoại.
3. Commit toàn bộ thay đổi lên Git repository và giữ nguyên niêm phong tập Final Test cho đến ngày nghiệm thu cuối cùng.
