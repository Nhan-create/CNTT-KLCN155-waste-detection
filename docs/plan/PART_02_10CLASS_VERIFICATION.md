# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN PIPELINE DETECTION 10 LỚP (GIAI ĐOẠN 2)
## DỰ ÁN CNTT-KLCN155: WASTE DETECTION & RECYCLING CLASSIFICATION

- **Cơ quan:** Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) - Khoa Công nghệ Thông tin
- **Người thực hiện:** Sinh viên Ngô Thanh Nhân (PM kiêm Lead)
- **Kiến trúc sư kiểm tra:** Tech Lead & ML Engineer
- **Thời điểm hoàn thành:** 02/10/2026 (GMT+7)
- **Cam kết phương pháp:** Tiếp cận chuẩn mực nghiên cứu ML - dựa trên bằng chứng toán học và dữ liệu thực nghiệm, không cảm tính, không võ đoán.

---

## 1. TRẢ LỜI CHI TIẾT 10 CÂU HỎI KHOA HỌC KÈM BẰNG CHỨNG TRUY XUẤT

### Câu 1: HEAD, branch và working tree hiện tại là gì? Những thay đổi mới nằm ở commit nào?
- **Trả lời:**
  - Branch: `main`
  - Commit HEAD: `9d385c289f56989e14594aa6c2965f43440085a3`
  - Working Tree: Sạch sẽ (clean), đã đồng bộ với `origin/main`.
  - Thay đổi mới nhất nằm ở commit `9d385c2`: *"Tính năng: Đồng bộ hóa toàn diện pipeline Detection 10 lớp, kiểm chứng nạp backbone SSDLite và thực nghiệm smoke training thành công"*.
- **Bằng chứng:** `git status --short`, `git rev-parse HEAD`, `git log -n 1 --stat 9d385c2`.

---

### Câu 2: Con số 5.750 boxes được tính từ những manifest và file nhãn nào? Có bao gồm dữ liệu chưa duyệt hoặc synthetic không?
- **Trả lời:**
  - Con số **5.750 bounding boxes** là tổng số boxes nằm trong toàn bộ **1.562 tệp nhãn vật lý** trên đĩa tại thư mục [`data/detection/labels/`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/labels/).
  - **Phân rã cấu trúc 5.750 boxes:**
    1. **Tập đã Manifest và đưa vào Split huấn luyện/đánh giá (`data/detection/manifest.jsonl`):** Gồm **1.327 ảnh với 4.370 boxes**:
       - 1.305 ảnh synthetic (nguồn Mendeley): chứa **4.095 boxes**, được phân bổ 100% vào tập `train`.
       - 22 ảnh thực tế (nguồn TACO) đã qua thẩm định thủ công nghiêm ngặt (`review_status == APPROVED`): chứa **275 boxes**, chia theo nhóm cảnh chụp (scene groups) vào: `train` (12 ảnh, 185 boxes), `val` (5 ảnh, 45 boxes), `test` (5 ảnh, 45 boxes).
    2. **Tập nhãn chưa duyệt / chưa manifest (Unmanifested / Unreviewed):** Gồm **235 ảnh thực tế với 1.380 boxes** nằm tại [`data/detection/labels/real/`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/labels/real/):
       - Chứa 511 boxes `shoes`, 58 boxes `clothes`, 321 boxes `plastic`, 204 boxes `trash`, 86 boxes `metal`, 72 boxes `glass`, 68 boxes `paper`, 58 boxes `cardboard`, 2 boxes `biological`.
       - **Trạng thái:** Toàn bộ 235 ảnh này nằm ở trạng thái `UNREVIEWED` (103 ảnh TACO, 114 ảnh OpenImages) hoặc `REJECTED`/`NEEDS_RELABEL` trong `data/audit/real_detection_source_manifest.csv`.
       - **Nguyên tắc học thuật:** **Tuyệt đối không tự tiện nâng trạng thái lên `APPROVED`** khi chưa qua quy trình thẩm định và kiểm tra chéo hai lượt qua CVAT. Vì vậy, 1.380 boxes này được bảo lưu trên đĩa nhưng KHÔNG được đưa vào split chính thức.
- **Bằng chứng:** Kết quả kiểm toán dữ liệu từ script `audit_per_class_splits.py` và `real_detection_source_manifest.csv`.

---

### Câu 3: Nhãn 10 lớp đã khôi phục từ bản gốc nào? Có mẫu nào mất nhãn sau lần chuyển sang 6 lớp không?
- **Trả lời:**
  - Nhãn 10 lớp đã được khôi phục nguyên trạng 100% từ bản sao lưu bảo toàn gốc tại [`data/detection/labels_10cls_backup/`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/labels_10cls_backup/).
  - **Bằng chứng toán học SHA-256:**
    - Tổng số tệp trong thư mục hiện hành `labels/`: **1.562 tệp**.
    - Tổng số tệp trong thư mục sao lưu `labels_10cls_backup/`: **1.562 tệp**.
    - Số tệp thiếu ở thư mục hiện hành: **0 tệp**.
    - Số tệp khác biệt mã băm SHA-256: **0 tệp**.
    - Toàn bộ 5.750 bounding boxes nguyên gốc của 10 lớp được bảo toàn toàn vẹn, không có bất kỳ mẫu nào bị mất nhãn hay bị chuyển nhãn sai lệch.

---

### Câu 4: Train/val/test có bao nhiêu ảnh, boxes và cảnh độc lập cho từng lớp? Lớp nào còn thiếu?
- **Trả lời:**
  - Thống kê chi tiết từ tập dữ liệu manifest chính thức (`manifest.jsonl`, 1.327 ảnh, 4.370 boxes):

| Split | Tổng ảnh | Số cảnh độc lập (Scenes) | Tổng Boxes | Lớp đầy đủ ($\ge 5$ boxes) | Lớp thiếu ($< 5$ boxes) | Lớp vắng mặt (0 box) |
|:---:|:---:|:---:|:---:|:---|:---|:---|
| **TRAIN** | 1.317 | 1.253 | 4.280 | 9 lớp: `battery` (463), `biological` (448), `cardboard` (458), `clothes` (462), `glass` (480), `metal` (461), `paper` (436), `plastic` (540), `trash` (531) | `shoes` (1 box / 1 ảnh) | *Không có* |
| **VAL** | 5 | 5 | 45 | `plastic` (11), `trash` (20) | `biological` (2), `cardboard` (4), `glass` (2), `metal` (1), `paper` (3), `shoes` (2) | **`battery` (0), `clothes` (0)** |
| **TEST** | 5 | 5 | 45 | `metal` (9), `plastic` (6), `trash` (19) | `battery` (1), `biological` (2), `cardboard` (1), `glass` (3), `paper` (4) | **`clothes` (0), `shoes` (0)** |

- **Nhận định khoa học:**
  - Tập Test và Val hiện tại bị thiếu nghiêm trọng các cảnh thực tế cho 3 lớp: `clothes` (vắng mặt cả Val và Test), `battery` (vắng mặt ở Val), `shoes` (vắng mặt ở Test).
  - Đây là lý do tại sao cấu hình [`configs/detection_dataset.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_dataset.yaml) khai báo: `classes_fully_evaluable: 7`, `classes_limited_evaluable: [battery, shoes]`, `classes_blocked: [clothes]`.
  - Để giải quyết triệt để, cần tiến hành thẩm định tập 235 ảnh real chưa duyệt hoặc tổ chức đợt thu thập bổ sung ảnh thực địa tại TP.HCM.

---

### Câu 5: 296 keys khác 258 keys trước đây ở những keys nào? Mapping thay đổi vì lý do gì?
- **Trả lời:**
  - **Số liệu gốc:** MobileNetV3-Large Classifier có 312 keys (308 feature keys + 4 classifier head keys). SSDLite320 có 476 keys.
  - **Sự khác biệt giữa 258 keys và 296 keys:** Chênh lệch đúng **+38 tensor keys**.
  - **Lý do thay đổi mapping:**
    - Cấu trúc mạng MobileNetV3 trong Torchvision SSDLite320 được tách thành hai cổng trích xuất đặc trưng đa quy mô:
      + Nhánh $C_4$ lấy từ sau block 12 (tại `backbone.features.0`): gồm 258 keys. Bản ánh xạ cũ chỉ nạp nhánh này và dừng lại.
      + Nhánh $C_5$ lấy từ sau block 15 (tại `backbone.features.1`): chứa các block sâu 13, 14, 15.
    - Trong bản ánh xạ giải phẫu mới tại [`scripts/audit_backbone_transfer.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py):
      + Block 13 (sub-blocks 1, 2, 3: depthwise, Squeeze-and-Excitation, projection) được ánh xạ chính xác sang `backbone.features.1.0`.
      + Block 14 được ánh xạ nguyên vẹn sang `backbone.features.1.1`.
      + Block 15 được ánh xạ nguyên vẹn sang `backbone.features.1.2`.
      + Riêng block 16 (1x1 conv 960 kênh) bị SSDLite thay bằng reduced-tail conv 672 kênh nên không chuyển giao.
    - Nhờ vậy, 38 keys của các khối inverted residual sâu đã được tái sử dụng thành công, nâng tổng số feature keys chuyển giao từ **258 / 308 (83.8%)** lên **296 / 308 (96.1%)**.
- **Bằng chứng:** [`data/audit/backbone_transfer_audit.json`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json) và [`data/audit/backbone_transfer_keys.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_keys.csv).

---

### Câu 6: `max_abs_diff = 0` được tính trực tiếp từ checkpoint nguồn và model đích sau nạp hay chỉ từ hai bản sao dữ liệu trung gian?
- **Trả lời:** Được tính **TRỰC TIẾP** giữa tensor lưu trong tệp checkpoint nguồn trên đĩa cứng và tensor của mô hình SSDLite trong bộ nhớ RAM/VRAM sau khi gọi `model.load_state_dict(...)`.
- **Thực nghiệm kiểm chứng:**
  - Script [`verify_max_abs_diff.py`](file:///D:/CNTT-KLCN155-waste-detection/scratch/verify_max_abs_diff.py) thực hiện:
    1. Đọc tệp checkpoint `artifacts/official_run/best_model.pt` từ đĩa, trích xuất `state_dict`.
    2. Khởi tạo mô hình SSDLite320 bằng hàm `build_ssdlite(pretrained=True, backbone_weights=ckpt_path)`.
    3. Duyệt từng tensor key, thực hiện phép trừ tensor: `diff = (model_state[target_k] - source_tensor).abs().max().item()`.
  - **Kết quả:** Kiểm tra 253 feature weight/bias tensors tương ứng, `max_abs_diff` đạt chính xác **`0.0`**.
  - Không có bất kỳ sự sai lệch số học nào xảy ra trong quá trình nạp trọng số backbone.

---

### Câu 7: Code augmentation đã được gọi ở đâu? Copy-Paste có mask đã kiểm tra và quy tắc xử lý che khuất chưa?
- **Trả lời:**
  - Module Augmentation chuẩn mực cho Detection đã được xây dựng tại [`src/detection/augmentation.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/augmentation.py) sử dụng thư viện `albumentations 2.0.8`.
  - Cài đặt đầy đủ 4 chiến lược theo đề cương:
    1. `none`: Baseline (giữ nguyên ảnh và hộp bao).
    2. `geometric`: Lật ngang (`HorizontalFlip`), dịch chuyển, xoay góc nhẹ, co giãn (`ShiftScaleRotate`), đồng bộ chặt chẽ với toạ độ Bbox.
    3. `photometric`: Biến đổi màu sắc quang học (`ColorJitter`), làm mờ camera (`GaussianBlur`, `MotionBlur`).
    4. `combined`: Phối hợp cả biến đổi hình học và quang học.
  - **Về kỹ thuật Copy-Paste [5] (Ghiasi et al., 2021):**
    - **BLOCKER KHOA HỌC:** Kỹ thuật Copy-Paste chuẩn mực đòi hỏi **mặt nạ phân đoạn đối tượng (pixel-level instance segmentation mask)** để tách rời hình dạng tự nhiên của vật thể rác và dán lên nền mới.
    - Bộ dữ liệu hiện tại chỉ có nhãn hình chữ nhật Bounding Box (YOLO format), hoàn toàn **CHƯA CÓ polygon mask** (`"foreground_masks": []`, `"mask_reviewed": false` trong `manifest.jsonl`).
    - **Nguyên tắc:** Tuyệt đối không dùng crop hình chữ nhật để dán, vì việc dán cả viền nền cũ sẽ tạo ra biên giả tạo (edge artifacts), làm sai lệch phân phối đặc trưng học của mạng nơ-ron và vi phạm tính chuẩn tắc học thuật.
    - Hàm gọi `strategy='copy_paste'` ném ngoại lệ `NotImplementedError` có kiểm soát với lý do nêu trên.
  - **Bằng chứng thực nghiệm:** Đã chạy thử nghiệm trên ảnh thật `taco_0081.jpg`, xuất các mẫu ảnh trước–sau tại [`artifacts/part02/augmentation_samples/`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/augmentation_samples/).

---

### Câu 8: Hai detector đã chạy một bước backward, lưu/nạp checkpoint và suy luận ảnh thật chưa?
- **Trả lời:** Cả hai mô hình đã hoàn thành chạy thực tế (Real Smoke Training) 2 epochs trên GPU NVIDIA GeForce RTX 2050 (4GB VRAM):
  - **SSDLite320 (Mô hình chính):**
    + Loss hội tụ giảm từ 9.93 xuống 6.13.
    + Validation mAP@0.5:0.95 tính qua `pycocotools.COCOeval` tăng từ 0.0018 lên 0.0073.
    + Lưu checkpoint thành công: [`artifacts/detection_smoke/smoke-ssdlite320/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-ssdlite320/weights/best.pt) (28.45 MB, SHA-256: `c9a079d630bdff79...`).
  - **YOLOv8n (Mô hình đối chứng):**
    + Loss box giảm từ 1.30 xuống 0.80, loss cls giảm từ 4.37 xuống 2.43.
    + Validation mAP@0.5:0.95 đạt 0.0064.
    + Lưu checkpoint thành công: [`artifacts/detection_smoke/smoke-yolov8n/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-yolov8n/weights/best.pt) (6.20 MB, SHA-256: `84b677fcb4de6cb0...`).
  - **Nạp lại và suy luận trên ảnh Validation thật:**
    + Cả hai checkpoint đã được nạp lại độc lập qua script `test_real_val_inference.py`.
    + Chạy suy luận thành công trên ảnh thực tế ngoài trời `data/detection/images/real/taco_0081.jpg`.
    + Xuất ảnh trực quan hoá bounding box tại:
      - [`artifacts/part02/smoke_ssdlite_real_val_inference.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/smoke_ssdlite_real_val_inference.png)
      - [`artifacts/part02/smoke_yolov8n_real_val_inference.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/smoke_yolov8n_real_val_inference.png)

---

### Câu 9: CVAT và dữ liệu TP.HCM đã có bằng chứng sử dụng chưa?
- **Trả lời:**
  - **Về CVAT:** Hiện tại trong repo **chưa có tệp dữ liệu nào được xuất trực tiếp từ nền tảng CVAT** (XML, JSON, COCO export). Các nhãn hiện có là kết quả chuyển đổi từ các bộ dữ liệu công khai (TACO, Mendeley, OpenImages) được rà soát nội bộ qua công cụ Review Tool. Quy trình kiểm tra chéo hai lượt trên CVAT trực tuyến sẽ được áp dụng trong đợt thu thập ảnh thực địa.
  - **Về dữ liệu TP.HCM:** Trong repo **chưa có ảnh chụp thực tế tại địa bàn TP.HCM**. Toàn bộ 257 ảnh thực tế hiện có là từ TACO và OpenImages. Kế hoạch thu thập 200–300 ảnh thực địa tại TP.HCM (HUIT, đường phố, điểm thu gom rác) đã được đưa vào Tờ trình điều chỉnh đề cương (Tuần 6–7).

---

### Câu 10: Những yêu cầu đề cương nào đã có kết quả, mới có code hoặc vẫn thiếu?
- **Trả lời phân định rõ ràng:**

| Nhóm trạng thái | Các hạng mục cụ thể | Diễn giải hiện trạng |
|:---|:---|:---|
| **ĐÃ CÓ KẾT QUẢ THỰC NGHIỆM**<br>*(Empirical Results)* | 1. Classifier 10 lớp Phần 1<br>2. Chuyển giao Backbone sang SSDLite (296 keys, diff = 0.0)<br>3. Smoke Training thật 2 epochs cả SSDLite & YOLOv8n trên GPU<br>4. Suy luận ảnh validation thật và xuất ảnh visualize<br>5. Tương tác kiểm chứng Web App qua MCP Chrome DevTools | - Classifier đạt 98.3% F1.<br>- Đã đo đạc tensor và kiểm chứng toán học.<br>- GPU RTX 2050 huấn luyện thành công, checkpoints lưu đầy đủ.<br>- Trực tiếp tương tác trên trình duyệt thật qua MCP. |
| **MỚI CÓ CODE CHUẨN BỊ**<br>*(Code Ready, Pending Full Run)* | 1. Thuật toán Hợp nhất WBF (`src/detection/fusion.py`)<br>2. Module tinh chỉnh ngưỡng WBF trên Validation (`src/detection/tune.py`)<br>3. Module đánh giá COCO AP và benchmark phần cứng (`src/detection/evaluate.py`)<br>4. Module Tăng cường dữ liệu Albumentations 4 chiến lược (`src/detection/augmentation.py`) | - Mã nguồn đã sẵn sàng 100%, vượt qua unit test.<br>- Chờ chạy toàn bộ ma trận huấn luyện chính thức (120 epochs). |
| **THIẾU / BLOCKER**<br>*(Missing / Blocked)* | 1. Tăng cường dữ liệu Copy-Paste [5]<br>2. Dữ liệu thực địa tự chụp tại TP.HCM<br>3. Quy trình gán nhãn kiểm tra chéo trên CVAT<br>4. Đánh giá phân tập điều kiện thách thức (thiếu sáng, ngược sáng, che khuất) | - Thiếu instance segmentation masks (chỉ có bbox).<br>- Chưa đi chụp thực địa TP.HCM theo kế hoạch Tuần 6–7.<br>- Chưa thiết lập project CVAT.<br>- Cần gắn thẻ `conditions` chi tiết cho từng ảnh thực tế. |

---

## 2. BẰNG CHỨNG KIỂM TRA ỨNG DỤNG WEB BẰNG MCP CHROME DEVTOOLS

Ứng dụng web Streamlit ([`streamlit_app.py`](file:///D:/CNTT-KLCN155-waste-detection/streamlit_app.py)) đã được khởi chạy tại cổng `8501` và kiểm tra tự động qua MCP `chrome-devtools`.

### Các kịch bản kiểm thử đã thực hiện và bằng chứng:
1. **Khởi tạo và hiển thị cấu hình ban đầu:**
   - Hiển thị danh mục đúng chuẩn 10 nhóm rác: *Pin, Rác hữu cơ, Bìa carton, Quần áo, Thủy tinh, Kim loại, Giấy, Nhựa, Giày dép, Rác khác*.
   - Khởi tạo mặc định với mô hình chính: `SSDLite320-MobileNetV3 (Mô hình chính)`.
   - **Bằng chứng:** [`artifacts/part02/mcp_streamlit_initial_view.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_streamlit_initial_view.png).

2. **Tải ảnh và cảnh báo khi chưa đạt ngưỡng tin cậy:**
   - Tải ảnh thực tế `taco_0081.jpg`, chạy nhận diện ở ngưỡng tin cậy mặc định $0.35$.
   - Thông báo hiển thị đúng yêu cầu đề cương: *"Không phát hiện vật thể rác đạt ngưỡng tin cậy trong ảnh này. Bạn có thể chọn ảnh rõ hơn hoặc điều chỉnh ngưỡng."*
   - Hiển thị đầy đủ độ trễ xử lý (Processing time).
   - **Bằng chứng:** [`artifacts/part02/mcp_streamlit_detection_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_streamlit_detection_result.png).

3. **Điều chỉnh ngưỡng tin cậy và phân loại thùng rác tại nguồn:**
   - Hạ ngưỡng tin cậy xuống $0.05$ (phù hợp với checkpoint smoke 2 epochs).
   - SSDLite phát hiện 47 vị trí tiềm năng, vẽ bounding box, hiển thị bảng số lượng và hướng dẫn bỏ rác vào các thùng màu sắc tương ứng.
   - **Bằng chứng:** [`artifacts/part02/mcp_streamlit_detection_active_005.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_streamlit_detection_active_005.png).

4. **Chuyển đổi sang Mô hình đối chứng (YOLOv8n):**
   - Lựa chọn `YOLOv8n (Mô hình đối chứng)` từ combobox trên giao diện.
   - Bấm nút "Nhận diện", mô hình đối chứng phát hiện vật thể Pin với độ tin cậy $13.2\%$ tại tọa độ $(163, 236, 391, 396)$, độ trễ $1.340\text{ ms}$.
   - Giao diện kích hoạt cảnh báo phân loại: *🔴 Thùng Đỏ / Cam (Rác nguy hại): Pin, ắc quy, bóng đèn, rác độc hại.*
   - **Bằng chứng:** [`artifacts/part02/mcp_streamlit_yolov8n_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_streamlit_yolov8n_result.png).

5. **Chuyển đổi sang Chế độ Hợp nhất dự đoán (WBF):**
   - Lựa chọn `Hợp nhất SSDLite + YOLOv8n (WBF)`.
   - Thuật toán WBF tự động chạy đồng thời SSDLite và YOLOv8n, hợp nhất dự đoán thành 2 bounding box đồng thuận.
   - Nút **"Tải ảnh kết quả"** hoạt động chính xác.
   - **Bằng chứng:** [`artifacts/part02/mcp_streamlit_wbf_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_streamlit_wbf_result.png).

> [!NOTE]
> Các checkpoint smoke hiện tại chỉ phục vụ kiểm chứng tính toàn vẹn và thông suốt của luồng dữ liệu (pipeline plumbing verification). Độ chính xác nhận diện sẽ được tối ưu khi hoàn thành huấn luyện chính thức 120 epochs.

---

## 3. ĐÍNH CHÍNH VÀ CHUẨN HÓA CÁC NHẬN ĐỊNH HỌC THUẬT

1. **Phân biệt rõ ràng giữa "Có mã nguồn" và "Đã có kết quả thực nghiệm":**
   - Các thuật toán WBF, COCOeval, Albumentations tuy đã được viết code hoàn chỉnh nhưng chưa được coi là "hoàn thành nghiên cứu" cho đến khi chạy xong toàn bộ ma trận thực nghiệm và ghi nhận số liệu.
2. **Bản chất của việc nạp trọng số Backbone Classifier:**
   - Việc chuyển giao trọng số MobileNetV3 từ Classifier Phần 1 sang SSDLite là một giải pháp kỹ thuật nhằm tận dụng đặc trưng miền rác thải và tăng tốc độ hội tụ; đây không phải là yêu cầu tiên quyết bắt buộc của đề cương mà là cải tiến kỹ thuật có kiểm chứng.
3. **Tính năng phân loại thùng rác:**
   - Khuyến nghị màu sắc thùng rác trên giao diện web là tính năng giá trị gia tăng (Value-added feature) hỗ trợ người dùng phân loại tại nguồn, không làm thay đổi bài toán gốc là Bounding Box Object Detection.
4. **Tuyệt đối không võ đoán kết quả trước thực nghiệm:**
   - Không điền trước các giá trị mAP, FPS, kích thước mô hình hay tuyên bố "WBF chắc chắn tốt hơn". Mô hình nào tối ưu hơn sẽ do thực nghiệm đo đạc khách quan quyết định.

---

## 4. BẢNG TỔNG KẾT TRẠNG THÁI KIỂM CHỨNG (PASS / FAIL / BLOCKED)

| STT | Hạng mục kiểm chứng | Trạng thái | Bằng chứng xác thực |
|:---:|:---|:---:|:---|
| 1 | Khôi phục toàn vẹn dữ liệu nhãn 10 lớp | **PASS** | 1.562 tệp nhãn khớp 100% SHA-256 với `labels_10cls_backup`. 0 tệp thiếu, 0 diff. |
| 2 | Kiểm toán phân rã 5.750 bounding boxes | **PASS** | 4.370 boxes trong split chính thức; 1.380 boxes bảo lưu an toàn ở trạng thái UNREVIEWED. Không gian lận dữ liệu. |
| 3 | Chuyển giao Backbone SSDLite 296 keys | **PASS** | `max_abs_diff = 0.0` tính trực tiếp từ tệp checkpoint vào bộ nhớ. Báo cáo tại `backbone_transfer_audit.json`. |
| 4 | Cấu hình kiến trúc SSDLite & YOLOv8n chuẩn 10 lớp | **PASS** | SSDLite num_classes=11 (10 fg + 1 bg). YOLOv8n depth=0.33, width=0.25, nc=10. |
| 5 | Smoke Training thật SSDLite320 trên GPU | **PASS** | Loss giảm 9.93 -> 6.13; Val mAP tăng; Checkpoint lưu tại `smoke-ssdlite320/weights/best.pt`. |
| 6 | Smoke Training thật YOLOv8n trên GPU | **PASS** | Loss giảm ổn định; Checkpoint lưu tại `smoke-yolov8n/weights/best.pt`. |
| 7 | Nạp checkpoint và suy luận trên ảnh Validation thật | **PASS** | Xuất ảnh có Bbox và nhãn tại `smoke_ssdlite_real_val_inference.png` và `smoke_yolov8n_real_val_inference.png`. |
| 8 | Tương tác và kiểm chứng Web App bằng MCP | **PASS** | 5 ảnh chụp màn hình kiểm chứng upload, chọn 3 model, chỉnh threshold, đếm vật thể và tải ảnh kết quả. |
| 9 | Pipeline Tăng cường dữ liệu Albumentations | **PASS** | Hoàn thành 4 chiến lược trong `src/detection/augmentation.py`. Xuất ảnh mẫu đồng bộ bbox tại `augmentation_samples/`. |
| 10 | Tăng cường dữ liệu Copy-Paste [5] | **BLOCKED** | Bị chặn do thiếu polygon mask segmentation. Tuyệt đối không dùng crop hình chữ nhật để thay thế. |
| 11 | Quy trình gán nhãn chéo trên nền tảng CVAT | **BLOCKED** | Chưa có tài khoản/project CVAT và file xuất nhãn. Dự kiến thực hiện trong đợt thu thập ảnh TP.HCM (Tuần 6–7). |
| 12 | Ảnh chụp thực tế tại địa bàn TP.HCM | **BLOCKED** | Chưa tổ chức chụp thực địa. Cần kế hoạch thu thập 200–300 ảnh cho các lớp `battery`, `clothes`, `shoes`. |
