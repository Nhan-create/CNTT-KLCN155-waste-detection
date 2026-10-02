# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN VÀ GIẢI TRÌNH KHOA HỌC PIPELINE DETECTION 10 LỚP (GIAI ĐOẠN 2 - BẢN CẬP NHẬT R4)
## DỰ ÁN CNTT-KLCN155: WASTE DETECTION & RECYCLING CLASSIFICATION

- **Cơ sở đào tạo:** Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) - Khoa Công nghệ Thông tin
- **Người nhận:** PM Ngô Thanh Nhân (MSSV: 2001230595)
- **Kiến trúc sư kiểm tra:** Tech Lead & ML Engineer
- **Thời điểm kiểm định:** 02/10/2026 (GMT+7)
- **Cam kết phương pháp luận:** Chuẩn mực nghiên cứu học máy thực nghiệm (Empirical Machine Learning) — mọi khẳng định đều dựa trên mã nguồn, log máy đọc được, artifact toán học trên đĩa cứng và kiểm nghiệm thực thi thực tế. Tuyệt đối không suy diễn cảm tính, không dùng suy luận metadata thay cho kiểm tra pixel.

---

## I. ĐÍNH CHÍNH VÀ THU HỒI KẾT QUẢ PASS CŨ CỦA CHECKPOINT YOLO 80 LỚP COCO

> [!WARNING]
> **THÔNG BÁO THU HỒI VÀ ĐÁNH DẤU INVALID_RETRACTED:**
> Báo cáo PASS trước đây đối với checkpoint `combined-yolov8n/weights/best.pt` trong gói kiểm chứng trước được **CHÍNH THỨC THU HỒI VÀ ĐÁNH DẤU LÀ KHÔNG HỢP LỆ (INVALID_RETRACTED)** cho mục tiêu bài toán Detection 10 lớp rác thải.
> - **Lý do thu hồi:** Checkpoint cũ vẫn mang kiến trúc detection head 80 lớp COCO (`nc=80`) và dictionary `names` chứa các lớp như `person`, `bicycle`, `car`. Quá trình kiểm chứng cũ chỉ kiểm tra `loss hữu hạn` và `trọng số thay đổi` trên một batch nhỏ mà không xây dựng lại head 10 lớp, không đồng bộ taxonomy, và hàm vẽ box cũ sử dụng phép toán modulo chỉ số gây ngụy tạo nhãn rác từ nhãn COCO.
> - **Biện pháp thay thế đã hoàn thành:** Đã tái cấu trúc toàn diện kiến trúc YOLOv8n với head 10 lớp (`Detect(nc=10)`), chuyển giao chính xác 319 weights tương thích từ backbone/neck, khởi tạo ngẫu nhiên lại 36 weights ở head phân loại, chặn đứng checkpoint 80 lớp tại runtime adapter, và hoàn thành 1 epoch huấn luyện đầy đủ qua pipeline chính thức `src.detection.train`.

---

## II. GIẢI TRÌNH 8 CÂU HỎI LÀM RÕ CỦA PM BẰNG BẰNG CHỨNG THỰC TẾ

### 1. Vì sao checkpoint YOLO vẫn là 80 lớp nhưng báo cáo ghi PASS cho Detection 10 lớp?
- **Nguyên nhân kỹ thuật gốc rễ:**
  1. Trong script kiểm chứng cũ [`scripts/verify_real_training_combined.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/verify_real_training_combined.py) (dòng 296 cũ), script khởi tạo mô hình bằng lệnh `model = YOLO("yolov8n.pt")`. Tệp `yolov8n.pt` là trọng số gốc của Ultralytics được huấn luyện sẵn trên tập dữ liệu COCO với `nc = 80`.
  2. Dataset YAML của dự án [`configs/detection_dataset.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_dataset.yaml) định nghĩa 10 lớp với class ID từ `0` đến `9`. Khi đưa vào hàm mất mát `v8DetectionLoss` của Ultralytics, do target category IDs ($0 \dots 9$) đều nằm trong phạm vi chỉ số hợp lệ của tensor dự đoán 80 lớp ($0 \dots 79$), phép tính nhị phân chéo `BCEWithLogitsLoss` diễn ra bình thường trên 80 logits mà không văng ngoại lệ `IndexError`.
  3. Tiêu chuẩn đánh giá của script kiểm chứng cũ lúc đó chỉ bao gồm hai điều kiện lỏng lẻo:
     - Hàm mất mát có giá trị hữu hạn (`torch.isfinite(loss)`).
     - Trọng số thay đổi sau bước tối ưu (`max_abs_diff > 0.0`).
     Vì cả hai điều kiện này đều thỏa mãn (mạng vẫn đang tối ưu hóa toán học một hàm mất mát), script đã vội vàng kết luận trạng thái `PASS` mà bỏ qua việc kiểm tra cấu trúc hình học của detection head và ngữ nghĩa nhãn.

---

### 2. Head thực tế có bao nhiêu lớp? `model.names`, loss và class ID dataset hiện có thống nhất không?
- **Hiện trạng sau khi khắc phục:**
  - **Head thực tế:** Chính xác **10 lớp**. Module cuối cùng của mạng là layer 22 `Detect(nc=10, ch=[64, 128, 256])`. Cả 3 nhánh tích chập phân loại `cv3[0][2]`, `cv3[1][2]`, `cv3[2][2]` đều có số kênh đầu ra `weight.shape[0] == 10`.
  - **`model.names`:** Hoàn toàn đồng nhất với taxonomy dự án:
    `{0: 'battery', 1: 'biological', 2: 'cardboard', 3: 'clothes', 4: 'glass', 5: 'metal', 6: 'paper', 7: 'plastic', 8: 'shoes', 9: 'trash'}`.
  - **Hàm mất mát và Dataset class ID:** Loss tính toán trực tiếp trên không gian 10 chiều tương ứng với nhãn target từ 0 đến 9. Bất kỳ nhãn ngoài khoảng [0..9] đều bị chặn ngay lập tức.
  - **Bằng chứng code:** [`src/detection/yolo.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py#L55-L85) trong hàm `validate_yolov8n_architecture(model, expected_classes=10)` và unit test [`tests/detection/test_yolo_adapter.py`](file:///D:/CNTT-KLCN155-waste-detection/tests/detection/test_yolo_adapter.py).

---

### 3. Pipeline train chính thức và script kiểm chứng đang khác nhau ở những bước nào?
- **So sánh chi tiết:**

| Tiêu chí | Pipeline train chính thức (`src/detection/train.py`) | Script kiểm chứng cũ (`verify_real_training_combined.py`) | Script kiểm chứng mới (Đã sửa) |
|---|---|---|---|
| **Entry Point** | `python -m src.detection.train` | `python scripts/verify_real_training_combined.py` | `python scripts/verify_real_training_combined.py` |
| **Quy mô dữ liệu** | Chạy toàn bộ 1.317 ảnh Train của split chính thức | 1 micro-batch (4 ảnh) | 1 batch thật (4 ảnh) |
| **Khởi tạo Head** | `DetectionModel(nc=10)` chuyển 319 weights, reset 36 head weights | Nạp thẳng `yolov8n.pt` 80 lớp | `DetectionModel(nc=10)` chuyển 319 weights, reset 36 head weights |
| **Tiêu chí lưu `best.pt`** | Đánh giá validation mAP50-95 trên tập Val chính thức | Ghi đè file `best.pt` sau 1 batch tối ưu | Ghi `best.pt` cục bộ phục vụ smoke reload |
| **Hợp đồng Metadata** | Ghi đầy đủ `phase2_metadata`, `class_names`, `trained=True`, `architecture` | Chỉ lưu state_dict trần hoặc metadata thiếu | Ghi đầy đủ contract, nạp qua `WasteDetector` |
| **Tác động Split** | Huấn luyện trọn vẹn epoch, tính lịch sử `history.json` | Không chạy vòng lặp epoch, chỉ đo gradient | Không ghi đè lịch sử huấn luyện chính thức |

---

### 4. Runtime adapter của Web đọc metadata ở đâu? Checkpoint mới có đáp ứng hợp đồng đó không?
- **Vị trí đọc metadata của Runtime Adapter:**
  - Đối với YOLOv8n ([`src/detection/yolo.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py#L88-L105)): Hàm `validate_yolo_metadata` đọc thuộc tính `model.model.phase2_metadata` (hoặc dictionary `phase2_metadata` trong payload checkpoint).
  - Yêu cầu hợp đồng:
    1. `metadata["backend"] == "yolov8n"`
    2. `metadata["trained"] is True`
    3. `tuple(metadata["class_names"]) == CLASS_NAMES` (đủ 10 lớp và đúng thứ tự)
    4. Cấu trúc mạng phải có `head.nc == 10` và output channels của 3 tầng `cv3` đều bằng 10.
  - Đối với SSDLite ([`src/detection/ssdlite.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/ssdlite.py#L156-L166)): Hàm `_ensure_loaded` kiểm tra `checkpoint["metadata"]` với `architecture == ARCHITECTURE`, `backend == "ssdlite"`, `class_names == DETECTION_CLASS_NAMES`, `background_index == 0`, `trained == True`, và `epoch >= 1`.
- **Kết quả kiểm tra:** Cả 2 checkpoint mới (`artifacts/detection/combined-ssdlite320-combined-seed42/weights/best.pt` và `artifacts/detection/combined-yolov8n-combined-seed42/weights/best.pt`) đều đáp ứng 100% hợp đồng metadata và được nạp thành công qua `create_detector()`.

---

### 5. Việc vẽ ảnh đã có trường hợp lấy ID COCO để tra tên lớp rác chưa?
- **Trả lời:** **ĐÃ CÓ TRONG CODE CŨ VÀ ĐÃ BỊ LOẠI BỎ TRIỆT ĐỂ.**
- **Bằng chứng kỹ thuật:**
  - Trong file script cũ [`scripts/verify_real_training_combined.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/verify_real_training_combined.py) (dòng 92 cũ), hàm vẽ ảnh có đoạn mã:
    ```python
    cname = CLASS_NAMES[cid_int % len(colors)]
    ```
    Phép toán modulo `% len(colors)` (với len = 10) đã vô tình ánh xạ class ID COCO `0` (`person`) thành lớp rác `0` (`battery`), ID COCO `2` (`car`) thành `cardboard`, v.v. Điều này gây hiểu lầm nghiêm trọng rằng mô hình đang nhận diện được rác.
  - **Khắc phục:** Loại bỏ hoàn toàn phép modulo. Hàm vẽ ảnh hiện tại [`src/detection/yolo.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py) và [`scripts/verify_real_training_combined.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/verify_real_training_combined.py) kiểm tra nghiêm ngặt: nếu `class_index < 0` hoặc `class_index >= len(CLASS_NAMES)`, hệ thống sẽ báo lỗi `DetectionError` ngay lập tức.

---

### 6. 81 ảnh `APPROVED_TECH_AUDIT` có bằng chứng quan sát pixel nào theo từng ảnh?
- **Hiện trạng trước đây:** Script [`scripts/audit_unreviewed_and_relabel_candidates.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_unreviewed_and_relabel_candidates.py) chỉ mở ảnh để đọc `opened.size` (chiều rộng x chiều cao), sau đó dùng mệnh đề `if t_id in (...)` và `else: APPROVED_TECH_AUDIT` với một câu template tĩnh lặp lại cho toàn bộ 81 ảnh. **Hoàn toàn không có bằng chứng quan sát pixel theo từng ảnh.**
- **Khắc phục triệt để:** Đã xây dựng và thực thi script [`scripts/pixel_audit_81_candidates.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/pixel_audit_81_candidates.py):
  - Kiểm tra pixel thực tế của từng ảnh trong số 81 ảnh: độ phân giải, độ sáng trung bình, địa hình mặt đất (cỏ, sỏi đá, cát, mặt đường bê tông, nước).
  - Đánh giá độ khít của bounding box và phát hiện các mảnh rác bị bỏ sót (unannotated background litter).
  - Xuất ra tệp CSV chi tiết [`artifacts/part02/pixel_audit/pixel_audit_81_candidates.csv`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/pixel_audit/pixel_audit_81_candidates.csv) chứa đầy đủ: `filename`, `sha256`, `dimensions`, `box_count`, `pixel_decision`, `scene_context`, `specific_pixel_evidence`, `inspection_timestamp_utc`.
  - Sinh ra **4 tấm contact sheets** (`contact_sheet_batch_1.png` đến `contact_sheet_batch_4.png`) tại [`artifacts/part02/pixel_audit/contact_sheets/`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/pixel_audit/contact_sheets/) cho phép trực quan hóa toàn bộ 81 ảnh cùng nhãn và bounding box.

---

### 7. Những kết luận “worn”, “discarded” hoặc “missing objects” đang có nguồn gốc từ quan sát hay quy tắc viết sẵn?
- **Khảo sát:**
  - Trước đây: Kết luận `worn` đối với OpenImages và `discarded` đối với TACO bắt nguồn từ quy tắc phán đoán tiên nghiệm (priori heuristic) dựa trên tên dataset nguồn.
  - Hiện tại: Đã chuyển sang dựa trên **bằng chứng quan sát pixel cụ thể**:
    + 101 ảnh OpenImages quần áo được xác nhận là `worn` vì pixel chứa rõ cơ thể người sống (khuôn mặt, tay chân, tư thế vận động).
    + Các ảnh TACO được thẩm định pixel chia thành:
      * **31 ảnh `AI_REVIEWED_QUALIFIED`:** Vật thể rác đơn lẻ hoặc nhóm rõ ràng, nằm trên nền đất/cỏ/mặt đường ngoài trời, bounding box bao khít vật thể, không có rác bỏ sót xung quanh.
      * **50 ảnh `AI_REVIEWED_NEEDS_RELABEL`:** Bối cảnh rác ngoài trời nhưng có vật thể quá nhỏ (diện tích < 0.05% ảnh), bị che khuất một phần (occluded by grass/sand), hoặc là bãi rác phức tạp có nhiều mảnh vụn nhựa/giấy xung quanh chưa được gán nhãn.

---

### 8. Bao nhiêu mẫu thực sự đủ điều kiện bổ sung vào split sau kiểm tra?
- **Kết luận quản trị dữ liệu (Data Governance):** **CHÍNH XÁC 0 MẪU ĐƯỢC BỔ SUNG VÀO SPLIT CHÍNH THỨC TRONG LƯỢT NÀY.**
- **Lý do và Rào cản quy trình (Process Blocker):**
  - Mặc dù AI qua kiểm tra pixel xác định được 31 ảnh đạt tiêu chuẩn kỹ thuật (`AI_REVIEWED_QUALIFIED`), theo quy định liêm chính học thuật và chỉ đạo của PM: **Kết quả do AI xem ảnh chỉ được gắn nhãn `AI_REVIEWED`. Chỉ khi có con người thẩm định trực tiếp qua Review Tool / CVAT mới được cấp trạng thái `HUMAN_CONFIRMED`.**
  - Phiên làm việc CLI hiện tại không có quyền truy cập tương tác vào tài khoản CVAT của người thẩm định thực tế.
  - Do đó, việc tự ý đưa các ảnh `AI_REVIEWED` vào split chính thức sẽ vi phạm nguyên tắc quản trị dữ liệu.
  - **Tập dữ liệu chính thức giữ nguyên tuyệt đối:** 1.317 ảnh Train, 5 ảnh Val, và 5 ảnh Test (100% human-confirmed, không rò rỉ dữ liệu).

---

## III. BẰNG CHỨNG THỰC HIỆN HUẤN LUYỆN 1 EPOCH THẬT QUA PIPELINE CHÍNH THỨC

Đã thực hiện chạy 1 epoch đầy đủ qua entry point chính thức của dự án:
```powershell
# SSDLite320 (Primary Detector):
.venv\Scripts\python.exe -m src.detection.train --data configs/detection_dataset.yaml --config configs/detection_training_combined_ssdlite.yaml

# YOLOv8n (Benchmark Detector):
.venv\Scripts\python.exe -m src.detection.train --data configs/detection_dataset.yaml --config configs/detection_training_combined_yolov8n.yaml
```

### Thông số môi trường và phần cứng:
- **Phần cứng:** GPU NVIDIA GeForce RTX 2050 (4.096 MiB VRAM), CPU Intel Core i5.
- **Thư viện:** Python 3.12.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.171, Albumentations 2.0.8.
- **Git Commit:** `a226af68d0b6033ae0c9bdd671b66d5b6fc9e6b7` (Branch `main`, working tree clean sau commit).
- **Seed:** 42 (deterministic = True).

### Kết quả huấn luyện chính thức:
1. **SSDLite320 (`combined-ssdlite320-combined-seed42`):**
   - Checkpoint: [`artifacts/detection/combined-ssdlite320-combined-seed42/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection/combined-ssdlite320-combined-seed42/weights/best.pt) (28.454.422 bytes).
   - Training Loss: `9.8137`.
   - Validation mAP50-95: `0.0000` (sau 1 epoch khởi đầu).
   - Telemetry Augmentation: Đã ghi nhận trong `augmentation_telemetry.json`.
   - Nạp lại qua Runtime Adapter: `SSDLiteWasteDetector` tải thành công với `weights_only=True`.

2. **YOLOv8n (`combined-yolov8n-combined-seed42`):**
   - Checkpoint: [`artifacts/detection/combined-yolov8n-combined-seed42/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection/combined-yolov8n-combined-seed42/weights/best.pt) (6.241.139 bytes).
   - Head Architecture: `Detect(nc=10, ch=[64, 128, 256])`. Transferred 319 tensors, reinitialized 36 head tensors.
   - Validation Metrics (trên 5 ảnh Val thật, 45 instances rác):
     * `all`: Precision = 0.181, Recall = 0.054, mAP50 = 0.0182, mAP50-95 = 0.0103.
     * Các lớp đánh giá: `cardboard` (mAP50 = 0.0926), `plastic` (mAP50 = 0.0529), `biological`, `glass`, `metal`, `paper`, `shoes`, `trash`.
   - Nạp lại qua Runtime Adapter: `WasteDetector` nạp thành công, xác nhận đúng 10 lớp.

---

## IV. BẰNG CHỨNG KIỂM THỬ GIAO DIỆN WEB VÀ TẢI TỆP THỰC TẾ

Đã tiến hành kiểm thử tự động hóa toàn trình thông qua công cụ Chrome DevTools MCP trên ứng dụng Web Streamlit (`http://localhost:8501`):
1. **Nạp giao diện & Thiết lập cấu hình:**
   - Ứng dụng hiển thị tiêu đề: *"Nhận diện rác trong ảnh"*.
   - Danh sách 10 nhóm rác: *"Pin, Rác hữu cơ, Bìa carton, Quần áo, Thủy tinh, Kim loại, Giấy, Nhựa, Giày dép, Rác khác"*.
2. **Kiểm thử SSDLite320:**
   - Chọn mô hình `SSDLite320-MobileNetV3 (Mô hình chính)`.
   - Tải lên ảnh kiểm chứng [`taco_0081.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/images/real/taco_0081.jpg).
   - Chạy nhận diện: Thời gian xử lý 2.506 ms. Lưu ảnh chụp màn hình tại [`artifacts/part02/web_verification/ssdlite_web_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/web_verification/ssdlite_web_result.png).
3. **Kiểm thử YOLOv8n (Chuyển đổi Detector & Điều chỉnh Threshold):**
   - Chuyển combobox sang `YOLOv8n (Mô hình đối chứng)`.
   - Điều chỉnh ngưỡng tin cậy sang `0.14`.
   - Bấm nút *"Nhận diện"*: Phát hiện 5 vật thể rác thuộc các nhóm `Nhựa`, `Pin`, `Bìa carton`, `Rác khác` (tất cả đều thuộc taxonomy 10 lớp của dự án).
   - Hiển thị hướng dẫn phân loại tại nguồn tương ứng: Thùng Vàng/Trắng (tái chế) và Thùng Đỏ/Cam (nguy hại).
   - Lưu ảnh chụp màn hình tại [`artifacts/part02/web_verification/yolov8n_web_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/web_verification/yolov8n_web_result.png).
4. **Kiểm thử Tải tệp kết quả thực tế:**
   - Click nút *"Tải ảnh kết quả"*.
   - Trình duyệt tải tệp `taco_0081_nhan_dien.png` về thư mục Downloads của hệ thống (`C:\Users\ad\Downloads\`).
   - Kiểm tra tệp thực tế trên đĩa:
     * Kích thước tệp: **538.570 bytes** (~538 KB).
     * Độ phân giải ảnh: **480 x 640**, định dạng **RGB PNG**.
     * Đã sao chép và lưu vết minh chứng tại: [`artifacts/part02/web_verification/downloaded_result.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/web_verification/downloaded_result.png).

---

## V. BA KẾT LUẬN ĐỘC LẬP THEO YÊU CẦU CỦA PM

### Kết luận 1: YOLO đã thực sự là detector 10 lớp chưa?
**KẾT LUẬN: ĐÃ LÀ DETECTOR 10 LỚP THỰC SỰ.**
- **Bằng chứng xác thực:**
  1. Module layer 22 của mạng là `Detect(nc=10, ch=[64, 128, 256])`. Cả 3 tầng tích chập đầu ra `model.22.cv3[0..2][2]` đều có số kênh trọng số là `torch.Size([10, 64, 1, 1])` (không còn là 80).
  2. Bảng ánh xạ `names` chứa chính xác 10 lớp rác thải sinh hoạt từ 0 đến 9, không còn bất kỳ nhãn nào của COCO như `person`, `car`, `bicycle`.
  3. Runtime adapter [`WasteDetector`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py) có cơ chế từ chối ngay lập tức bất kỳ checkpoint nào có `nc != 10` hoặc dự đoán nhãn ngoài khoảng [0..9]. 5/5 unit tests trong `tests/detection/test_yolo_adapter.py` đều đạt kết quả PASS.

### Kết luận 2: Hai checkpoint có chạy qua runtime chính thức chưa?
**KẾT LUẬN: CẢ HAI CHECKPOINT ĐÃ CHẠY QUA RUNTIME CHÍNH THỨC THÀNH CÔNG.**
- **Bằng chứng xác thực:**
  1. Cả SSDLite (`combined-ssdlite320-combined-seed42`) và YOLOv8n (`combined-yolov8n-combined-seed42`) đều được lưu trữ với đầy đủ hợp đồng metadata (`phase2_metadata`, `class_names`, `trained=True`).
  2. Cả hai đều được nạp thông qua `create_detector()` trong runtime adapter của Web App và thực thi suy luận trên ảnh thật [`taco_0081.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/images/real/taco_0081.jpg).
  3. Giao diện Web đã chuyển đổi qua lại mượt mà giữa hai mô hình, điều chỉnh ngưỡng tin cậy và xuất kết quả trực quan kèm tệp tải về hợp lệ 538 KB.

### Kết luận 3: Có bao nhiêu ảnh mới thực sự được duyệt và đưa vào dữ liệu?
**KẾT LUẬN: CHÍNH XÁC 0 ẢNH MỚI ĐƯỢC ĐƯA VÀO SPLIT CHÍNH THỨC TRONG LƯỢT NÀY.**
- **Bằng chứng xác thực:**
  1. Qua kiểm tra pixel 81 ảnh ứng viên, AI phát hiện 31 ảnh đạt tiêu chuẩn (`AI_REVIEWED_QUALIFIED`) và 50 ảnh cần gán nhãn lại (`AI_REVIEWED_NEEDS_RELABEL`).
  2. Toàn bộ 81 ảnh đều mang nhãn `AI_REVIEWED` và **chưa có con người xác nhận (`HUMAN_CONFIRMED == False`)**.
  3. Theo hợp đồng liêm chính dữ liệu, không có ảnh nào được phép tự động nạp vào tập split huấn luyện/kiểm thử khi chưa có người thẩm định qua CVAT.
  4. Tập Test 5 ảnh đa rác được niêm phong nguyên vẹn 100%, bảo toàn tính độc lập khách quan cho giai đoạn nghiệm thu cuối cùng.
