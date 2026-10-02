# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN VÀ GIẢI TRÌNH KHOA HỌC PIPELINE DETECTION 10 LỚP (GIAI ĐOẠN 2)
## DỰ ÁN CNTT-KLCN155: WASTE DETECTION & RECYCLING CLASSIFICATION

- **Cơ sở đào tạo:** Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) - Khoa Công nghệ Thông tin
- **Người thực hiện:** Sinh viên Ngô Thanh Nhân (PM kiêm Lead, MSSV: 200123059)
- **Kiến trúc sư kiểm tra:** Tech Lead & ML Engineer
- **Thời điểm kiểm định:** 02/10/2026 (GMT+7)
- **Cam kết phương pháp luận:** Chuẩn mực nghiên cứu học máy (Empirical Machine Learning) — mọi khẳng định đều dựa trên mã nguồn, log máy đọc được, artifact toán học trên đĩa cứng và kiểm nghiệm thực thi thực tế. Tuyệt đối không suy diễn cảm tính, không dùng suy luận metadata thay cho kiểm tra pixel.

---

## 1. GIẢI TRÌNH 8 CÂU HỎI KỸ THUẬT CỐT LÕI BẰNG CODE, LOG VÀ ARTIFACT

### Câu 1: Vì sao `sample_none.png`, `sample_geometric.png` và `sample_combined.png` trong gói bàn giao trước có SHA-256 và pixel giống hệt nhau?
- **Nguyên nhân kỹ thuật gốc rễ (Root Cause):**
  1. **Do xác suất ngẫu nhiên không kích hoạt (Stochastic Bypass):**
     Trong file script thử nghiệm cũ [`test_augmentation_pipeline.py`](file:///C:/Users/ad/.gemini/antigravity-cli/brain/2c27965b-1604-457d-b2ba-985a0d9a342d/scratch/test_augmentation_pipeline.py) (dòng 50), lệnh `np.random.seed(42)` được gọi cho từng chiến lược. Tuy nhiên, thư viện `Albumentations 2.0.8` sử dụng generator nội bộ `Generator(PCG64)` độc lập chứ không phụ thuộc vào `np.random.RandomState` legacy.
     Ở chiến lược `geometric`, pipeline gồm `HorizontalFlip(p=0.5)` và `ShiftScaleRotate(p=0.5)`. Xác suất cả hai phép đều không kích hoạt là $(1 - 0.5) \times (1 - 0.5) = 25\%$.
     Khi không có phép biến đổi nào kích hoạt, Albumentations thực hiện **phép biến đổi đồng nhất (Identity Transform)**: trả về nguyên vẹn 100% mảng pixel và bounding boxes của ảnh gốc.
  2. **Thiếu cơ chế Telemetry / Trace Log ghi nhận:**
     Trước đây, pipeline không ghi log xem phép biến đổi nào đã kích hoạt thực tế (`applied=True/False`), dẫn đến việc xuất ra một mẫu ngẫu nhiên mà không hề hay biết rằng nó chưa trải qua phép biến đổi nào.
  3. **Hàm vẽ box có tính tất định (Deterministic Drawing):**
     Hàm `draw_yolo_boxes` khi vẽ cùng các ground-truth boxes lên cùng một ảnh gốc sẽ tạo ra các tệp PNG có chuỗi byte giống nhau $100\%$, dẫn đến mã băm SHA-256 hoàn toàn trùng khớp: `d6595295845ee87f34c26110e4b58787a3f9ef15ce8a92719dd149179d0e730f`.
- **Giải pháp và Khắc phục hoàn toàn:**
  - Nâng cấp `src/detection/augmentation.py` chuyển sang dùng `A.ReplayCompose` để tự động ghi lại trạng thái thực thi của từng transform (`applied: bool`, tham số góc xoay, tỷ lệ scale, ma trận biến đổi).
  - Tách bạch 2 chế độ:
    + **Chế độ bình thường (Stochastic Training Mode):** Giữ nguyên xác suất ngẫu nhiên chuẩn ($p=0.5, p=0.7$) cho training.
    + **Chế độ kiểm tra ép buộc (Forced Verification Mode, `force_apply=True`):** Đặt $p=1.0$ để kiểm chứng chắc chắn mọi phép biến đổi đều kích hoạt, kiểm tra đồng bộ tọa độ box và xuất ảnh minh chứng đối soát.

---

### Câu 2: Augmentation đã được gọi trong training loader của cả hai detector chưa? Những phép biến đổi thực tế nào chạy ở mỗi strategy?
- **Khảo sát tại commit cũ `9d385c2` / `31afdba`:**
  - **SSDLite320 ([`src/detection/ssdlite_train.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/ssdlite_train.py)):**
    Trong `YoloBoxDataset.__getitem__`, dữ liệu chỉ được đọc từ đĩa, chuyển đổi tọa độ và trả về tensor. **Hoàn toàn chưa import hay gọi hàm `apply_augmentation`**. Thực chất cả 2 epochs smoke training trước đây mới chỉ chạy ở mức baseline `none`.
  - **YOLOv8n ([`src/detection/train.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/train.py)):**
    Trong `_train_yolo`, toàn bộ tham số augmentation mặc định của Ultralytics đã bị tắt triệt để (`augment=False, mosaic=0.0, mixup=0.0, degrees=0.0, fliplr=0.0, ...`). Tuy nhiên, pipeline `apply_augmentation` của dự án cũng chưa được chèn vào `dataset.transforms` của Ultralytics.
- **Hiện trạng sau khi sửa chữa:**
  - Đã tích hợp trực tiếp `apply_augmentation` vào `YoloBoxDataset` của SSDLite320 (chỉ kích hoạt trên split `train`, không áp dụng cho `val`/`test`).
  - Đã xây dựng class wrapper `AblationAugmentationTransform` và chèn vào đầu chuỗi transform trong `Phase2Trainer.build_dataset` của YOLOv8n.
  - **Các phép biến đổi thực tế chạy ở từng strategy:**
    1. `none`: Identity transform (không biến đổi).
    2. `geometric`: `HorizontalFlip(p=0.5)`, `ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)` với `clip=True, min_visibility=0.2`.
    3. `photometric`: `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7)`, `OneOf([GaussianBlur(3,5), MotionBlur(3,5)], p=0.3)`.
    4. `combined`: Phối hợp `HorizontalFlip` + `ShiftScaleRotate` + `ColorJitter` + `OneOf(Blur)` + `Simple Copy-Paste` từ DonorBank tập Train.

---

### Câu 3: YOLO có còn augmentation mặc định ngoài chiến lược đang khảo sát không?
- **Trả lời:** **KHÔNG CÒN.**
- **Bằng chứng kỹ thuật ([`src/detection/train.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/train.py#L165-L175)):**
  - Trong `_train_yolo`, toàn bộ các cờ biến đổi ngầm của Ultralytics được ghi đè và khóa chặt bằng 0:
    ```python
    args.update(
        augment=False, multi_scale=False, close_mosaic=0, mosaic=0.0,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0, auto_augment=None, erasing=0.0
    )
    ```
  - Trong `Phase2Trainer.build_dataset`: thuộc tính `dataset.augment = False` đảm bảo Ultralytics không gọi hàm sinh mosaic/mixup nội bộ.
  - Phép xử lý duy nhất còn lại của Ultralytics là `LetterBox(new_shape=(320, 320), scaleup=False)` để đưa ảnh về kích thước chuẩn mà không làm biến dạng tỷ lệ khung hình.
  - Nhờ vậy, YOLOv8n và SSDLite320 được đối chứng **hoàn toàn đồng nhất về mặt dữ liệu đầu vào** trong ma trận thực nghiệm 2x4.

---

### Câu 4: Dữ liệu nguồn hiện có segmentation/polygon hoặc mask sử dụng được không? Có thể lấy mask hợp lệ từ nguồn hiện hữu thay vì chờ chụp thực địa không?
- **Trả lời:** **HOÀN TOÀN CÓ VÀ ĐÃ TRÍCH XUẤT THÀNH CÔNG.**
- **Bằng chứng thực nghiệm:**
  - Tệp [`data/audit/taco_annotations_raw.json`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/taco_annotations_raw.json) (dung lượng 3.022.022 bytes) chứa toàn bộ 4.784 annotations gốc của tập dữ liệu TACO ở định dạng COCO.
  - Kiểm tra đối soát 105 ảnh TACO có trên đĩa cứng: **100% (1.041/1.041 boxes) đều có trường `'segmentation'` chứa polygon points chuẩn xác**.
  - **Trích xuất DonorBank phục vụ Copy-Paste ([`src/detection/copy_paste.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/copy_paste.py)):**
    + Tuân thủ nghiêm ngặt nguyên tắc cách ly dữ liệu: **CHỈ trích xuất từ 12 ảnh TACO thuộc tập `train`** (`manifest.jsonl` split == 'train'). Tuyệt đối không lấy donor từ `val` hoặc `test`.
    + Đã giải quyết bài toán sai lệch kích thước: tính tỷ lệ scale chính xác giữa ảnh trên đĩa và ảnh gốc metadata (`sx = w_img / raw_w`, `sy = h_img / raw_h`).
    + Trích xuất được **68 foreground donor objects có mask nhị phân hoàn chỉnh** bao gồm 8 lớp: `battery` (1), `biological` (2), `cardboard` (5), `glass` (4), `metal` (6), `paper` (6), `plastic` (35), `trash` (9).
    + Có cơ chế **xử lý che khuất (Occlusion Handling)**: tự động loại bỏ bounding box nền nếu bị vật thể dán đè che khuất $\ge 80\%$ diện tích.
  - Như vậy, nhánh Copy-Paste trong chiến lược `combined` đã có thể vận hành hợp lệ ngay trên dữ liệu hiện hữu mà không cần chờ chụp thực địa.

---

### Câu 5: Trong 235 ảnh ngoài split, bao nhiêu UNREVIEWED, REJECTED và NEEDS_RELABEL? Bao nhiêu còn phù hợp để thẩm định?
- **Số liệu kiểm toán thực tế trên đĩa cứng:**
  - Tổng số ảnh thực tế trên đĩa ([`data/detection/images/real/`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/images/real/)): **257 ảnh** (105 ảnh TACO + 152 ảnh OpenImages).
  - Số ảnh đã qua thẩm định và đưa vào split chính thức ([`data/detection/manifest.jsonl`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/manifest.jsonl)): **22 ảnh** (toàn bộ từ TACO).
  - Số ảnh thực tế ngoài split: $257 - 22 = \mathbf{235\text{ ảnh}}$.
- **Phân rã trạng thái chi tiết của 235 ảnh ngoài split:**
  1. **114 ảnh OpenImages đã qua kiểm toán thị giác (Visual Audit):**
     - Lưu tại [`artifacts/part02/real_data_audit/real_images_audit_table.csv`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/real_data_audit/real_images_audit_table.csv) với kết luận `AUDIT_OI_PENDING_PM_SCOPE_DECISION`:
       + **101 ảnh `WORN_BY_PERSON`:** Quần áo và giày dép đang được con người mặc khi sinh hoạt/dạo phố. **HOÀN TOÀN KHÔNG PHÙ HỢP** làm dữ liệu rác thải (nếu đưa vào sẽ dạy mô hình nhận diện quần áo người bình thường là rác).
       + **10 ảnh `COMMERCIAL_PRODUCT_STUDIO`:** Ảnh chụp sản phẩm giày/áo quảng cáo studio trên nền trắng sạch. **KHÔNG PHÙ HỢP**.
       + **2 ảnh `UNRESOLVED`:** Bối cảnh phức tạp cần PM xem xét.
       + **1 ảnh `APPEARS_DISCARDED_OUTDOORS`:** Giày dép vứt ngoài trời, **PHÙ HỢP ĐỂ THẨM ĐỊNH VÀO RÁC THẢI**.
  2. **103 ảnh TACO và OpenImages ở trạng thái `UNREVIEWED`:**
     - Nằm trong [`data/audit/real_detection_source_manifest.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/real_detection_source_manifest.csv).
     - **Toàn bộ 103 ảnh này ĐỀU PHÙ HỢP ĐỂ THẨM ĐỊNH** qua Review Tool / CVAT để bổ sung cảnh rác thực tế ngoài trời.
  3. **14 ảnh `REJECTED`:**
     - Bị từ chối vì lý do nhãn mơ hồ, ảnh mờ nhòe hoặc không thuộc phạm vi rác thải sinh hoạt. Cần giữ nguyên lý do loại trừ, không đưa vào tập dữ liệu.
  4. **4 ảnh `NEEDS_RELABEL`:**
     - Các ảnh ly cốc, chai lọ có bối cảnh phù hợp nhưng nhãn bbox bị thiếu hoặc sai lệch. **CẢ 4 ẢNH ĐỀU PHÙ HỢP ĐỂ CHỈNH SỬA VÀ ĐƯA VÀO DỮ LIỆU**.
- **Tổng kết khả năng thẩm định:** Có $103 + 4 + 1 = \mathbf{108\text{ ảnh}}$ hoàn toàn phù hợp để thẩm định và tái sử dụng cho bài toán phát hiện rác thải.

---

### Câu 6: Nguồn nào có khả năng bổ sung rác quần áo, pin và giày dép đúng ngữ cảnh?
- **Phân tích bản chất thiếu hụt:**
  - Khảo sát trực tiếp trong toàn bộ 4.784 annotations của TACO gốc: `Battery` chỉ có đúng **2 annotations**, `Shoe` chỉ có **7 annotations**, và TACO hoàn toàn không có danh mục `Clothing`.
  - Bộ dữ liệu OpenImages chứa nhiều nhãn `Clothing` và `Footwear` nhưng hơn $95\%$ là ảnh con người đang mặc đồ hoặc sản phẩm thời trang thương mại.
- **Nguồn bổ sung đúng ngữ cảnh rác thải sinh hoạt (Discards & Litter Context):**
  1. **Pin cũ (`battery`):**
     - Chụp thực địa tại các điểm đặt thùng thu gom pin nguy hại chuyên dụng tại TP.HCM (chuỗi siêu thị Co.opmart, Go!, các sảnh giảng đường HUIT, chung cư cao tầng).
     - Tìm kiếm các dataset rác thải điện tử công khai: *E-Waste Detection Dataset* hoặc các bộ ảnh chụp thiết bị pin nhỏ thải bỏ.
  2. **Giày dép cũ (`shoes`):**
     - Chụp thực tế dép tổ ong, dép lê, giày rách vứt ở các bãi đất trống, lề đường, thùng rác công cộng, cống rãnh tại TP.HCM.
     - Lọc thủ công các ảnh OpenImages có bối cảnh "discarded footwear" (loại bỏ ảnh người mang giày).
  3. **Quần áo thải bỏ (`clothes`):**
     - Chụp thực địa các bao tải quần áo cũ, vải vụn, giẻ lau tại các điểm tập kết rác dân cư hoặc điểm thu nhận quần áo cũ từ thiện tại TP.HCM.
     - Tuyệt đối không cào dữ liệu ảnh người mẫu hoặc người đi đường trên mạng vì sẽ làm sai lệch phân phối đặc trưng học của mô hình.

---

### Câu 7: Log train thô, config, checkpoint và kết quả đánh giá smoke hiện nằm ở đâu? Vì sao chưa có trong ZIP trước?
- **Vị trí vật lý cụ thể trên đĩa:**
  - **SSDLite320 (Mô hình chính):**
    + Weights: [`artifacts/detection_smoke/smoke-ssdlite320/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-ssdlite320/weights/best.pt) (28.456.918 bytes, SHA-256: `c9a079d630bdff7919d476c960784af885662fd12fa28fee7c8fef03bd551c9b`).
    + Checkpoint last: [`artifacts/detection_smoke/smoke-ssdlite320/weights/last.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-ssdlite320/weights/last.pt) (28.456.918 bytes).
    + Metrics & History: [`artifacts/detection_smoke/smoke-ssdlite320/history.json`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-ssdlite320/history.json), [`training_summary.json`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-ssdlite320/training_summary.json).
    + Config: [`configs/detection_training_smoke_ssdlite.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_training_smoke_ssdlite.yaml).
  - **YOLOv8n (Mô hình đối chứng):**
    + Weights: [`artifacts/detection_smoke/smoke-yolov8n/weights/best.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-yolov8n/weights/best.pt) (6.208.931 bytes, SHA-256: `84b677fcb4de6cb0725973aa7de21f0a594dfea57c20c875256f393fff2a06dd`).
    + Metrics & Curves: [`artifacts/detection_smoke/smoke-yolov8n/results.csv`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/smoke-yolov8n/results.csv), `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`.
    + Config: [`configs/detection_training_smoke_yolov8n.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_training_smoke_yolov8n.yaml), `effective_train_args.json`.
- **Vì sao chưa có trong file ZIP trước:**
  - Script đóng gói trước đó ([`export_report_and_zip.py`](file:///C:/Users/ad/.gemini/antigravity-cli/brain/2c27965b-1604-457d-b2ba-985a0d9a342d/scratch/export_report_and_zip.py)) chỉ liệt kê việc sao chép các tệp báo cáo Markdown, ảnh chụp Web App và 4 tệp kiểm toán CSV/JSON nhẹ. Nó đã bỏ quên thư mục `artifacts/detection_smoke/` chứa tệp weights nhị phân và log huấn luyện máy đọc được.
  - Sơ suất này đã được khắc phục trong script đóng gói mới, bổ sung đầy đủ weights và logs vào gói bàn giao chính thức.

---

### Câu 8: Con số classifier “F1 98,3%” trong báo cáo trước lấy từ artifact nào? Có khớp kết quả Phần 1 đã nghiệm thu không?
- **Khảo sát đối chiếu nguồn gốc:**
  - Tra cứu trong toàn bộ repo: Con số "98,3%" chỉ xuất hiện duy nhất dưới dạng text tại một dòng trong báo cáo nháp trước đó mà không có bất kỳ artifact JSON nào chứng thực.
  - **Đối soát với Biên bản nghiệm thu chính thức Phần 1 ([`docs/plan/PART_01_ACCEPTANCE.md`](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/PART_01_ACCEPTANCE.md#L67-L87)):**
    + Chỉ số nghiệm thu chính thức tại checkpoint tốt nhất (Epoch 12, [`artifacts/official_run/best_model.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/best_model.pt)):
      * **Validation Accuracy: 96,18% (0.9618)**
      * **Validation Macro-Averaged F1: 95,59% (0.9559)**
      * Validation Loss: 0.6248
  - Tra cứu trong [`artifacts/official_run/official_training_metrics.json`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_metrics.json):
    `"val_accuracy": 0.9618`, `"val_macro_f1": 0.9559`.
- **Kết luận:** Con số "98,3%" là do sơ suất ghi chép trong văn bản nháp trước đó (có thể nhầm lẫn với F1 của lớp `clothes` đạt 99,47% hoặc Train Accuracy 99,9%). **Kết quả chính thức của Phần 1 đã được chuẩn hóa và đính chính lại đúng số liệu nghiệm thu: Validation Macro-F1 = 95,59% và Accuracy = 96,18%.**

---

## 2. BẢNG TỔNG KẾT KIỂM CHỨNG KỸ THUẬT (PASS / FAIL / BLOCKED)

| STT | Vấn đề / Hạng mục kiểm chứng | Nguyên nhân kỹ thuật gốc rễ | Tệp mã nguồn đã chỉnh sửa | Phép kiểm tra thực thi | Bằng chứng vật lý xác thực | Kết luận |
|:---:|:---|:---|:---|:---|:---|:---:|
| 1 | Ảnh mẫu augmentation bị trùng SHA-256 và pixel | Xác suất ngẫu nhiên $p=0.5$ không kích hoạt do Albumentations dùng `default_rng()` độc lập; identity transform giữ nguyên pixel ảnh gốc. | [`src/detection/augmentation.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/augmentation.py) | `verify_augmentation_and_loaders.py`: chạy đối chiếu Normal vs Forced mode trên 3 ảnh thật (`taco_0081`, `0082`, `0853`). | 24 ảnh mẫu khác biệt SHA-256 tại `augmentation_samples/` kèm telemetry diff pixel trong `augmentation_verification_report.json`. | **PASS** |
| 2 | Augmentation chưa được gọi trong Training Loader | File `ssdlite_train.py` và `train.py` chỉ đọc dữ liệu thô, chưa kết nối pipeline biến đổi vào Dataset. | [`src/detection/ssdlite_train.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/ssdlite_train.py)<br>[`src/detection/train.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/train.py) | Chạy 1 batch forward + backward qua SSDLite loader (`combined`) và YOLO ablation transform. | SSDLite: Loss = 12.67, GradNorm = 136.99.<br>YOLO: Ablation transform verified với boxes đồng bộ. | **PASS** |
| 3 | Tăng cường dữ liệu Simple Copy-Paste [5] | Bị báo BLOCKED do hiểu lầm toàn bộ dataset chỉ có bounding box hình chữ nhật. | [`src/detection/copy_paste.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/copy_paste.py) | Khảo sát `taco_annotations_raw.json`: trích xuất 68 donor objects có polygon mask từ 12 ảnh TACO train; xử lý che khuất (occlusion). | `test_copy_paste_synthetic_donor_and_occlusion` passed trong pytest (18/18 tests pass). | **PASS** |
| 4 | Kiểm chứng toán học chuyển giao Backbone | Cần chứng minh tính chính xác tuyệt đối khi nạp 296 feature keys từ Classifier sang SSDLite. | [`scripts/audit_backbone_transfer.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py) | Đo hiệu số tensor trực tiếp giữa checkpoint đĩa và RAM: $\max \|T_{\text{model}} - T_{\text{source}}\|$. | 253 weight/bias tensors đạt `max_abs_diff = 0.0`. 43 BatchNorm buffers. Báo cáo tại `data/audit/backbone_transfer_audit.json`. | **PASS** |
| 5 | Thống kê số keys backbone trong tài liệu | File JSON cũ còn sót chuỗi mô tả cố định "258/168 keys" dù đã chuyển giao 296 keys. | [`scripts/audit_backbone_transfer.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py) | Chạy lại audit script, tạo lại file JSON với số liệu động. | `backbone_transfer_audit.json` cập nhật: 296/308 feature keys, 253 tensors đối chiếu, không còn số 258/168. | **PASS** |
| 6 | Đính chính F1 Classifier Phần 1 | Ghi nhầm 98,3% F1 trong văn bản nháp không khớp với biên bản nghiệm thu. | [`docs/plan/PART_02_10CLASS_VERIFICATION.md`](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/PART_02_10CLASS_VERIFICATION.md) | Đối soát chéo với `PART_01_ACCEPTANCE.md` và `official_training_metrics.json`. | Khớp chuẩn 100%: Val Acc 96,18%, Val Macro-F1 95,59% (0.9559). | **PASS** |
| 7 | Tờ trình điều chỉnh đề cương | Dùng sai tên đề tài chính thức; lập luận sai rằng 6 lớp làm mất khả năng chuyển backbone. | [`docs/proposal/TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md`](file:///D:/CNTT-KLCN155-waste-detection/docs/proposal/TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md) | Soạn thảo lại tờ trình bám sát tên đề tài gốc, giải thích mở rộng bằng nhu cầu thực tế và kế hoạch dữ liệu. | Tờ trình hoàn chỉnh gửi GVHD ThS. Huỳnh Thị Châu Lan. | **PASS** |
| 8 | Bằng chứng kiểm tra tính năng tải tệp Web App | Giao diện web có nút tải nhưng chưa chứng minh tệp tải về có hợp lệ trên đĩa không. | [`streamlit_app.py`](file:///D:/CNTT-KLCN155-waste-detection/streamlit_app.py)<br>[`src/web/detection_logic.py`](file:///D:/CNTT-KLCN155-waste-detection/src/web/detection_logic.py) | Mô phỏng luồng tải ảnh: gọi `analyze_image_file`, xuất bytes ra đĩa và giải mã lại bằng Pillow. | Tệp `downloaded_annotated_sample.png` lưu hợp lệ (480x640, 396.647 bytes, format PNG). | **PASS** |
| 9 | Thẩm định 235 ảnh thực tế ngoài split | 101 ảnh OpenImages là người mặc quần áo; 103 ảnh TACO/OpenImages chưa duyệt. | [`data/audit/real_detection_source_manifest.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/real_detection_source_manifest.csv) | Phân rã 235 ảnh: 108 ảnh phù hợp thẩm định (103 UNREVIEWED + 4 NEEDS_RELABEL + 1 discarded). | Danh sách phân loại và kế hoạch thẩm định chi tiết. | **PENDING THẨM ĐỊNH** |
| 10 | Bổ sung dữ liệu thực địa TP.HCM cho 3 lớp thiếu | Thiếu bối cảnh rác thực tế ngoài trời cho `battery`, `clothes`, `shoes` trên tập Val/Test. | Kế hoạch thực địa Tuần 6–7 | Lập danh mục địa điểm và đối tượng chụp cụ thể tại TP.HCM. | Cam kết không đưa toàn bộ ảnh thực địa vào test mà chia theo cảnh độc lập (scene-based). | **BLOCKED (CHỜ CHỤP)** |

---

## 3. PHÂN BIỆT RÕ RÀNG GIỮA SMOKE TRAINING VÀ DETECTION CHÍNH THỨC

- **Bản chất của các thực nghiệm Smoke hiện tại:**
  - Hai lượt chạy 2 epochs của SSDLite320 và YOLOv8n trên GPU RTX 2050 chỉ có giá trị duy nhất là **kiểm tra thông suốt đường ống kỹ thuật (Pipeline Plumbing Verification)**:
    + Chứng minh dữ liệu nạp đúng định dạng tensor.
    + Chứng minh hàm mất mát tính toán ra giá trị hữu hạn và lan truyền ngược (backward pass) thành công.
    + Chứng minh checkpoint lưu trữ và nạp lại suy luận được trên ảnh PIL.
  - **Tuyệt đối không dùng kết quả smoke này để công bố độ chính xác nhận diện của hệ thống.** Các giá trị mAP thấp (0.007) là hoàn toàn bình thường đối với mô hình mới học 2 epochs trên vài nghìn boxes.
- **Nguyên tắc đối với tập kiểm thử cuối cùng (Final Held-out Test Set):**
  - Tập Test được **NIÊM PHONG TUYỆT ĐỐI** trong suốt quá trình nghiên cứu và tối ưu.
  - Không bao giờ dùng tập Test để:
    + Chọn checkpoint tối ưu (`best.pt`).
    + Lựa chọn chiến lược tăng cường dữ liệu (`none`, `geometric`, `photometric`, `combined`).
    + Tinh chỉnh ngưỡng confidence vận hành hay siêu tham số hợp nhất WBF.
  - Tập Test chỉ được mở ra đánh giá đúng một lần duy nhất khi toàn bộ quá trình huấn luyện và chọn mô hình trên tập Validation đã kết thúc.

---

## 4. BẰNG CHỨNG THỰC NGHIỆM VÀ KHẢ NĂNG TÁI HIỆN (REPRODUCIBILITY EVIDENCE)

Toàn bộ minh chứng kỹ thuật đã được tạo lập và lưu trữ có thể kiểm tra trực tiếp:
1. **Báo cáo kiểm chứng Augmentation & DataLoader:** [`artifacts/part02/augmentation_verification_report.json`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/augmentation_verification_report.json).
2. **Bộ ảnh đối chứng Augmentation (24 ảnh mẫu):** [`artifacts/part02/augmentation_samples/`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/augmentation_samples/).
3. **Mẫu ảnh tải về từ Web App:** [`artifacts/part02/downloaded_annotated_sample.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/downloaded_annotated_sample.png).
4. **Báo cáo nạp trọng số Backbone:** [`data/audit/backbone_transfer_audit.json`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json) và [`data/audit/backbone_transfer_keys.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_keys.csv).
5. **Checkpoints và Log Smoke Training:** [`artifacts/detection_smoke/`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/detection_smoke/).
6. **Mã nguồn và Unit Tests (18/18 Passed):** Chạy lệnh `.venv\Scripts\pytest.exe tests/detection/ -v`.
