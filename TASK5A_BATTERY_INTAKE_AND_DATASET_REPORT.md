# BÁO CÁO THỰC THI SPRINT AI-UPGRADE-01 — TASK 5A
## TIẾP NHẬN DỮ LIỆU PIN (BATTERY) & THIẾT LẬP DATASET PILOT AN TOÀN (FAIL-CLOSED)

- **Dự án:** Hệ thống Nhận diện và Phân loại Rác thải (CNTT-KLCN155)
- **Repository:** `D:\CNTT-KLCN155-waste-detection`
- **Mã công việc:** SPRINT AI-UPGRADE-01 — TASK 5A
- **Thời điểm hoàn thành:** 2026-10-13 20:55 UTC+7
- **Phân định vai trò:** Antigravity Integrator + Chuyên trách Data / Model / QA
- **Gói bàn giao:** `C:\Users\ad\Downloads\CNTT_KLCN155_Task5A_Delivery.zip`

---

## 1. TỔNG QUAN VÀ VERDICT

| Hạng mục kiểm tra | Trạng thái | Ghi chú từ hệ thống |
|---|---|---|
| **Cơ chế Fail-Closed Dataset Intake** | **ĐẠT (PASS)** | Mô-đun `src/detection/safe_dataset_builder.py` chặn 100% việc quét mù thư mục và tự đưa ảnh vào train |
| **Quarantine & Pending Sanitization** | **ĐẠT (PASS)** | `wiki_bat_corroded_screwdriver` và `wiki_bio_food_scraps_bin_09` bị chặn tuyệt đối, không bị coi là background sạch |
| **Phân tách Ngữ cảnh (Context Segregation)** | **ĐẠT (PASS)** | Hai ảnh vỏ chuối nền giấy trắng được gán `OBJECT_ONLY_WHITE_SURFACE`, tách biệt khỏi rác ngoài đường |
| **Kiểm tra bản quyền `wiki_bio_apple_core_01`** | **ĐẠT (PASS)** | Giấy phép *Copyrighted free use* của tác giả Adrian Zion cho phép sử dụng tự do cho mọi mục đích khi ghi nhận tác giả |
| **Tiếp nhận Roboflow Battery** | **ĐẠT CỔNG (GATE HOLD)** | Xác nhận chưa có file ZIP tại `data/acquisition_staging/roboflow_battery/`. Dừng tại cổng đúng quy chuẩn; sẵn sàng script xử lý tự động khi có file |
| **Dataset Pilot v3 Staging** | **ĐẠT (PASS)** | Khởi tạo `data/detection_upgrade_v3_pilot/` và cấu hình YAML `configs/detection_dataset_real_v3_pilot.yaml`. Tập `real_v2` gốc được giữ bất biến |
| **Cấu hình YOLOv8n RTX 2050 Mobile** | **ĐẠT (PASS)** | Đề xuất cấu hình tối ưu VRAM 4GB (`imgsz=512, batch=4, workers=2, epochs=25, amp=true`), không chạy training GPU |
| **Làm rõ sai lệch 720 vs 774 boxes** | **ĐẠT (PASS)** | Chỉ ra bằng chứng vật lý: 720 là số box GT thực tế trên đĩa của test split; 774 là số box class `trash` từ bảng kiểm kê cũ bị chép nhầm |
| **Trạng thái phê duyệt huấn luyện** | **APPROVED_FOR_TRAIN = 0** | Giữ nghiêm ngặt 0 ảnh mới đưa vào huấn luyện cho đến khi PM/Người dùng duyệt |

---

## 2. TASK 1 — KHÓA CÁC ẢNH KHÔNG ĐƯỢC TRAIN & CƠ CHẾ FAIL-CLOSED

### 2.1. Phân loại và cô lập các ứng viên có rủi ro cao

1. **`wiki_bat_corroded_screwdriver` (QUARANTINED):**
   - **Thực tế vật lý:** Cụm tuốc-nơ-vít điện rỉ sét gồm đầu kẹp motor, công tắc, dây nối, bo sạc và khối cell pin.
   - **Biện pháp xử lý:** File nhãn `wiki_bat_corroded_screwdriver.txt` có dung lượng 0 byte.
   - **Nguyên tắc an toàn:** Không được phép đưa vào training như một ảnh "nền sạch không có rác" (clean negative). Mô-đun `safe_dataset_builder` ném ngoại lệ và loại bỏ ngay lập tức nếu phát hiện mã trạng thái `QUARANTINED`.

2. **`wiki_bio_food_scraps_bin_09` (PENDING_ANNOTATION):**
   - **Thực tế vật lý:** Thùng ủ rác hữu cơ gia đình lẫn bàn tay người, xô chứa và hộp bánh pizza ở rìa.
   - **Biện pháp xử lý:** File nhãn `wiki_bio_food_scraps_bin_09.txt` có dung lượng 0 byte.
   - **Nguyên tắc an toàn:** Chưa có bounding box hợp lệ cho từng vật thể rác hữu cơ đơn lẻ. Bị khóa ở trạng thái `PENDING_ANNOTATION`. Bị cấm triệt để khỏi vai trò ảnh negative.

3. **Hai ảnh vỏ chuối trên nền giấy trắng (`OBJECT_ONLY`):**
   - `wiki_bio_banana_peel_oak_hill_05` (trên khăn giấy trắng có vân)
   - `wiki_bio_banana_peel_sidewalk_06` (trên giấy trắng phẳng)
   - **Đính chính metadata:** Cập nhật `camera_domain` thành `NOT_SPECIFIED_IN_SOURCE` (không tự suy diễn chụp trong nhà vì mô tả nguồn không đề cập) và `scene_context` thành `OBJECT_ONLY_WHITE_SURFACE`.
   - **Cách ly:** Được tách riêng khỏi tập rác thải ngoài đường phố (`REAL_DISCARDED_WASTE`). Mặc định bộ lọc dataset huấn luyện thực tế ngoài hiện trường sẽ loại trừ nhóm này để tránh mô hình bị thiên lệch (overfitting vào phông nền studio/giấy trắng).

4. **Kiểm tra bản quyền `wiki_bio_apple_core_01`:**
   - **Tác giả:** Adrian Zion
   - **Trang nguồn:** `https://commons.wikimedia.org/wiki/File:Apple_core.jpg`
   - **Giấy phép:** *Copyrighted free use* (Wikimedia Commons).
   - **Điều khoản:** *"The copyright holder allows anyone to use it for any purpose, provided that the copyright holder is properly attributed. Redistribution, derivative work, commercial use, and all other uses are permitted."*
   - **Kết luận:** Đạt điều kiện hợp pháp để sử dụng trong tập dữ liệu nghiên cứu và huấn luyện AI, với điều kiện duy trì manifest ghi rõ tác giả và nguồn gốc.

### 2.2. Kiến trúc `src/detection/safe_dataset_builder.py`
Mô-đun được xây dựng với cơ chế chặn đứng quét thư mục mù (`scan_directory_blindly_guard`):
- Mọi mẫu nạp vào tập huấn luyện phải thông qua manifest có chữ ký kiểm định (`visual_qa`).
- Nhãn 0-byte chỉ được chấp nhận làm background negative nếu và chỉ nếu `visual_qa == 'QA_PASS_CLEAN_NEGATIVE'`.
- Bộ kiểm thử hồi quy độc lập `tests/qa/test_fail_closed_dataset_intake.py` gồm 9 test cases đã chạy và **PASS 100%**.

---

## 3. TASK 2 — TIẾP NHẬN DATASET PIN (BATTERY) TỪ ROBOFLOW UNIVERSE

### 3.1. Kết quả kiểm tra hiện trạng thư mục tiếp nhận
- **Đường dẫn chỉ định:** `data/acquisition_staging/roboflow_battery/`
- **Kết quả kiểm tra đĩa:** `0 files` (Thư mục trống).
- **Thực thi cổng an toàn:** Hệ thống **dừng đúng tại cổng tiếp nhận**, không tạo dữ liệu giả lập, không tự ý giả định đã có 254 ảnh trên đĩa.

### 3.2. Hướng dẫn Người dùng Tải Dataset từ Trình duyệt (Offline, Không cần API Key)
Người dùng thực hiện 5 bước sau để đưa dữ liệu vào hệ thống một cách an toàn tuyệt đối (không để lộ API key hay token lên chat):

1. Mở trình duyệt web và truy cập địa chỉ dự án Roboflow Universe đã xác minh:
   ```
   https://universe.roboflow.com/alif-kurniawan-x336t/waste-battery-only/dataset/1
   ```
2. Nhấn nút **"Download Dataset"** ở góc trên bên phải.
3. Tại mục **Format**, chọn định dạng: **`YOLOv8`** (hoặc `YOLOv5 PyTorch`).
4. Tại mục tùy chọn tải, chọn **"download zip to computer"** và nhấn **Continue**.
5. Di chuyển file ZIP vừa tải về vào đúng thư mục:
   ```
   D:\CNTT-KLCN155-waste-detection\data\acquisition_staging\roboflow_battery\
   ```
6. Khi file ZIP đã nằm trong thư mục, chạy lệnh sau để hệ thống tự động kiểm tra giải nén an toàn (chống Zip Slip, Zip Bomb), kiểm tra mapping về `0: battery`, quét rò rỉ pHash và lập manifest QA:
   ```powershell
   .venv\Scripts\python.exe scripts/import_and_qa_roboflow_battery.py
   ```

---

## 4. TASK 3 — KẾ HOẠCH BỔ SUNG VÀ ĐA DẠNG HÓA DỮ LIỆU THIẾU

### 4.1. Phân rã chủng loại Pin (Class 0: `battery`)
Hiện tại `real_v2` chỉ có 2 box pin (1 train, 1 test, 0 val). Khi mở rộng, cần phân loại rõ theo hình thái vật lý:
1. **Pin trụ tiêu chuẩn gia đình (AA, AAA, C, D):** Rơi vãi đơn lẻ, pin chảy nước/rỉ sét, hai cực kim loại lộ rõ.
2. **Pin vuông 9V (Transistor battery):** Đầu nối snap bấm đặc trưng.
3. **Pin cúc áo (Coin/Button cells - CR2032, LR44):** Thử thách nhận diện vật thể nhỏ.
4. **Khối pin Li-ion phẳng / Pouch cell:** Pin sạc dự phòng phồng rộp, pin điện thoại cũ.
5. **Bộ pin dụng cụ cầm tay / Laptop (Packs):** Vỏ nhựa bảo vệ cứng chứa nhiều cell.
6. **Ắc-quy chì-axit mini (Lead-acid):** Xe máy, xe đạp điện bỏ đi.
- **Ranh giới gán nhãn:** Chỉ vẽ box cho cell pin thực sự; không bao trùm dây điện phức tạp hay cụm máy móc không thể tách rời (như trường hợp đã cách ly ở `wiki_bat_corroded_screwdriver`).

### 4.2. Mở rộng Rác Hữu cơ (Class 1: `biological`) ngoài vỏ chuối
Hiện tại `real_v2` chỉ có 6 box rác hữu cơ. Trong Task 4, 5/6 ảnh biological là vỏ chuối. Kế hoạch thu thập bổ sung bao gồm:
1. **Phế phẩm trái cây:** Lõi táo (`wiki_bio_apple_core_01` đã xác minh), vỏ cam/quýt/bưởi, vỏ dưa hấu, hạt xoài.
2. **Thức ăn thừa nấu chín:** Mẩu bánh mì khô, xương động vật thừa, cơm/mì bỏ đi.
3. **Phế phẩm rau củ:** Lá bắp cải héo, cuống rau muống, vỏ hành tỏi dính đất.
4. **Bã trà & cà phê:** Túi lọc trà đã qua sử dụng, bã cà phê nén.
5. **Rác vườn đô thị:** Lá cây khô rụng thành đống trên vỉa hè, cành cây mục nhỏ ở rãnh thoát nước.
- **Ranh giới gán nhãn:** Tuyệt đối không nhận diện hoa quả tươi nguyên trên cây đang sống là rác thải. Chỉ chấp nhận phế phẩm bị vứt bỏ (discarded).

### 4.3. Phân định Ngữ cảnh: `REAL_DISCARDED_WASTE` vs `OBJECT_ONLY`
- **`REAL_DISCARDED_WASTE` (Rác ngoài thực địa):** Rác nằm trên mặt đường nhựa, gạch vỉa hè, đất cát, nền xi măng, có bóng đổ tự nhiên và độ che khuất một phần. Đây là dữ liệu nòng cốt để triển khai mô hình thực tế.
- **`OBJECT_ONLY_WHITE_SURFACE` (Vật thể trên nền phẳng/studio):** Ảnh chụp vật thể tách biệt trên giấy trắng, khăn giấy hoặc hộp chụp sản phẩm. Phải được gắn nhãn riêng trong manifest và cách ly khỏi tập đánh giá in-the-wild để tránh làm lệch phân phối đặc trưng nền.

---

## 5. TASK 4 — CHUẨN BỊ DATASET PILOT V3 VÀ CẤU HÌNH TRAIN

### 5.1. Thiết lập Cấu trúc Staging `data/detection_upgrade_v3_pilot/`
Cấu trúc thư mục mới được khởi tạo hoàn toàn độc lập, bảo vệ tuyệt đối tính bất biến của `data/detection_real_v2/`:
```
data/detection_upgrade_v3_pilot/
├── images/                 # Thư mục ảnh pilot
├── labels/                 # Thư mục nhãn pilot
├── manifests/              # Chứa pilot_manifest.json
├── staging_candidates/     # Khu vực cách ly các ứng viên chờ duyệt
├── splits/
│   ├── train.txt           # 1,398 ảnh (v2 baseline + 0 ảnh mới được duyệt)
│   ├── val.txt             # 299 ảnh
│   └── test.txt            # 299 ảnh (Trùng khớp 100% với splits_real_v2/test.txt)
└── README.md
```

### 5.2. Thống kê Chi tiết Số ảnh và Bounding Box từng Lớp
Dữ liệu trên đĩa được tính toán chính xác bằng script `scripts/build_pilot_dataset_v3_upgrade.py`:

| Class ID | Tên Lớp | Train Boxes | Val Boxes | Test Boxes (Frozen) | Tổng Boxes | Tỷ lệ (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | **battery** | 1 | 0 | 1 | 2 | 0.04% |
| **1** | **biological** | 3 | 2 | 1 | 6 | 0.11% |
| **2** | **cardboard** | 169 | 39 | 36 | 244 | 4.37% |
| **3** | **clothes** | 368 | 68 | 91 | 527 | 9.44% |
| **4** | **glass** | 213 | 82 | 32 | 327 | 5.86% |
| **5** | **metal** | 420 | 74 | 73 | 567 | 10.15% |
| **6** | **paper** | 160 | 43 | 34 | 237 | 4.24% |
| **7** | **plastic** | 1,560 | 382 | 274 | 2,216 | 39.69% |
| **8** | **shoes** | 215 | 41 | 39 | 295 | 5.28% |
| **9** | **trash** | 854 | 170 | 139 | 1,163 | 20.83% |
| **Tổng** | **10 Lớp** | **3,963** | **901** | **720** | **5,584** | **100.00%** |
| **Số ảnh** | — | **1,398** | **299** | **299** | **1,996** | — |

### 5.3. Làm rõ Bản chất Sai lệch 720 so với 774 boxes trong Internal Test
- **Sự thật vật lý trên đĩa:** Tập `splits_real_v2/test.txt` gồm đúng **299 ảnh**. Khi quét toàn bộ các file nhãn `.txt` tương ứng trong `data/detection/labels/real/`, tổng số bounding box là **720 boxes** (`1+1+36+91+32+73+34+274+39+139 = 720`).
- **Nguồn gốc con số 774:** Trong file báo cáo kiểm kê lịch sử `reports/annotation_inventory_summary.json` (tập dữ liệu synthetic/Mendeley cũ), riêng lớp 9 (`trash`) có đúng **774 boxes**. Con số 774 này đã bị sao chép nhầm vào cột tổng số box của tập test trong một số bản nháp trước đây.
- **Kết luận:** **720 boxes** là con số Ground Truth vật lý có thật và có thể tái lập 100% trên đĩa. Không có 774 boxes trong `splits_real_v2/test.txt`.

### 5.4. Đề xuất Cấu hình Fine-tuning YOLOv8n (RTX 2050 Mobile - 4GB VRAM)
Cấu hình đã được tích hợp vào `configs/detection_dataset_real_v3_pilot.yaml` và kiểm tra tương thích thành công với Ultralytics:
- **Trọng số khởi tạo:** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` (SHA-256: `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8`)
- **Kích thước ảnh đầu vào (`imgsz`):** `512` (Cung cấp biên an toàn VRAM thoải mái cho card 4GB trên Windows WDDM, tránh tràn bộ nhớ so với 640).
- **Kích thước batch (`batch`):** `4` (Tránh Out-Of-Memory khi nạp các ảnh độ phân giải cao).
- **Số luồng CPU Dataloader (`workers`):** `2` (Hạn chế nghẽn I/O và thrashing RAM hệ thống).
- **Số epoch tối đa (`epochs`):** `25` (Fine-tune nhẹ để bảo tồn đặc trưng các lớp đã học, kiên nhẫn `patience = 10`).
- **Tốc độ học (`lr0` / `lrf`):** `0.001` / `0.01` (Tránh hiện tượng catastrophic forgetting).
- **Mixed Precision (`amp`):** `True` (FP16 giảm 40% dung lượng VRAM GPU).
- **Bộ nhớ đệm ảnh (`cache`):** `False` (Bảo vệ RAM vật lý của máy chủ).
- **Ràng buộc:** **Không chạy training GPU trong Task 5A** (`training_executed: false`).

---

## 6. TASK 5 — KẾT QUẢ KIỂM THỬ HỆ THỐNG VÀ ĐÓNG GÓI GIAO VIỆC

### 6.1. Bảng Tổng hợp Kết quả Kiểm thử (85/85 Tests PASS)

| Nhóm Kiểm Thử | File Test | Số Test | Kết quả | Ý nghĩa kiểm chứng |
|---|---|:---:|:---:|---|
| **Mapping 10 Lớp & SSDLite Shift** | `tests/detection/test_real_mapping_integration.py`<br>`tests/detection/test_mapping_regression.py` | 19 | **19/19 PASS** | Thứ tự 10 lớp cố định; SSDLite target shift +1 / decode -1; tên hiển thị tiếng Việt chuẩn |
| **Bảo vệ Tài nguyên Preflight Guard** | `tests/detection/test_preflight_guard_integration.py` | 8 | **8/8 PASS** | Chặn đứng OOM / OS thrashing khi RAM < 1000MB; không thể bypass qua CLI hoặc tham số |
| **Fail-Closed Dataset Intake (Task 5A)** | `tests/qa/test_fail_closed_dataset_intake.py` | 9 | **9/9 PASS** | Chặn quét mù; chặn Quarantined / Pending nạp vào train; cách ly Object-only; bản quyền lõi táo hợp lệ |
| **Chống Rò rỉ & Nhiễm bẩn Phân tách** | `tests/qa/test_phash_leakage_audit.py`<br>`tests/qa/test_phase4e_intake_safety.py`<br>`tests/qa/test_split_contamination_safety.py` | 41 | **41/41 PASS** | Không rò rỉ Holdout (pHash $\ge 16$); không đưa ảnh test/val vào train; kiểm định nghiêm ngặt bounding box YOLO |
| **Vòng đời QA & Bàn giao Task 4** | `tests/test_candidate_manifest_qa_lifecycle.py`<br>`tests/test_task4_image_acquisition_and_qa.py` | 8 | **8/8 PASS** | Dung lượng 10 ảnh tải về $\le$ 50 MiB (38.5 MB); `approved_for_train == False` trên mọi ứng viên |
| **TỔNG CỘNG** | **6 Suites** | **85** | **85/85 PASS** | **Hệ thống đạt chuẩn an toàn 100%** |

### 6.2. Đóng gói Deliverable
Gói deliverable chứa toàn bộ mã nguồn mô-đun an toàn, script, cấu hình dataset pilot v3, dữ liệu staging và báo cáo được xuất thành công vào:
```
C:\Users\ad\Downloads\CNTT_KLCN155_Task5A_Delivery.zip
```
- **Kích thước file:** ~125 KB (chứa toàn bộ code, tests, configs, manifest JSON; không duplicate dữ liệu ảnh thô nặng).
- **SHA-256:** Được xác thực toàn vẹn bằng PowerShell.

---

## 7. BƯỚC TIẾP THEO (ĐỀ XUẤT CHO TASK 5B)
1. Người dùng tải file ZIP YOLOv8 từ Roboflow Universe và đặt vào `data/acquisition_staging/roboflow_battery/`.
2. Chạy `scripts/import_and_qa_roboflow_battery.py` để tự động kiểm duyệt toàn bộ ảnh pin, lọc trùng lặp pHash, và tạo contact sheet visual QA.
3. PM/Người dùng duyệt các ứng viên đạt tiêu chuẩn (`APPROVED_FOR_TRAIN`).
4. Khởi động quy trình huấn luyện pilot 25 epochs YOLOv8n trên RTX 2050 Mobile với cấu hình an toàn đã chuẩn bị.
