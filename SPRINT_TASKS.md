# KẾ HOẠCH PHÂN BỔ CÔNG VIỆC THEO SPRINT (SPRINT TASKS & ROADMAP)
**Phân bổ Nhiệm vụ, Thứ tự Ưu tiên và Ma trận Trách nhiệm Multi-Agent**  
**Dự án:** `CNTT-KLCN155-waste-detection`  
**Giai đoạn:** Sprint 0 Hoàn tất — Chuẩn bị Sẵn sàng Sprint 1  
**Tác giả:** Antigravity Integrator  
**Ngày lập:** 2026-10-12  

---

## 1. KHẢO SÁT CHỨC NĂNG: ĐÃ CÓ VS CÒN THIẾU (GAP ANALYSIS)

| Thành phần hệ thống | Trạng thái hiện tại trong Repo | Khoảng trống cần hoàn thiện (Gap) |
|---|---|---|
| **Taxonomy 10 lớp** | Đã đóng băng cố định tại `src/detection/schema.py`. | Đã hoàn chỉnh. Giữ nguyên bất biến. |
| **Dữ liệu huấn luyện** | `splits_real_v2` (1.996 ảnh, 5.585 box). Đã loại bỏ sạch dữ liệu Mendeley/Synthetic. | Thiếu nghiêm trọng ảnh Pin (Recall 0%) và Rác sinh học (Recall 0%). Cần bổ sung Đợt A (100–200 ảnh). |
| **Checkpoints mô hình** | YOLOv8n E1 (`best.pt`, 5.94 MB), SSDLite320 E2 (27.14 MB), MobileNetV3 (48.57 MB, 96.18%). | Cần fine-tuning pilot trên tập dữ liệu Đợt A với cấu hình RTX 2050 (imgsz 512, batch 2-4). |
| **Kiểm thử độc lập** | 27 ảnh nghiệm thu `D:\Hinh_Anh_Rac` (57 GT objects), module `practical_metrics.py`. | Cần thu thập thêm 2 tập kiểm thử mới: Fixed Camera Holdout và Mobile Camera Holdout. |
| **Pipeline Tọa độ** | `src/detection/coordinates.py` đã có `ensure_exif_orientation`, `letterbox_image`, `scale_coords_back`. | Cần tích hợp vào vòng lặp đọc camera để xử lý xoay màn hình điện thoại tự động. |
| **Cơ chế Tracking** | `src/detection/tracking.py` đã có `TemporalDetectionSmoother`. | Cần mở rộng thêm gán `track_id` để chống đếm trùng một vật thể rác nhiều lần trong 1 giờ. |
| **Camera Cố định** | `cv2.VideoCapture(0)` đã kiểm tra hoạt động tốt trên máy trạm. | Chưa có background worker tự động reconnect khi mất kết nối USB/RTSP và hàng đợi drop frame chống lag. |
| **Camera Di động** | Có prototype WebRTC thô trong `src/web/detection_live.py`. Thiếu `streamlit-webrtc`. | Cần giải pháp truyền luồng ổn định (WebRTC hoặc Lightweight WebSocket/HTTP multipart) kết nối điện thoại. |
| **Giao diện Streamlit** | Đang có 2 Tab xử lý ảnh tĩnh (`streamlit_app.py`). | Cần bổ sung Tab 3 (Camera Cố định) và Tab 4 (Camera Di động) kèm dọn dẹp VRAM khi chuyển tab. |

---

## 2. MA TRẬN PHÂN CÔNG TRÁCH NHIỆM SUB-AGENTS (RACI MATRIX)

Tuân thủ quy chuẩn `AGENTS.md`:
- **Integrator (Main):** Điều phối, quản lý dependencies, kiểm soát tài nguyên GPU, hợp nhất code tuần tự.
- **Agent A (Data / Mapping):** Thu thập, tiền xử lý, quản lý manifest và gắn nhãn (`src/detection/schema.py`, `configs/`, `data/`).
- **Agent B (Model / Inference):** Huấn luyện mô hình, kiểm soát VRAM, tối ưu hóa suy luận (`src/detection/yolo.py`, `src/detection/ssdlite.py`, `configs/detection_training*.yaml`).
- **Agent C (Coordinates / Rendering / Web):** Xử lý luồng camera, tọa độ xoay, WebRTC/RTSP, giao diện Streamlit (`src/detection/coordinates.py`, `src/web/`, `streamlit_app.py`).
- **Agent D (QA Independent):** Giám sát chất lượng, thẩm định Ground Truth độc lập, chạy benchmark và phát hiện lỗi thực địa (`reports/qa_acceptance/`).

---

## 3. DANH MỤC NHIỆM VỤ CHI TIẾT THEO SPRINT

```
[SPRINT 0: THIẾT KẾ & CHUẨN BÓA] (HOÀN THÀNH)
  - Khảo sát hiện trạng, checkpoints, phần cứng RTX 2050
  - Biên soạn 5 tài liệu kỹ thuật cốt lõi
  - Chốt hiện trạng 4 ảnh Wikimedia Phase 4E (2 ảnh QA PASS chờ duyệt)
       |
       v
[SPRINT 1: CAMERA CORE & THU THẬP ĐỢT A] (CHUẨN BỊ TRIỂN KHAI)
  - Xây dựng Shared Camera Worker (Auto-reconnect, Drop frame queue)
  - Thu thập và Visual QA Đợt A (100–200 ảnh: Pin, Rác sinh học, Hard-Negative)
  - Tích hợp Tab Camera Cố định và Mobile Stream prototype vào Streamlit
  - Chuẩn bị môi trường Preflight cho Huấn luyện Pilot
       |
       v
[SPRINT 2: HUẤN LUYỆN PILOT & ĐỐI SOÁT TEST MỚI]
  - Huấn luyện Pilot YOLOv8n (20–40 epochs, imgsz 512, batch 2-4)
  - Đánh giá trên Holdout đóng băng 27 ảnh và 2 tập Test Camera mới
  - So sánh đối chứng với SSDLite320
       |
       v
[SPRINT 3: KIỂM THỬ THỰC ĐỊA & HOÀN THIỆN HỆ THỐNG]
  - Chạy thực nghiệm liên tục >= 1 giờ trên Camera cố định
  - Kiểm thử cơ động nhiều góc độ trên Camera điện thoại
  - Hoàn thiện báo cáo nghiệm thu và đóng gói sản phẩm
```

### 3.1 Sprint 0: Khảo sát Hiện trạng & Đặc tả Kỹ thuật (*Trạng thái: HOÀN TẤT*)
- [x] **TASK-0.1:** Khảo sát toàn diện hệ thống: Checkpoints, Dataset real v2, v3 candidate, phần cứng GPU/RAM, camera (`Integrator`).
- [x] **TASK-0.2:** Kiểm kê và chốt trạng thái 4 ảnh Wikimedia Phase 4E: 2 ảnh QA PASS giữ nguyên chờ người dùng duyệt, 2 ảnh loại bỏ (`Agent A` & `Agent D`).
- [x] **TASK-0.3:** Xây dựng `REAL_WORLD_DEPLOYMENT_SPEC.md` (`Integrator` & `Agent C`).
- [x] **TASK-0.4:** Xây dựng `DATA_COLLECTION_PLAN.md` (`Agent A` & `Integrator`).
- [x] **TASK-0.5:** Xây dựng `ANNOTATION_POLICY.md` (`Agent A` & `Agent D`).
- [x] **TASK-0.6:** Xây dựng `TRAINING_AND_EVALUATION_PLAN.md` (`Agent B` & `Agent D`).
- [x] **TASK-0.7:** Xây dựng `SPRINT_TASKS.md` (`Integrator`).

---

### 3.2 Sprint 1: Xây dựng Camera Core & Thu thập Dữ liệu Đợt A (*Chuẩn bị triển khai*)

| Mã Task | Tên Nhiệm vụ | Người phụ trách | Độ ưu tiên | Phụ thuộc | File ảnh hưởng / Đầu ra |
|---|---|:---:|:---:|:---:|---|
| **TASK-1.1** | **Trình duyệt 2 ứng viên Phase 4E:** Trình người dùng phê duyệt 2 ảnh QA PASS (`wiki_bio_banana_peel_ground`, `wiki_neg_junction_boxes`). | Integrator | **P0** | Sprint 0 | Báo cáo phê duyệt người dùng. |
| **TASK-1.2** | **Camera Core Engine:** Xây dựng `VideoStreamWorker` hỗ trợ USB Webcam và RTSP, có hàng đợi drop frame kích thước 1 và auto-reconnect exponential backoff. | Agent C | **P0** | TASK-0.3 | `src/web/camera_stream.py`, `tests/web/test_camera_stream.py`. |
| **TASK-1.3** | **Anti-Duplicate Tracking:** Nâng cấp `TemporalDetectionSmoother` sang tracker có gán `track_id` và logic chống đếm trùng vật thể tĩnh. | Agent C | **P1** | TASK-1.2 | `src/detection/tracking.py`, `tests/detection/test_tracking_ids.py`. |
| **TASK-1.4** | **Thu thập Đợt A (Batch A Pilot):** Thu thập 100–200 ảnh thật theo whitelist (Pin 40-60, Rác sinh học 40-60, Hard-Negative 30-50, Multi-waste 20-30). | Agent A | **P0** | TASK-1.1, ANNOTATION_POLICY | `data/acquisition_staging/batch_a/`, `batch_a_manifest.json`. |
| **TASK-1.5** | **Visual QA & Gán nhãn Đợt A:** Soi pixel 100% full-frame, gán nhãn snug bbox, đối soát trùng lặp SHA-256/pHash với 3.842 hash đóng băng. | Agent D & Agent A | **P0** | TASK-1.4 | `batch_a_qa_manifest.json`, visual overlay contact sheets. |
| **TASK-1.6** | **Tích hợp UI Streamlit Tab 3 (Fixed Camera):** Thêm giao diện chạy Webcam/RTSP trực tiếp trên Streamlit với cơ chế ngắt stream dọn VRAM khi đổi tab. | Agent C | **P1** | TASK-1.2, TASK-1.3 | `streamlit_app.py`, `src/web/detection_settings.py`. |
| **TASK-1.7** | **Nguyên mẫu Mobile Stream Tab 4:** Xây dựng giải pháp stream điện thoại qua trình duyệt (WebRTC / HTTP multipart fallback). | Agent C | **P1** | TASK-1.6 | `src/web/mobile_bridge.py`, giao diện mobile QR code. |
| **TASK-1.8** | **Chuẩn bị Training Preflight:** Xây dựng file cấu hình `detection_dataset_real_v3_pilot.yaml` và kiểm thử dry-run dataloader không tốn GPU. | Agent B | **P1** | TASK-1.5 | `configs/detection_training_pilot.yaml`, `tests/qa/test_pilot_dry_run.py`. |

---

### 3.3 Sprint 2: Huấn luyện Pilot & Đánh giá Hai Tập Test Mới

| Mã Task | Tên Nhiệm vụ | Người phụ trách | Độ ưu tiên | Phụ thuộc | File ảnh hưởng / Đầu ra |
|---|---|:---:|:---:|:---:|---|
| **TASK-2.1** | **Xin phép Người dùng chạy Training Pilot:** Báo cáo thông số VRAM, thời gian ước tính và xin duyệt lệnh train GPU. | Integrator | **P0** | Sprint 1 | User Approval Gate. |
| **TASK-2.2** | **Huấn luyện Pilot YOLOv8n:** Chạy fine-tune 20–40 epochs (imgsz 512, batch 2-4, AMP, AdamW) có preflight guard và OOM fallback. | Agent B | **P0** | TASK-2.1 | `artifacts/detection_pilot/yolov8n_pilot/weights/best.pt`, `train_provenance.json`. |
| **TASK-2.3** | **Huấn luyện So sánh SSDLite320:** Chạy song song độc lập (tuần tự) để đối chứng theo quy chuẩn học thuật. | Agent B | **P1** | TASK-2.2 | `artifacts/detection_pilot/ssdlite_pilot/weights/best.pt`. |
| **TASK-2.4** | **Xây dựng 2 Tập Test Mới:** Thu thập 50–100 video/frame cho Fixed Camera Holdout và Mobile Camera Holdout. | Agent D | **P0** | TASK-1.2 | `data/test_fixed_camera/`, `data/test_mobile_camera/`, GT manifests. |
| **TASK-2.5** | **Đánh giá Độc lập Toàn diện:** Chạy benchmark trên cả 3 tập (Holdout cũ 27 ảnh, Fixed Test, Mobile Test), đo mAP, Recall Pin/Bio, False Positives. | Agent D | **P0** | TASK-2.2, TASK-2.4 | `reports/qa_acceptance/PILOT_PRACTICAL_METRICS_REPORT.md`. |

---

### 3.4 Sprint 3: Kiểm thử Thực địa $\ge 1$ Giờ & Đóng gói Sản phẩm

| Mã Task | Tên Nhiệm vụ | Người phụ trách | Độ ưu tiên | Phụ thuộc | File ảnh hưởng / Đầu ra |
|---|---|:---:|:---:|:---:|---|
| **TASK-3.1** | **Thực nghiệm Camera Cố định $\ge 1$ giờ:** Giám sát liên tục điểm rác/thùng rác trong 60 phút, đo drift, false positives và rò rỉ RAM/VRAM. | Agent C & Agent D | **P0** | Sprint 2 | Log tài nguyên 1 giờ, bảng thống kê phát hiện. |
| **TASK-3.2** | **Thực nghiệm Camera Di động Ngoài thực địa:** Kiểm thử quét rác bằng điện thoại tại nhiều địa điểm, ánh sáng và góc độ khác nhau. | Agent C & Agent D | **P0** | Sprint 2 | Video clip kiểm thử, nhật ký nhận diện. |
| **TASK-3.3** | **Hoàn thiện UI/UX & Triệt tiêu Bug:** Tối ưu hóa độ mượt, nhãn tiếng Việt, bảng cảnh báo thùng rác tương ứng. | Agent C | **P1** | TASK-3.1, TASK-3.2 | `streamlit_app.py`, tài liệu hướng dẫn sử dụng. |
| **TASK-3.4** | **Đóng gói và Báo cáo Tổng kết:** Xuất báo cáo nghiệm thu Definition of Done và tạo bản backup an toàn. | Integrator | **P0** | Toàn bộ tasks | `FINAL_REAL_WORLD_ACCEPTANCE_REPORT.md`. |

---

## 4. DỰ TOÁN NGÂN SÁCH THỜI GIAN, BỘ NHỚ VÀ TÀI NGUYÊN TÍNH TOÁN

| Giai đoạn | Thời gian ước tính | Dung lượng đĩa trống yêu cầu | Mức tiêu thụ GPU / RAM |
|---|:---:|:---:|:---:|
| **Sprint 0** *(Hiện tại)* | Đã hoàn tất | ~2 MiB (tài liệu md) | 0% GPU, RAM bình thường |
| **Sprint 1** | 2 – 3 ngày | ~500 MiB – 1 GiB | < 10% GPU (chỉ chạy camera inference test) |
| **Sprint 2** | 2 – 3 ngày | ~2 – 3 GiB (checkpoints & logs) | 70–90% GPU trong 45–90 phút (có kiểm soát và xin phép) |
| **Sprint 3** | 2 – 3 ngày | ~3 – 5 GiB (video logs thực địa) | 30–50% GPU (chạy inference camera liên tục) |
| **Tổng thể dự án** | **~7 – 10 ngày** | **$\le 10\text{ GiB}$** | **Hoàn toàn khả thi và an toàn trên RTX 2050 4GB** |
