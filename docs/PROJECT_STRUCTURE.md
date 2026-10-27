# HƯỚNG DẪN CẤU TRÚC MÃ NGUỒN & ĐIỀU HƯỚNG DỰ ÁN (`CNTT-KLCN155`)

> **Tên đề tài:** Hệ thống phát hiện và phân loại đa đối tượng rác thải sinh hoạt (`CNTT-KLCN155-waste-detection`)
> **Mục đích tài liệu:** Cung cấp bản đồ điều hướng kỹ thuật chính xác 100% theo trạng thái mã nguồn trên đĩa sau đợt tái cấu trúc `PRE-GD01-SOURCE-REFACTOR`, giúp thành viên nhóm, giảng viên hướng dẫn và kiểm định viên tra cứu nhanh chức năng từng thư mục, từng module Python và từng luồng thực thi.

---

## A. Tổng quan cấu trúc dự án

### A.1. Cây thư mục thực tế của dự án

*(Đã lược bỏ thư mục môi trường ảo `.venv/`, bộ nhớ đệm `__pycache__/`, `.pytest_cache/` và các tệp ảnh dữ liệu thô)*

```text
CNTT-KLCN155-waste-detection/
├── streamlit_app.py                          # Entrypoint ứng dụng Web Streamlit (2 Tab: Phân loại đơn vật thể & Phát hiện đa vật thể)
├── app.py                                    # Entrypoint ứng dụng Desktop PyQt5 (phân loại ảnh tĩnh, camera trực tiếp & lưu lịch sử SQLite)
├── README.md                                 # Tài liệu hướng dẫn cài đặt, tải trọng số Release v1.0.0-models và vận hành toàn hệ thống
├── ARCHITECTURE_AND_NEXT_STEPS.md            # Sơ đồ kiến trúc 5 tầng (Giai đoạn 1 & Giai đoạn 2) và lộ trình nâng cấp
├── GLOSSARY.md                               # Từ điển thuật ngữ và định nghĩa 10 lớp rác thải chuẩn (0..9)
├── AGENTS.md                                 # Quy chuẩn điều phối Multi-Agent, phân quyền Sub-Agent A/B/C/D và ràng buộc bất biến
├── CLAUDE.md                                 # Quy tắc kỹ thuật cốt lõi dành cho AI Agent làm việc trên repository
├── Tienxulidulieu.md                         # Đặc tả thuật toán tiền xử lý dữ liệu, khử trùng SHA-256/pHash và chia tách cụm
├── pytest.ini                                # Cấu hình chạy kiểm thử tự động pytest
├── requirements.txt                          # Danh sách thư viện phụ thuộc chính của dự án
├── requirements.lock                         # Bản khóa phiên bản chính xác của toàn bộ môi trường Python
├── yolov8n.pt                                # Trọng số khởi tạo gốc Ultralytics YOLOv8n phục vụ huấn luyện/kiểm thử ngoại tuyến
│
├── src/                                      # Mã nguồn cốt lõi của hệ thống (63 files .py, chia làm 3 phân hệ rõ ràng)
│   ├── __init__.py                           # Khởi tạo gói gốc src
│   ├── detection/                            # [35 files .py] Tầng mô hình AI (SSDLite320, YOLOv8n, MobileNetV3), WBF, Tiling, Two-Stage, Huấn luyện & Đánh giá
│   │   ├── __init__.py                       # Khởi tạo gói detection và xuất các giao diện chính
│   │   ├── types.py                          # Kiểu dữ liệu bất biến chuẩn hóa đầu ra: BoundingBox, Detection, DetectionResult
│   │   ├── schema.py                         # Lược đồ 10 lớp phát hiện (DETECTION_CLASS_NAMES) và hàm ánh xạ nhãn nguồn map_source_label
│   │   ├── coordinates.py                    # Chuẩn hóa xoay chiều EXIF, đệm tỉ lệ letterbox, chiếu ngược tọa độ về ảnh gốc và cắt biên
│   │   ├── ssdlite.py                        # Tầng 3A (Mô hình chính): Kiến trúc SSDLite320-MobileNetV3 (320x320), nạp backbone Stage-1 & suy luận
│   │   ├── ssdlite_train.py                  # Vòng lặp huấn luyện SSDLite320-MobileNetV3, YoloBoxDataset và chọn best.pt theo Validation mAP
│   │   ├── yolo.py                           # Tầng 3B (Mô hình đối chứng): Bộ bọc WasteDetector cho YOLOv8n (imgsz=640) kèm kiểm tra kiến trúc
│   │   ├── factory.py                        # Hàm nhà máy create_detector khởi tạo thống nhất backend ssdlite, yolov8n hoặc wbf
│   │   ├── fusion.py                         # Tầng 4: Thuật toán hợp nhất khung bao Weighted Boxes Fusion (WBF) theo từng lớp (FusionDetector)
│   │   ├── tune.py                           # Tìm kiếm lưới (grid search) siêu tham số ngưỡng confidence/IoU/WBF chỉ trên tập Validation
│   │   ├── tiling.py                         # Suy luận cắt ô cửa sổ trượt (SAHI-style TiledDetector) giúp bắt vật thể rác kích thước nhỏ
│   │   ├── two_stage.py                      # Thẩm định 2 giai đoạn (TwoStageDetector): dùng MobileNetV3-Large chấm lại vùng cắt proposal
│   │   ├── tracking.py                       # Làm mượt khung bao theo thời gian (TemporalDetectionSmoother) cho luồng camera/video
│   │   ├── visualization.py                  # Vẽ khung bao (Highlight Box / Bounding Box) và nhãn tiếng Việt lên ảnh kết quả
│   │   ├── augmentation.py                   # 4 cấu hình tăng cường dữ liệu Albumentations (none, light, medium,full) kèm khử suy biến bbox
│   │   ├── copy_paste.py                     # Tăng cường dữ liệu Copy-Paste (DonorBank, apply_copy_paste) cho các lớp hiếm (battery, glass...)
│   │   ├── balance_split.py                  # Cân bằng tần suất lặp ảnh huấn luyện (Repeat-Factor Sampling) cho lớp thiểu số
│   │   ├── dataset.py                        # Kiểm định nghiêm ngặt cấu trúc YAML, thư mục ảnh/nhãn và hình học bbox chuẩn YOLO
│   │   ├── manifest.py                       # Quản lý manifest kiểm duyệt từng ảnh phát hiện (DetectionRecord) và dấu vân tay SHA-256
│   │   ├── prepare.py                        # Xuất dữ liệu bbox đã duyệt sang định dạng YOLO 70/15/15 theo cụm cảnh chụp (scene-isolated)
│   │   ├── safe_dataset_builder.py           # Cổng bảo vệ chống quét thư mục mù, chống ghi đè baseline và kiểm tra Visual QA trước khi nạp
│   │   ├── export_train_manifest.py          # Lọc và xuất manifest huấn luyện sạch, chặn rò rỉ tập Test/Holdout và ảnh chưa duyệt
│   │   ├── preflight_guard.py                # Kiểm tra an toàn phần cứng (RAM, VRAM, ổ đĩa, quyền ghi) trước khi huấn luyện
│   │   ├── training_common.py                # Tiện ích cố định seed, tính dataset fingerprint SHA-256 và ghi metadata huấn luyện chung
│   │   ├── train.py                          # CLI điều phối huấn luyện thống nhất cho cả SSDLite320 và YOLOv8n (chặn tuyệt đối tập Test)
│   │   ├── evaluate.py                       # CLI tính toán COCO mAP@0.50, mAP@0.50:0.95, Precision, Recall, F1 và độ trễ suy luận
│   │   ├── export.py                         # CLI xuất mô hình phát hiện sang định dạng ONNX phục vụ triển khai tăng tốc
│   │   ├── mobilenetv3.py                    # Định nghĩa duy nhất kiến trúc phân loại MobileNetV3-Large 10 lớp (Giai đoạn 1 & Stage-2)
│   │   ├── checkpoint.py                     # Lưu và nạp checkpoint MobileNetV3 tự mô tả (kèm danh sách lớp, chuẩn hóa, SHA-256 dataset)
│   │   ├── classifier_data.py                # Tạo ImageFolder Dataset, DataLoader và tensor trọng số lớp cho huấn luyện Classifier
│   │   ├── classifier_engine.py              # Các hàm nguyên tử chạy 1 epoch huấn luyện/đánh giá và đóng/mở băng backbone 2 pha
│   │   ├── classifier_train.py               # CLI huấn luyện 2 pha MobileNetV3-Large (5 epoch đóng băng + 25 epoch fine-tune toàn mạng)
│   │   ├── classifier_evaluate.py            # CLI đánh giá độc lập MobileNetV3-Large trên tập Test và vẽ ma trận nhầm lẫn
│   │   ├── metrics.py                        # Tính toán Accuracy, Macro-F1, Weighted-F1, Precision/Recall từng lớp và Macro AUC
│   │   └── predict.py                        # CLI và lớp WastePredictor suy luận phân loại ảnh đơn lẻ (trả về Top-1, Top-K, cảnh báo độ tự tin)
│   │
│   ├── data/                                 # [15 files .py] Tầng tiền xử lý dữ liệu, khử trùng lặp SHA-256/pHash & kiểm định chống rò rỉ
│   │   ├── __init__.py                       # Khởi tạo gói data
│   │   ├── schema.py                         # Định nghĩa 10 lớp chuẩn (CLASS_NAMES), ManifestRecord và đọc bảng ánh xạ label_mapping.csv
│   │   ├── merge_datasets.py                 # Hợp nhất các nguồn dữ liệu gốc (VN Trash, Garbage v2) về cây thư mục raw an toàn
│   │   ├── dedup.py                          # Phát hiện ảnh hỏng, trùng lặp tuyệt đối (SHA-256) và gần trùng (pHash + BK-Tree + Union-Find)
│   │   ├── clean_data.py                     # Kiểm toán manifest thô và xuất báo cáo cụm trùng lặp / xung đột nhãn giữa các nguồn
│   │   ├── split.py                          # Thuật toán chia tập 70/15/15 phân tầng theo nhóm cụm pHash (Group-Stratified Split)
│   │   ├── split_dataset.py                  # Hiện thực hóa (materialize) các thư mục ảnh train/val/test trên đĩa theo manifest đã chia
│   │   ├── validation.py                     # CLI & bộ kiểm định độc lập sau chia tập (đủ 10 lớp, 0 ảnh hỏng, 0 rò rỉ cụm qua train/val/test)
│   │   ├── balance_classes.py                # Tính trọng số cân bằng lớp (class weights) không tạo bản sao ảnh vật lý trên đĩa
│   │   ├── dataset.py                        # Khám phá danh sách lớp, đếm số lượng mẫu và tạo tensor trọng số lớp cho huấn luyện
│   │   ├── dataloader.py                     # Xây dựng phép biến đổi ảnh (transforms) chuẩn ImageNet và PyTorch DataLoader
│   │   ├── augmentation.py                   # Định nghĩa nhóm phép tăng cường dữ liệu ảnh cho nhánh phân loại
│   │   ├── eda.py                            # Tổng hợp thống kê khám phá dữ liệu (EDA) và xuất báo cáo phân bố lớp/kích thước ảnh
│   │   ├── manifest.py                       # Đọc/ghi nguyên tử (atomic write) tệp CSV manifest và băm kiểm tra toàn vẹn
│   │   └── run_pipeline.py                   # CLI chạy trọn gói pipeline tiền xử lý dữ liệu đầu-cuối có tính tái lập 100%
│   │
│   └── app/                                  # [12 files .py] Tầng giao diện người dùng (Streamlit Web & PyQt5 Desktop) và dịch vụ suy luận
│       ├── __init__.py                       # Khởi tạo gói app
│       ├── detection_settings.py             # Quản lý cấu hình runtime Streamlit, đường dẫn trọng số mặc định (E4/E2/E1) & ngưỡng từng lớp
│       ├── detection_service.py              # Dịch vụ bọc mô hình phát hiện đảm bảo truy cập tuần tự an toàn luồng (thread-safe)
│       ├── detection_logic.py                # Xây dựng pipeline phát hiện đa tầng (Base -> Tiling -> TwoStage -> PerClassThreshold) & giải mã ảnh
│       ├── classifier_service.py             # Dịch vụ phân loại MobileNetV3-Large (PyTorch / ONNX Runtime) kèm hướng dẫn phân loại rác tại nguồn
│       ├── inference_service.py              # Dịch vụ suy luận an toàn luồng dùng chung cho phân loại ảnh tĩnh
│       ├── app_logic.py                      # Chuyển đổi kết quả dự đoán thành mô hình hiển thị (PredictionView) độc lập giao diện
│       ├── presentation.py                   # Ánh xạ mã lớp tiếng Anh (battery..trash) sang tên hiển thị tiếng Việt
│       ├── style.py                          # Các thành phần định dạng giao diện và biểu tượng trạng thái cho Streamlit
│       ├── main_window.py                    # Cửa sổ chính ứng dụng Desktop PyQt5 (WasteClassifierWindow)
│       ├── workers.py                        # Các luồng nền QThread (ImageInferenceWorker, CameraInferenceWorker) tách biệt khỏi UI thread
│       └── history.py                        # Lưu trữ lịch sử dự đoán ảnh tĩnh vào cơ sở dữ liệu SQLite (HistoryRepository)
│
├── configs/                                  # [23 files] Cấu hình YAML/JSON cho dataset, huấn luyện SSDLite/YOLO, WBF, ngưỡng hiệu chuẩn & manifest release
├── scripts/                                  # [37 files .py hoạt động] Công cụ CLI tải trọng số, tải dữ liệu, kiểm toán chống rò rỉ, vẽ biểu đồ khóa luận
│   └── archive/                              # [183 files .py lưu trữ] Các script lịch sử một lần được phân vào 6 thư mục con theo giai đoạn
├── tests/                                    # [93 files kiểm thử] Bộ kiểm thử tự động (519 passed) chia thành 5 phân hệ: detection, data, app, integration, qa
├── docs/                                     # Tài liệu kiến trúc (architecture/), thực nghiệm (experiments/), kế hoạch (plan/), cổng chất lượng PM (pm/), đề cương (proposal/)
├── reports/                                  # Báo cáo đánh giá thực nghiệm, biểu đồ khóa luận (figures/), kiểm định độc lập (qa_acceptance/) & lưu trữ (archive/)
├── notebooks/                                # [2 files .ipynb] Notebook huấn luyện trên Kaggle GPU và Google Colab
├── data/                                     # Bảng ánh xạ nhãn gốc (data/metadata/label_mapping.csv) và dữ liệu huấn luyện/kiểm thử cục bộ (bị ignore trên Git)
├── artifacts/                                # Nơi lưu trữ trọng số mô hình (.pt, .onnx), nhật ký huấn luyện và các gói bàn giao (deliveries/)
└── weights/                                  # Trọng số nền phụ trợ (bị ignore trên Git)
```

---

### A.2. Bảng chức năng từng thư mục cấp cao

| Thư mục | Số lượng tệp chính | Trách nhiệm cốt lõi trong hệ thống | Trạng thái Git |
|---|---|---|---|
| [`src/detection/`](../src/detection) | `35` files `.py` | Trái tim AI của dự án: chứa mô hình chính **SSDLite320-MobileNetV3** (Tầng 3A), mô hình đối chứng **YOLOv8n** (Tầng 3B), mô hình phân loại **MobileNetV3-Large**, hợp nhất **WBF** (Tầng 4), cắt ô **Tiling**, thẩm định **Two-Stage**, chiếu ngược tọa độ EXIF/letterbox và toàn bộ CLI huấn luyện/đánh giá. | Theo dõi bởi Git (`tracked`) |
| [`src/data/`](../src/data) | `15` files `.py` | Pipeline tiền xử lý dữ liệu chuẩn hóa 10 lớp: hợp nhất nguồn dữ liệu, kiểm tra ảnh lỗi, khử trùng lặp SHA-256 & pHash (BK-Tree), chia tập Group-Stratified 70/15/15 và kiểm định chống rò rỉ dữ liệu (`validation.py`). | Theo dõi bởi Git (`tracked`) |
| [`src/app/`](../src/app) | `12` files `.py` | Tầng dịch vụ ứng dụng và giao diện người dùng (Tầng 5): điều phối cấu hình detector/classifier, quản lý ngưỡng hiệu chuẩn theo lớp, vẽ kết quả cho Web **Streamlit** (`streamlit_app.py`) và Desktop **PyQt5** (`app.py`). | Theo dõi bởi Git (`tracked`) |
| [`configs/`](../configs) | `23` files (`.yaml`/`.json`) | Lưu trữ toàn bộ cấu hình tái lập thí nghiệm: cấu hình tiền xử lý, siêu tham số huấn luyện SSDLite320/YOLOv8n/MobileNetV3, cấu hình hợp nhất WBF, ngưỡng hiệu chuẩn từng lớp (`detection_thresholds_calibrated.json`) và manifest phát hành trọng số (`model_release_manifest.json`). | Theo dõi bởi Git (`tracked`) |
| [`scripts/`](../scripts) | `37` files `.py` hoạt động | Chứa các công cụ dòng lệnh đang hoạt động phục vụ tải trọng số (`download_models.py`), tải dataset (`download_datasets.py`), chạy ma trận thí nghiệm (`run_experiment_matrix.py`), kiểm tra rò rỉ (`check_split_leakage.py`) và sinh biểu đồ khóa luận (`generate_thesis_charts.py`). | Theo dõi bởi Git (`tracked`) |
| [`scripts/archive/`](../scripts/archive) | `183` files `.py` (6 thư mục con) | Kho lưu trữ các script lịch sử chạy một lần (Phase 4, Task 5–7, Task 8, kiểm toán cũ, module giao diện cũ không còn gọi). Được bảo tồn để truy vết lịch sử học thuật mà không làm nhiễu thư mục `scripts/` chính. | Theo dõi bởi Git (`tracked`) |
| [`tests/`](../tests) | `93` files `.py` (5 thư mục con + 8 file gốc) | Hệ thống kiểm thử tự động (`519 passed`): đồng bộ cấu trúc với `src/` gồm `tests/detection/`, `tests/data/`, `tests/app/`, `tests/integration/` và `tests/qa/`. | Theo dõi bởi Git (`tracked`) |
| [`docs/`](../docs) | `6` thư mục con + tài liệu gốc | Chứa tài liệu kiến trúc (`architecture/`), nhật ký thực nghiệm (`experiments/`), kế hoạch (`plan/`), tiêu chuẩn nghiệm thu Quality Gates của PM (`pm/`), tờ trình đề cương (`proposal/`) và tài liệu cấu trúc này (`PROJECT_STRUCTURE.md`). | Theo dõi bởi Git (`tracked`) |
| [`reports/`](../reports) | Nhiều thư mục chuyên đề | Chứa kết quả kiểm định độc lập (`qa_acceptance/`), biểu đồ khóa luận (`figures/`), script tái lập báo cáo (`reproduce_scripts/`), bằng chứng kiểm toán (`multi_waste_detection_audit/`) và báo cáo các đợt trước (`archive/`). | Theo dõi có chọn lọc |
| [`notebooks/`](../notebooks) | `2` files `.ipynb` | Notebook huấn luyện chuẩn hóa trên đám mây GPU: `train_mobilenetv3_kaggle.ipynb` (Giai đoạn 1) và `phase2_detection_colab.ipynb` (Giai đoạn 2). | Theo dõi bởi Git (`tracked`) |
| [`data/`](../data) | `data/metadata/label_mapping.csv` + dữ liệu cục bộ | Lưu trữ bảng ánh xạ nhãn gốc (được commit) và các tập dữ liệu ảnh/nhãn huấn luyện cục bộ (`data/processed/`, `data/detection_upgrade_v4_expanded/`... được loại trừ khỏi Git qua `.gitignore`). | Chỉ track `label_mapping.csv` |
| `artifacts/` | Trọng số `.pt`, `.onnx`, `.zip` | Chứa các checkpoint mô hình đã huấn luyện (`official_run/`, `detection_v2/`, `detection_v4_expanded/`...) được tải tự động qua `python scripts/download_models.py`. | Loại trừ qua `.gitignore` (phát hành qua GitHub Releases) |
| `weights/` | Trọng số khởi tạo phụ | Lưu trữ các file trọng số nền thử nghiệm ngoài thư mục gốc. | Loại trừ qua `.gitignore` |

---

### A.3. Cách tổ chức theo các tầng kiến trúc (5-Tier Architecture)

```text
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5 — GIAO DIỆN NGƯỜI DÙNG, DỊCH VỤ ỨNG DỤNG & THỐNG KÊ                                │
│ • Entrypoints: streamlit_app.py (Web 2 Tabs), app.py (PyQt5 Desktop)                      │
│ • Package: src/app/* (detection_logic, detection_settings, classifier_service, ...)       │
└─────────────────────────────────────────────▲─────────────────────────────────────────────┘
                                              │ DetectionResult (boxes [x1,y1,x2,y2], scores, class_ids 0..9)
┌─────────────────────────────────────────────┴─────────────────────────────────────────────┐
│ TẦNG 4 — HỢP NHẤT KHUNG BAO (WBF) & HẬU XỬ LÝ ĐA GIAI ĐOẠN                                │
│ • Weighted Boxes Fusion: src/detection/fusion.py (FusionDetector, fuse_detections)        │
│ • Cắt ô ảnh lớn (SAHI): src/detection/tiling.py (TiledDetector)                           │
│ • Thẩm định 2 giai đoạn: src/detection/two_stage.py (TwoStageDetector + MobileNetV3)      │
│ • Làm mượt & Hiển thị: src/detection/tracking.py, src/detection/visualization.py          │
└──────────────────────▲─────────────────────────────────────────────▲──────────────────────┘
                       │ Chuẩn hóa qua Prediction Adapter            │ Chuẩn hóa qua Prediction Adapter
                       │ (types.py, factory.py, coordinates.py)      │ (types.py, factory.py, coordinates.py)
┌──────────────────────┴──────────────────────┐ ┌────────────────────┴──────────────────────┐
│ TẦNG 3A — NHÁNH MÔ HÌNH CHÍNH (PRIMARY)     │ │ TẦNG 3B — NHÁNH ĐỐI CHỨNG (CONTROL)       │
│ • SSDLite320-MobileNetV3-Large (320×320)    │ │ • Ultralytics YOLOv8n (Native imgsz=640)  │
│ • Kế thừa trọng số backbone từ Stage-1      │ │ • Đầu ra Decoupled Head Anchor-Free       │
│ • Nhãn nội bộ: 0=BG, 1..10=FG (decode -1)   │ │ • Nhãn nội bộ: 0..9 khớp trực tiếp        │
│ • Code: src/detection/{ssdlite,             │ │ • Code: src/detection/{yolo,              │
│         ssdlite_train,mobilenetv3}.py       │ │         train,export}.py                  │
└──────────────────────▲──────────────────────┘ └────────────────────▲──────────────────────┘
                       └──────────────────────┬──────────────────────┘
                                              │ Dữ liệu sạch đã kiểm định (0 rò rỉ cụm)
┌─────────────────────────────────────────────┴─────────────────────────────────────────────┐
│ TẦNG 1 & 2 — TIỀN XỬ LÝ DỮ LIỆU, KHỬ TRÙNG LẶP & TĂNG CƯỜNG DỮ LIỆU                       │
│ • Khử trùng SHA-256 + pHash BK-Tree & Group Split 70/15/15: src/data/*                    │
│ • Kiểm định YOLO bbox, EXIF, Albumentations & Copy-Paste: src/detection/{dataset,         │
│   prepare,safe_dataset_builder,augmentation,copy_paste,balance_split,coordinates}.py      │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## B. Giải thích chi tiết từng file source chính

### B.1. Hai điểm vào ứng dụng tại thư mục gốc (`2` files) và gói gốc `src/__init__.py` (`1` file)

| Đường dẫn file | Chức năng chính | Thành phần / Class / Hàm quan trọng | File liên quan | Khi nào cần sửa file đó |
|---|---|---|---|---|
| [`streamlit_app.py`](../streamlit_app.py) | Điểm vào chính của ứng dụng Web Streamlit gồm 2 Tab: Tab 1 (Phân loại rác đơn vật thể bằng `MobileNetV3-Large`) và Tab 2 (Phát hiện đa vật thể bằng `SSDLite320` / `YOLOv8n` / `WBF`, thống kê số lượng từng loại rác và xuất gói `.zip`). | `cached_classifier()`, `cached_detector()`, `render_classification_result()`, `render_detection_result()`, `main()` | `src/app/detection_settings.py`, `src/app/detection_logic.py`, `src/app/classifier_service.py`, `src/app/style.py` | Khi cần thay đổi bố cục giao diện Web, thêm bộ lọc thanh bên (sidebar), hoặc bổ sung biểu đồ/bảng thống kê trên Streamlit. |
| [`app.py`](../app.py) | Điểm vào ứng dụng Desktop PyQt5 phục vụ phân loại ảnh tĩnh từ đĩa, phân loại luồng camera trực tiếp và ghi lịch sử vào SQLite. | `build_parser()`, `parse_args()`, `main()` | `src/detection/predict.py`, `src/app/main_window.py`, `src/app/history.py`, `src/app/inference_service.py` | Khi cần thêm tham số dòng lệnh CLI cho ứng dụng Desktop PyQt5 hoặc thay đổi cách khởi tạo cửa sổ `QApplication`. |
| [`src/__init__.py`](../src/__init__.py) | Đánh dấu `src` là gói Python gốc để các lệnh `python -m src.<subpackage>.<module>` hoạt động đồng nhất trên mọi hệ điều hành. | *(Module docstring)* | Toàn bộ `src/detection/*`, `src/data/*`, `src/app/*` | Hầu như không cần sửa; chỉ cập nhật nếu muốn xuất phiên bản toàn cục (`__version__`). |

---

### B.2. Phân hệ `src/detection/` — Mô hình AI, Huấn luyện, Đánh giá, Suy luận & Hợp nhất WBF (`35` files)

| Đường dẫn file | Chức năng chính | Thành phần / Class / Hàm quan trọng | File liên quan | Khi nào cần sửa file đó |
|---|---|---|---|---|
| [`src/detection/__init__.py`](../src/detection/__init__.py) | Khởi tạo gói `src.detection` và cung cấp lối tắt import cho các kiểu dữ liệu và bộ phát hiện cốt lõi. | `__all__` (`BoundingBox`, `Detection`, `DetectionResult`, `DETECTION_CLASS_NAMES`, `WasteDetector`) | `src/detection/types.py`, `src/detection/schema.py`, `src/detection/yolo.py` | Khi cần xuất thêm lớp hoặc hàm public ở cấp gói `src.detection`. |
| [`src/detection/types.py`](../src/detection/types.py) | Định nghĩa hợp đồng dữ liệu đầu ra độc lập framework cho toàn bộ các mô hình phát hiện (Tầng 3A, 3B, Tầng 4, Tầng 5). | `BoundingBox` (`as_xyxy`, `width`, `height`, `area`, `clip`), `Detection`, `DetectionResult` (`class_counts`) | `src/detection/ssdlite.py`, `src/detection/yolo.py`, `src/detection/fusion.py`, `src/app/detection_logic.py` | Khi cần bổ sung thuộc tính mới vào kết quả phát hiện (ví dụ: mã theo dõi đối tượng hoặc điểm phụ trợ) mà mọi tầng đều dùng chung. |
| [`src/detection/schema.py`](../src/detection/schema.py) | Khóa thứ tự 10 lớp phát hiện chuẩn (`0: battery` ➔ `9: trash`) và ánh xạ tên nhãn từ các bộ dữ liệu nguồn mở (TACO, Mendeley, Roboflow, OpenImages) về 10 lớp chuẩn. | `DETECTION_CLASS_NAMES`, `DetectionDatasetError`, `map_source_label()`, `_normalize_token()` | `configs/detection_source_mapping.yaml`, `src/detection/prepare.py`, `src/data/schema.py` | Khi tích hợp bộ dữ liệu nguồn mới có tên lớp thô chưa có trong từ điển ánh xạ. *(Lưu ý: Tuyệt đối không tự ý đổi thứ tự 10 lớp chuẩn khi chưa có lệnh PM).* |
| [`src/detection/coordinates.py`](../src/detection/coordinates.py) | Xử lý toàn bộ hình học tọa độ: xoay ảnh đúng chiều EXIF trước khi suy luận, đệm ảnh giữ tỉ lệ (`letterbox_image`), chiếu ngược tọa độ dự đoán về kích thước ảnh gốc và cắt khung bao tràn biên. | `ensure_exif_orientation()`, `letterbox_image()`, `scale_coords_back()`, `scale_coords_direct()`, `clip_and_validate_box()`, `tile_to_image()` | `src/detection/ssdlite.py`, `src/detection/yolo.py`, `src/detection/tiling.py`, `src/app/detection_logic.py` | Khi cần tinh chỉnh thuật toán đổi hệ tọa độ giữa ảnh tiền xử lý (`320×320` hoặc `640×640`) và ảnh gốc của người dùng. |
| [`src/detection/ssdlite.py`](../src/detection/ssdlite.py) | **Tầng 3A (Mô hình chính):** Xây dựng mô hình `SSDLite320-MobileNetV3-Large`, truyền trọng số backbone từ `MobileNetV3-Large` Giai đoạn 1, xử lý quy ước nhãn `0=Background, 1..10=Foreground` (decode trừ `1` về `0..9`) và chạy suy luận ở kích thước chuẩn `320×320`. | `SSDLiteWasteDetector` (`predict`, `predict_image`), `build_ssdlite()`, `_transfer_stage1_backbone_weights()`, `tensor_detections()`, `resolve_device()` | `src/detection/ssdlite_train.py`, `src/detection/mobilenetv3.py`, `src/detection/factory.py`, `src/detection/coordinates.py` | Khi cần nâng cấp kiến trúc head/backbone của SSDLite320, thay đổi cách nạp checkpoint Stage-1 hoặc tối ưu hậu xử lý NMS của nhánh SSDLite. |
| [`src/detection/ssdlite_train.py`](../src/detection/ssdlite_train.py) | Huấn luyện `SSDLite320-MobileNetV3` trên tập dữ liệu định dạng YOLO (tự động cộng `+1` vào class ID khi nạp target để dành `0` cho Background), đánh giá Validation sau mỗi epoch và lưu `best.pt` / `last.pt`. | `YoloBoxDataset`, `build_ssd_augmentation()`, `collate_detection_batch()`, `evaluate_validation()`, `train_ssdlite()` | `src/detection/ssdlite.py`, `src/detection/train.py`, `src/detection/augmentation.py`, `src/detection/copy_paste.py` | Khi cần điều chỉnh chiến lược tối ưu hóa (optimizer, scheduler, warmup, đóng băng backbone) hoặc hàm mất mát của SSDLite320. |
| [`src/detection/yolo.py`](../src/detection/yolo.py) | **Tầng 3B (Mô hình đối chứng):** Bọc mô hình Ultralytics `YOLOv8n` với cơ chế kiểm tra nghiêm ngặt thứ tự 10 lớp, xác thực kiến trúc Decoupled Head và giữ nguyên độ phân giải gốc của checkpoint (`imgsz=640`, không ép xuống `320`). | `WasteDetector` (`predict`, `predict_image`), `DetectionError`, `validate_yolov8n_architecture()`, `validate_yolo_metadata()`, `_checkpoint_train_imgsz()` | `src/detection/factory.py`, `src/detection/train.py`, `src/detection/types.py` | Khi cần thay đổi cách gọi Ultralytics YOLO runtime, bổ sung kiểm tra tương thích checkpoint YOLO hoặc truyền tham số NMS/agnostic_nms. |
| [`src/detection/factory.py`](../src/detection/factory.py) | Cung cấp một điểm khởi tạo duy nhất (`create_detector`) cho cả 3 chế độ backend: `ssdlite` (Tầng 3A), `yolov8n` (Tầng 3B) và `wbf` (Tầng 4 hợp nhất SSDLite + YOLO). | `create_detector()` | `src/detection/ssdlite.py`, `src/detection/yolo.py`, `src/detection/fusion.py`, `src/app/detection_logic.py`, `src/detection/evaluate.py` | Khi cần thêm một backend phát hiện mới hoặc thay đổi tham số khởi tạo mặc định khi ghép nối `FusionDetector`. |
| [`src/detection/fusion.py`](../src/detection/fusion.py) | **Tầng 4 (Hợp nhất dự đoán):** Cài đặt thuật toán Weighted Boxes Fusion (WBF) theo từng lớp (`class-wise`) để kết hợp khung bao từ `SSDLite320` và `YOLOv8n`, kèm lọc khung bao lồng nhau (containment suppression). | `FusionDetector`, `fuse_detections()`, `_normalize_for_wbf()`, `_restore_and_filter_fused_boxes()`, `box_iou()`, `box_intersection_over_smaller()` | `src/detection/factory.py`, `src/detection/tune.py`, `src/detection/types.py`, `configs/detection_fusion.yaml` | Khi cần điều chỉnh công thức tính điểm hợp nhất WBF, trọng số giữa các mô hình (`weights`) hoặc ngưỡng loại bỏ khung bao trùng/lồng nhau. |
| [`src/detection/tune.py`](../src/detection/tune.py) | Tự động dò tìm bộ siêu tham số suy luận tối ưu (ngưỡng confidence, ngưỡng IoU, trọng số WBF) **chỉ sử dụng tập Validation** (tuyệt đối không chạm tập Test) và ghi ra tệp JSON cấu hình đã chọn. | `tune()`, `main()` | `src/detection/evaluate.py`, `src/detection/fusion.py`, `scripts/run_experiment_matrix.py` | Khi cần mở rộng không gian tìm kiếm lưới (grid search) cho ngưỡng suy luận hoặc thay đổi hàm mục tiêu chọn cấu hình trên tập Validation. |
| [`src/detection/tiling.py`](../src/detection/tiling.py) | Cắt ảnh độ phân giải lớn thành các ô cửa sổ trượt có chồng lấp (`SAHI-style sliced inference`), chạy detector trên cả ảnh toàn cục lẫn từng ô cắt, rồi hợp nhất và khử các khung bao bị cắt cụt ở mép ô nội bộ. | `TileWindow`, `TiledDetector`, `compute_tiles()`, `compute_adaptive_tile_size()`, `suppress_truncated()` | `src/app/detection_logic.py`, `src/detection/coordinates.py`, `src/detection/fusion.py` | Khi cần tối ưu khả năng phát hiện vật thể rác nhỏ ở xa, thay đổi kích thước ô cắt (`tile_size`), độ chồng lấp (`overlap`) hoặc ngưỡng kích hoạt cắt ô. |
| [`src/detection/two_stage.py`](../src/detection/two_stage.py) | Thẩm định hai giai đoạn (`Two-Stage Refinement`): cắt từng vùng ứng viên (proposal) từ detector, đưa qua `MobileNetV3-Large` phân loại theo lô (`MobileNetBatchAdapter`) và tính lại điểm tin cậy theo công thức Bayesian (`product` hoặc `blend`). | `TwoStageDetector`, `MobileNetBatchAdapter`, `RefinedDetection`, `build_class_mapping()` | `src/detection/predict.py`, `src/app/detection_logic.py`, `src/detection/types.py` | Khi cần tinh chỉnh cơ chế kết hợp xác suất giữa Detector và Classifier (`alpha`, `min_proposal_gate`, `padding_ratio`) để giảm dương tính giả trên nền phức tạp. |
| [`src/detection/tracking.py`](../src/detection/tracking.py) | Làm mượt vị trí và điểm số khung bao giữa các khung hình liên tiếp bằng trung bình trượt hàm mũ (EMA) và khớp IoU tham lam. | `TemporalDetectionSmoother`, `_Track`, `box_iou()`, `_blend_box()` | `src/detection/types.py` | Khi cần cải thiện độ ổn định khung bao trên luồng video hoặc camera thời gian thực. |
| [`src/detection/visualization.py`](../src/detection/visualization.py) | Vẽ khung bao phát hiện lên ảnh kết quả với 2 chế độ trực quan: *Khung viền nổi bật* (`highlight` — làm tối nhẹ vùng nền ngoài vật thể) và *Hộp chữ nhật viền nét* (`bbox`), kèm nhãn tiếng Việt và phần trăm độ tin cậy. | `draw_detections()`, `detection_label()`, `_rgb_color()`, `_label_font()` | `src/app/presentation.py`, `src/app/detection_logic.py`, `src/detection/types.py` | Khi cần thay đổi bảng màu 10 lớp, phông chữ tiếng Việt, độ dày nét vẽ hoặc hiệu ứng hiển thị khung bao trên giao diện. |
| [`src/detection/augmentation.py`](../src/detection/augmentation.py) | Cung cấp 4 mức tăng cường dữ liệu dựa trên Albumentations (`none`, `light`, `medium`, `full`) cho bài toán phát hiện đa đối tượng, kèm bộ lọc loại bỏ các khung bao bị biến dạng hoặc suy biến sau biến đổi hình học. | `get_detection_transform()`, `apply_augmentation()`, `apply_augmentation_with_telemetry()`, `_sanitize_yolo_boxes()` | `src/detection/ssdlite_train.py`, `src/detection/train.py`, `configs/detection_training_combined_ssdlite.yaml` | Khi cần thêm phép biến đổi màu sắc/hình học mới vào quá trình huấn luyện detector hoặc thay đổi ngưỡng diện tích tối thiểu của bbox sau cắt. |
| [`src/detection/copy_paste.py`](../src/detection/copy_paste.py) | Kỹ thuật tăng cường dữ liệu **Copy-Paste** cấp đối tượng: trích xuất các mẫu vật thể sạch thuộc lớp hiếm (`battery`, `glass`, `metal`...) vào ngân hàng mẫu (`DonorBank`) và ghép ngẫu nhiên không chồng lấp lên ảnh huấn luyện. | `DonorObject`, `DonorBank`, `apply_copy_paste()` | `src/detection/ssdlite_train.py`, `src/detection/train.py` | Khi cần tăng tần suất xuất hiện của các lớp thiểu số trong quá trình huấn luyện hoặc điều chỉnh quy tắc kiểm tra chồng lấp IoU khi dán vật thể. |
| [`src/detection/balance_split.py`](../src/detection/balance_split.py) | Tạo tệp danh sách ảnh huấn luyện cân bằng lớp theo thuật toán **Repeat-Factor Sampling (RFS)** (tương tự LVIS), giúp tăng số lần lấy mẫu các ảnh chứa lớp hiếm mà không làm thay đổi tập Validation hay Test. | `compute_image_repeat_factor()`, `build_balanced_split()`, `_resolve_split_line()` | `src/detection/train.py`, `src/detection/ssdlite_train.py` | Khi cần điều chỉnh ngưỡng tần suất lặp (`repeat_threshold`) hoặc mức trần lặp tối đa (`max_repeat`) cho các lớp ít mẫu. |
| [`src/detection/dataset.py`](../src/detection/dataset.py) | Kiểm tra tính hợp lệ toàn diện của một bộ dữ liệu phát hiện chuẩn YOLO (kiểm tra tệp `dataset.yaml`, sự tồn tại của thư mục `train/val/test`, tính đối xứng ảnh–nhãn và tọa độ chuẩn hóa `0..1`). | `DetectionDatasetReport`, `load_detection_dataset_yaml()`, `read_yolo_boxes()`, `validate_detection_dataset()` | `src/detection/train.py`, `src/detection/ssdlite_train.py`, `src/detection/evaluate.py` | Khi cần bổ sung quy tắc kiểm tra chất lượng tệp nhãn YOLO trước khi cho phép bắt đầu huấn luyện hoặc đánh giá. |
| [`src/detection/manifest.py`](../src/detection/manifest.py) | Quản lý việc đọc/ghi tệp manifest CSV chứa thông tin nguồn gốc, trạng thái kiểm duyệt (`reviewed`) và mã băm SHA-256 của từng ảnh trong bộ dữ liệu phát hiện Giai đoạn 2. | `DetectionRecord`, `load_manifest()`, `write_manifest()`, `load_reviewed_manifest()`, `image_fingerprints()` | `src/detection/prepare.py`, `src/detection/schema.py` | Khi cần thêm trường siêu dữ liệu (metadata column) mới vào sổ cái quản lý ảnh phát hiện. |
| [`src/detection/prepare.py`](../src/detection/prepare.py) | CLI xuất các ảnh và khung bao đã được kiểm duyệt (`reviewed=True`) thành bộ dữ liệu định dạng YOLO chia sẵn `70/15/15` theo cụm bối cảnh chụp (`scene_id`) để chống rò rỉ nền cảnh giữa các tập. | `split_records()`, `write_dataset_yaml()`, `copy_record()`, `prepare_dataset()`, `main()` | `src/detection/manifest.py`, `src/detection/dataset.py`, `src/detection/schema.py` | Khi cần thay đổi tỉ lệ chia tập phát hiện hoặc cách thức tổ chức cây thư mục đầu ra khi chuẩn bị dữ liệu. |
| [`src/detection/safe_dataset_builder.py`](../src/detection/safe_dataset_builder.py) | Cổng an toàn dữ liệu (Dataset Safety Gate): ngăn chặn việc quét thư mục tự do không có manifest, cấm ghi đè lên các bộ dữ liệu baseline đã khóa và bắt buộc mọi ảnh nạp mới phải có bằng chứng Visual QA hợp lệ. | `SafeDatasetBuilderError`, `verify_visual_qa_pass_evidence()`, `validate_candidate_for_intake()`, `scan_directory_blindly_guard()`, `build_safe_pilot_dataset_staging()` | `scripts/build_dataset_v3_candidate.py`, `src/detection/export_train_manifest.py` | Khi cần cập nhật danh sách các thư mục baseline được bảo vệ chống ghi đè hoặc siết chặt tiêu chuẩn nạp ảnh bổ sung. |
| [`src/detection/export_train_manifest.py`](../src/detection/export_train_manifest.py) | Xuất danh sách ảnh đủ điều kiện đưa vào tập Train sau khi đã đối chiếu chéo để loại bỏ hoàn toàn mọi ảnh trùng SHA-256 với tập Val/Test/Holdout và các ảnh chưa được phê duyệt. | `load_authoritative_splits()`, `validate_image_eligibility()`, `filter_manifest_for_train()`, `export_train_manifest()`, `main()` | `src/detection/safe_dataset_builder.py`, `src/detection/training_common.py` | Khi cần thay đổi quy trình phê duyệt manifest đầu vào trước các đợt huấn luyện mở rộng. |
| [`src/detection/preflight_guard.py`](../src/detection/preflight_guard.py) | Kiểm tra an toàn tài nguyên hệ thống trước khi chạy huấn luyện hoặc đánh giá nặng (kiểm tra dung lượng RAM khả dụng, VRAM GPU, dung lượng ổ đĩa trống và quyền ghi thư mục đầu ra). | `PreflightReport`, `PreflightSafetyError`, `check_training_safety()`, `enforce_preflight_guard()`, `main()` | `src/detection/train.py`, `src/detection/ssdlite_train.py`, `README.md` | Khi cần điều chỉnh ngưỡng RAM/VRAM tối thiểu cho phù hợp với cấu hình máy trạm mới. |
| [`src/detection/training_common.py`](../src/detection/training_common.py) | Tập hợp các hàm dùng chung đảm bảo tính tái lập khoa học giữa hai nhánh SSDLite320 và YOLOv8n: cố định seed ngẫu nhiên, băm SHA-256 toàn bộ danh sách ảnh/nhãn (`dataset_fingerprint`) và ghi tệp JSON metadata của lần chạy. | `seed_everything()`, `dataset_fingerprint()`, `split_images()`, `run_metadata()`, `write_json()` | `src/detection/ssdlite_train.py`, `src/detection/train.py`, `src/detection/evaluate.py` | Khi cần ghi bổ sung thông tin môi trường phần cứng hoặc phiên bản thư viện vào `run_metadata.json` của mỗi đợt huấn luyện. |
| [`src/detection/train.py`](../src/detection/train.py) | CLI điều phối huấn luyện trung tâm cho Giai đoạn 2 (`python -m src.detection.train`): hỗ trợ cả `--model ssdlite320_mobilenetv3` và `--model yolov8n`, tự động tạo bản sao `dataset.yaml` đã gỡ bỏ đường dẫn tập Test để đảm bảo quá trình huấn luyện không bao giờ đọc tập Test. | `AblationAugmentationTransform`, `_resolved_dataset_file()`, `train_detector()`, `main()` | `src/detection/ssdlite_train.py`, `src/detection/yolo.py`, `src/detection/training_common.py`, `configs/detection_*.yaml` | Khi cần thêm tham số CLI cho quá trình huấn luyện detector hoặc tích hợp thêm chế độ thử nghiệm ablation. |
| [`src/detection/evaluate.py`](../src/detection/evaluate.py) | CLI đánh giá chất lượng mô hình phát hiện độc lập (`python -m src.detection.evaluate`): tính toán đầy đủ bộ chỉ số chuẩn **COCO mAP@0.50**, **mAP@0.50:0.95**, Precision, Recall, F1 tại ngưỡng vận hành, bảng chỉ số riêng từng lớp (`per_class_ap50`) và tốc độ suy luận (ms/ảnh). | `coco_metrics()`, `operating_metrics()`, `evaluate_detector()`, `load_evaluation_split()`, `main()` | `src/detection/factory.py`, `src/detection/tune.py`, `scripts/run_experiment_matrix.py` | Khi cần bổ sung độ đo đánh giá mới (ví dụ: AP theo kích thước vật thể nhỏ/vừa/lớn hoặc xuất thêm biểu đồ PR curve). |
| [`src/detection/export.py`](../src/detection/export.py) | CLI xuất checkpoint mô hình phát hiện đã huấn luyện sang định dạng mở `ONNX` (hoặc `TorchScript`) sau khi kiểm tra hợp lệ siêu dữ liệu 10 lớp. | `export_detector()`, `main()` | `src/detection/yolo.py`, `src/detection/schema.py` | Khi cần xuất trọng số mô hình sang các định dạng tối ưu hóa phần cứng biên (ONNX, TensorRT, OpenVINO). |
| [`src/detection/mobilenetv3.py`](../src/detection/mobilenetv3.py) | Định nghĩa kiến trúc **MobileNetV3-Large** 10 lớp duy nhất trong toàn dự án (thay thế lớp `Linear` cuối cùng bằng `num_classes=10`), được dùng chung cho huấn luyện phân loại Giai đoạn 1, khởi tạo backbone SSDLite320 và bộ thẩm định Two-Stage. | `MODEL_ID`, `build_model()` | `src/detection/checkpoint.py`, `src/detection/ssdlite.py`, `src/detection/classifier_train.py`, `src/detection/predict.py` | Khi cần thay đổi tỉ lệ `dropout` hoặc cấu trúc tầng phân loại cuối của MobileNetV3-Large. |
| [`src/detection/checkpoint.py`](../src/detection/checkpoint.py) | Quản lý định dạng lưu trữ và kiểm tra tính toàn vẹn của checkpoint `MobileNetV3-Large` (`format_version=1`), đóng gói kèm danh sách 10 lớp, kích thước ảnh đầu vào, thông số chuẩn hóa ImageNet và fingerprint SHA-256 của dataset. | `CheckpointMetadata`, `LoadedCheckpoint`, `CheckpointCompatibilityError`, `save_checkpoint()`, `load_checkpoint()` | `src/detection/mobilenetv3.py`, `src/detection/classifier_train.py`, `src/detection/predict.py`, `scripts/import_ecovision_checkpoint.py` | Khi cần nâng cấp phiên bản schema của tệp checkpoint phân loại hoặc thêm trường kiểm tra tính tương thích mới. |
| [`src/detection/classifier_data.py`](../src/detection/classifier_data.py) | Xây dựng `ImageFolder` Dataset và PyTorch `DataLoader` cho quá trình huấn luyện `MobileNetV3-Large`, đồng thời kiểm tra nghiêm ngặt thứ tự thư mục lớp phải khớp tuyệt đối với `CLASS_NAMES`. | `load_model_config()`, `build_transforms()`, `create_datasets()`, `create_dataloaders()`, `ordered_class_weight_tensor()` | `src/data/dataset.py`, `src/data/validation.py`, `src/detection/classifier_train.py`, `configs/model_config.yaml` | Khi cần thay đổi phép biến đổi ảnh (data augmentation) hoặc cấu hình nạp dữ liệu (`num_workers`, `pin_memory`) cho bộ phân loại. |
| [`src/detection/classifier_engine.py`](../src/detection/classifier_engine.py) | Cung cấp các hàm huấn luyện cấp thấp cho `MobileNetV3-Large`: đóng băng backbone ở Pha 1 (`freeze_backbone`), mở băng toàn mạng với learning rate phân tầng ở Pha 2 (`unfreeze_all`) và chạy vòng lặp 1 epoch hỗ trợ AMP (`GradScaler`). | `EpochResult`, `configure_phase()`, `optimizer_for_phase()`, `run_epoch()` | `src/detection/classifier_train.py`, `src/detection/metrics.py` | Khi cần thay đổi thuật toán tối ưu hóa (AdamW), cách cắt gradient (`grad_clip_norm`) hoặc lịch trình đóng/mở băng các tầng mạng phân loại. |
| [`src/detection/classifier_train.py`](../src/detection/classifier_train.py) | CLI điều phối huấn luyện 2 pha cho mô hình phân loại `MobileNetV3-Large` (`python -m src.detection.classifier_train`), hỗ trợ tiếp tục huấn luyện (`--resume last.pt`), dừng sớm (early stopping) và chọn `best.pt` theo **Validation Macro-F1**. | `TrainingResult`, `fit()`, `resolve_device()`, `seed_everything()`, `main()` | `src/detection/classifier_data.py`, `src/detection/classifier_engine.py`, `src/detection/checkpoint.py`, `configs/model_config.yaml` | Khi cần thay đổi quy trình huấn luyện tổng thể của nhánh Classifier hoặc thêm tệp nhật ký huấn luyện đầu ra. |
| [`src/detection/classifier_evaluate.py`](../src/detection/classifier_evaluate.py) | CLI đánh giá độc lập checkpoint `MobileNetV3-Large` trên tập Test (`python -m src.detection.classifier_evaluate`), xuất báo cáo `metrics.json`, `per_class_metrics.csv` và vẽ 2 biểu đồ ma trận nhầm lẫn (raw & normalized). | `EvaluationResult`, `evaluate_checkpoint()`, `_write_metrics_json()`, `_write_per_class_csv()`, `_write_confusion_plot()`, `main()` | `src/detection/checkpoint.py`, `src/detection/metrics.py`, `src/data/schema.py` | Khi cần thay đổi giao diện biểu đồ Confusion Matrix hoặc bổ sung chỉ số đánh giá cho báo cáo phân loại Giai đoạn 1. |
| [`src/detection/metrics.py`](../src/detection/metrics.py) | Tính toán thuần túy toàn bộ các độ đo phân loại đa lớp 10 nhãn: Accuracy, Macro-F1, Weighted-F1, Precision/Recall/F1/Support từng lớp, One-vs-Rest Macro AUC và ma trận nhầm lẫn. | `MetricBundle`, `compute_multiclass_metrics()` | `src/detection/classifier_engine.py`, `src/detection/classifier_evaluate.py` | Khi cần bổ sung công thức thống kê mới cho bài toán phân loại đa lớp. |
| [`src/detection/predict.py`](../src/detection/predict.py) | Cung cấp lớp `WastePredictor` và CLI (`python -m src.detection.predict`) để suy luận phân loại ảnh đơn lẻ hoặc theo lô tensor, trả về nhãn Top-1, danh sách Top-K kèm cờ cảnh báo `low_confidence` khi xác suất dưới ngưỡng (`0.55`). | `WastePredictor` (`predict`, `predict_image`, `predict_tensor_batch`), `Prediction`, `ScoredClass`, `PredictionError`, `main()` | `src/detection/checkpoint.py`, `src/detection/two_stage.py`, `src/app/inference_service.py`, `app.py` | Khi cần tối ưu tốc độ suy luận theo lô (`predict_tensor_batch`) phục vụ Two-Stage hoặc thay đổi định dạng JSON trả về của CLI phân loại. |

---

### B.3. Phân hệ `src/data/` — Tiền xử lý Dữ liệu, Khử trùng lặp & Chống rò rỉ (`15` files)

| Đường dẫn file | Chức năng chính | Thành phần / Class / Hàm quan trọng | File liên quan | Khi nào cần sửa file đó |
|---|---|---|---|---|
| [`src/data/__init__.py`](../src/data/__init__.py) | Khởi tạo gói `src.data` chứa các hợp đồng dữ liệu và pipeline tiền xử lý. | *(Module docstring)* | Toàn bộ `src/data/*` | Không cần sửa trong vận hành thông thường. |
| [`src/data/schema.py`](../src/data/schema.py) | Định nghĩa nguồn sự thật cho 10 lớp rác chuẩn (`CLASS_NAMES`), cấu trúc bản ghi `ManifestRecord` (18 trường) và hàm đọc/kiểm tra bảng ánh xạ nhãn `data/metadata/label_mapping.csv`. | `CLASS_NAMES`, `ManifestRecord`, `PipelineError`, `LabelMappingError`, `canonicalize_label()`, `load_label_mapping()`, `map_to_unified_label()` | `data/metadata/label_mapping.csv`, `src/data/merge_datasets.py`, `src/data/manifest.py`, `src/detection/schema.py` | Khi bổ sung bộ dữ liệu phân loại mới vào `label_mapping.csv` hoặc thêm cột thuộc tính vào `ManifestRecord`. |
| [`src/data/merge_datasets.py`](../src/data/merge_datasets.py) | Quét các thư mục dataset nguồn (VN Trash Classification, Garbage Classification v2), ánh xạ nhãn về 10 lớp chuẩn, cấp mã định danh ổn định (`stable_image_id`) và sao chép/liên kết an toàn vào cây `data/raw/`. | `SourceSpec`, `stable_image_id()`, `ingest_sources()` | `src/data/schema.py`, `src/data/run_pipeline.py`, `scripts/download_datasets.py` | Khi cần tích hợp thêm một nguồn dữ liệu phân loại mới có cấu trúc thư mục khác với VN Trash hoặc Garbage v2. |
| [`src/data/dedup.py`](../src/data/dedup.py) | Kiểm tra khả năng giải mã từng tệp ảnh, tính mã băm **SHA-256** (bắt trùng lặp tuyệt đối) và **64-bit Perceptual Hash (`pHash`)**, sau đó gom cụm ảnh gần trùng bằng cấu trúc **BK-Tree** + **Union-Find** và cách ly các cụm xung đột nhãn (`conflict`). | `ImageSignature`, `DuplicateReport`, `_BKTree`, `_UnionFind`, `sha256_file()`, `phash_bits()`, `hamming_distance()`, `find_duplicate_clusters()` | `src/data/clean_data.py`, `src/data/validation.py`, `Tienxulidulieu.md` | Khi cần điều chỉnh ngưỡng khoảng cách Hamming của `pHash` (`phash_hamming_threshold`, mặc định `<= 6`) hoặc thuật toán gom cụm trùng lặp. |
| [`src/data/clean_data.py`](../src/data/clean_data.py) | Điều phối bước làm sạch trên manifest thô và xuất các báo cáo CSV xác định (`exact_duplicates.csv`, `near_duplicates.csv`, `label_conflicts.csv`, `corrupt_files.csv`). | `CleaningOutputs`, `run_cleaning()`, `_write_duplicate_reports()` | `src/data/dedup.py`, `src/data/manifest.py`, `src/data/run_pipeline.py` | Khi cần xuất thêm báo cáo thống kê chi tiết về các ảnh bị loại bỏ trong bước làm sạch. |
| [`src/data/split.py`](../src/data/split.py) | Thực hiện thuật toán chia tập **Group-Stratified Split (70% Train / 15% Val / 15% Test)**: giữ toàn bộ các ảnh có cùng `cluster_id` (gần trùng pHash) nằm trong cùng một tập để triệt tiêu hoàn toàn rò rỉ dữ liệu qua biên giới split. | `SplitConfig`, `assign_splits()`, `materialize_splits()`, `_stable_group_key()`, `_allocate_bucket()` | `src/data/split_dataset.py`, `src/data/validation.py`, `src/data/run_pipeline.py` | Khi cần thay đổi tỉ lệ chia tập hoặc thuật toán phân bổ cụm vào các thùng `train/val/test`. |
| [`src/data/split_dataset.py`](../src/data/split_dataset.py) | Gói bước gán nhãn split và hiện thực hóa các tệp ảnh đã được chọn (`keep_for_Output=True`) ra cây thư mục vật lý `data/processed/<version>/{train,val,test}/<class_name>/`. | `run_split()` | `src/data/split.py`, `src/data/manifest.py`, `src/data/run_pipeline.py` | Khi cần thay đổi cơ chế tạo tệp vật lý (hardlink vs copy) khi xuất bộ dữ liệu đã chia. |
| [`src/data/validation.py`](../src/data/validation.py) | Bộ kiểm định độc lập sau khi chia tập (`python -m src.data.validation`): xác minh đủ 10 lớp ở mỗi split, không có ảnh hỏng, không có trùng SHA-256 và không có bất kỳ cụm pHash nào xuất hiện ở nhiều hơn một split. | `ValidationReport`, `validate_processed_dataset()`, `require_valid_processed_dataset()`, `main()` | `src/data/dedup.py`, `src/data/schema.py`, `src/detection/classifier_data.py` | Khi cần bổ sung điều kiện nghiệm thu bắt buộc mới đối với bộ dữ liệu trước khi cho phép huấn luyện. |
| [`src/data/balance_classes.py`](../src/data/balance_classes.py) | Tính toán trọng số nghịch đảo tần suất lớp (`N / (K * n_c)`) từ tập Train và lưu vào tệp JSON để dùng trong hàm mất mát `CrossEntropyLoss` mà không cần nhân bản ảnh trên đĩa. | `write_class_weights()` | `src/data/dataset.py`, `src/data/run_pipeline.py` | Khi cần thay đổi định dạng tệp JSON lưu trọng số cân bằng lớp. |
| [`src/data/dataset.py`](../src/data/dataset.py) | Khám phá các thư mục lớp trong tập dữ liệu đã xử lý, đếm phân bố mẫu theo lớp và tính toán tensor trọng số lớp theo đúng thứ tự chuẩn `CLASS_NAMES`. | `discover_class_names()`, `compute_class_counts()`, `compute_class_weights()`, `ordered_weight_tensor()` | `src/data/balance_classes.py`, `src/detection/classifier_data.py` | Khi cần thay đổi công thức tính trọng số lớp (ví dụ: chuyển sang Effective Number of Samples). |
| [`src/data/dataloader.py`](../src/data/dataloader.py) | Tạo phép biến đổi ảnh chuẩn hóa ImageNet (kèm RandomResizedCrop, ColorJitter cho tập Train) và khởi tạo PyTorch `DataLoader` từ thư mục `ImageFolder`. | `load_config()`, `get_transforms()`, `make_imagefolder_dataloader()` | `src/data/augmentation.py`, `configs/preprocessing_config.yaml` | Khi cần thay đổi chuỗi biến đổi `torchvision.transforms` mặc định cho dữ liệu phân loại. |
| [`src/data/augmentation.py`](../src/data/augmentation.py) | Cung cấp cấu hình nhóm các phép tăng cường dữ liệu ảnh (xoay, lật ngang, biến đổi màu) cho pipeline tiền xử lý phân loại. | `augmentation_group()` | `src/data/dataloader.py` | Khi cần tinh chỉnh tham số biên độ xoay hoặc biến đổi màu sắc của nhánh phân loại. |
| [`src/data/eda.py`](../src/data/eda.py) | Phân tích khám phá dữ liệu (EDA) từ danh sách bản ghi manifest và xuất báo cáo JSON tổng hợp số lượng mẫu theo nguồn, theo lớp, theo split và thống kê kích thước ảnh. | `build_eda_summary()`, `write_eda_report()` | `src/data/run_pipeline.py`, `src/data/schema.py` | Khi cần bổ sung thêm chỉ số thống kê mô tả vào báo cáo đầu ra của pipeline dữ liệu. |
| [`src/data/manifest.py`](../src/data/manifest.py) | Thực hiện đọc/ghi nguyên tử (ghi vào tệp tạm rồi `os.replace`) cho tệp `split_manifest.csv` và tạo mã băm SHA-256 đại diện cho toàn bộ trạng thái manifest. | `write_manifest()`, `read_manifest()`, `fingerprint_manifest()`, `_atomic_replace_text()` | `src/data/schema.py`, `src/data/clean_data.py`, `src/data/split_dataset.py` | Khi cần thay đổi cơ chế tuần tự hóa CSV hoặc cách tính fingerprint của manifest. |
| [`src/data/run_pipeline.py`](../src/data/run_pipeline.py) | CLI điều phối toàn bộ pipeline tiền xử lý dữ liệu từ đầu đến cuối (`python -m src.data.run_pipeline`): Hợp nhất nguồn ➔ Khử trùng SHA-256/pHash ➔ Chia tập 70/15/15 ➔ Kiểm định chống rò rỉ ➔ Tính trọng số lớp & Xuất báo cáo EDA. | `PipelineResult`, `run_pipeline()`, `_load_config()`, `main()` | Toàn bộ các module trong `src/data/*`, `configs/preprocessing_config.yaml` | Khi cần thêm một bước xử lý mới vào chuỗi pipeline dữ liệu tự động. |

---

### B.4. Phân hệ `src/app/` — Dịch vụ Ứng dụng & Giao diện Người dùng Tầng 5 (`12` files)

| Đường dẫn file | Chức năng chính | Thành phần / Class / Hàm quan trọng | File liên quan | Khi nào cần sửa file đó |
|---|---|---|---|---|
| [`src/app/__init__.py`](../src/app/__init__.py) | Khởi tạo gói `src.app` chứa toàn bộ logic dịch vụ và giao diện cho Streamlit Web và PyQt5 Desktop. | *(Module docstring)* | Toàn bộ `src/app/*`, `streamlit_app.py`, `app.py` | Không cần sửa trong vận hành thông thường. |
| [`src/app/detection_settings.py`](../src/app/detection_settings.py) | Quản lý toàn bộ cấu hình thời gian chạy cho giao diện phát hiện đa đối tượng: tự động chọn checkpoint mặc định theo thứ tự ưu tiên (`E4 v4_expanded` ➔ `E2/E1 Release v1.0.0-models`), nạp bảng ngưỡng hiệu chuẩn riêng từng lớp (`detection_thresholds_calibrated.json`) và đọc biến môi trường/CLI. | `DetectionWebSettings`, `DEFAULT_SSDLITE_PATH`, `DEFAULT_YOLO_PATH`, `DEFAULT_CLASSIFIER_PATH`, `load_per_class_thresholds()`, `parse_detection_settings()`, `validate_detection_settings()` | `streamlit_app.py`, `src/app/detection_logic.py`, `configs/detection_thresholds_calibrated.json` | Khi cần đổi đường dẫn checkpoint mặc định của ứng dụng, thay đổi ngưỡng mặc định (`confidence_threshold`, `iou_threshold`, `two_stage_alpha`) hoặc thêm tùy chọn cấu hình mới trên UI. |
| [`src/app/detection_service.py`](../src/app/detection_service.py) | Bọc đối tượng detector bên dưới một khóa luồng (`threading.Lock`) để đảm bảo nhiều phiên truy cập trên Streamlit không gọi đồng thời vào cùng một instance mô hình PyTorch/Ultralytics. | `DetectionService` (`predict_image`, `class_names`), `Detector` (Protocol) | `src/app/detection_logic.py`, `streamlit_app.py` | Khi cần bổ sung cơ chế đo thời gian suy luận hoặc quản lý hàng đợi yêu cầu ở tầng dịch vụ. |
| [`src/app/detection_logic.py`](../src/app/detection_logic.py) | Lắp ráp chuỗi pipeline phát hiện hoàn chỉnh từ `DetectionWebSettings` (`Base Detector` ➔ `TiledDetector` ➔ `TwoStageDetector` ➔ `PerClassThresholdDetector`), giải mã ảnh tải lên từ trình duyệt (xử lý EXIF), vẽ khung bao và tạo bảng tổng hợp số lượng rác theo lớp. | `PerClassThresholdDetector`, `ImageDetectionOutput`, `DetectionMediaError`, `build_detection_pipeline()`, `detect_image_bytes()`, `detection_rows()`, `count_rows()` | `src/app/detection_settings.py`, `src/detection/factory.py`, `src/detection/tiling.py`, `src/detection/two_stage.py`, `src/detection/visualization.py` | Khi cần thay đổi thứ tự kết nối các bộ lọc hậu xử lý trong ứng dụng hoặc thêm cột thông tin vào bảng kết quả phát hiện trên Streamlit. |
| [`src/app/classifier_service.py`](../src/app/classifier_service.py) | Cung cấp dịch vụ phân loại rác đơn vật thể độ chính xác cao (`MobileNetClassifier`) cho Tab 1 của Streamlit, tự động ưu tiên chạy qua `ONNX Runtime` nếu có tệp `.onnx` cùng thư mục (hoặc fallback về PyTorch `.pt`) và đính kèm hướng dẫn phân loại rác tại nguồn bằng tiếng Việt. | `MobileNetClassifier` (`classify`), `ClassificationOutput`, `CLASS_INFO_VI` | `streamlit_app.py`, `src/detection/mobilenetv3.py`, `src/detection/coordinates.py` | Khi cần cập nhật nội dung hướng dẫn màu thùng rác/cách xử lý tại nguồn (`CLASS_INFO_VI`) hoặc tinh chỉnh cơ chế nạp ONNX Runtime. |
| [`src/app/inference_service.py`](../src/app/inference_service.py) | Dịch vụ bọc `WastePredictor` an toàn luồng (`threading.Lock`) dùng cho luồng phân loại ảnh tĩnh của ứng dụng Desktop PyQt5 và các bài kiểm thử dịch vụ. | `InferenceService` (`predict_image`, `predict_bgr_frame`), `Predictor` (Protocol) | `src/detection/predict.py`, `src/app/app_logic.py`, `app.py` | Khi cần thay đổi giao diện gọi dự đoán giữa tầng UI Desktop và `WastePredictor`. |
| [`src/app/app_logic.py`](../src/app/app_logic.py) | Giải mã byte ảnh tải lên và chuyển đổi đối tượng `Prediction` thô thành `PredictionView` đã gắn nhãn hiển thị tiếng Việt, sẵn sàng để hiển thị lên giao diện mà không phụ thuộc vào thư viện UI. | `PredictionView`, `ScoredClassView`, `MediaInputError`, `prediction_view()`, `classify_image_bytes()` | `src/app/inference_service.py`, `src/app/presentation.py`, `src/detection/predict.py` | Khi cần thay đổi cấu trúc dữ liệu hiển thị kết quả phân loại Top-1 / Top-K. |
| [`src/app/presentation.py`](../src/app/presentation.py) | Bảng ánh xạ chuẩn từ 10 mã lớp tiếng Anh (`battery`..`trash`) sang tên hiển thị tiếng Việt (`Pin`..`Rác khác`) dùng chung cho cả Streamlit, PyQt5 và bộ vẽ khung bao `visualization.py`. | `LabelPresentation`, `DISPLAY_NAMES_VI`, `present_label()` | `src/detection/visualization.py`, `src/app/app_logic.py`, `src/app/detection_logic.py`, `src/app/main_window.py` | Khi cần chỉnh sửa thuật ngữ hiển thị tiếng Việt của bất kỳ lớp nào trong 10 lớp rác. |
| [`src/app/style.py`](../src/app/style.py) | Cung cấp các hàm sinh mã HTML/CSS và favicon trong suốt giúp giao diện Streamlit hiển thị đồng nhất, rõ ràng. | `transparent_favicon()`, `status_markup()` | `streamlit_app.py` | Khi cần thay đổi kiểu dáng thẻ trạng thái (badge) hoặc CSS tùy chỉnh trên Streamlit. |
| [`src/app/main_window.py`](../src/app/main_window.py) | Cài đặt cửa sổ chính `WasteClassifierWindow` của ứng dụng Desktop PyQt5: quản lý nút chọn ảnh, bật/tắt camera, hiển thị kết quả Top-1/Top-3 và bảng lịch sử phân loại. | `WasteClassifierWindow`, `HistoryWriter` | `app.py`, `src/app/workers.py`, `src/app/history.py`, `src/app/presentation.py` | Khi cần thay đổi bố cục các nút bấm hoặc bảng hiển thị trên ứng dụng Desktop PyQt5. |
| [`src/app/workers.py`](../src/app/workers.py) | Các lớp worker chạy trên luồng nền (`QThread` / `QObject`) của PyQt5 để đọc khung hình OpenCV và chạy suy luận mô hình mà không làm đơ giao diện chính. | `ImageInferenceWorker`, `CameraInferenceWorker`, `FramePrediction` | `src/app/main_window.py`, `src/app/inference_service.py` | Khi cần điều chỉnh tốc độ lấy mẫu khung hình camera hoặc cơ chế giải phóng thiết bị webcam khi đóng ứng dụng. |
| [`src/app/history.py`](../src/app/history.py) | Lưu trữ và truy vấn lịch sử các lần phân loại ảnh tĩnh của người dùng bằng cơ sở dữ liệu nhẹ SQLite (`outputs/history.sqlite3`). | `HistoryEntry`, `HistoryRepository` (`record`, `recent`, `close`) | `app.py`, `src/app/main_window.py` | Khi cần thêm trường thông tin mới vào bảng lịch sử SQLite của ứng dụng Desktop. |

---

### B.5. Danh mục 37 Script CLI & Tiện ích đang hoạt động tại `scripts/*.py`

| Đường dẫn script | Nhóm chức năng | Mô tả nhiệm vụ chính |
|---|---|---|
| [`scripts/__init__.py`](../scripts/__init__.py) | Khởi tạo gói | Đánh dấu `scripts` là gói để các bài test trong `tests/integration/` và `tests/qa/` có thể import trực tiếp các hàm tiện ích. |
| [`scripts/download_models.py`](../scripts/download_models.py) | Tải & Kiểm định Trọng số | Tải tự động các gói trọng số từ GitHub Release `v1.0.0-models`, đối chiếu kích thước byte + mã băm SHA-256 và giải nén an toàn vào `artifacts/`. |
| [`scripts/download_datasets.py`](../scripts/download_datasets.py) | Tải Dữ liệu Nguồn | Tải đúng phiên bản đã khóa của 2 bộ dữ liệu Kaggle (VN Trash v1, Garbage v2 v12) qua KaggleHub và sinh tệp cấu hình cục bộ. |
| [`scripts/import_ecovision_checkpoint.py`](../scripts/import_ecovision_checkpoint.py) | Chuyển đổi Trọng số | Nhập trọng số MobileNetV3-Large nguồn mở sang định dạng checkpoint tự mô tả chuẩn của dự án (`src/detection/checkpoint.py`). |
| [`scripts/train_official_mobilenetv3.py`](../scripts/train_official_mobilenetv3.py) | Huấn luyện GĐ1 | Script chạy huấn luyện chính thức mô hình phân loại `MobileNetV3-Large` trên Split V2 (`artifacts/official_run/best_model.pt`). |
| [`scripts/reproduce_validation.py`](../scripts/reproduce_validation.py) | Kiểm định GĐ1 | Tái lập độc lập kết quả đánh giá trên tập Validation của `MobileNetV3-Large` (xác minh SHA-256 checkpoint và manifest). |
| [`scripts/evaluate_final_test.py`](../scripts/evaluate_final_test.py) | Kiểm định GĐ1 | Đánh giá độc lập cổng nghiệm thu cuối cùng (Gate A) của `MobileNetV3-Large` trên tập Test và đo độ trễ suy luận CPU. |
| [`scripts/benchmark_classifier_runtime.py`](../scripts/benchmark_classifier_runtime.py) | Đo hiệu năng GĐ1 | Xuất `MobileNetV3-Large` sang ONNX và đo tốc độ suy luận thực tế giữa PyTorch CPU, PyTorch GPU và ONNX Runtime. |
| [`scripts/run_experiment_matrix.py`](../scripts/run_experiment_matrix.py) | Điều phối Thực nghiệm GĐ2 | Chạy ma trận thực nghiệm phát hiện đa đối tượng (`E1 YOLOv8n`, `E2 SSDLite320`, `WBF`), tự động gọi `tune` trên Val và `evaluate` trên Test. |
| [`scripts/train_detection_v2.py`](../scripts/train_detection_v2.py) | Huấn luyện GĐ2 | Trình chạy huấn luyện và đánh giá các mô hình phát hiện trên bộ dữ liệu thực tế `detection_real_v2`. |
| [`scripts/benchmark_inference.py`](../scripts/benchmark_inference.py) | Đo hiệu năng GĐ2 | Đo thời gian suy luận trung bình (ms/ảnh) và FPS của `SSDLite320` và `YOLOv8n`. |
| [`scripts/compare_eval_v1_vs_v2.py`](../scripts/compare_eval_v1_vs_v2.py) | So sánh Thực nghiệm | So sánh định lượng kết quả đánh giá giữa các phiên bản pipeline phát hiện. |
| [`scripts/generate_thesis_charts.py`](../scripts/generate_thesis_charts.py) | Biểu đồ Khóa luận | Sinh toàn bộ các biểu đồ chất lượng xuất bản (Loss/mAP curves, Per-class AP50, Confusion Matrix...) vào `reports/figures/`. |
| [`scripts/generate_authoritative_dataset_metrics.py`](../scripts/generate_authoritative_dataset_metrics.py) | Thống kê Dữ liệu | Tính toán bảng thống kê số lượng ảnh và số lượng khung bao chuẩn xác từ các tệp nhãn trên đĩa. |
| [`scripts/check_split_leakage.py`](../scripts/check_split_leakage.py) | Kiểm toán Chống rò rỉ | Kiểm tra chéo toàn bộ mã băm SHA-256 và khoảng cách Hamming pHash giữa các tập `train/val/test` để đảm bảo 0 rò rỉ dữ liệu. |
| [`scripts/verify_phash_cross_split_and_holdout.py`](../scripts/verify_phash_cross_split_and_holdout.py) | Kiểm toán Chống rò rỉ | Kiểm tra khoảng cách pHash giữa tập huấn luyện mở rộng và tập đối chứng độc lập `DEV-27` (`data/holdout`). |
| [`scripts/validate_detection_annotations.py`](../scripts/validate_detection_annotations.py) | Kiểm định Nhãn BBox | Kiểm tra hình học khung bao YOLO (tọa độ `0..1`, diện tích tối thiểu, chồng lấp trùng lặp IoU cao cùng lớp). |
| [`scripts/validate_dataset_v3_quality_gates.py`](../scripts/validate_dataset_v3_quality_gates.py) | Cổng Chất lượng Dữ liệu | Thư viện kiểm tra các tiêu chuẩn Quality Gates đối với hình học khung bao, taxonomy 10 lớp và trùng lặp bbox. |
| [`scripts/run_dataset_quality_gates_audit.py`](../scripts/run_dataset_quality_gates_audit.py) | Cổng Chất lượng Dữ liệu | Chạy kiểm toán tự động các cổng chất lượng dữ liệu trên bộ dữ liệu phát hiện. |
| [`scripts/validate_pilot_quality_gates.py`](../scripts/validate_pilot_quality_gates.py) | Cổng Chất lượng Dữ liệu | Kiểm tra chất lượng các lô ảnh thử nghiệm (pilot batches) trước khi hợp nhất vào tập huấn luyện. |
| [`scripts/validate_pilot_v3_pretraining_readiness.py`](../scripts/validate_pilot_v3_pretraining_readiness.py) | Cổng Chất lượng Dữ liệu | Kiểm tra tính sẵn sàng toàn diện trước khi chạy huấn luyện thử nghiệm Pilot v3. |
| [`scripts/verify_yolo_v3_compatibility.py`](../scripts/verify_yolo_v3_compatibility.py) | Kiểm định Tương thích | Xác minh tính tương thích của cấu trúc dữ liệu và nhãn đối với bộ nạp dữ liệu Ultralytics YOLOv8. |
| [`scripts/verify_yolo_dataloader_batches.py`](../scripts/verify_yolo_dataloader_batches.py) | Kiểm định Tương thích | Kiểm tra trực tiếp các batch tensor sinh ra từ DataLoader phát hiện để đảm bảo không lỗi nhãn/tọa độ. |
| [`scripts/prepare_detection_dataset.py`](../scripts/prepare_detection_dataset.py) | Chuẩn bị Dữ liệu GĐ2 | Chuẩn bị và kiểm tra cấu trúc bộ dữ liệu phát hiện YOLO từ nguồn ảnh và nhãn đã quy hoạch. |
| [`scripts/build_detection_splits.py`](../scripts/build_detection_splits.py) | Chia tập Dữ liệu GĐ2 | Xây dựng các tập chia `train/val/test` cho bài toán phát hiện đối tượng. |
| [`scripts/build_real_v2_splits.py`](../scripts/build_real_v2_splits.py) | Chia tập Dữ liệu GĐ2 | Tạo bản chia tập `splits_real_v2` khử trùng lặp pHash và cô lập theo cụm bối cảnh. |
| [`scripts/build_dataset_v3_candidate.py`](../scripts/build_dataset_v3_candidate.py) | Mở rộng Dữ liệu | Xây dựng bộ dữ liệu ứng viên v3 sử dụng cổng an toàn `safe_dataset_builder.py`. |
| [`scripts/audit_v3_splits_and_exclusions.py`](../scripts/audit_v3_splits_and_exclusions.py) | Kiểm toán Dữ liệu | Kiểm toán các tệp chia tập v3 và danh sách các ảnh bị loại trừ (quarantine/exclusions). |
| [`scripts/collect_real_detection_data.py`](../scripts/collect_real_detection_data.py) | Thu thập Dữ liệu | Thu thập và chuyển đổi nhãn COCO sang YOLO 10 lớp từ các nguồn dữ liệu rác thực tế. |
| [`scripts/collect_real_detection_v2.py`](../scripts/collect_real_detection_v2.py) | Thu thập Dữ liệu | Phiên bản v2 của công cụ thu thập và chuẩn hóa ảnh phát hiện rác thực tế. |
| [`scripts/import_and_qa_roboflow_battery.py`](../scripts/import_and_qa_roboflow_battery.py) | Bổ sung Lớp Hiếm | Giải nén an toàn (chống Zip-Slip) và kiểm định chất lượng ảnh lớp `battery` từ nguồn bổ sung. |
| [`scripts/ingest_roboflow_battery_export.py`](../scripts/ingest_roboflow_battery_export.py) | Bổ sung Lớp Hiếm | Chuẩn hóa khung bao và nạp dữ liệu lớp `battery` đã qua kiểm định vào kho dữ liệu. |
| [`scripts/stage_phase4e_whitelist_candidates.py`](../scripts/stage_phase4e_whitelist_candidates.py) | Bổ sung Dữ liệu | Đưa các ảnh ứng viên nằm trong danh sách trắng (whitelist) đã duyệt Visual QA vào khu vực staging. |
| [`scripts/acquire_and_qa_task8f_scarce_classes.py`](../scripts/acquire_and_qa_task8f_scarce_classes.py) | Bổ sung Lớp Hiếm | Thu thập và kiểm định chống trùng lặp (dHash/SHA-256) cho các lớp rác hiếm thuộc Task 8F. |
| [`scripts/correct_task8f_semantic_qa.py`](../scripts/correct_task8f_semantic_qa.py) | Chuẩn hóa Ngữ nghĩa Nhãn | Khử khung bao trùng lặp và rà soát tính chính xác ngữ nghĩa 10 lớp cho dữ liệu mở rộng Task 8F. |
| [`scripts/field_capture_import_and_qa.py`](../scripts/field_capture_import_and_qa.py) | Nhập Ảnh Thực địa | Nhập và kiểm tra chất lượng ảnh chụp thực địa (first-party field capture), đảm bảo không rò rỉ sang tập Val/Test. |
| [`scripts/recompile_clean_bytecode.py`](../scripts/recompile_clean_bytecode.py) | Vệ sinh Môi trường | Xóa toàn bộ tệp `.pyc` cũ có chứa đường dẫn ngoại lai và biên dịch lại bytecode sạch cho toàn bộ workspace. |

---

## C. Mapping theo kiến trúc đồ án

### C.1. Tầng Dữ liệu (Data Preprocessing, EXIF, Dedup, Split & Augmentation)
- **Chuẩn hóa 10 lớp & Ánh xạ nhãn nguồn:**
  - Nhánh Phân loại (Giai đoạn 1): [`src/data/schema.py`](../src/data/schema.py) (`CLASS_NAMES`, `load_label_mapping`, `map_to_unified_label`) + [`data/metadata/label_mapping.csv`](../data/metadata/label_mapping.csv).
  - Nhánh Phát hiện đa đối tượng (Giai đoạn 2): [`src/detection/schema.py`](../src/detection/schema.py) (`DETECTION_CLASS_NAMES`, `map_source_label`) + [`configs/detection_source_mapping.yaml`](../configs/detection_source_mapping.yaml).
- **Khử trùng lặp SHA-256 / pHash BK-Tree & Chia tập chống rò rỉ (Group-Stratified Split 70/15/15):**
  - [`src/data/merge_datasets.py`](../src/data/merge_datasets.py) ➔ [`src/data/dedup.py`](../src/data/dedup.py) (`sha256_file`, `phash_bits`, `find_duplicate_clusters`) ➔ [`src/data/clean_data.py`](../src/data/clean_data.py) ➔ [`src/data/split.py`](../src/data/split.py) (`assign_splits`) ➔ [`src/data/split_dataset.py`](../src/data/split_dataset.py) ➔ [`src/data/validation.py`](../src/data/validation.py) (`validate_processed_dataset`).
  - Điều phối toàn trình bởi [`src/data/run_pipeline.py`](../src/data/run_pipeline.py).
- **Kiểm định hình học BBox, Chuẩn hóa EXIF & Tăng cường dữ liệu Detection:**
  - Xử lý chiều ảnh EXIF & phép biến đổi tọa độ: [`src/detection/coordinates.py`](../src/detection/coordinates.py) (`ensure_exif_orientation`, `letterbox_image`, `scale_coords_back`, `clip_and_validate_box`).
  - Kiểm định dataset YOLO & Cổng an toàn nạp dữ liệu: [`src/detection/dataset.py`](../src/detection/dataset.py), [`src/detection/prepare.py`](../src/detection/prepare.py), [`src/detection/safe_dataset_builder.py`](../src/detection/safe_dataset_builder.py), [`src/detection/export_train_manifest.py`](../src/detection/export_train_manifest.py).
  - Tăng cường dữ liệu & Cân bằng lớp hiếm: [`src/detection/augmentation.py`](../src/detection/augmentation.py) (4 mức Albumentations), [`src/detection/copy_paste.py`](../src/detection/copy_paste.py) (`DonorBank`, `apply_copy_paste`) và [`src/detection/balance_split.py`](../src/detection/balance_split.py) (`build_balanced_split`).

### C.2. Tầng 3A — Mô hình Chính: `SSDLite320-MobileNetV3` (Primary Detector)
- **Kiến trúc & Suy luận (`320×320`):** [`src/detection/ssdlite.py`](../src/detection/ssdlite.py) (`SSDLiteWasteDetector`, `build_ssdlite`, `_transfer_stage1_backbone_weights`).
  - Kế thừa trọng số trích xuất đặc trưng từ mô hình phân loại [`src/detection/mobilenetv3.py`](../src/detection/mobilenetv3.py).
  - Tuân thủ nghiêm ngặt quy ước nhãn nội bộ Torchvision SSD: `0 = Background`, `1..10 = Foreground` (cộng `+1` khi nạp nhãn huấn luyện trong `YoloBoxDataset`, trừ `-1` khi giải mã dự đoán trong `tensor_detections` để trả về `class_index` chuẩn `0..9`).
- **Huấn luyện & Kiểm soát tài nguyên:** [`src/detection/ssdlite_train.py`](../src/detection/ssdlite_train.py) (`train_ssdlite`, `evaluate_validation`), [`src/detection/training_common.py`](../src/detection/training_common.py) (`dataset_fingerprint`, `seed_everything`) và [`src/detection/preflight_guard.py`](../src/detection/preflight_guard.py).

### C.3. Tầng 3B — Mô hình Đối chứng: `YOLOv8n` (Control Detector)
- **Bộ bọc mô hình & Kiểm tra kiến trúc (`imgsz=640`):** [`src/detection/yolo.py`](../src/detection/yolo.py) (`WasteDetector`, `validate_yolov8n_architecture`, `validate_yolo_metadata`).
  - Sử dụng trực tiếp chỉ số lớp `0..9` trùng khớp với `DETECTION_CLASS_NAMES`.
  - Giữ nguyên độ phân giải suy luận gốc của checkpoint (`effective_image_size = 640`), không bị ép xuống `320×320`.
- **Huấn luyện, Tối ưu ngưỡng & Xuất ONNX:** [`src/detection/train.py`](../src/detection/train.py), [`src/detection/tune.py`](../src/detection/tune.py), [`src/detection/export.py`](../src/detection/export.py).

### C.4. Prediction Adapter & Hợp đồng Đầu ra Chung (Output Contract)
- Mọi mô hình phát hiện (`SSDLiteWasteDetector`, `WasteDetector`, `FusionDetector`, `TiledDetector`, `TwoStageDetector`, `PerClassThresholdDetector`) đều tuân thủ cùng một giao diện `predict_image(...) -> DetectionResult` được định nghĩa tại:
  - [`src/detection/types.py`](../src/detection/types.py):
    - `BoundingBox(x1, y1, x2, y2)` — tọa độ pixel tuyệt đối trên hệ quy chiếu ảnh gốc (sau khi đã chuẩn hóa EXIF).
    - `Detection(box, class_index, class_name, confidence)` — đảm bảo `class_index in 0..9` và `class_name == DETECTION_CLASS_NAMES[class_index]`.
    - `DetectionResult(image_width, image_height, detections, inference_ms)` — cung cấp thuộc tính `class_counts` đếm số vật thể theo từng lớp.
  - [`src/detection/factory.py`](../src/detection/factory.py): Hàm `create_detector(model_path, backend=..., ...)` giúp Tầng 4, Tầng 5 và CLI đánh giá (`evaluate.py`) chuyển đổi giữa `ssdlite`, `yolov8n` và `wbf` mà không cần sửa code gọi hàm.

### C.5. Tầng 4 — Hợp nhất Khung bao WBF & Tinh chỉnh Đa giai đoạn (Refinement Layer)
- **Weighted Boxes Fusion (WBF):** [`src/detection/fusion.py`](../src/detection/fusion.py) (`FusionDetector`, `fuse_detections`) hợp nhất các dự đoán từ `SSDLite320` và `YOLOv8n` theo từng lớp (`class-wise`), chuẩn hóa tọa độ về `[0, 1]` khi chạy WBF và khôi phục về tọa độ pixel ảnh gốc kèm bước lọc khung bao lồng nhau (`containment_threshold`).
- **Suy luận cắt ô cửa sổ trượt (SAHI Tiling):** [`src/detection/tiling.py`](../src/detection/tiling.py) (`TiledDetector`) chia ảnh lớn thành các ô chồng lấp để tăng độ nhạy với rác kích thước nhỏ.
- **Thẩm định 2 giai đoạn (Two-Stage Refinement):** [`src/detection/two_stage.py`](../src/detection/two_stage.py) (`TwoStageDetector`, `MobileNetBatchAdapter`) dùng mô hình phân loại `MobileNetV3-Large` chấm điểm lại từng vùng cắt proposal để khử dương tính giả.
- **Làm mượt & Trực quan hóa:** [`src/detection/tracking.py`](../src/detection/tracking.py) (`TemporalDetectionSmoother`) và [`src/detection/visualization.py`](../src/detection/visualization.py) (`draw_detections`).

### C.6. Tầng 5 — Giao diện Ứng dụng (Streamlit Web & PyQt5 Desktop) và Thống kê
- **Ứng dụng Web Streamlit 2 Tabs:** [`streamlit_app.py`](../streamlit_app.py) kết hợp với [`src/app/detection_settings.py`](../src/app/detection_settings.py), [`src/app/detection_logic.py`](../src/app/detection_logic.py), [`src/app/detection_service.py`](../src/app/detection_service.py), [`src/app/classifier_service.py`](../src/app/classifier_service.py), [`src/app/presentation.py`](../src/app/presentation.py) và [`src/app/style.py`](../src/app/style.py).
- **Ứng dụng Desktop PyQt5:** [`app.py`](../app.py) kết hợp với [`src/app/main_window.py`](../src/app/main_window.py), [`src/app/workers.py`](../src/app/workers.py), [`src/app/history.py`](../src/app/history.py), [`src/app/inference_service.py`](../src/app/inference_service.py) và [`src/app/app_logic.py`](../src/app/app_logic.py).

---

## D. Cách chạy hệ thống & Vị trí tra cứu Cấu hình / Trọng số / Metrics

### D.1. Các lệnh CLI thực tế (Đã xác minh hoạt động 100%)

*(Chạy tại thư mục gốc repository sau khi đã kích hoạt môi trường ảo `.\.venv\Scripts\Activate.ps1`)*

#### 1. Tải trọng số mô hình và khởi chạy ứng dụng
```powershell
# Tải và xác minh SHA-256 bộ 3 trọng số mô hình chạy ứng dụng từ GitHub Release v1.0.0-models
python scripts/download_models.py

# Chỉ kiểm tra lại SHA-256 các trọng số hiện có trên đĩa
python scripts/download_models.py --verify-only

# Khởi chạy ứng dụng Web Streamlit (2 Tab: Phân loại MobileNetV3-Large & Phát hiện đa đối tượng)
streamlit run streamlit_app.py

# Khởi chạy ứng dụng Desktop PyQt5
python app.py --checkpoint artifacts/official_run/best_model.pt --device auto
python app.py --help
```

#### 2. Pipeline tiền xử lý dữ liệu & Kiểm định chống rò rỉ (Giai đoạn 1)
```powershell
# Xem hướng dẫn tham số CLI tiền xử lý và kiểm định
python -m src.data.run_pipeline --help
python -m src.data.validation --help

# Chạy pipeline tiền xử lý dữ liệu đầu-cuối
python -m src.data.run_pipeline `
  --config data/sources/preprocessing_config.local.yaml `
  --raw-root data/raw `
  --metadata-root data/metadata/v1 `
  --processed-root data/processed/v1 `
  --report-root outputs/data/v1

# Kiểm định độc lập bộ dữ liệu đã chia (đủ 10 lớp, 0 ảnh lỗi, 0 rò rỉ cụm pHash)
python -m src.data.validation `
  --dataset-root data/processed/v1 `
  --manifest data/metadata/v1/split_manifest.csv
```

#### 3. Huấn luyện, Đánh giá & Dự đoán với Bộ phân loại `MobileNetV3-Large` (Giai đoạn 1)
```powershell
# Xem hướng dẫn CLI của nhánh Classifier
python -m src.detection.classifier_train --help
python -m src.detection.classifier_evaluate --help
python -m src.detection.predict --help

# Dự đoán phân loại trên 1 ảnh bất kỳ bằng CLI
python -m src.detection.predict `
  --checkpoint artifacts/official_run/best_model.pt `
  --image path/to/image.jpg `
  --top-k 3 `
  --confidence-threshold 0.55 `
  --device auto
```

#### 4. Chuẩn bị dữ liệu, Huấn luyện & Đánh giá Phát hiện Đa đối tượng (Giai đoạn 2 — Tầng 3A, 3B, Tầng 4)
```powershell
# Xem hướng dẫn CLI của nhánh Object Detection
python -m src.detection.prepare --help
python -m src.detection.train --help
python -m src.detection.evaluate --help

# Huấn luyện mô hình chính SSDLite320-MobileNetV3 (Tầng 3A)
python -m src.detection.train `
  --data configs/detection_dataset.yaml `
  --config configs/detection_training_combined_ssdlite.yaml `
  --model ssdlite320_mobilenet_v3_large `
  --device cuda

# Huấn luyện mô hình đối chứng YOLOv8n (Tầng 3B)
python -m src.detection.train `
  --data configs/detection_dataset.yaml `
  --config configs/detection_training_combined_yolov8n.yaml `
  --model yolov8n.pt `
  --device cuda

# Tìm ngưỡng suy luận tối ưu trên tập Validation (không chạm tập Test)
python -m src.detection.tune `
  --data configs/detection_dataset.yaml `
  --backend wbf `
  --model artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt `
  --secondary-model artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt `
  --output artifacts/detection_v2/WBF_selected_validation_config.json

# Đánh giá độc lập trên tập Validation hoặc Test với cấu hình đã khóa
python -m src.detection.evaluate `
  --data configs/detection_dataset.yaml `
  --selection artifacts/detection_v2/WBF_selected_validation_config.json `
  --split test `
  --output artifacts/detection_v2/WBF_test_evaluation.json
```

#### 5. Chạy bộ kiểm thử tự động & Kiểm tra biên dịch Bytecode
```powershell
# Kiểm tra cú pháp và biên dịch bytecode toàn bộ mã nguồn
python -m compileall -q src tests app.py streamlit_app.py scripts

# Chạy toàn bộ 519 bài kiểm thử tự động (chế độ Qt headless)
$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; pytest -q
```

---

### D.2. Bảng tra cứu nhanh vị trí Cấu hình, Trọng số và Báo cáo Metrics

| Nội dung cần tìm | Vị trí chính xác trong Repository |
|---|---|
| **Bảng ánh xạ nhãn 10 lớp chuẩn** | [`data/metadata/label_mapping.csv`](../data/metadata/label_mapping.csv) (Phân loại) & [`configs/detection_source_mapping.yaml`](../configs/detection_source_mapping.yaml) (Phát hiện) |
| **Cấu hình tiền xử lý dữ liệu GĐ1** | [`configs/preprocessing_config.yaml`](../configs/preprocessing_config.yaml) |
| **Cấu hình huấn luyện MobileNetV3-Large** | [`configs/model_config.yaml`](../configs/model_config.yaml) |
| **Cấu hình huấn luyện SSDLite320 (Tầng 3A)** | [`configs/detection_training.yaml`](../configs/detection_training.yaml) & [`configs/detection_training_combined_ssdlite.yaml`](../configs/detection_training_combined_ssdlite.yaml) |
| **Cấu hình huấn luyện YOLOv8n (Tầng 3B)** | [`configs/detection_training_yolov8n.yaml`](../configs/detection_training_yolov8n.yaml) & [`configs/detection_training_combined_yolov8n.yaml`](../configs/detection_training_combined_yolov8n.yaml) |
| **Cấu hình hợp nhất WBF (Tầng 4)** | [`configs/detection_fusion.yaml`](../configs/detection_fusion.yaml) |
| **Ngưỡng nhạy hiệu chuẩn riêng từng lớp** | [`configs/detection_thresholds_calibrated.json`](../configs/detection_thresholds_calibrated.json) |
| **Danh mục trọng số phát hành & SHA-256** | [`configs/model_release_manifest.json`](../configs/model_release_manifest.json) |
| **Trọng số mặc định khi chạy ứng dụng** | `artifacts/official_run/best_model.pt` (Classifier), `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` (hoặc bản `E4`), `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` (hoặc bản `E4`) — được quản lý tự động trong [`src/app/detection_settings.py`](../src/app/detection_settings.py) |
| **Tiêu chuẩn nghiệm thu Quality Gates của PM** | [`docs/pm/SSDLITE320_QUALITY_GATES.md`](pm/SSDLITE320_QUALITY_GATES.md) |
| **Biểu đồ khóa luận & Báo cáo QA độc lập** | [`reports/figures/`](../reports/figures) và [`reports/qa_acceptance/`](../reports/qa_acceptance) |

---

## E. Các thư mục lịch sử & Quy tắc bảo tồn bằng chứng

Để giữ cho thư mục gốc và `src/`, `scripts/` luôn gọn gàng mà vẫn bảo toàn 100% tính khả kiểm (traceability) của khóa luận qua các giai đoạn, các tệp lịch sử được cách ly rõ ràng tại 4 khu vực sau:

1. **[`scripts/archive/`](../scripts/archive) (`183` files `.py` chia vào `6` thư mục con):**
   - `scripts/archive/legacy_src_unused/` (`8` files): Các module giao diện thử nghiệm cũ không còn được gọi trong luồng chính (`lighting_enhance.py`, `review_tool.py`, `detection_live.py`, `detection_video.py`, `live.py`, `video.py`, `smoothing.py`, `settings.py`).
   - `scripts/archive/scratch_diagnostics/` (`21` files): Các script cắt ảnh, đo đạc contour và chẩn đoán cục bộ một lần.
   - `scripts/archive/phase4_delivery/` (`38` files): Các script đóng gói và xuất báo cáo bàn giao thuộc Phase 4.
   - `scripts/archive/task5_to_7_battery/` (`25` files): Các script thu thập, gán nhãn lại và kiểm toán chuyên biệt cho lớp `battery` ở Task 5–7.
   - `scripts/archive/task8_expansion/` (`35` files): Các script mở rộng dữ liệu đa lớp quy mô lớn thuộc chuỗi Task 8A–8I.
   - `scripts/archive/legacy_audits/` (`56` files): Các script kiểm toán nguồn gốc dữ liệu và thử nghiệm smoke test từ các giai đoạn đầu.
2. **[`reports/multi_waste_detection_audit/20261008_192500/source_snapshot/`](../reports/multi_waste_detection_audit/20261008_192500/source_snapshot) (`9` files `.py`):**
   - Là ảnh chụp mã nguồn lịch sử (historical audit snapshot) được Git theo dõi tại thời điểm kiểm toán `20261008_192500` và được tham chiếu bởi bài kiểm tra toàn vẹn `reports/qa_acceptance/T1_holdout_integrity.py`. Giữ nguyên tại chỗ để bảo toàn lịch sử Git và bằng chứng kiểm toán.
3. **[`reports/archive/task_reports/`](../reports/archive/task_reports) (`17` files `.md`):**
   - Lưu trữ các báo cáo tiến độ theo từng Task lịch sử (`TASK4_*.md` ➔ `TASK8A_*.md`, `AI_UPGRADE_BASELINE.md`, `BASELINE_CORRECTION.md`, `DATASET_QA_REPORT.md`, `FAILURE_ANALYSIS.md`, `BAO_CAO_PHAN_1_MAPPING_10_LOP.md`) đã được chuyển từ thư mục gốc vào khu vực lưu trữ.
4. **[`reports/archive/delivery_manifests/`](../reports/archive/delivery_manifests) (`2` files `.csv`):**
   - Lưu trữ các tệp manifest kiểm kê gói bàn giao Phần 1 (`CNTT-KLCN155_PART01_VERIFIED_R1_manifest.csv`, `CNTT-KLCN155_PART01_VERIFIED_R2_manifest.csv`).

> ⚠️ **KHUYẾN CÁO QUAN TRỌNG DÀNH CHO THÀNH VIÊN MỚI:**
> Toàn bộ các tệp trong `scripts/archive/` và `reports/multi_waste_detection_audit/.../source_snapshot/` chỉ phục vụ mục đích **lưu trữ và đối chiếu lịch sử thực nghiệm**. Khi phát triển tính năng mới, huấn luyện mô hình hoặc chạy ứng dụng, **chỉ sử dụng mã nguồn chính thức trong `src/`, `streamlit_app.py`, `app.py` và 37 script đang hoạt động tại cấp cao nhất `scripts/*.py`**.
