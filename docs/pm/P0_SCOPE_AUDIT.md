# BÁO CÁO P0 — KIỂM KÊ HIỆN TRẠNG & ĐỐI CHIẾU ĐỀ CƯƠNG KHÓA LUẬN CNTT-KLCN155

- **Dự án:** Khóa luận cử nhân ngành CNTT năm học 2026 – 2027 (Mã đề tài: `CNTT-KLCN155`)
- **Tên đề tài chính thức:** *Xây dựng hệ thống phát hiện và phân loại đa đối tượng rác thải sinh hoạt trong ảnh chụp thực tế bằng mô hình học sâu nhẹ và kỹ thuật tăng cường dữ liệu*
- **Tài liệu nguồn yêu cầu:** `CNTT-KLCN155_da_bo_sung.docx` (Lưu tại: `C:\Users\ad\Downloads\CNTT-KLCN155_da_bo_sung.docx`)
- **Người thực hiện:** Dev Anti
- **Người nhận:** Project Manager (PM)
- **Thời điểm kiểm kê:** 2026-09-27T19:43:00+07:00
- **Trạng thái Task P0:** `DONE` (Đã xác minh đầy đủ hiện vật, không bị chặn bởi việc thiếu file Word hay thiếu repo).

---

## 1. Xác định Thư mục gốc Dự án & Hiện trạng Git

### 1.1. Thông tin Kho lưu trữ (Repository)
* **Đường dẫn thư mục gốc thực tế:** `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3`
* **Git Remote URL:** `https://github.com/Nhan-create/waste-classifier-mobilenetv3.git` (Trùng khớp sinh viên Ngô Thanh Nhân - MSSV: 2001230595)
* **Branch hiện tại:** `feat/mobilenetv3-10-class` (Đang đi trước `origin/feat/mobilenetv3-10-class` 4 commit: `35b28e4`, `e92ca84`, `b159541`, `f778460`, `1fa0d97`)
* **HEAD Commit:** `1fa0d9797e73bfce82e876cd6cacecddebebd543`
  * *Author:* `Nhan-create <n***@gmail.com>` *(thông tin cá nhân đã che)*
  * *Subject:* `feat(web): make Streamlit presentation icon-free`
  * *Date:* `Thu Sep 3 18:56:51 2026 +0700`

### 1.2. Trạng thái Working Tree (`git status --short`)
```text
 M .github/workflows/ci.yml
 M .gitignore
 M README.md
 M requirements.txt
 M src/web/style.py
 M streamlit_app.py
 M tests/test_repository_contract.py
 M tests/web/test_streamlit_contract.py
?? configs/detection_dataset.yaml
?? configs/detection_fusion.yaml
?? configs/detection_training.yaml
?? configs/detection_training_rtx2050.yaml
?? docs/object-detection-training.md
?? docs/pm/P0_SCOPE_AUDIT.md
?? src/detection/
?? src/web/detection_live.py
?? src/web/detection_logic.py
?? src/web/detection_service.py
?? src/web/detection_settings.py
?? src/web/detection_video.py
?? tests/detection/
?? tests/web/test_detection_image_outputs.py
?? tests/web/test_detection_settings.py
```

### 1.3. Phân loại các tệp hiện hữu theo từng nhóm chức năng
* **Tài liệu đề cương & Hướng dẫn:**
  * File gốc Word: `C:\Users\ad\Downloads\CNTT-KLCN155_da_bo_sung.docx`
  * Tài liệu đào tạo detection nội bộ: `docs/object-detection-training.md`
  * Quy chuẩn provenance cũ: `docs/model-provenance.md`
* **Cấu hình (Configs):**
  * Dataset detection 6 lớp: `configs/detection_dataset.yaml`
  * WBF Fusion parameters: `configs/detection_fusion.yaml`
  * Huấn luyện YOLO: `configs/detection_training.yaml`, `configs/detection_training_rtx2050.yaml`
* **Dữ liệu & Nhãn (Data & Labels):**
  * Dữ liệu hiện có: `data/detection/v1/` (và mirror tại `D:\waste-training\dataset-v1\`) gồm `images/` và `labels/` định dạng YOLO `.txt`.
  * Mapping cũ: `data/metadata/label_mapping.csv`
* **Mã nguồn Huấn luyện & Đánh giá (Detection Core - `src/detection/`):**
  * Chuẩn dữ liệu: `dataset.py`, `schema.py`, `prepare.py`, `manifest.py`
  * Mô hình SSDLite: `ssdlite.py`, `ssdlite_train.py`
  * Mô hình YOLO: `yolo.py`
  * Hợp nhất WBF: `fusion.py`, `tune.py`
  * Huấn luyện chung: `train.py`, `training_common.py`
  * Đánh giá COCO: `evaluate.py`
  * Trực quan hóa: `visualization.py`, `types.py`, `tracking.py`, `factory.py`
* **Giao diện Web Streamlit (`src/web/` & `streamlit_app.py`):**
  * `streamlit_app.py`, `src/web/detection_settings.py`, `src/web/detection_logic.py`, `src/web/detection_service.py`, `src/web/style.py`
* **Bộ kiểm thử (Tests):**
  * `tests/detection/test_detection_dataset.py`, `test_fusion_evaluation.py`, `test_train.py`, `test_types_and_tracking.py`, `test_visualization.py`, `test_yolo_adapter.py`
  * `tests/web/test_detection_image_outputs.py`, `test_detection_settings.py`, `test_streamlit_contract.py`

---

## 2. Bảng Đối chiếu Yêu cầu Đề cương Word (`CNTT-KLCN155_da_bo_sung.docx`) vs Hiện trạng Dự án

*Ghi chú vị trí:* Dựa trên thứ tự đoạn văn bản (paragraphs) trích xuất trực tiếp từ XML gốc của `CNTT-KLCN155_da_bo_sung.docx`.

| STT | Vị trí trong Word | Yêu cầu trong Đề cương Word (Diễn giải ngắn gọn) | Mức độ | Hiện trạng thực tế trong repo | Bằng chứng kiểm tra | Đánh giá |
|:---:|:---|:---|:---:|:---|:---|:---:|
| 1 | Mục "Mục tiêu" (Đoạn 15, 21) | Phát hiện đa đối tượng trên ảnh chụp thực tế: Bounding Box, nhãn lớp, confidence, thống kê số lượng. Giới hạn ở ảnh tĩnh; **không** làm segmentation, **không** robot, **không** video real-time. | `WORD_BẮT_BUỘC` | Đã có schema Bounding Box, đếm số lượng, giao diện ảnh tĩnh. Code live video cũ đang được loại bỏ trong `streamlit_app.py`. | `src/detection/schema.py:27`, `streamlit_app.py:165-210`, `tests/web/test_streamlit_contract.py:10` | **ĐANG TRIỂN KHAI** (Khớp hướng) |
| 2 | Mục "Mục tiêu" & "Yêu cầu" (Đoạn 16, 22) | **Ba nguồn dữ liệu:** `VN-trash` [2], `Garbage Classification V2` [3] và ảnh tự thu thập tại TP.HCM. Thống nhất **sáu lớp:** Nhựa, Giấy/bìa, Kim loại, Thủy tinh, Hữu cơ, Rác nguy hại. | `WORD_BẮT_BUỘC` | `configs/detection_dataset.yaml` đã khai báo 6 lớp. **Tuy nhiên**, dữ liệu trên máy hiện là 10 lớp (`dataset-v1`, nguồn OpenImages + Mendeley). 3 nguồn theo Word **chưa có trên máy**. | `configs/detection_dataset.yaml:10-17`, `data/detection/v1/labels/` (đếm ra 10 lớp 0..9) | **LỆCH HIỆN TRẠNG** (Chưa nạp 3 nguồn Word) |
| 3 | Mục "Yêu cầu" (Đoạn 22) | **Quy tắc ánh xạ và loại bỏ:** Garbage V2 ánh xạ plastic, paper/cardboard, metal, glass, biological, battery vào 6 lớp; **loại shoes, clothes, trash**. VN-trash: organic -> hữu cơ, medical -> nguy hại; inorganic phải review từng ảnh hoặc loại nếu mơ hồ. | `WORD_BẮT_BUỘC` | File `data/metadata/label_mapping.csv` hiện tại là file mapping 10 lớp cũ (vẫn còn gán nhãn `trash`). Chưa có file ánh xạ và kịch bản loại bỏ theo đúng quy tắc đoạn 22. | `data/metadata/label_mapping.csv:1-10` | **CHƯA ĐẠT** (Cần viết script theo đoạn 22) |
| 4 | Mục "Yêu cầu" (Đoạn 23) | Mọi ảnh phải có bounding box được gán và kiểm tra chéo qua **CVAT**. Phân chia theo cảnh chụp (**Scene-based**) tỷ lệ **70%–15%–15%**, seed cố định, không trùng lặp cảnh giữa các tập. | `WORD_BẮT_BUỘC` | Script `src/detection/prepare.py` đã cài đặt logic chia 70/15/15 theo scene (`split_ratios: [0.70, 0.15, 0.15]`), kiểm tra hash duplicate. Nhưng chưa có nhãn export từ CVAT cho 3 nguồn Word. | `src/detection/prepare.py:70-130`, `configs/detection_dataset.yaml:7-9` | **MÃ NGUỒN CÓ SẴN - THIẾU DỮ LIỆU ĐẦU VÀO** |
| 5 | Mục "Yêu cầu" (Đoạn 24) | Tiền xử lý và tăng cường: Chuẩn hóa ảnh, dùng `Albumentations` biến đổi hình học, ánh sáng, màu sắc, mờ, nhiễu. Khảo sát **Copy-Paste** chỉ sử dụng vật thể có mặt nạ tiền cảnh (mask) đã kiểm tra; box tính lại từ mask. | `WORD_BẮT_BUỘC` | Đã cài `albumentations==2.0.8`. `configs/detection_dataset.yaml` mới để `augmentation_variant: none`. **Chưa có kho polygon mask** nào cho rác để chạy Copy-Paste. | `requirements.txt:24`, `configs/detection_dataset.yaml:9`, scan toàn bộ repo không có file polygon/mask | **THIẾU HIỆN VẬT MASK** |
| 6 | Mục "Yêu cầu" (Đoạn 25) | Huấn luyện **hai mô hình:** Mô hình chính là `SSDLite320-MobileNetV3` [6], [7]; mô hình đối chứng là `YOLOv8n` [8]. | `WORD_BẮT_BUỘC` | Mã nguồn adapter đã có: `src/detection/ssdlite.py`, `src/detection/ssdlite_train.py` và `src/detection/yolo.py`. **Mâu thuẫn:** Configs lại ghi `yolo26s.pt` và `yolo26n.pt`. Chưa có checkpoint weights `.pt`/`.pth` nào. | `configs/detection_training.yaml:2`, `configs/detection_training_rtx2050.yaml:3`, `artifacts/detection/*/weights/` (rỗng) | **MÂU THUẪN TÊN MODEL & CHƯA CÓ TRỌNG SỐ** |
| 7 | Mục "Yêu cầu" (Đoạn 26) | **Thiết kế thực nghiệm Ablation 4 nhánh:** (1) Không tăng cường, (2) Tăng cường hình học, (3) Tăng cường ánh sáng/màu sắc, (4) Kết hợp hình học - ánh sáng - Copy-Paste. | `WORD_BẮT_BUỘC` | Chưa chạy thực nghiệm. Cấu hình mới chỉ có `augmentation_variant: none`. Chưa tạo 4 ma trận config cho 4 nhánh. | `configs/detection_dataset.yaml:9` | **CHƯA THỰC HIỆN** |
| 8 | Mục "Mục tiêu" & "Yêu cầu" (Đoạn 17, 26) | **Weighted Boxes Fusion (WBF):** Hợp nhất box cùng lớp theo WBF [1]. Trọng số, threshold IoU, confidence chọn bằng Grid Search trên tập **Validation**; cấm chỉnh trên Test. | `WORD_BẮT_BUỘC` | Module `src/detection/fusion.py` đã dùng `ensemble_boxes.weighted_boxes_fusion`. File `configs/detection_fusion.yaml` đã có lưới tham số. Script `src/detection/tune.py` thực hiện tìm kiếm trên val. | `src/detection/fusion.py:1-90`, `configs/detection_fusion.yaml:1-12`, `src/detection/tune.py:30-80` | **ĐÃ CÀI ĐẶT CODE THUẬT TOÁN** (Chờ weights để chạy) |
| 9 | Mục "Yêu cầu" (Đoạn 27) | Báo cáo mAP@0.5, mAP@0.5:0.95, AP từng lớp theo COCO; Precision, Recall, F1. Stress-test (thiếu sáng, ngược sáng, nhỏ, che khuất). Đo latency batch 1, FPS, model size, tài nguyên. | `WORD_BẮT_BUỘC` | Module `src/detection/evaluate.py` đã tích hợp `pycocotools` tính AP, mAP50, mAP50-95 và đo latency batch 1. Chưa có tập test stress-test thực tế để chạy. | `src/detection/evaluate.py:50-180`, `requirements.txt:26` | **MÃ NGUỒN CÓ SẴN - CHỜ MÔ HÌNH** |
| 10 | Mục "Yêu cầu" (Đoạn 28) | **Ứng dụng Web:** Tải 1 hoặc nhiều ảnh, vẽ bounding box, nhãn lớp, confidence, thống kê số lượng theo nhóm, thanh trượt chỉnh confidence, đo thời gian xử lý, thông báo khi không có rác, nút tải ảnh kết quả. | `WORD_BẮT_BUỘC` | `streamlit_app.py` đã triển khai: upload đa ảnh (`accept_multiple_files=True`), slider confidence, bảng thống kê nhóm, hiển thị latency, nút tải zip ảnh annotated. | `streamlit_app.py:130-220`, `src/web/detection_settings.py:150-230` | **ĐÃ TRIỂN KHAI PHẦN LỚN UI** |
| 11 | Mục "Môi trường" (Đoạn 31-35) | GPU T4 16GB (Colab) / máy cá nhân CPU 4 nhân, RAM 8GB. HĐH Windows/Linux. Ngôn ngữ **Python 3.10**. Thư viện: PyTorch, Torchvision, Ultralytics, Albumentations, Streamlit... | `WORD_BẮT_BUỘC` | Máy cục bộ hiện tại chạy **Python 3.12.10** (Venv: Python 3.12.13). GPU cục bộ là NVIDIA RTX 2050 4GB. Thư viện trong venv cài `torch==2.3.1+cpu` (chưa nhận GPU CUDA). | Log `python --version`, `nvidia-smi`, `pip list` | **LỆCH PHIÊN BẢN PYTHON & CUDA** |
| 12 | Mục "Sản phẩm" (Đoạn 29) | ONNX hoặc TensorFlow Lite là nội dung mở rộng nếu còn thời gian. | `WORD_CHO_PHÉP` | Repo có file `src/detection/export.py` hỗ trợ export LiteRT/TFLite, đúng tính chất mở rộng. | `src/detection/export.py:1-45` | **KHỚP ĐỀ CƯƠNG** (Mở rộng tùy chọn) |

---

## 3. Kiểm tra Riêng 9 Khẳng định Rủi ro Cao

| STT | Khẳng định cần kiểm tra | Phân loại | Căn cứ trong file Word (`CNTT-KLCN155_da_bo_sung.docx`) | Bằng chứng thực tế trong Repository | Đánh giá & Rủi ro |
|:---:|:---|:---:|:---|:---|:---|
| 1 | **Ba nguồn dữ liệu** (VN-trash, Garbage V2, tự chụp TP.HCM) | `WORD_BẮT_BUỘC` | Đoạn 22: *"Sử dụng VN-trash [2], Garbage Classification V2 [3] và ảnh do nhóm tự thu thập tại TP.HCM."* | Trong repo chỉ có `data/detection/v1` (nguồn OpenImages + Mendeley). Ba nguồn Word chưa tải về. | **Rủi ro lệch đề cương nghiêm trọng.** Phải thu thập đúng 3 nguồn Word. |
| 2 | **Ánh xạ 6 lớp & loại bỏ shoes, clothes, trash** | `WORD_BẮT_BUỘC` | Đoạn 22: Thống nhất 6 lớp; *"loại shoes, clothes và trash khỏi tập nếu không thể gán chắc chắn. Với VN-trash, organic và medical được ánh xạ... nhãn inorganic phải được xem lại từng ảnh..."* | Repo hiện có `data/metadata/label_mapping.csv` vẫn còn map nhãn `trash`. Thư mục `v1` trên máy có cả class shoes, clothes, trash. | **Rủi ro sai lệch số lượng lớp và logic tiền xử lý.** Cần tạo bộ lọc 6 lớp mới. |
| 3 | **CVAT và Bounding Box** | `WORD_BẮT_BUỘC` | Đoạn 23 & 33: Bounding box phải gán và kiểm tra chéo; công cụ ghi rõ là **CVAT**. | Trong repo chưa có file annotation XML/JSON/ZIP nào từ CVAT. | **Khối lượng việc gán nhãn thủ công rất lớn.** |
| 4 | **Chia 70/15/15 theo bối cảnh (Scene-based)** | `WORD_BẮT_BUỘC` | Đoạn 23: *"Chia dữ liệu theo cảnh chụp... tỷ lệ 70%–15%–15%, dùng seed cố định và không để ảnh cùng cảnh... xuất hiện ở nhiều tập."* | `src/detection/prepare.py` đã hiện thực logic `split_ratios: [0.70, 0.15, 0.15]` và nhóm theo `scene_id`. | **Khớp thiết kế code**, nhưng chưa có dữ liệu scene thực tế để kiểm tra. |
| 5 | **Bốn nhánh Ablation Study** | `WORD_BẮT_BUỘC` | Đoạn 26: Huấn luyện với 4 chiến lược: None, Geometric, Color/Light, Combined+Copy-Paste. | Repo mới chỉ có cờ `augmentation_variant: none` trong config. Chưa có script chạy hàng loạt 4 nhánh. | **Chưa thực hiện.** Cần lên kế hoạch thí nghiệm ở các task sau. |
| 6 | **50–100 Polygon Mask cho Copy-Paste** | `ĐỀ_XUẤT_THÊM` | Đoạn 24 chỉ ghi: *"Khảo sát Copy-Paste [5]... kỹ thuật này chỉ sử dụng các vật thể có mặt nạ tiền cảnh đã được trích xuất và kiểm tra..."* — **Hoàn toàn KHÔNG có con số 50–100 mask.** | Trong repo và máy tính: Không có bất kỳ file mask `.png`/`.json` nào. | Số lượng 50–100 chỉ là khuyến nghị AI trước đây, không phải yêu cầu cứng của Word. Tuy nhiên, **phải có mask thật** mới làm được Copy-Paste theo Word. |
| 7 | **Hai mô hình: SSDLite320-MobileNetV3 và YOLOv8n** | `WORD_BẮT_BUỘC` | Đoạn 25: *"Sử dụng SSDLite320-MobileNetV3 [6], [7] làm mô hình chính và YOLOv8n [8] làm đối chứng..."* | `src/detection/ssdlite.py` dùng torchvision SSDLite. Nhưng `configs/detection_training.yaml` lại ghi `yolo26s.pt` và `configs/detection_training_rtx2050.yaml` ghi `yolo26n.pt`! | **Mâu thuẫn cấu hình.** Cần sửa config về đúng `yolov8n.pt`. |
| 8 | **Hợp nhất Weighted Boxes Fusion (WBF)** | `WORD_BẮT_BUỘC` | Đoạn 17, 26: Áp dụng WBF [1], chọn tham số bằng Grid Search trên Validation. | `src/detection/fusion.py` và `src/detection/tune.py` đã viết đúng theo thư viện `ensemble-boxes` của Solovyev et al. | **Đã sẵn sàng về mặt giải thuật.** |
| 9 | **Toàn bộ chức năng Streamlit** | `WORD_BẮT_BUỘC` | Đoạn 28: Tải nhiều ảnh, vẽ box, nhãn, confidence, đếm nhóm, slider, đo latency, báo không rác, nút tải ảnh. | `streamlit_app.py` và `src/web/` đã có đủ các phần tử UI này. | **Khớp yêu cầu giao diện.** |

---

## 4. Kiểm kê Dữ liệu & Môi trường Hiện có trên Máy

### 4.1. Dữ liệu thực tế trên máy (Không tải thêm ở P0)
1. **Tập dữ liệu `data/detection/v1` (và mirror tại `D:\waste-training\dataset-v1`):**
   * *Đường dẫn:* `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3\data\detection\v1`
   * *Nguồn gốc:* Tổng hợp từ OpenImages (`oi_...`) và Mendeley Synthetic (`syn_...`). Không phải VN-trash hay Garbage Classification V2.
   * *Số lượng file:*
     * `images/train`: 1,004 ảnh | `labels/train`: 1,004 file `.txt`
     * `images/val`: 223 ảnh | `labels/val`: 223 file `.txt`
     * `images/test`: 192 ảnh | `labels/test`: 192 file `.txt`
     * **Tổng cộng: 1,419 ảnh** và 1,419 file nhãn YOLO.
   * *Định dạng nhãn:* Text file YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (đã chuẩn hóa [0, 1]).
   * *Phân bố lớp (Đếm thực tế từ toàn bộ 1,419 file nhãn):*
     * Class 0 (`battery`): 462 boxes
     * Class 1 (`biological`): 446 boxes
     * Class 2 (`cardboard`): 452 boxes
     * Class 3 (`clothes`): 462 boxes
     * Class 4 (`glass`): 476 boxes
     * Class 5 (`metal`): 448 boxes
     * Class 6 (`paper`): 422 boxes
     * Class 7 (`plastic`): 472 boxes
     * Class 8 (`shoes`): 507 boxes
     * Class 9 (`trash`): 455 boxes
     * *Nhận xét:* Đây là **dataset 10 lớp**, có Bounding Box, **không có Polygon/Mask**.
2. **Tập nén `C:\Users\ad\Downloads\Dataset_Garbage.zip`:**
   * Kích thước: 42,822,633 bytes (~40.8 MB).
   * Cấu trúc: Thư mục `dataset-resized/` chứa 2,527 ảnh phân loại đơn nhãn (TrashNet). Không có bounding box.
3. **Thư mục dự kiến cho đề cương: `data/detection/v2-six-class`:**
   * Trạng thái: **CHƯA TỒN TẠI** trên toàn bộ hệ thống (`C:\` và `D:\`).

### 4.2. Kiểm kê Môi trường Thực thi (Python & Phần cứng)
* **Hệ điều hành:** Windows 11 Home Single Language (Build 26100)
* **Python hệ thống:** Python 3.12.10 (tại `C:\Users\ad\AppData\Local\Programs\Python\Python312\python.exe`)
* **Môi trường ảo (.venv):** `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3\.venv` (Python 3.12.13)
  * *Lưu ý đối chiếu:* Đề cương Word (Đoạn 34) ghi **Python 3.10**. Local đang lệch 2 phiên bản minor (3.12 vs 3.10).
* **Phần cứng GPU cục bộ:**
  * Model: **NVIDIA GeForce RTX 2050** (4096 MiB VRAM)
  * Driver Version: 591.86 | CUDA Version hỗ trợ tối đa: 13.1
* **Thư viện AI cốt lõi trong `.venv` hiện tại:**
  * `torch==2.3.1+cpu` và `torchvision==0.18.1+cpu` -> **ĐANG CHẠY BẢN CPU, CHƯA BẬT CUDA GPU!**
  * `ultralytics==8.4.150`
  * `streamlit==1.63.0`
  * `albumentations==2.0.8`
  * `ensemble-boxes==1.0.9` (đã cài đặt)
  * `opencv-python==4.10.0`
  * `numpy==1.26.4`
  * `pandas==3.0.1`
  * `scikit-learn==1.9.0`
* **Trọng số mô hình (Checkpoints):**
  * `artifacts/detection/waste-yolo26s/weights/`: **RỖNG** (chưa có `best.pt` hay `best.pth`).
  * Hệ thống chưa có bất kỳ checkpoint nào đã huấn luyện cho detection.

---

## 5. Hiện vật Đã Hoàn thành vs Phần Còn Thiếu

### 5.1. Phần ĐÃ HOÀN THÀNH (Có hiện vật kiểm chứng)
1. **Kiến trúc mã nguồn Detection (`src/detection/`):**
   * Đã có code load dataset YOLO, kiểm tra rò rỉ dữ liệu (`dataset.py`).
   * Đã có code adapter chạy suy luận SSDLite320 (`ssdlite.py`) và YOLO (`yolo.py`).
   * Đã có module WBF Fusion (`fusion.py`) và module Grid Search tối ưu tham số trên tập Val (`tune.py`).
   * Đã có module tính mAP COCO chuẩn qua pycocotools (`evaluate.py`).
2. **Giao diện Web Streamlit (`streamlit_app.py`, `src/web/`):**
   * Đã có giao diện tương tác upload ảnh, slider confidence, renderer bounding box tiếng Việt, bảng thống kê 6 nhóm rác và nút download zip ảnh kết quả.
3. **Cấu hình thí nghiệm:**
   * Đã có template cấu hình WBF (`configs/detection_fusion.yaml`) và dataset (`configs/detection_dataset.yaml`).

### 5.2. Phần CÒN THIẾU (Chưa có hiện vật)
1. **Dữ liệu chuẩn theo Word:** Chưa tải và tiền xử lý 3 bộ dữ liệu: `VN-trash`, `Garbage Classification V2` và ảnh tự chụp TP.HCM.
2. **Nhãn Bounding Box 6 lớp:** Chưa có nhãn chuẩn 6 lớp cho 3 bộ dữ liệu Word.
3. **Mặt nạ tiền cảnh (Polygon Masks) cho Copy-Paste:** Hoàn toàn chưa có.
4. **Trọng số đã huấn luyện (Model Weights):** Chưa có checkpoint của `SSDLite320-MobileNetV3` và `YOLOv8n`.
5. **Thực nghiệm Ablation 4 nhánh:** Chưa chạy và chưa có bảng số liệu.
6. **Môi trường huấn luyện cục bộ có GPU:** PyTorch trong `.venv` đang là bản CPU (`+cpu`).
7. **Quyển báo cáo khóa luận & slide:** Chưa có tài liệu văn bản chính thức trong repo.

### 5.3. Định hướng Khảo sát Gán nhãn cho P1
* *Không suy đoán số ảnh tùy tiện:* Cần khảo sát cụ thể ở P1 số lượng ảnh thô tải về từ `VN-trash` và `Garbage Classification V2`.
* Đo đạc tốc độ gán nhãn trung bình (số giây/box hoặc số phút/ảnh) trên công cụ CVAT để xây dựng kế hoạch phân công công việc khả thi cho nhóm 3 sinh viên trong 12 tuần (Đoạn 58-59 trong Word dành 2 tuần cho khâu gán nhãn).

---

## 6. Các Mâu thuẫn & Chỗ Chưa Xác Minh (Contradictions & Unverified)

1. **Mâu thuẫn số lượng lớp (10 lớp vs 6 lớp):**
   * Repo cũ đang kế thừa pipeline 10 lớp (`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`).
   * File Word bắt buộc chỉ giữ **6 lớp** vật liệu và yêu cầu loại bỏ `shoes`, `clothes`, `trash`.
2. **Mâu thuẫn nguồn dữ liệu:**
   * Dữ liệu hiện có trong `data/detection/v1` là OpenImages + Mendeley Synthetic.
   * File Word bắt buộc là `VN-trash` + `Garbage Classification V2` + Ảnh chụp TP.HCM.
3. **Mâu thuẫn tên mô hình YOLO:**
   * Configs trong repo ghi `yolo26s.pt` / `yolo26n.pt` (tên gọi nội bộ hoặc thử nghiệm).
   * File Word quy định rõ mã định danh: `YOLOv8n` [8] (kèm trích dẫn GitHub Ultralytics YOLOv8).
4. **Mâu thuẫn môi trường PyTorch & GPU:**
   * Máy có GPU rời RTX 2050 4GB nhưng venv đang cài `torch+cpu`. Nếu train trên máy local sẽ cực kỳ chậm.
5. **Chưa xác minh tình trạng ảnh tự chụp TP.HCM:**
   * Cần PM và nhóm sinh viên xác nhận nhóm đã đi chụp ảnh thực tế tại TP.HCM chưa, đã lưu ở đâu và số lượng là bao nhiêu.

---

## 7. Rủi ro Tiến độ Dự án (Schedule Risks)

1. **Nghẽn cổ chai gán nhãn (Data Labeling Bottleneck):**
   * Theo kế hoạch tuần trong Word (Đoạn 58-59), Tuần 2 và Tuần 3 phải hoàn thành thu thập và gán nhãn kiểm tra chéo. Nếu phải gán nhãn hàng nghìn ảnh thô từ đầu bằng CVAT, nguy cơ trễ hạn là rất cao.
2. **Rủi ro rò rỉ dữ liệu khi không chia theo bối cảnh (Leakage Risk):**
   * Garbage Classification V2 có nhiều góc chụp của cùng một vật thể rác. Nếu không gom cụm theo scene trước khi split 70/15/15, chỉ số mAP trên tập Test sẽ bị ảo.
3. **Rủi ro tính toán (Compute Resource Risk):**
   * Đề cương Word ghi train trên Google Colab T4 16GB. Nếu phụ thuộc Colab miễn phí dễ bị ngắt kết nối giữa chừng; trong khi máy local chưa cài PyTorch CUDA để tận dụng GPU RTX 2050.

---

## 8. Danh sách Đầu vào Cần PM / Người dùng Quyết định Trước P1

1. **Quyết định về Nguồn ảnh chụp TP.HCM:**
   * Nhóm sinh viên đã có ảnh chụp thực tế tại TP.HCM chưa? Nếu có, đường dẫn lưu trữ ở đâu? Nếu chưa, kế hoạch đi chụp bao nhiêu ảnh và bối cảnh ở đâu?
2. **Quyết định về Phương án Gán nhãn cho 2 bộ Kaggle:**
   * Đối với `VN-trash` và `Garbage Classification V2`, nhóm sẽ:
     * (A) Gán nhãn thủ công 100% trên CVAT?
     * Hay (B) Sử dụng mô hình Pretrained (như YOLOv8x/Grounding-DINO) để gán nhãn bán tự động (Pre-labeling), sau đó người chỉ việc review/chỉnh sửa trên CVAT để tiết kiệm thời gian?
3. **Quyết định về Môi trường Huấn luyện Cục bộ:**
   * Có cho phép cài đặt lại `torch` và `torchvision` bản CUDA tương thích với card NVIDIA GeForce RTX 2050 (Driver 591.86, CUDA 12.x/13.x) trong local `.venv` để chạy thử nghiệm nhanh không?
4. **Quyết định về Phiên bản Python:**
   * Chấp nhận chạy Python 3.12 cục bộ (và ghi chú trong báo cáo) hay bắt buộc phải cài đúng Python 3.10 theo câu chữ trong Word?
5. **Quyết định về Quản lý Git Branch:**
   * Đề xuất tạo branch mới sạch sẽ: `feat/phase2-detection-six-class` tách khỏi branch `feat/mobilenetv3-10-class` hiện tại để tránh lẫn lộn giữa hệ thống 10 lớp cũ và 6 lớp mới.

---

## 9. Đề xuất DUY NHẤT MỘT Task P1 Tiếp theo

> **TÊN TASK P1:**  
> **P1 — Khảo sát Nguồn Dữ liệu, Thiết lập Pipeline Intake 3 Nguồn & Thử nghiệm Giao thức Gán nhãn 6 Lớp**

**Mục tiêu cụ thể của P1:**
1. Chạy script tải 2 nguồn dataset chính thức từ Kaggle (`VN-trash` và `Garbage Classification V2`) về thư mục `data/sources/`.
2. Kiểm kê chính xác số lượng ảnh, định dạng ảnh, phân bố thư mục lớp gốc của 2 bộ dữ liệu này.
3. Thiết lập file mapping chính thức ánh xạ và loại bỏ lớp (`shoes`, `clothes`, `trash`) đúng 100% theo Đoạn 22 trong Word.
4. Xây dựng tài liệu hướng dẫn gán nhãn (Annotation Guidelines) trên CVAT và thử nghiệm gán nhãn mẫu cho 20 ảnh đầu tiên để đo thời gian chuẩn làm cơ sở lập tiến độ chi tiết.
5. Chưa can thiệp vào huấn luyện mô hình hay sửa đổi code thuật toán.

---

## 10. Phụ lục: Lệnh Kiểm kê Đã Thực thi (Audit Execution Logs)

*(Toàn bộ mật khẩu, token và thông tin cá nhân đã được che)*

```powershell
# 1. Xác định thư mục, branch và commit
cd C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3
git branch -v
# Output: * feat/mobilenetv3-10-class 1fa0d97 [ahead 4] feat(web): make Streamlit presentation icon-free

git status --short
# Output: M .github/workflows/ci.yml, M requirements.txt, M streamlit_app.py ... ?? src/detection/ ...

git log -n 1 --stat
# Output: commit 1fa0d9797e73bfce82e876cd6cacecddebebd543 (Author: Nhan-create <n***@gmail.com>)

# 2. Kiểm tra môi trường Python & GPU
python --version
# Output: Python 3.12.10
nvidia-smi
# Output: NVIDIA GeForce RTX 2050 (4096 MiB VRAM), Driver: 591.86, CUDA: 13.1

# 3. Kiểm tra gói trong virtual environment
.\.venv\Scripts\python.exe -c "import torch, torchvision, ultralytics, streamlit; print(torch.__version__, torchvision.__version__, ultralytics.__version__, streamlit.__version__)"
# Output: 2.3.1+cpu 0.18.1+cpu 8.4.150 1.63.0

# 4. Kiểm kê dữ liệu hiện có
python -c "import os; print(len(os.listdir('data/detection/v1/images/train')))"
# Output: 1004 ảnh train (tương ứng 223 val, 192 test)
```
