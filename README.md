# CNTT-KLCN155 — Hệ thống phát hiện và phân loại đa đối tượng rác thải sinh hoạt

[![Architecture & Roadmap](https://img.shields.io/badge/Architecture%20%26%20Roadmap-View%20Guide-blue?style=for-the-badge)](ARCHITECTURE_AND_NEXT_STEPS.md)
[![Alignment Review](https://img.shields.io/badge/Proposal%20Alignment-Reviewed-green?style=for-the-badge)](docs/plan/PROPOSAL_ARCHITECTURE_ALIGNMENT_REVIEW.md)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Khóa luận Cử nhân Công nghệ Thông tin (2026 – 2027)**  
> **Trường Đại học Công Thương TP.HCM (HUIT)**  
> **GVHD:** ThS. Huỳnh Thị Châu Lan  
> **Sinh viên thực hiện:** Ngô Thanh Nhân (Nhóm trưởng / PM), Võ Gia Ninh, Vũ Trường Vinh  
> 
> 📖 **Xem hướng dẫn kiến trúc và các việc cần làm tiếp theo tại:** [**`ARCHITECTURE_AND_NEXT_STEPS.md`**](ARCHITECTURE_AND_NEXT_STEPS.md)  
> 📋 **Xem báo cáo đối chiếu đề cương chi tiết tại:** [**`docs/plan/PROPOSAL_ARCHITECTURE_ALIGNMENT_REVIEW.md`**](docs/plan/PROPOSAL_ARCHITECTURE_ALIGNMENT_REVIEW.md)

---

## Tổng quan dự án

Dự án gồm hai giai đoạn nghiên cứu khoa học chặt chẽ:
1. **Giai đoạn 1 (Đã hoàn thành):** Tiền xử lý, khử trùng 15.754 ảnh thành 14.829 ảnh sạch Split V2 (không rò rỉ cụm) và huấn luyện mô hình nhẹ **MobileNetV3-Large** phân loại rác đơn thể 10 lớp (đạt **Test Acc 96.13%**, **Macro-F1 0.9578**, tối ưu hóa ONNX Runtime đạt **7.40 ms/ảnh** trên CPU).
2. **Giai đoạn 2 (Đang triển khai):** Phát hiện đa đối tượng rác thải thực tế bằng **SSDLite320-MobileNetV3 (mô hình chính)** kế thừa backbone và **YOLOv8n (mô hình đối chứng)**, kết hợp kỹ thuật tăng cường dữ liệu mở rộng (**Albumentations & Copy-Paste**) và hợp nhất dự đoán **Weighted Boxes Fusion (WBF)** trên tập Validation.


Đây là bài toán **phân loại ảnh đơn nhãn, 10 lớp**: mỗi ảnh được biến đổi thành tensor RGB, MobileNetV3-Large tạo 10 logits, rồi Softmax đổi logits thành xác suất. Class ID có xác suất lớn nhất là top-1; giao diện đồng thời hiển thị top-3 và cảnh báo khi top-1 thấp hơn ngưỡng mặc định `0.55`.

```text
Hai dataset Kaggle
        │
        ▼
manifest nguồn + ánh xạ nhãn
        │
        ▼
kiểm tra ảnh ─ SHA-256 ─ pHash ─ cách ly xung đột
        │
        ▼
group-stratified split 70/15/15 ─ validator chống leakage
        │
        ▼
MobileNetV3-Large (ImageNet) ─ weighted CE ─ AdamW ─ 2 phase
        │
        ├── best.pt + last.pt
        ├── metrics JSON/CSV + confusion matrix
        └── CLI / Streamlit ảnh-camera-video / PyQt5
```

MobileNetV3-Large dùng các khối inverted residual, depthwise convolution, pointwise convolution, Squeeze-and-Excitation và hàm kích hoạt h-swish để giảm chi phí tính toán. Classification layer cuối được thay bằng `Linear(..., 10)`. Huấn luyện transfer learning gồm:

1. đóng băng feature extractor, học classifier 5 epoch với LR `1e-3`;
2. mở toàn bộ backbone, fine-tune tối đa 25 epoch với LR backbone `1e-4` và LR head `3e-4`;
3. dùng weighted cross-entropy, label smoothing `0.1`, AdamW, cosine schedule, gradient clipping và AMP khi có CUDA;
4. chọn `best.pt` theo **validation macro-F1**, không dùng test để chọn model.

Định nghĩa model chỉ có một nguồn tại `src/models/mobilenetv3.py`. Train, evaluate, CLI và GUI đều đọc chung checkpoint tự mô tả để không lệch thứ tự lớp.

## Mười lớp chuẩn

Thứ tự dưới đây được khóa trong manifest và checkpoint:

| Chỉ số | Class ID | Tên hiển thị |
|---:|---|---|
| 0 | `battery` | Pin |
| 1 | `biological` | Rác hữu cơ |
| 2 | `cardboard` | Bìa carton |
| 3 | `clothes` | Quần áo |
| 4 | `glass` | Thủy tinh |
| 5 | `metal` | Kim loại |
| 6 | `paper` | Giấy |
| 7 | `plastic` | Nhựa |
| 8 | `shoes` | Giày dép |
| 9 | `trash` | Rác khác |

Tên tiếng Việt chỉ là lớp trình bày. Logic, thư mục dữ liệu và checkpoint luôn dùng class ID tiếng Anh ổn định.

## Nguồn dữ liệu, phiên bản và giấy phép

| Nguồn | Phiên bản khóa | Giấy phép do trang nguồn công bố |
|---|---|---|
| [VN Trash Classification](https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification/versions/1) | `mrgetshjtdone/vn-trash-classification/versions/1` | MIT |
| [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/versions/12) | `sumn2u/garbage-classification-v2/versions/12` | MIT |

Người sử dụng cần đọc data card, điều khoản Kaggle và thông tin trích dẫn tại từng trang nguồn trước khi phân phối lại dữ liệu hoặc sản phẩm dẫn xuất. Phiên bản được ghi rõ trong cấu hình và notebook để một bản cập nhật upstream không âm thầm làm thay đổi thí nghiệm.

### Chính sách ánh xạ nhãn

`data/metadata/label_mapping.csv` là nguồn sự thật được review và commit. Garbage Classification v2 đã có đúng 10 class ID nên giữ nguyên sau khi chuẩn hóa tên. Chín nhãn của VN Trash được ánh xạ như sau:

| Nhãn VN Trash | Class ID chuẩn |
|---|---|
| `Alu` | `metal` |
| `Carton` | `cardboard` |
| `Foam_box` | `plastic` |
| `Milk_box` | `cardboard` |
| `Other` | `trash` |
| `PET` | `plastic` |
| `Paper` | `paper` |
| `Paper_cup` | `paper` |
| `Plastic_cup` | `plastic` |

Nhãn không có trong bảng không được đoán tự động: pipeline dừng hoặc đánh dấu `unmapped` để người dùng kiểm tra.

## Cài đặt môi trường

Khuyến nghị Python 3.11. Python 3.10–3.12 cũng phù hợp nếu các wheel trong `requirements.txt` có sẵn cho hệ điều hành.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`numpy==1.26.4` và `scipy==1.13.1` được khóa để tương thích ABI với `torch==2.3.1`. Máy CUDA cần driver tương thích; notebook Kaggle là đường chạy khuyến nghị cho huấn luyện đầy đủ.

## Tải dữ liệu cục bộ

Đăng nhập KaggleHub bên ngoài repository; không dán thông tin xác thực vào mã hoặc notebook:

```powershell
python -c "import kagglehub; kagglehub.login()"
python scripts/download_datasets.py `
  --output-root data/sources `
  --config-output data/sources/preprocessing_config.local.yaml
```

KaggleHub tải đúng hai handle đã khóa. Helper tự tìm thư mục thực sự chứa `Train/Test` hoặc `original`, in các root đã phân giải và ghi `data/sources/preprocessing_config.local.yaml`. Layout sau khi phân giải là:

```text
data/sources/vn_trash/     # có Train/ và Test/
data/sources/garbage_v2/   # có original/ hoặc các thư mục lớp
```

Nếu archive có thêm một thư mục bọc ngoài, helper xử lý mà không di chuyển hay nhân đôi ảnh. Dữ liệu, config local và credential đều nằm dưới đường dẫn bị `.gitignore` loại khỏi Git.

## Chạy trên Kaggle GPU

1. Tạo Kaggle Notebook và bật GPU.
2. Attach đúng hai dataset/version trong bảng nguồn ở trên.
3. Mở `notebooks/train_mobilenetv3_kaggle.ipynb` và chạy tuần tự toàn bộ cell.
4. Tải `/kaggle/working/waste-classifier-output.zip` sau khi hoàn tất.

Notebook tự tìm layout input, ghi cấu hình đường dẫn đã phân giải, chạy pipeline, xác nhận `validation.is_valid`, huấn luyện hai phase, đánh giá `best.pt` trên test đúng một lần, rồi đóng gói checkpoint, manifest, cấu hình, metric và biểu đồ. Notebook không chứa credential.

## Pipeline dữ liệu

Khi dùng downloader ở trên, truyền config local mà nó tạo:

```powershell
python -m src.preprocessing.run_pipeline `
  --config data/sources/preprocessing_config.local.yaml `
  --raw-root data/raw `
  --metadata-root data/metadata/v1 `
  --processed-root data/processed/v1 `
  --report-root outputs/data/v1
```

Trên bash, thay dấu tiếp dòng PowerShell bằng `\`. Pipeline tạo một version mới và cố ý từ chối ghi đè version đã tồn tại. Kiểm tra độc lập:

```bash
python -m src.data.validation \
  --dataset-root data/processed/v1 \
  --manifest data/metadata/v1/split_manifest.csv
```

Các kiểm tra gồm: file corrupt, nhãn không ánh xạ, exact duplicate bằng SHA-256, near-duplicate bằng pHash, cluster xung đột nhãn, đủ 10 lớp và duplicate cluster không đi qua ranh giới train/val/test. Chi tiết thuật toán nằm trong `Tienxulidulieu.md`.

## Huấn luyện và resume

```bash
python -m src.training.train \
  --data-root data/processed/v1 \
  --manifest data/metadata/v1/split_manifest.csv \
  --config configs/model_config.yaml \
  --preprocessing-config data/sources/preprocessing_config.local.yaml \
  --output-dir artifacts/run-001 \
  --device cuda
```

Nếu thiếu VRAM, thêm `--batch-size 16`. Tiếp tục một lần chạy bị ngắt bằng checkpoint `last.pt`:

```bash
python -m src.training.train \
  --data-root data/processed/v1 \
  --manifest data/metadata/v1/split_manifest.csv \
  --config configs/model_config.yaml \
  --preprocessing-config data/sources/preprocessing_config.local.yaml \
  --output-dir artifacts/run-001 \
  --resume artifacts/run-001/last.pt \
  --device cuda
```

`best.pt` là checkpoint triển khai tốt nhất theo validation macro-F1; `last.pt` chứa thêm optimizer, scheduler và scaler để resume. Mỗi checkpoint lưu model ID, 10 class names theo thứ tự, input size, normalization, epoch, metrics, training config và fingerprint SHA-256 của dataset.

## Đánh giá

Chỉ đánh giá sau khi model đã được chọn bằng validation:

```bash
python -m src.evaluation.evaluate \
  --checkpoint artifacts/run-001/best.pt \
  --test-root data/processed/v1/test \
  --output-dir outputs/evaluation/run-001 \
  --device cuda
```

Output gồm accuracy, macro-F1, weighted-F1, precision/recall/F1/support từng lớp, one-vs-rest macro AUC khi đủ điều kiện, confusion matrix raw và normalized.

## Dự đoán bằng CLI

```bash
python -m src.inference.predict \
  --checkpoint artifacts/run-001/best.pt \
  --image path/to/image.jpg \
  --top-k 3 \
  --confidence-threshold 0.55 \
  --device auto
```

Kết quả là một JSON object chứa `top1`, `topk` và `low_confidence`. Transform và class order được lấy từ metadata checkpoint, không hard-code riêng ở CLI.

## Ứng dụng Streamlit: Phân loại & Phát hiện đa đối tượng (`streamlit_app.py`)

Sau khi đã cài đặt môi trường, chạy ứng dụng tại thư mục gốc repository:

```powershell
streamlit run streamlit_app.py
```

Mặc định ứng dụng tự động nạp bộ trọng số 10 lớp đã huấn luyện ([`src/web/detection_settings.py`](src/web/detection_settings.py)):
- **YOLOv8n (E1 - Đối chứng):** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`
- **SSDLite320-MobileNetV3 (E2 - Mô hình chính):** `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt`
- **MobileNetV3-Large Classifier (Stage-1 / Stage-2):** `artifacts/official_run/best_model.pt`

Có thể tùy chỉnh đường dẫn detector hoặc chế độ chạy bằng biến môi trường hoặc CLI:

```powershell
$env:WASTE_DETECTOR='path/to/best.pt'
$env:WASTE_DETECTOR_BACKEND='yolov8n'   # yolov8n | ssdlite | wbf
streamlit run streamlit_app.py
# hoặc: streamlit run streamlit_app.py -- --backend ssdlite --detector path/to/best.pt
```

Giao diện Web gồm **2 Tab chuyên biệt** kết hợp thanh điều khiển cấu hình bên trái:

1. **Tab 1 — Phân loại rác độ chính xác cao (`MobileNetV3-Large` 96.18%):**
   - Dành cho ảnh chụp cận cảnh một vật thể hoặc một cụm rác đồng nhất.
   - Hiển thị nhãn tiếng Việt, độ tin cậy, thời gian xử lý (ms), hướng dẫn màu thùng rác tại nguồn và biểu đồ xác suất Top-5 lớp.
2. **Tab 2 — Phát hiện vị trí đa vật thể (`YOLOv8n` / `SSDLite320` / `WBF`):**
   - Hỗ trợ tải lên đồng thời một hoặc nhiều ảnh thực tế, thống kê số lượng từng loại rác và xuất gói `.zip` ảnh kết quả.
   - **Cắt ô ảnh (`Tiling` — [`src/detection/tiling.py`](src/detection/tiling.py)):** Tự động chia ảnh lớn ($\ge 800\text{px}$) thành các ô $640 \times 640$ có chồng lấp để bắt rác kích thước nhỏ, kèm bộ lọc loại bỏ box nhiễu bao trọn ô cắt.
   - **Thẩm định hai giai đoạn (`Two-Stage` — [`src/detection/two_stage.py`](src/detection/two_stage.py)):** Dùng `MobileNetV3-Large` phân loại lại từng vùng cắt proposal theo xác suất Bayesian (`product`, cổng `min_proposal_gate=0.28`) hoặc `blend` để triệt tiêu dương tính giả trên nền đường phố/tòa nhà.
   - **Ngưỡng hiệu chuẩn riêng từng lớp ([`configs/detection_thresholds_calibrated.json`](configs/detection_thresholds_calibrated.json)):** Tùy chọn điều chỉnh ngưỡng nhạy riêng cho từng nhóm vật liệu (`plastic`, `metal`, `battery`, `clothes`, `shoes`...).
   - **Hai kiểu vẽ khung bao ([`src/detection/visualization.py`](src/detection/visualization.py)):** *Khung viền nổi bật (Highlight Box)* và *Hộp chữ nhật viền nét (Bounding Box)* sắc nét, hiển thị rõ vật thể bên trong.
   - **Chuẩn hóa tọa độ & bảo vệ phần cứng:** Tự động xử lý xoay chiều EXIF, đệm tỉ lệ letterbox và chiếu ngược tọa độ chính xác về ảnh gốc ([`src/detection/coordinates.py`](src/detection/coordinates.py)), kiểm tra an toàn RAM/VRAM qua [`src/detection/preflight_guard.py`](src/detection/preflight_guard.py).


## Ứng dụng PyQt5 classification cũ

```bash
python app.py \
  --checkpoint artifacts/run-001/best.pt \
  --device auto \
  --history-db outputs/history.sqlite3
```

- “Chọn ảnh” chạy inference ở worker thread và lưu một bản ghi SQLite.
- “Bật camera” đọc BGR, chuyển RGB và dự đoán tuần tự hoàn toàn trong bộ nhớ; không ghi JPEG tạm và không thêm lịch sử mỗi frame.
- Camera lỗi không vô hiệu hóa phân loại ảnh tĩnh.
- Đóng cửa sổ yêu cầu camera dừng và giải phóng thiết bị.

Ứng dụng chỉ phân loại hình ảnh, không đưa ra chỉ dẫn xử lý rác mang tính pháp lý hoặc phụ thuộc địa phương.

## Bố cục artifact

```text
data/metadata/label_mapping.csv       # được commit
data/metadata/v1/                     # manifest, duplicate/conflict reports, weights
data/processed/v1/{train,val,test}/   # 10 thư mục lớp mỗi split
artifacts/run-001/
  best.pt
  last.pt
  history.csv
  resolved_config.yaml
outputs/evaluation/run-001/
  metrics.json
  per_class_metrics.csv
  confusion_matrix_raw.png
  confusion_matrix_normalized.png
outputs/history.sqlite3
```

Trừ bảng mapping, toàn bộ dữ liệu, checkpoint, SQLite và output sinh ra đều bị ignore.

## Kiểm thử và CI

```powershell
ruff check src tests app.py streamlit_app.py scripts
python -m compileall src app.py streamlit_app.py scripts
$env:QT_QPA_PLATFORM='offscreen'
$env:MPLBACKEND='Agg'
pytest -q
```

GitHub Actions chạy cùng quality gate trên Python 3.11, CPU, Qt headless; CI không tải dataset, mở camera hoặc chạy huấn luyện đầy đủ. Test hiện có dùng dữ liệu tổng hợp nhỏ để kiểm tra mapping, dedup, group split, checkpoint, train smoke, evaluation, inference, SQLite, worker Qt, GUI và notebook contract.

## Giới hạn

- Đây là closed-set classifier: ảnh ngoài 10 lớp vẫn bị xếp vào một lớp; `low_confidence` chỉ là cảnh báo, không phải bộ phát hiện out-of-distribution.
- Một ảnh chứa nhiều loại rác vẫn chỉ nhận một class ID.
- Chất lượng thực tế phụ thuộc dữ liệu, góc chụp, ánh sáng và domain shift; phải báo metric từ lần chạy Kaggle thực tế, không suy diễn từ dataset khác.
- pHash là heuristic cho ảnh gần trùng; cluster xung đột được cách ly để review thay vì tự sửa nhãn.
- Webcam và tốc độ inference phụ thuộc CPU/GPU, driver và camera của máy chạy.
