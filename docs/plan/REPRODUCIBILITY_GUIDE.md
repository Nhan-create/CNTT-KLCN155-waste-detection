# HƯỚNG DẪN TÁI HIỆN VÀ KIỂM CHỨNG ĐỘC LẬP (REPRODUCIBILITY GUIDE)

**Mã đề tài:** `CNTT-KLCN155` — Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3  
**Phiên bản bàn giao:** `CNTT-KLCN155_PART01_VERIFIED_R1`  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead ML:** Antigravity Engineering Team  
**Thời điểm lập:** 02/10/2026  

---

## 1. Tổng quan Gói Bàn giao và Tính Độc lập

Gói bàn giao `CNTT-KLCN155_PART01_VERIFIED_R1.zip` được đóng gói độc lập, tự chứa (self-contained) toàn bộ mã nguồn, cấu hình, siêu dữ liệu (metadata), log gốc, bảng quyết định thẩm định trực quan, bảng dự đoán chi tiết từng mẫu và checkpoint mô hình tốt nhất.

Người đánh giá có thể giải nén sang bất kỳ thư mục nào trên bất kỳ máy tính nào (Windows, Linux, macOS) mà **không bị phụ thuộc cứng vào cấu trúc ổ đĩa hay đường dẫn tuyệt đối của máy huấn luyện**. Mọi script đều hỗ trợ tham số dòng lệnh CLI (`--checkpoint`, `--val-dir`, `--manifest`, `--dataset-info`, `--device`, `--output-dir`).

---

## 2. Yêu cầu Môi trường và Cài đặt Dependencies

### 2.1. Cấu hình phần cứng tối thiểu
- **CPU:** 4 lõi trở lên (Intel Core i5 / AMD Ryzen 5).
- **RAM:** Tối thiểu 8 GB (khuyến nghị 16 GB).
- **GPU (Tùy chọn):** NVIDIA GPU hỗ trợ CUDA (RTX 2050 4GB trở lên). Nếu không có GPU, script tự động chuyển sang chế độ CPU (`--device cpu`).
- **Ổ đĩa trống:** Tối thiểu 2 GB cho mã nguồn, checkpoint và dữ liệu val.

### 2.2. Khởi tạo môi trường ảo Python
Khuyến nghị sử dụng Python 3.12 (dự án được kiểm thử trên Python 3.12.10):

```powershell
# 1. Khởi tạo virtual environment
python -m venv .venv

# 2. Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS: source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc chính xác
pip install -r requirements.txt
```

Nội dung thư viện chính trong `requirements.txt`:
```text
torch>=2.5.1
torchvision>=0.20.1
numpy>=2.1.0
pandas>=2.2.0
pillow>=11.0.0
scikit-learn>=1.5.0
networkx>=3.4.0
```

---

## 3. Các Bước Tái hiện Kỹ thuật Độc lập

### Bước 1: Kiểm tra tính toàn vẹn của tệp Checkpoint và Manifest
Trước khi chạy suy luận, hãy kiểm tra mã băm SHA-256 của hai tệp quan trọng nhất:

```powershell
# Kiểm tra mã băm checkpoint
Get-FileHash -Path artifacts/official_run/best_model.pt -Algorithm SHA256

# Kiểm tra mã băm manifest V2
Get-FileHash -Path data/audit/split_manifest_v2.csv -Algorithm SHA256
```

**Mã băm chuẩn đối chiếu:**
- `best_model.pt` SHA-256: `c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424`
- `split_manifest_v2.csv` SHA-256: `1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96`

---

### Bước 2: Tái hiện Toàn bộ Kết quả Validation từ Checkpoint
Chạy script tái hiện độc lập [scripts/reproduce_validation.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/reproduce_validation.py):

```powershell
python scripts/reproduce_validation.py `
  --checkpoint artifacts/official_run/best_model.pt `
  --manifest data/audit/split_manifest_v2.csv `
  --val-dir data/processed_v2/val `
  --output-dir artifacts/official_run `
  --batch-size 64
```

*Nếu chạy trên máy không có GPU NVIDIA, thêm cờ `--device cpu`:*
```powershell
python scripts/reproduce_validation.py --device cpu
```

**Kết quả đầu ra kỳ vọng:**
1. Xuất bảng dự đoán 2.223 dòng: `artifacts/official_run/val_predictions.csv` (ghi nhận `filename, true_label, predicted_label, confidence, is_correct, sha256`).
2. Xuất ma trận nhầm lẫn: `artifacts/official_run/val_confusion_matrix.csv`.
3. Xuất danh sách 85 ca lỗi: `artifacts/official_run/val_error_analysis.csv`.
4. Xuất JSON tóm tắt: `artifacts/official_run/reproduced_validation_summary.json`.
5. **Tiêu chuẩn kiểm chứng:**
   - Số mẫu chính xác: **2.138 / 2.223** mẫu.
   - Top-1 Accuracy: **96.18%** (sai lệch $= 0.00e+00$).
   - Macro-F1: **0.9559** (sai lệch $= 0.00e+00$).
   - Số lỗi: **85** lỗi (sai lệch $= 0$).
   - Battery Recall: **94.69%** (Precision: 100.00%).

---

### Bước 3: Kiểm chứng Kiểm toán Rò rỉ Dữ liệu (Leakage Audit)
Chạy script kiểm toán độc lập [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py):

```powershell
python scripts/check_split_leakage.py `
  --manifest data/audit/split_manifest_v2.csv `
  --dataset-info data/metadata/dataset_info.csv `
  --decision-table data/audit/visual_audit_decision_table.csv `
  --processed-dir data/processed_v2
```

**Kết quả đầu ra kỳ vọng:**
- Kiểm tra SHA-256 trùng xuyên split: **0 cặp trùng**.
- Kiểm tra toàn vẹn ảnh trên đĩa: **100% khớp mã băm manifest**.
- Ứng viên pHash $\le 4$ xuyên split: **3 cặp ứng viên** (đều được đối soát với bảng thẩm định trực quan xác nhận là các vật thể khác loại trùng ngẫu nhiên hình học nền trắng).
- Trạng thái kiểm toán: **`VERIFIED_NO_LEAKAGE_WITHIN_HASH_SCOPE`**.

---

### Bước 4: Kiểm chứng Chuỗi Liên thông và Cụm Bắc cầu
Chạy script kiểm toán thành phần liên thông [scripts/find_all_transitive_clusters.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/find_all_transitive_clusters.py):

```powershell
python scripts/find_all_transitive_clusters.py `
  --manifest data/audit/split_manifest_v2.csv `
  --dataset-info data/metadata/dataset_info.csv `
  --decision-table data/audit/visual_audit_decision_table.csv
```

**Kết quả đầu ra kỳ vọng:**
- 29 cụm gần trùng cùng lớp.
- 0 cụm vắt ngang qua nhiều split (**100% cụm được cô lập trọn vẹn trong một split duy nhất**).
- Bậc liên thông tối đa: 1 (không có chuỗi mắt xích bắc cầu phức tạp).

---

### Bước 5: Đối chứng Lịch sử Huấn luyện từ Log Gốc
Chạy script xuất báo cáo tự động từ log gốc [scripts/export_official_results.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/export_official_results.py):

```powershell
python scripts/export_official_results.py `
  --raw-log artifacts/official_run/official_training_raw_execution.log `
  --checkpoint artifacts/official_run/best_model.pt `
  --manifest data/audit/split_manifest_v2.csv
```

**Kết quả đầu ra kỳ vọng:**
- Phân tích cú pháp 12 epochs từ log thô `official_training_raw_execution.log` (SHA-256: `a1c6ca0c5beca94b7b8e22f3bdfbc9645b8579df981602d4198211cf9355f7ed`).
- Xuất tệp `artifacts/official_run/training_history.csv` và `official_training_metrics.json`.
- Hoàn toàn không có dữ liệu gán cứng (zero hardcoding).

---

## 4. Hướng dẫn Thu thập và Khởi tạo Dữ liệu Thô (Nếu cần tạo lại từ đầu)

Nếu người đánh giá muốn tái tạo toàn bộ thư mục `data/processed_v2/` từ dữ liệu thô:
1. Tải hai dataset từ Kaggle:
   - [VN Trash Classification](https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification)
   - [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
2. Đặt dữ liệu thô vào thư mục:
   - `data/raw/garbage_v2/` (12.259 ảnh)
   - `data/raw/vn_trash/` (3.495 ảnh)
3. Chạy script khử trùng và tạo inventory:
   ```powershell
   python scripts/audit_and_reconcile_data.py
   ```
4. Chạy script phân chia Split V2:
   ```powershell
   python scripts/create_split_v2.py
   ```
Mã nguồn sẽ tự động tạo thư mục vật lý `data/processed_v2/` với đầy đủ liên kết cứng (NTFS hardlink) và xuất manifest V2 khớp chính xác mã SHA-256 đã công bố.
