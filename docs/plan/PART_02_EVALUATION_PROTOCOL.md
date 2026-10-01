# QUY TRÌNH ĐÁNH GIÁ ĐÓNG BĂNG CHO CỔNG GATE A (EVALUATION PROTOCOL)
## ĐÁNH GIÁ TÍNH TỔNG QUÁT HÓA CỦA MOBILENETV3 TRÊN TẬP FINAL TEST ĐỘC LẬP

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Tài liệu:** Đóng băng Quy trình Đánh giá (Frozen Evaluation Protocol)  
**Phiên:** Task 2 — Giai đoạn A Acceptance Gate  
**Thời điểm đóng băng:** 02/10/2026 01:15:00+07:00  
**Trạng thái protocol:** **`FROZEN_LOCKED`** (Không được phép sửa đổi sau khi mở niêm phong Test Set)

---

## 1. Định danh Tài nguyên Huấn luyện Đã Đóng băng

Toàn bộ tài nguyên phục vụ đánh giá Gate A được cố định bất biến:

| Mục | Đường dẫn tệp | SHA-256 Checksum | Ghi chú |
|:---|:---|:---|:---|
| **Checkpoint Mô hình** | [`artifacts/official_run/best_model.pt`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/best_model.pt) | `c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424` | Checkpoint tốt nhất tại Epoch 12 của Phase 2 fine-tuning. |
| **Phân vùng Dữ liệu** | [`data/audit/split_manifest_v2.csv`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/split_manifest_v2.csv) | `1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96` | Phân chia sạch Split V2 (10.383 train, 2.223 val, 2.223 test). |
| **Thư mục Final Test** | `data/processed_v2/test/` | Khớp 100% mã băm từng file với manifest | 2.223 ảnh vật lý, lần đầu tiên nạp vào mô hình. |

---

## 2. Quy chuẩn Danh mục Lớp và Tiền xử lý (Preprocessing & Class Mapping)

### 2.1. Thứ tự Danh mục Lớp (Alphabetical Indexing 0-9)
```python
CLASS_NAMES = [
    "battery",      # Index 0
    "biological",   # Index 1
    "cardboard",    # Index 2
    "clothes",      # Index 3
    "glass",        # Index 4
    "metal",        # Index 5
    "paper",        # Index 6
    "plastic",      # Index 7
    "shoes",        # Index 8
    "trash",        # Index 9
]
```

### 2.2. Chuỗi Biến đổi Tiền xử lý Chuẩn (Deterministic Transforms)
Khớp chính xác 100% với chuỗi biến đổi trên tập Validation:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### 2.3. Quy tắc Suy luận và Quyết định Dự đoán
```python
logits = model(inputs)
probs = torch.softmax(logits, dim=1)
predicted_class_id = torch.argmax(probs, dim=1)
confidence = torch.max(probs, dim=1)
```

---

## 3. Giao thức Đo lường Độ trễ CPU (CPU Latency Protocol)

Để bảo đảm tính khách quan và khoa học khi đánh giá khả năng triển khai biên:
* **Thiết bị phần cứng:** CPU AMD Ryzen 5 6600H (6 cores, 12 threads).
* **Số luồng CPU kích hoạt:** `torch.set_num_threads(4)` (mô phỏng CPU 4 lõi của máy tính bảng hoặc máy tính nhúng mini PC).
* **Kích thước Batch:** `batch_size = 1` (chế độ suy luận đơn ảnh thời gian thực).
* **Số lượt Warmup:** **50 lượt** forward liên tiếp (loại bỏ hiệu ứng khởi động bộ đệm và JIT compile).
* **Số lượt Benchmark:** **200 lượt** forward độc lập bằng hàm `time.perf_counter()`.
* **Số liệu thống kê bắt buộc báo cáo:**
  * Thời gian Forward trung bình (`mean_latency_ms`).
  * Thời gian Forward trung vị (`median_latency_ms`).
  * Phân vị 95 (`p95_latency_ms`).
  * Độ trễ nhỏ nhất và lớn nhất (`min_ms`, `max_ms`).
  * Thời gian toàn luồng Pipeline (`full_pipeline_ms` = Resize + ToTensor + Normalize + Forward + Softmax).
  * Tốc độ xử lý khung hình tương đương (`FPS` = $1000 / \text{mean\_latency\_ms}$).

---

## 4. Kiểm chứng Tính Toàn vẹn Dữ liệu Đọc Thực tế (Physical Byte Provenance)

Trong quá trình DataLoader đọc từng ảnh từ đĩa vật lý:
1. Script tính trực tiếp hàm băm SHA-256 từ byte stream:
   $$H_{\text{disk}} = \text{SHA256}(\text{read}(P_{\text{file}}))$$
2. Đối chiếu trực tiếp với mã băm kỳ vọng trong manifest:
   $$\Delta_{\text{hash}} = (H_{\text{disk}} == H_{\text{manifest}})$$
3. Nếu phát hiện bất kỳ file nào không khớp mã băm ($\Delta_{\text{hash}} = \text{False}$), script dừng ngay lập tức và hủy bỏ kết quả Gate A.

---

## 5. Tiêu chuẩn Đánh giá Vượt Cổng Gate A (Pass/Fail Criteria)

Mô hình được xác nhận **`GATE_A_PASSED`** khi và chỉ khi thỏa mãn đồng thời toàn bộ 6 tiêu chí:

$$\begin{aligned}
1. &\quad \text{Top-1 Accuracy} \ge 88.0\% \\
2. &\quad \text{Macro-Averaged F1} \ge 0.850 \\
3. &\quad \text{Recall}_{\text{battery}} \ge 88.0\% \\
4. &\quad \min_{c} \text{Recall}_{c} \ge 80.0\% \\
5. &\quad \text{Mean CPU Latency} < 20.0 \text{ ms/ảnh (Batch size 1)} \\
6. &\quad \text{Data Leakage / Integrity Mismatch} = 0
\end{aligned}$$

Nếu có bất kỳ tiêu chí nào không đạt, trạng thái kết luận là **`GATE_A_FAILED`**. Nhóm kỹ thuật giữ nguyên hiện trạng, không được can thiệp vào test set để điều chỉnh mô hình.
