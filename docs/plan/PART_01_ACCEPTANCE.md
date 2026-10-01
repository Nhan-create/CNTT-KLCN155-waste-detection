# BIÊN BẢN NGHIỆM THU KỸ THUẬT PHẦN 1 (PART 01 ACCEPTANCE)

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3 phân loại rác  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead ML:** Antigravity Engineering Team  
**Ngày lập biên bản:** 01/10/2026  

---

## 1. Trả lời và Đối soát 12 Câu hỏi Kỹ thuật Bắt buộc của PM

| STT | Câu hỏi làm rõ của PM | Bằng chứng Kỹ thuật & Số liệu Đo đạc | Trạng thái | Ảnh hưởng đến Bước tiếp theo |
|:---:|:---|:---|:---:|:---|
| **1** | Tiêu chí nhận một ảnh vào dữ liệu classification là gì? | Xác lập chính sách rõ ràng: (1) Ảnh đơn vật thể rác hoặc ảnh chứa nhiều vật thể đồng nhất cùng một lớp vật liệu (Homogeneous Multi-object) được CHẤP NHẬN vào classification vì đặc trưng bề mặt vật liệu mang tính đại diện đồng nhất; (2) Ảnh hỗn tạp nhiều vật thể thuộc các lớp khác nhau (Heterogeneous Multi-object) KHÔNG thuộc phạm vi phân loại đơn rác, được chuyển sang Giai đoạn B (Detector Bounding Box); (3) Ảnh vật liệu phức hợp được phân loại theo lớp vật liệu chiếm ưu thế hoặc công năng tái chế (Hộp sữa -> `cardboard`, Hộp xốp -> `plastic`, Pin -> `battery`). | `VERIFIED` | Ngăn chặn việc loại bỏ dữ liệu tùy tiện hoặc gọi sai bản chất dữ liệu; phân định rõ ranh giới giữa Classification và Detection. |
| **2** | Hai dataset ánh xạ vào 10 lớp như thế nào? Những nhãn nào còn mơ hồ? | Bảng ánh xạ chuẩn tại [data/metadata/label_mapping.csv](file:///D:/CNTT-KLCN155-waste-detection/data/metadata/label_mapping.csv): (a) `garbage_v2` có sẵn 10 lớp khớp 1:1; (b) `vn_trash` ánh xạ 9 lớp: Alu -> `metal`, Carton -> `cardboard`, Foam_box -> `plastic`, Milk_box -> `cardboard`, PET -> `plastic`, Paper & Paper_cup -> `paper`, Plastic_cup -> `plastic`, Other -> `trash`. Điểm mơ hồ: Cặp 10 (`train_cardboard438.jpg` vs `paper_582.jpg`) cùng một vật thể hình trụ nhưng `vn_trash` gán Carton còn `garbage_v2` gán paper $\to$ Đã cách ly 2 mẫu này vào [data/audit/quarantined_samples.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/quarantined_samples.csv). | `VERIFIED` | Triệt tiêu xung đột nhãn giữa hai nguồn; bảo đảm tính nhất quán của dữ liệu huấn luyện. |
| **3** | Vì sao file thống kê cũ vẫn ghi `cross_source_duplicates = 0` dù script đã sửa? | Ở phiên làm việc trước, nhóm kỹ sư đã sửa code dòng 427 của [scripts/audit_and_reconcile_data.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_and_reconcile_data.py) nhưng chỉ chạy `audit_backbone_transfer.py`, `check_split_leakage.py` và `run_smoke_test.py` mà chưa kích hoạt lệnh chạy lại `audit_and_reconcile_data.py` trước khi nén zip. Trong phiên này, script đã được thực thi lại thành công, cập nhật [data/audit/audit_summary.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/audit_summary.json) chính xác: `cross_source_duplicates = 890` (trong đó 879 exact MD5) với timestamp `16:26:46+0700`. | `VERIFIED` | Số liệu thống kê trong JSON, CSV và báo cáo khớp nhau 100%. |
| **4** | Với mỗi cặp trong 18 ứng viên pHash, bằng chứng nào xác nhận cùng ảnh, cùng cảnh, cùng vật thể hoặc chỉ giống hình thức? | Đã thẩm định trực quan toàn bộ 18 cặp bằng script [scripts/inspect_plate_details.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/inspect_plate_details.py) kết hợp ảnh đối chiếu song song: (1) Cặp 10: cùng vật thể cốc giấy/bìa nhưng xung đột nhãn `cardboard` vs `paper` $\to$ cách ly; (2) Cặp 12: hai lon kim loại khác hẳn nhau (lon có bóng tối bên phải vs lon đứng giữa) $\to$ trùng ngẫu nhiên hình học; (3) Cặp 18: hai hộp xốp khác hình thái (hộp kín góc thấp vs hộp mở) $\to$ trùng ngẫu nhiên hình học; (4) Cặp 11, 14, 16: vật thể khác loại trên nền trắng $\to$ trùng ngẫu nhiên cross-class; (5) Các cặp còn lại: chuỗi burst shot cùng vật thể hoặc ảnh resize độ phân giải khác nhau. | `VERIFIED` | Phân loại chính xác bản chất từng cặp, không phụ thuộc vào MAE hay pHash một cách máy móc. |
| **5** | Có ứng viên gần trùng trong cùng split tạo thành chuỗi nối sang split khác không? | Đã kiểm toán bằng giải thuật Connected Components trong [scripts/find_all_transitive_clusters.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/find_all_transitive_clusters.py) trên toàn bộ 14.831 ảnh: Phát hiện 29 cụm gần trùng cùng lớp ($\text{size} = 2$). Trong split cũ: 14 cụm nằm trọn vẹn nội bộ trong 1 split, 15 cụm vắt ngang qua 2 split khác nhau. KHÔNG CÓ chuỗi bắc cầu bậc 3 trở lên (không có chuỗi $A \sim B \sim C$ kéo từ trong split ra ngoài split). Trong Split V2 mới: 100% cụm (29/29) được gom nguyên tử vào 1 split duy nhất. | `VERIFIED` | Giải quyết triệt để rò rỉ cụm và liên kết bắc cầu. |
| **6** | Có ảnh thiếu pHash, thiếu đường dẫn hoặc không đọc được bị script bỏ qua âm thầm không? | Kiểm toán tự động xác nhận: 100% trong số 14.831 ảnh sạch đều có mã pHash 64-bit hợp lệ (0 ảnh thiếu); 100% tệp tin vật lý đều tồn tại và giải mã ảnh RGB thành công bằng thư viện Pillow (0 tệp hỏng). Script kiểm tra không bỏ qua bất kỳ tệp tin nào. | `VERIFIED` | Toàn vẹn dữ liệu được bảo đảm không có lỗ hổng ẩn. |
| **7** | Split mới được xây dựng theo nhóm nguồn gốc thế nào để tránh chia các bản sao sang nhiều tập? | Cài đặt trong [scripts/create_split_v2.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/create_split_v2.py) sử dụng giải thuật Disjoint Set Union (Union-Find) gom tất cả 29 cặp gần trùng thành các đơn vị nguyên tử (Atomic Clusters). Sau đó áp dụng Stratified Group Split phân bổ theo tỷ lệ 70% Train (10.383 ảnh), 15% Val (2.223 ảnh), 15% Test (2.223 ảnh) với seed 42 cố định. Mọi ảnh trong cùng một cụm được cam kết nằm trong cùng một split duy nhất. | `VERIFIED` | Triệt tiêu hoàn toàn rò rỉ cụm xuyên split. |
| **8** | Pipeline train có đọc đúng manifest mới, đúng nhãn và đúng preprocessing không? | Đã kiểm định trong [scripts/verify_preprocessing_pipeline.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/verify_preprocessing_pipeline.py): Thư mục vật lý `data/processed_v2/` tạo bằng NTFS hardlink khớp chính xác $100\%$ từng tệp với [data/audit/split_manifest_v2.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/split_manifest_v2.csv) (mã băm SHA-256: `1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96`). Class mapping `class_to_idx` đồng nhất trên mọi split. Augmentation kiểm soát đúng chuẩn (RandomResizedCrop 0.8-1.0 chỉ chạy trên Train; Val chỉ Resize cố định). | `VERIFIED` | Pipeline đọc đúng dữ liệu, đúng nhãn, đúng tiền xử lý. |
| **9** | Checkpoint chạy thử cũ đã dùng split lỗi sẽ được đánh dấu và cách ly thế nào? | Checkpoint chạy thử của R2.1 được chuyển vào thư mục lưu trữ lịch sử [artifacts/legacy_smoke_test_r2_1/](file:///D:/CNTT-KLCN155-waste-detection/artifacts/legacy_smoke_test_r2_1/) kèm nhãn cảnh báo rõ ràng trong metadata: `LEGACY_SPLIT_FLAGGED_WITH_BURST_LEAKAGE`. Tuyệt đối không sử dụng làm checkpoint khởi tạo hay benchmark cho lần huấn luyện chính thức mới. | `VERIFIED` | Cách ly hoàn toàn dấu vết thực nghiệm cũ có rò rỉ. |
| **10** | Cấu hình train chính thức, tiêu chí chọn checkpoint và điều kiện dừng được chốt trước khi chạy ra sao? | Đã đóng băng trong [scripts/train_official_mobilenetv3.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/train_official_mobilenetv3.py): Kiến trúc MobileNetV3-Large khởi tạo từ ImageNet-1K chuẩn; Chiến lược Two-phase (Phase 1: 3 epochs warmup head $lr=10^{-3}$; Phase 2: 9 epochs fine-tuning $lr_{backbone}=10^{-4}, lr_{head}=3 \times 10^{-4}$ kèm Cosine Annealing); Batch size = 64; Loss = CrossEntropyLoss(label_smoothing=0.1); Tiêu chí chọn checkpoint: **Validation Macro-Averaged F1 cao nhất**; Điều kiện dừng sớm: Early stopping patience = 4 epochs không tăng Val Macro-F1. | `VERIFIED` | Cấu hình minh bạch, chốt chặt trước khi chạy, không điều chỉnh cảm tính sau khi xem kết quả. |
| **11** | Gói bàn giao có đủ dữ liệu metadata và dependencies để chạy lại các script không? | Gói bàn giao chứa đầy đủ mã nguồn `scripts/`, cấu hình `configs/`, metadata `data/metadata/`, manifest `data/audit/split_manifest_v2.csv`, báo cáo JSON/CSV, và `requirements.txt`. Mọi script đều có thể tái hiện độc lập trên môi trường Python 3.12 với PyTorch CUDA. | `VERIFIED` | Tính tái hiện (Reproducibility) được bảo đảm toàn diện. |
| **12** | Điều kiện nào cho phép kết luận hoàn thành phần 1, điều kiện nào buộc báo BLOCKED? | **Điều kiện HOÀN THÀNH:** (1) Data Gate xác nhận `ZERO_LEAKAGE_VERIFIED` trên Split V2; (2) Pipeline tiền xử lý và DataLoader được kiểm định khớp manifest; (3) Smoke test P1.5 chạy thông suốt và nạp lại checkpoint không sai lệch; (4) Huấn luyện chính thức P1.6 hoàn thành, lưu checkpoint tốt nhất theo Val Macro-F1 và vượt qua bài test reload; (5) Test set được khóa chặt 100%. **Điều kiện BÁO BLOCKED:** Nếu phát hiện rò rỉ cụm không thể giải quyết, hoặc GPU bị lỗi OOM không thể chạy, hoặc quá trình nạp lại checkpoint bị sai lệch giá trị dự đoán. | `VERIFIED` | Tiêu chí nghiệm thu rõ ràng, đo lường được bằng bằng chứng vật lý. |

---

## 2. Đối chiếu Tiến độ và Trạng thái Cổng Nghiệm thu 6 Subtasks

1. **P1.1 — Kiểm kê và chuẩn hóa taxonomy:** `PASSED`
2. **P1.2 — Làm sạch và thẩm định ảnh trùng:** `PASSED`
3. **P1.3 — Tạo split mới và kiểm tra rò rỉ:** `PASSED` (`ZERO_LEAKAGE_VERIFIED`)
4. **P1.4 — Pipeline tiền xử lý & DataLoader:** `PASSED`
5. **P1.5 — Chạy thử trên split mới (Smoke Test):** `PASSED`
6. **P1.6 — Huấn luyện chính thức MobileNetV3:** `PASSED` (Đạt Val Acc 96.18%, Val Macro-F1 0.9559, Battery Recall 94.69%)

---

## 3. Kết quả Thực nghiệm và Đo đạc Thực tế Mô hình Chính thức (P1.6)

### 3.1. Thông số Kỹ thuật & Môi trường Chạy thực tế
- **Mô hình:** MobileNetV3-Large (Khởi tạo ImageNet-1K pretrained weights chuẩn của Torchvision).
- **Phần cứng:** NVIDIA GeForce RTX 2050 (4.095,5 MB VRAM), CUDA 12.1, PyTorch 2.5.1+cu121.
- **Chiến lược Two-Phase:**
  - *Phase 1 (Warmup):* 3 epochs đóng băng backbone, chỉ tối ưu Head ($lr=10^{-3}$, peak VRAM: 312,7 MB).
  - *Phase 2 (Fine-tuning):* 9 epochs mở toàn bộ mạng ($lr_{backbone}=10^{-4}, lr_{head}=3 \times 10^{-4}$ kèm Cosine Annealing, peak VRAM: 1.514,5 MB).
- **Hàm mất mát:** `CrossEntropyLoss(label_smoothing=0.1)`.
- **Tập dữ liệu:** Split V2 độc lập hoàn toàn (Train: 10.383 ảnh; Val: 2.223 ảnh; Test: 2.223 ảnh **LOCKED**).

### 3.2. Lịch sử Huấn luyện Toàn bộ 12 Epochs ([artifacts/official_run/training_history.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/training_history.csv))

| Epoch | Giai đoạn (Phase) | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro-F1 | Thời gian (s) | Peak VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Phase 1 (Warmup) | 0.9685 | 82.75% | 0.7757 | 90.51% | 0.8997 | 120.6s | 312.7 MB |
| 2 | Phase 1 (Warmup) | 0.7756 | 90.78% | 0.7463 | 92.53% | 0.9215 | 127.2s | 312.7 MB |
| 3 | Phase 1 (Warmup) | 0.7227 | 93.48% | 0.7530 | 92.08% | 0.9140 | 128.9s | 312.7 MB |
| 4 | Phase 2 (Fine-tune) | 0.6554 | 96.16% | 0.6841 | 94.33% | 0.9375 | 114.0s | 1514.5 MB |
| 5 | Phase 2 (Fine-tune) | 0.5933 | 98.77% | 0.6681 | 94.65% | 0.9415 | 115.7s | 1514.5 MB |
| 6 | Phase 2 (Fine-tune) | 0.5659 | 99.45% | 0.6496 | 95.10% | 0.9450 | 119.3s | 1514.5 MB |
| 7 | Phase 2 (Fine-tune) | 0.5503 | 99.77% | 0.6373 | 95.95% | 0.9527 | 131.9s | 1514.5 MB |
| 8 | Phase 2 (Fine-tune) | 0.5396 | 99.92% | 0.6312 | 95.86% | 0.9514 | 143.7s | 1514.5 MB |
| 9 | Phase 2 (Fine-tune) | 0.5340 | 99.91% | 0.6285 | 95.86% | 0.9516 | 130.6s | 1514.5 MB |
| 10 | Phase 2 (Fine-tune) | 0.5312 | 99.94% | 0.6263 | 96.13% | 0.9555 | 139.8s | 1514.5 MB |
| 11 | Phase 2 (Fine-tune) | 0.5286 | 99.97% | 0.6256 | 96.04% | 0.9548 | 135.4s | 1514.5 MB |
| **12** | **Phase 2 (Fine-tune)** | **0.5282** | **99.92%** | **0.6248** | **96.18%** | **0.9559** | **142.2s** | **1514.5 MB** |

> **Checkpoint tốt nhất:** Lưu tại **Epoch 12** ([artifacts/official_run/best_model.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/best_model.pt), dung lượng 48,57 MB).  
> **Kiểm tra nạp lại (Reload Test):** Đánh giá độc lập mô hình nạp lại trên toàn bộ 2.223 ảnh Val cho kết quả khớp $100\%$ không sai lệch ($\Delta \text{Acc} = 0.00e+00, \Delta \text{Macro-F1} = 0.00e+00$).

### 3.3. Bảng Hiệu năng Chi tiết Từng Lớp trên Tập Validation (Best Checkpoint)

| Tên Lớp (Class) | Precision | Recall | F1-Score | Số mẫu (Support) | Đánh giá so với Ngưỡng Cổng |
|:---|:---:|:---:|:---:|:---:|:---|
| **battery** | **100.00%** | **94.69%** | **0.9727** | 113 | **VƯỢT XA** (Ngưỡng yêu cầu $\ge 88.0\%$) |
| **biological** | 99.01% | 95.24% | 0.9709 | 105 | Đạt rất cao |
| **cardboard** | 96.45% | 96.45% | 0.9645 | 310 | Đồng đều tuyệt đối |
| **clothes** | 98.95% | **100.00%** | **0.9947** | 284 | Không bỏ sót bất kỳ mẫu nào |
| **glass** | 97.25% | 95.38% | 0.9631 | 260 | Rất tốt |
| **metal** | 95.07% | 96.76% | 0.9591 | 339 | Vượt trội |
| **paper** | 92.79% | 92.79% | 0.9279 | 222 | Cân bằng |
| **plastic** | 96.28% | 95.64% | 0.9596 | 298 | Rất tốt |
| **shoes** | 95.18% | **100.00%** | **0.9753** | 217 | Không bỏ sót bất kỳ mẫu nào |
| **trash** | 88.89% | 85.33% | 0.8707 | 75 | Đạt ngưỡng an toàn ($\ge 80.0\%$) |
| **TOÀN BỘ (Macro Avg)** | **96.09%** | **95.23%** | **0.9559** | **2.223** | **VƯỢT CỔNG** (Ngưỡng yêu cầu $\ge 0.850$) |
| **TOÀN BỘ (Accuracy)** | — | — | **96.18%** | **2.223** | **VƯỢT CỔNG** (Ngưỡng yêu cầu $\ge 88.0\%$) |

### 3.4. Phân tích Ma trận Nhầm lẫn ([artifacts/official_run/val_confusion_matrix.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_confusion_matrix.csv))

| Nhãn Thực Tế \ Nhãn Dự Đoán | battery | bio | card | cloth | glass | metal | paper | plas | shoes | trash | Tổng |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **battery** | **107** | 0 | 1 | 0 | 0 | **5** | 0 | 0 | 0 | 0 | 113 |
| **biological** | 0 | **100** | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 105 |
| **cardboard** | 0 | 0 | **299** | 0 | 1 | 1 | **6** | 1 | 1 | 1 | 310 |
| **clothes** | 0 | 0 | 0 | **284** | 0 | 0 | 0 | 0 | 0 | 0 | 284 |
| **glass** | 0 | 1 | 0 | 0 | **248** | 5 | 0 | 4 | 2 | 0 | 260 |
| **metal** | 0 | 0 | 2 | 0 | 2 | **328** | 2 | 3 | 1 | 1 | 339 |
| **paper** | 0 | 0 | **6** | 1 | 1 | 3 | **206** | 2 | 2 | 1 | 222 |
| **plastic** | 0 | 0 | 0 | 0 | 3 | 3 | 2 | **285** | 2 | 3 | 298 |
| **shoes** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **217** | 0 | 217 |
| **trash** | 0 | 0 | 1 | 2 | 0 | 0 | **6** | 1 | 1 | **64** | 75 |

**Nhận xét quy luật nhầm lẫn:**
1. **Pin (Battery) nhầm sang Kim loại (Metal):** 5 mẫu pin dạng trụ kim loại sáng bóng bị phân loại thành metal. Tuy nhiên, không có mẫu pin nào bị bỏ qua sang các lớp rác thông thường nguy hại như trash hay biological. Độ chính xác Precision của Battery đạt tuyệt đối $100\%$ (không có rác khác bị đoán nhầm là pin).
2. **Giao thoa Cardboard và Paper:** Có 6 mẫu bìa cứng bị nhầm thành giấy và 6 mẫu giấy bị nhầm thành bìa cứng. Đây là ranh giới thị giác tự nhiên giữa các sản phẩm từ bột gỗ nghiền (bìa mỏng vs giấy dày).
3. **Độ nhạy tuyệt đối:** Hai lớp `clothes` và `shoes` đạt Recall tuyệt đối $100\%$ (284/284 mẫu quần áo và 217/217 mẫu giày dép được phân loại chính xác hoàn toàn).

### 3.5. Phân tích Các Mẫu Lỗi Hàng Đầu ([artifacts/official_run/val_error_analysis.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_error_analysis.csv))
Tổng số mẫu dự đoán sai trên toàn bộ tập Validation: **85 / 2.223** mẫu (tỷ lệ lỗi chỉ $3,82\%$).
Các lỗi có độ tự tin cao nhất bao gồm:
1. `garbage_v2_paper_984.jpg` (Thực tế: `paper`, Dự đoán: `trash`, Confidence: $0.9136$): Ảnh mảnh giấy bị vò nát lẫn vết bẩn bùn đất.
2. `garbage_v2_paper_507.jpg` (Thực tế: `paper`, Dự đoán: `metal`, Confidence: $0.8516$): Giấy gói kẹo tráng bạc phản chiếu ánh kim loại.
3. `garbage_v2_metal_367.jpg` (Thực tế: `metal`, Dự đoán: `trash`, Confidence: $0.8389$): Miếng kim loại rỉ sét nặng bị phân rã bề mặt.
4. `vn_trash_train_cardboard 177.jpg` (Thực tế: `cardboard`, Dự đoán: `paper`, Confidence: $0.8286$): Bìa carton mỏng dạng phẳng không có lớp sóng giữa.

---

## 4. Cam kết Nghiêm ngặt về Bảo toàn Tập Test (Test Isolation Statement)

1. **Khóa chặt Test Set 100%:** Toàn bộ 2.223 ảnh tại `data/processed_v2/test/` không hề được nạp, đọc, hay đánh giá trong suốt quá trình chạy Phần 1.
2. **Không có phản hồi ngược (Zero Snooping Loop):** Mọi quyết định lựa chọn mô hình, cấu hình siêu tham số, dừng sớm đều căn cứ hoàn toàn trên tập Validation (2.223 ảnh).
3. **Sẵn sàng cho Cổng Đánh giá Độc lập (Evaluation Gate A):** Tập Test được giữ nguyên vẹn để phục vụ bài kiểm tra độc lập một lần (Single-Shot Benchmark) ở giai đoạn nghiệm thu chuyển tiếp.

---

## 5. Kết luận Nghiệm thu Kỹ thuật Phần 1

> **KẾT LUẬN CỦA TECH LEAD:**  
> Toàn bộ 6 nhiệm vụ con **P1.1 đến P1.6 đã hoàn thành xuất sắc**. Mọi số liệu báo cáo đều có bằng chứng trực tiếp (`VERIFIED`), mã băm SHA-256 đối soát khớp 100%, checkpoint mô hình thực nghiệm thật và vượt xa các chỉ tiêu cổng của đề tài. Phần 1 đủ điều kiện nghiệm thu chính thức: **`PART_01_ACCEPTED`**.

