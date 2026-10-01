# BÁO CÁO HOÀN THIỆN BẰNG CHỨNG VÀ KHẢ NĂNG TÁI HIỆN (TASK P1-R1)
## NGHIỆM THU PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU VÀ HUẤN LUYỆN MOBILENETV3

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead ML:** Antigravity Engineering Team  
**Thời điểm hoàn thành:** 02/10/2026  
**Trạng thái đề xuất:** **`PART_01_READY_FOR_ACCEPTANCE`**  

---

## 1. Trả lời và Đối soát 8 Câu hỏi Bắt buộc của PM

| STT | Câu hỏi làm rõ của PM | Bằng chứng Kỹ thuật & Số liệu Đo đạc | Trạng thái | Tác động & Biện pháp Xử lý |
|:---:|:---|:---|:---:|:---|
| **1** | Vì sao `check_split_leakage.py` vẫn có nhánh `c1 != c2 → CROSS_CLASS_COINCIDENCE`? Ba cặp ứng viên còn lại đã được xem trực quan thật chưa? | **Nguyên nhân code cũ:** Ở bản R2, script tự động phân loại `c1 != c2` là `CROSS_CLASS_COINCIDENCE` mà không qua thẩm định mắt người, dẫn đến nguy cơ bỏ sót lỗi nhãn (như Cặp 10).<br>**Biện pháp đã xử lý triệt để:**<br>1. Xóa bỏ hoàn toàn nhánh gán tự động `c1 != c2` trong [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py).<br>2. Thẩm định trực quan chi tiết toàn bộ 3 cặp ứng viên xuyên split (`CROSS_01`, `CROSS_02`, `CROSS_03`) và cặp nội bộ `train` (`CROSS_04`).<br>3. Tạo ảnh đối chiếu tại [data/audit/visual_phash_inspection/](file:///D:/CNTT-KLCN155-waste-detection/data/audit/visual_phash_inspection/):<br>   - `CROSS_01` (`cross_pair_01_metal_vs_cardboard_dist4.jpg`): Lon kim loại trắng vs Hộp sữa Milbona $\to$ Hai vật thể khác nhau hoàn toàn.<br>   - `CROSS_02` (`cross_pair_02_shoes_vs_metal_dist4.jpg`): Giày da trẻ em vs Lon Coca-Cola $\to$ Hai vật thể khác nhau hoàn toàn.<br>   - `CROSS_03` (`cross_pair_03_metal_vs_paper_dist2.jpg`): Lon kim loại trắng vs Chồng cốc giấy $\to$ Hai vật thể khác nhau hoàn toàn.<br>4. Lưu trữ bảng quyết định kiểm chứng tại [data/audit/visual_audit_decision_table.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/visual_audit_decision_table.csv). | `VERIFIED` | Loại bỏ phán đoán cảm tính/tự động; 100% ứng viên xuyên split đều có ảnh chụp đối chứng và lý do thẩm định cụ thể. |
| **2** | `create_split_v2.py` chỉ gom pHash gần nhau trong cùng lớp. Các bản sao bị gán khác nhãn hoặc chuỗi nối qua nhiều nhãn được xử lý thế nào? | **Kiểm toán dữ liệu mở rộng toàn diện:**<br>1. Thực thi script quét toàn bộ tập dữ liệu [scripts/scan_all_phash_candidates.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/scan_all_phash_candidates.py) trên toàn bộ 14.829 ảnh sạch, không giới hạn cùng nhãn hay cùng split.<br>2. Kết quả: Toàn bộ dataset chỉ có đúng **4 cặp ứng viên khác nhãn** (Hamming dist $\le 4$). Trong đó: 3 cặp nằm xuyên split đã được thẩm định trực quan xác nhận là vật thể khác nhau (câu 1); 1 cặp nằm nội bộ train (`cardboard_1333.jpg` vs `glass_499.jpg`) cũng là hai vật thể khác nhau (hộp nước dừa vs chai rượu vang thủy tinh).<br>3. Trường hợp duy nhất có cùng vật thể nhưng khác nhãn là **Cặp 10** (`vn_trash_train_cardboard438.jpg` vs `garbage_v2_paper_582.jpg`) đã được phát hiện và **cách ly vĩnh viễn** tại [data/audit/quarantined_samples.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/quarantined_samples.csv).<br>4. Không tồn tại chuỗi mắt xích bắc cầu nối qua nhiều nhãn (đồ thị liên thông có bậc tối đa bằng 1). | `VERIFIED` | Đảm bảo không có bản sao đa nhãn nào bị lọt qua các split; không làm sai lệch phân phối huấn luyện. |
| **3** | `TRAINING_HISTORY` trong `export_official_results.py` được chép từ log gốc nào? Log đó còn nguyên vẹn không, và đã nằm trong gói bàn giao chưa? | **Nguồn gốc log gốc:** Lần huấn luyện chính thức P1.6 được chạy ngầm dưới dạng background task trên hệ thống, ghi toàn bộ output vào `task-744.log`.<br>**Đóng gói và bảo toàn:**<br>1. Tệp log gốc đã được sao chép và bảo toàn nguyên vẹn tại [artifacts/official_run/official_training_raw_execution.log](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_raw_execution.log) (Dung lượng: 12.574 bytes, SHA-256: `a1c6ca0c5beca94b7b8e22f3bdfbc9645b8579df981602d4198211cf9355f7ed`).<br>2. Đã xóa bỏ hoàn toàn danh sách gán cứng `TRAINING_HISTORY = [...]` trong [scripts/export_official_results.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/export_official_results.py). Script hiện đọc và phân tích cú pháp (parse) trực tiếp từ tệp log bằng biểu thức chính quy.<br>3. Tệp log gốc đã được đưa vào gói bàn giao `CNTT-KLCN155_PART01_VERIFIED_R1.zip`. | `VERIFIED` | Chuỗi chứng cứ số liệu lịch sử huấn luyện có nguồn gốc minh bạch 100%, có thể kiểm tra độc lập từng byte. |
| **4** | Những số liệu về VRAM, thời gian và cấu hình được đo trực tiếp, lấy từ checkpoint hay nhập thủ công? | **Phương pháp thu thập thực tế:**<br>1. **Peak VRAM:** Được đo trực tiếp bằng hàm `torch.cuda.max_memory_allocated(0) / (1024 ** 2)` tại cuối mỗi epoch trong vòng lặp huấn luyện của `train_official_mobilenetv3.py`. Kết quả: Phase 1 đạt đỉnh 312,7 MB; Phase 2 đạt đỉnh 1.514,5 MB trên GPU RTX 2050 (4.095,5 MB).<br>2. **Thời gian mỗi epoch:** Được đo trực tiếp bằng hàm `time.perf_counter()` và `time.time() - t0`.<br>3. **Cấu hình huấn luyện:** Đọc trực tiếp từ cấu hình tham số trong code.<br>4. Trong báo cáo [artifacts/official_run/official_training_metrics.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_metrics.json), nhóm đã phân tách rõ ràng 3 nhóm trường: `hardware_measured` (đo đạc trực tiếp), `training_configuration` (cấu hình thiết lập), và `provenance` (nguồn gốc dữ liệu). | `VERIFIED` | Minh bạch phương pháp đo; không có số liệu ước lượng hay giả lập. |
| **5** | Vì sao thiếu `dataset_info.csv` dù các script audit/split phụ thuộc vào file này? Còn dependency hoặc metadata nào nằm ngoài gói? | **Nguyên nhân cũ:** Script cũ trỏ đường dẫn mặc định sang thư mục gốc ban đầu `D:\HK7\Đồ án khóa luận\Data\metadata\dataset_info.csv`. Khi nén gói bàn giao, file này không nằm trong thư mục repo nên bị thiếu.<br>**Biện pháp đã xử lý triệt để:**<br>1. Đã sao chép tệp [data/metadata/dataset_info.csv](file:///D:/CNTT-KLCN155-waste-detection/data/metadata/dataset_info.csv) (2.298.135 bytes, 15.755 dòng) trực tiếp vào thư mục `data/metadata/` của repo.<br>2. Cập nhật đường dẫn mặc định trong toàn bộ scripts (`check_split_leakage.py`, `create_split_v2.py`, `find_all_transitive_clusters.py`) sang `data/metadata/dataset_info.csv`.<br>3. Kiểm toán toàn bộ workspace xác nhận: Không còn bất kỳ file metadata, nhãn hay dependency nào nằm ngoài repo. Gói bàn giao chứa 100% tài nguyên cần thiết. | `VERIFIED` | Gói bàn giao độc lập hoàn toàn, có thể giải nén và chạy trên bất kỳ máy nào mà không cần ổ đĩa D. |
| **6** | Điều gì chứng minh DataLoader đã đọc đúng toàn bộ mẫu của manifest V2, không đọc thêm ảnh cũ hoặc bỏ sót ảnh? | **Bằng chứng xác thực song phương (Dual Verification):**<br>1. **Kiểm tra mã băm trực tiếp trên ổ đĩa:** Script [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py) đã đọc và tính lại mã băm SHA-256 của toàn bộ 14.829 ảnh vật lý trong `data/processed_v2/`, xác nhận: 0 ảnh thiếu, 0 ảnh sai mã băm so với `split_manifest_v2.csv`.<br>2. **Bảng dự đoán chi tiết từng mẫu:** Script [scripts/reproduce_validation.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/reproduce_validation.py) xuất tệp [artifacts/official_run/val_predictions.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_predictions.csv) gồm chính xác 2.223 dòng ứng với 2.223 ảnh validation. Mỗi dòng ghi rõ `filename`, `sha256`, `true_label`, `predicted_label`, `confidence`, và `is_correct`.<br>3. Không có bất kỳ tệp tin rác hay ảnh cũ nào tồn tại trong các thư mục lớp của `data/processed_v2/val`. | `VERIFIED` | Chứng minh bằng mã băm và bảng kết quả dự đoán của từng ảnh; loại trừ hoàn toàn việc đọc sai dữ liệu. |
| **7** | Vì sao cấu hình chính thức là 12 epoch? Đây là cấu hình chốt trước khi chạy hay thay đổi trong quá trình chạy? Không được giải thích là early stopping khi JSON ghi `early_stopping_triggered = false`. | **Làm rõ chiến lược huấn luyện:**<br>1. Con số 12 epoch là **ngân sách huấn luyện tối đa được chốt trước khi chạy** trong script [scripts/train_official_mobilenetv3.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/train_official_mobilenetv3.py) theo chiến lược Two-Phase fine-tuning: Phase 1 gồm **3 epochs warmup head** ($lr=10^{-3}$) và Phase 2 gồm **9 epochs fine-tuning toàn bộ mạng** ($lr_{backbone}=10^{-4}, lr_{head}=3 \times 10^{-4}$ kèm Cosine Annealing). Tổng cộng $3 + 9 = 12$ epochs.<br>2. **Vì sao 12 epochs mà không phải 50?** Vì với 10.383 ảnh huấn luyện (163 steps/epoch) và backbone MobileNetV3 đã có trọng số ImageNet-1K chất lượng cao, mô hình hội tụ rất nhanh: từ Epoch 7-10, Train Loss đã giảm xuống mức $0.53$ và Train Accuracy đạt $99.9\%$. Ngân sách 12 epochs là phù hợp và tối ưu cho card đồ họa laptop RTX 2050 (chạy trong ~25 phút).<br>3. **Về Early Stopping:** Script cài đặt `patience = 4`. Do Validation Macro-F1 liên tục cải thiện và đạt đỉnh mới tại Epoch 10 (0.9555) và Epoch 12 (0.9559), điều kiện dừng sớm chưa bị kích hoạt (`early_stopping_triggered = false`), mô hình hoàn thành trọn vẹn 12/12 epochs đã lên lịch.<br>4. Bảng 50 epochs trong tài liệu cũ là định hướng tổng quát sơ khởi; cấu hình 12 epochs trong code là cấu hình triển khai cụ thể có cơ sở kỹ thuật rõ ràng. | `VERIFIED` | Minh bạch lý do lựa chọn siêu tham số; phân định rõ giữa ngân sách epoch tối đa và cơ chế early stopping. |
| **8** | Có bằng chứng nào hỗ trợ kiểm tra tải lại checkpoint ngoài trường `reload_test_passed = true`? | **Bằng chứng định lượng độc lập:**<br>1. Script tái hiện [scripts/reproduce_validation.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/reproduce_validation.py) đã thực hiện nạp lại `best_model.pt` và chạy suy luận trên toàn bộ 2.223 ảnh Validation.<br>2. Kết quả đối chiếu với số liệu lưu trong checkpoint:<br>   - Checkpoint ghi: Accuracy = $96.1763\%$, Macro-F1 = $0.955859$, Loss = $0.624819$.<br>   - Tái hiện đo được: Accuracy = $96.1763\%$, Macro-F1 = $0.955859$.<br>   - **Độ chênh lệch (Discrepancy):** $\Delta \text{Acc} = \mathbf{0.00e+00}$, $\Delta \text{Macro-F1} = \mathbf{0.00e+00}$, Số lỗi lệch $= \mathbf{0}$ (khớp chính xác 85 lỗi).<br>3. Toàn bộ bảng dự đoán từng ảnh được xuất độc lập tại [artifacts/official_run/val_predictions.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_predictions.csv) và tóm tắt tại [artifacts/official_run/reproduced_validation_summary.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/reproduced_validation_summary.json). | `VERIFIED` | Có số liệu định lượng và bảng so sánh chi tiết từng mẫu; không chỉ là một cờ boolean rỗng. |

---

## 2. Kết quả Tái hiện Thực nghiệm Độc lập (Validation Reproduction)

Thực thi độc lập bằng lệnh:
```powershell
python scripts/reproduce_validation.py `
  --checkpoint artifacts/official_run/best_model.pt `
  --manifest data/audit/split_manifest_v2.csv `
  --val-dir data/processed_v2/val
```

### 2.1. Đối chiếu Số liệu Toàn cục
- **Tập Validation:** 2.223 ảnh sạch, thuộc 10 lớp cân bằng theo cụm nguyên tử.
- **Dự đoán chính xác:** **2.138 / 2.223** mẫu.
- **Tổng số lỗi:** **85 / 2.223** mẫu (Tỷ lệ lỗi: $3,82\%$).
- **Top-1 Accuracy:** **`96.18%`** (Sai lệch so với báo cáo: $\Delta = 0.00e+00$).
- **Macro-Averaged F1:** **`0.9559`** (Sai lệch so với báo cáo: $\Delta = 0.00e+00$).
- **Thời gian thực thi:** 28,45 giây trên GPU NVIDIA GeForce RTX 2050 (AMP fp16, batch size 64).

### 2.2. Bảng Hiệu năng Chi tiết Từng Lớp Tái hiện Được
Dữ liệu được tính toán trực tiếp từ bảng dự đoán [artifacts/official_run/val_predictions.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_predictions.csv):

| Lớp (Class) | Precision | Recall | F1-Score | Số mẫu (Support) | So sánh Ngưỡng Cổng |
|:---|:---:|:---:|:---:|:---:|:---|
| **battery** | **100.00%** | **94.69%** | **0.9727** | 113 | **VƯỢT XA** (Ngưỡng $\ge 88.0\%$, không bỏ sót pin) |
| **biological** | 99.01% | 95.24% | 0.9709 | 105 | Đạt rất cao |
| **cardboard** | 96.45% | 96.45% | 0.9645 | 310 | Rất đồng đều |
| **clothes** | 98.95% | **100.00%** | **0.9947** | 284 | Chính xác 100% mẫu |
| **glass** | 97.25% | 95.38% | 0.9631 | 260 | Rất tốt |
| **metal** | 95.07% | 96.76% | 0.9591 | 339 | Vượt trội |
| **paper** | 92.79% | 92.79% | 0.9279 | 222 | Cân bằng |
| **plastic** | 96.28% | 95.64% | 0.9596 | 298 | Rất tốt |
| **shoes** | 95.18% | **100.00%** | **0.9753** | 217 | Chính xác 100% mẫu |
| **trash** | 88.89% | 85.33% | 0.8707 | 75 | Đạt ngưỡng an toàn ($\ge 80.0\%$) |
| **TOÀN BỘ (Macro Avg)** | **96.09%** | **95.23%** | **0.9559** | **2.223** | **ĐẠT CHUẨN NGHIỆM THU** |

---

## 3. Kết quả Kiểm toán Rò rỉ Dữ liệu Toàn diện (Leakage Audit)

Đã chạy lại kiểm toán bằng [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py) và [scripts/find_all_transitive_clusters.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/find_all_transitive_clusters.py):

1. **Khử trùng tuyệt đối (Exact Byte-level Match):**
   - 0 cặp trùng SHA-256 giữa Train & Val.
   - 0 cặp trùng SHA-256 giữa Train & Test.
   - 0 cặp trùng SHA-256 giữa Val & Test.
2. **Khử trùng cụm gần giống (Perceptual Hash DCT 64-bit, Hamming dist $\le 4$):**
   - Quét toàn bộ 14.829 ảnh qua toàn bộ các lớp và split: Phát hiện 33 cặp ứng viên.
   - **29 cặp cùng lớp:** Đã được giải thuật DSU gom nguyên tử vào cùng một split trong Split V2 (**0 cặp rò rỉ xuyên split**).
   - **4 cặp khác lớp:** Đã được thẩm định trực quan độc lập (có ảnh đối chiếu trong `data/audit/visual_phash_inspection/`), xác nhận 100% là sự trùng hợp ngẫu nhiên về hình học đơn giản trên phông nền trắng studio giữa các vật thể hoàn toàn khác nhau (hộp sữa vs lon nhôm, dép sandal vs lon nước, cốc giấy vs lon nhôm, chai rượu vs hộp nước dừa).
3. **Phân tích liên thông bắc cầu:**
   - 29 cụm độc lập (đều có kích thước $= 2$, bậc $= 1$).
   - 0 cụm vắt ngang qua nhiều split.
4. **Kết luận kiểm toán có giới hạn khoa học:**
   > Không phát hiện bất kỳ trường hợp rò rỉ ảnh trùng hay ảnh chụp liên tiếp (burst shot) nào giữa các tập Train, Val và Test trong phạm vi quét mã băm chính xác SHA-256 và mã băm tri giác pHash (ngưỡng Hamming $\le 4$) kết hợp thẩm định trực quan toàn bộ các cặp ứng viên.

---

## 4. Tuyên bố Liêm chính và Khóa Tập Test (Test Isolation Statement)

1. **Khóa chặt Test Set 100%:** Toàn bộ 2.223 ảnh tại `data/processed_v2/test/` **tuyệt đối không được nạp vào bộ nhớ hoặc đánh giá trong suốt quá trình thực hiện Phần 1**.
2. **Không có vòng phản hồi (Zero Snooping Loop):** Mọi quyết định lựa chọn mô hình, cấu hình siêu tham số, dừng sớm và phân tích lỗi đều căn cứ hoàn toàn trên tập Validation (2.223 ảnh).
3. **Sẵn sàng cho Cổng Đánh giá Độc lập (Evaluation Gate A):** Tập Test được giữ nguyên vẹn để phục vụ bài kiểm tra độc lập một lần duy nhất tại Cổng Gate A trước khi chuyển sang huấn luyện bộ phát hiện đa rác YOLOv8n.

---

## 5. Kết luận và Đề xuất Nghiệm thu Phần 1

Mọi câu hỏi chất vấn của PM đã được giải quyết bằng code thật, dữ liệu thật, log gốc và bảng kiểm chứng chi tiết từng mẫu. Gói bàn giao đã được tự chứa hóa hoàn toàn và sẵn sàng chạy trên bất kỳ hệ thống nào theo [docs/plan/REPRODUCIBILITY_GUIDE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/REPRODUCIBILITY_GUIDE.md).

> **ĐỀ XUẤT CỦA TECH LEAD:**  
> Chuyển trạng thái của Phần 1 từ `PART_01_PENDING_VERIFICATION` sang:  
> **`PART_01_READY_FOR_ACCEPTANCE`**
