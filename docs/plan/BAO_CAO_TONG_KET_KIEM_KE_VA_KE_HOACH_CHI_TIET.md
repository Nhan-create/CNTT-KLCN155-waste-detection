# BÁO CÁO TỔNG KẾT KIỂM KÊ KỸ THUẬT VÀ KẾ HOẠCH CHI TIẾT (TASK R2.1)

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Cơ quan / Đơn vị:** Khoa Công nghệ Thông tin — Trường Đại học Công Thương TP.HCM (HUIT)  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead & Machine Learning Engineer:** Antigravity Engineering Team  
**Phiên:** Task R2.1 — Data Gate Audit, Leakage Verification & Smoke Test  
**Thời điểm hoàn thành:** 01/10/2026  
**Trạng thái phê duyệt:** `SMOKE_TEST_PASSED` | `DATA_GATE_FLAGGED_WITH_LEAKAGE`

---

## 1. Tuyên ngôn Liêm chính Kỹ thuật và Phản biện Kế hoạch

Trước khi đi vào các con số thực nghiệm, nhóm kỹ sư ML xác lập lại bài toán theo đúng định hướng chỉ đạo của PM:

> *"Làm sao chứng minh mô hình nhận dạng được đặc trưng của rác trên ảnh chưa từng thấy, thay vì nhớ ảnh trùng, nền chụp hoặc dấu hiệu riêng của dataset?"*

### 1.1. Phản biện sâu 5 cạm bẫy học máy trong dự án
1. **Độ chính xác (Accuracy) cao có thể che giấu lỗi nghiêm trọng nào?**
   - Phân bố dữ liệu giữa các lớp bị lệch đáng kể (ví dụ: `metal` có 1.581 ảnh train trong khi `trash` chỉ có 352 ảnh, `biological` có 489 ảnh). Một mô hình đạt $92\%$ Accuracy tổng thể hoàn toàn có thể có Recall lớp `battery` (pin độc hại) hoặc `trash` dưới $50\%$.
   - Trong bối cảnh phân loại rác, bỏ sót một viên pin gây nguy cơ cháy nổ lò ép rác, nghiêm trọng hơn rất nhiều so với việc phân loại nhầm giấy thành bìa carton. Do đó, Accuracy cao không phản ánh đủ mức độ an toàn thực tế; bắt buộc phải đánh giá qua **Macro-Averaged F1 Score** và **Recall riêng biệt cho từng lớp**.
2. **Nhãn `trash` được định nghĩa thế nào để người gán nhãn thống nhất?**
   - Khái niệm `trash` trong Kaggle Garbage Classification V2 vốn là nhãn rác hỗn tạp còn sót lại không thuộc các nhóm tái chế thông thường (tương đương "rác vô cơ không tái chế / rác sinh hoạt khác" theo phân loại của Bộ Tài nguyên & Môi trường).
   - Nếu không có bảng mô tả quy chuẩn (Rubric), người gán nhãn sẽ rất dễ nhầm lẫn: khi thấy một hộp xốp bẩn dính dầu mỡ, người thì gán `plastic`, người thì gán `trash`. Quy chuẩn thống nhất: vật liệu đơn chất, sạch thì gán vào lớp vật liệu (`plastic`, `paper`, `metal`); vật liệu phức hợp, rác bẩn khó phân hủy không thể tái chế thì gán vào `trash`.
3. **Mô hình có thể nhầm nền ảnh với đặc trưng vật thể không?**
   - Hoàn toàn có thể. Đặc biệt trong Kaggle Garbage V2, nhiều ảnh lớp `cardboard` được chụp trên nền sàn gỗ công nghiệp, lớp `shoes` chụp trên thảm cỏ, hoặc lớp `battery` chụp trên bàn làm việc có vân gỗ đặc thù. Nếu không kiểm soát nền hoặc không áp dụng các phép tăng cường biến đổi bối cảnh, mạng tích chập (CNN) sẽ trích xuất đặc trưng của mặt bàn hoặc sàn nhà thay vì học hình dạng cục pin hay vỏ chai.
4. **Chuyển backbone sang detector giúp gì và không chứng minh được điều gì?**
   - **Giúp gì:** Giúp tầng trích xuất đặc trưng (Backbone Features 258/308 keys) kế thừa các bộ lọc biên cạnh, họa tiết bề mặt và cấu trúc hình học đã học được từ ảnh rác, giúp detector hội tụ nhanh hơn so với khởi tạo ngẫu nhiên.
   - **Không chứng minh được gì:** Việc nạp thành công trọng số backbone **hoàn toàn không chứng minh được mô hình sẽ phát hiện (detect) tốt đa rác**. Các tầng tháp đặc trưng (Extra Feature Pyramid) và các đầu dự đoán bounding box (Head 168 keys) là những tầng hoàn toàn mới, bắt buộc phải học từ đầu khả năng định vị tọa độ vật thể ($x, y, w, h$) và phân biệt đa vật thể chen chúc nhau. Hiệu quả phát hiện chỉ có thể chứng minh bằng chỉ số mAP@0.5 đo trên dữ liệu gán nhãn thực tế.
5. **Hai dataset có tên khác nhau có thực sự tạo thành hai nguồn độc lập không?**
   - **Hoàn toàn không.** Kiểm toán thực nghiệm đã chứng minh: Kaggle VN Trash Classification chứa tới **879 ảnh trùng khớp mã băm MD5/SHA-256 tuyệt đối với Kaggle Garbage Classification V2**. Do đó, chúng có sự giao thoa dữ liệu nguồn rất lớn, không thể xem là hai tập dữ liệu độc lập.

---

## 2. Báo cáo Trả lời 12 Câu hỏi Làm rõ theo Bằng chứng Thực nghiệm

Dưới đây là 12 câu trả lời được đối soát trực tiếp từ mã nguồn, dữ liệu vật lý và log thực thi trên máy:

| STT | Câu hỏi làm rõ của PM | Bằng chứng thực tế & Số liệu đo được | Trạng thái | Ảnh hưởng đến bước tiếp theo |
|:---:|:---|:---|:---:|:---|
| **1** | Workspace mới có phải thư mục vật lý độc lập? Git root, venv và đường dẫn ghi đầu ra là gì? | Thư mục `D:\CNTT-KLCN155-waste-detection` là thư mục vật lý độc lập 100% (đã unlinked junction). Git root độc lập tại `D:\CNTT-KLCN155-waste-detection` (branch `main`). Virtualenv độc lập tại `D:\CNTT-KLCN155-waste-detection\.venv` (Python 3.12.10). Đường dẫn ghi đầu ra cục bộ: `artifacts/`, `data/audit/`, `outputs/`. Repo cũ tại `C:\Users\ad\Downloads\Do-an-deeplearning\...` được giữ nguyên vẹn. | `VERIFIED` | Toàn bộ script và huấn luyện chạy độc lập, không làm thay đổi hay xung đột với repo cũ. |
| **2** | Những script được nhắc trong R2 đã tồn tại chưa? Vì sao gói bàn giao trước thiếu script? | Các script đã tồn tại đầy đủ tại `D:\CNTT-KLCN155-waste-detection\scripts\` (`audit_and_reconcile_data.py`, `check_split_leakage.py`, `generate_phash_visual_audit.py`, `audit_backbone_transfer.py`, `run_smoke_test.py`). Lý do gói R2 trước thiếu: lệnh nén zip trong script cũ chỉ chọn `docs/plan/` và `data/audit/` mà bỏ sót thư mục `scripts/`. R2.1 sẽ đóng gói toàn bộ thư mục `scripts/` vào ZIP bàn giao. | `VERIFIED` | Cho phép PM hoặc bất kỳ kỹ sư nào chạy lại và tái hiện 100% kết quả audit và smoke test. |
| **3** | `duplicate_groups.csv` có 890 dòng cặp trùng khác nguồn, trong đó 879 dòng exact MD5. Vì sao báo cáo cũ ghi `cross_source_duplicates = 0`? | Do lỗi gán cứng tại dòng 427 của script cũ: `"cross_source_duplicates": 0`. Về mặt giải thuật: script duyệt qua `garbage_v2` trước, nên khi gặp ảnh trong `vn_trash` bị trùng, script ghi nhận `primary_path` ở `garbage_v2` và `duplicate_path` ở `vn_trash`. Kỹ sư trước hiểu nhầm rằng "tất cả ảnh bị loại nằm trong vn_trash nên là nội bộ vn_trash", trong khi thực tế 890 ảnh này bị loại khỏi `vn_trash` **chính vì chúng là bản sao của `garbage_v2`**. Đã sửa code và loại bỏ hoàn toàn bug này. | `VERIFIED` | Minh bạch số liệu trùng lặp liên nguồn; không còn báo cáo sai lệch. |
| **4** | Dữ liệu đã được khử trùng chung giữa hai nguồn trước khi chia tập chưa? | Phân tích quy trình cho thấy: Dữ liệu raw trước đây được quét khử trùng, nhưng khâu phân chia split cũ dựa trên danh sách tệp chưa xử lý triệt để các chuỗi ảnh chụp liên tiếp (burst shots) giữa hai nguồn, dẫn tới việc vẫn còn các cặp ảnh gần trùng bị lọt qua các split khác nhau. | `VERIFIED` | Đặt trạng thái `DATA_GATE_FLAGGED_WITH_LEAKAGE` cho split cũ và yêu cầu xử lý trước khi train chính thức. |
| **5** | Việc kiểm tra gần trùng đã duyệt mọi cặp cần thiết hay chỉ một phần? Ngưỡng pHash, thuật toán và hạn chế là gì? | Script kiểm tra cũ chỉ so sánh chuỗi băm chính xác (exact string match), bỏ sót các cặp có khoảng cách Hamming từ 1 đến 4. Script mới [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py) đã giải mã chuỗi hex thành mảng 64 bit và duyệt ma trận khoảng cách bitwise trên toàn bộ 14.831 ảnh qua các split. Ngưỡng sử dụng là Hamming $\le 4$. Hạn chế của pHash là có thể báo dương tính giả (false positive) đối với ảnh nền trắng đơn giản của các vật thể khác loại. | `VERIFIED` | Phát hiện chính xác toàn bộ 18 cặp ứng viên rò rỉ tiềm ẩn trên ổ đĩa. |
| **6** | PM từng phát hiện 18 cặp ứng viên pHash $\le 4$ nằm khác split. Bạn đã tái hiện được chưa? Bao nhiêu cặp là rò rỉ thật? | **Đã tái hiện chính xác 100% 18 cặp ứng viên của PM** (lưu tại `data/audit/cross_split_phash_verdicts.csv` và ảnh so sánh tại `data/audit/visual_phash_inspection/`). Kết quả thẩm định: **10 cặp là burst-shot cùng vật thể** (MAE < 20.0), **5 cặp là cùng vật thể xoay/đổi góc** (MAE 20.0–38.0), **3 cặp là trùng ngẫu nhiên khác lớp** trên nền trắng. Tổng cộng có **15 cặp rò rỉ dữ liệu thật**. | `VERIFIED` | Xác nhận rò rỉ trên split cũ; chặn không cho phép dùng split này để báo cáo benchmark chính thức. |
| **7** | VN Trash có thực sự là tập test ngoại bộ độc lập không? | **Hoàn toàn KHÔNG**. Trong tổng số 2.572 ảnh sạch của VN Trash: **1.801 ảnh nằm trong tập Train**, 386 ảnh trong Val, 385 ảnh trong Test; đồng thời **879 ảnh trùng khớp tuyệt đối MD5 với Garbage V2**. Do đó, VN Trash đã bị mô hình "học thuộc" trong khâu huấn luyện, không thể đóng vai trò tập test ngoại bộ. | `CONFLICT -> RESOLVED` | Sửa toàn bộ tài liệu; định vị VN Trash là dữ liệu gộp chung; tập test ngoại bộ bắt buộc phải là ảnh chụp thực tế TP.HCM mới. |
| **8** | Có chứng minh được Ecovision chưa từng thấy ảnh của VN Trash không? | **Không thể chứng minh được**. Do VN Trash chứa 879 ảnh giống hệt Garbage V2, mà Ecovision lại được train trên Garbage V2 không công bố split, nên Ecovision nhiều khả năng đã nhìn thấy các ảnh này trong quá trình pre-training. | `VERIFIED` | Bỏ qua Ecovision trong luồng chính; chuyển sang khởi tạo 100% từ ImageNet-1K chuẩn Torchvision. |
| **9** | Quy trình R2 có luồng "nếu fail test thì chỉnh loss và train lại". Đã sửa thế nào? | Đã sửa triệt để trong [06_SINGLE_OBJECT_EVALUATION_GATE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/06_SINGLE_OBJECT_EVALUATION_GATE.md). Xóa bỏ hoàn toàn mũi tên quay đầu từ Test Set về Huấn luyện. Toàn bộ việc chọn mô hình, tinh chỉnh siêu tham số và early stopping được thực hiện 100% trên Validation. Tập Test bị khóa kín và chỉ mở đúng 1 lần duy nhất sau khi mô hình đã đóng băng. Nếu fail test, mô hình bị REJECTED. | `VERIFIED` | Bảo toàn tính liêm chính học thuật và độ khách quan của bài báo cáo khóa luận. |
| **10** | Chuyển backbone sang SSDLite thực tế khớp bao nhiêu keys? Tỷ lệ khớp có chứng minh được detector hoạt động không? | Đã thực nghiệm trong [scripts/audit_backbone_transfer.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py): Khớp chính xác **258 / 308 keys backbone features (83,8%)**; 50 keys cuối không khớp do lớp chiếu khác kênh; 168 keys của detector (Extra layers + Detection heads) bắt buộc phải huấn luyện mới từ đầu. Đã nạp thử nghiệm và chạy forward pass thành công trên tensor `(1, 3, 320, 320)`, xuất ra đúng 300 boxes. **Tỷ lệ khớp keys không chứng minh được mô hình phát hiện tốt**; mAP bắt buộc phải đo trên tập dữ liệu gán nhãn bounding box. | `VERIFIED` | Có bảng ánh xạ CSV rõ ràng từng tầng; không còn kết luận vội vã. |
| **11** | Máy có GPU không? Đã cài PyTorch hỗ trợ GPU chưa? | Máy có card **NVIDIA GeForce RTX 2050 Laptop GPU (4.095,5 MB VRAM)**. Môi trường venv trước đó chỉ có PyTorch bản CPU. Nhóm đã thực hiện cài đặt thành công `torch==2.5.1+cu121` và `torchvision==0.20.1+cu121`. Lệnh kiểm tra `torch.cuda.is_available()` trả về `True`. | `VERIFIED` | Mở khóa toàn bộ năng lực tính toán GPU cho dự án. |
| **12** | Có ảnh chụp thực tế ngoài đời nào đã được đưa vào dữ liệu sạch chưa? | **Chưa có bất kỳ ảnh thực tế nào**. Toàn bộ 14.831 ảnh sạch hiện tại đều lấy từ 2 bộ dữ liệu Kaggle. Báo cáo trước ghi "đã kiểm chứng trên ảnh thực tế" là nhầm lẫn với kế hoạch thu thập tương lai. Đã đính chính rõ trong tài liệu: bộ ảnh thực tế TP.HCM đang trong kế hoạch thu thập và sẽ được thẩm định riêng. | `VERIFIED` | Minh bạch ranh giới giữa dữ liệu benchmark Kaggle và dữ liệu thực tế đời thực. |

---

## 3. Báo cáo Kết quả Thực nghiệm Smoke Test 3 Epochs trên GPU RTX 2050

Nhóm kỹ sư đã thực thi script chạy thử [scripts/run_smoke_test.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/run_smoke_test.py) trên tập dữ liệu `Data/processed/train` (10.381 ảnh) và `Data/processed/val` (2.225 ảnh). **Tập Test (2.225 ảnh) hoàn toàn bị khóa chặt và không nạp vào bộ nhớ để bảo toàn tính độc lập tuyệt đối**.

### 3.1. Bảng số liệu thực thi đo đạc trực tiếp:
- **Phần cứng:** NVIDIA GeForce RTX 2050 Laptop GPU (4.095,5 MB VRAM), CUDA 12.1.
- **Kiến trúc:** MobileNetV3-Large (Khởi tạo ImageNet-1K, output head 10 lớp logits).
- **Kỹ thuật tăng tốc:** Automatic Mixed Precision (AMP fp16) với `torch.amp.GradScaler`.

| Epoch | Thời gian thực thi | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro-F1 | Đỉnh VRAM GPU | Trạng thái Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 227.2 s | 0.8835 | 85.80% | 0.7084 | 92.99% | 0.9303 | 1.511,2 MB | Lưu mới best checkpoint |
| **2** | 194.2 s | 0.6509 | 95.50% | 0.6669 | 94.34% | 0.9421 | 1.511,2 MB | Cập nhật best checkpoint |
| **3** | 194.4 s | 0.5889 | 98.42% | 0.6562 | 94.56% | 0.9430 | 1.511,2 MB | Cập nhật best checkpoint |

### 3.2. Đánh giá chất lượng kỹ thuật:
1. **Khả năng tối ưu hóa & Gradient Flow:** Loss giảm ổn định từ $0.8835 \to 0.5889$; Val Accuracy đạt $94,56\%$; Val Macro-F1 đạt $0.9430$.
2. **Kiểm soát bộ nhớ GPU:** Đỉnh VRAM tiêu thụ đạt **1.511,2 MB** (chỉ chiếm $36,9\%$ dung lượng VRAM 4 GB). Hoàn toàn không xảy ra lỗi tràn bộ nhớ (CUDA OOM).
3. **Kiểm tra nạp lại Checkpoint (Reload Test):**
   - Checkpoint lưu tại [artifacts/smoke_test/best_smoke_checkpoint.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/smoke_test/best_smoke_checkpoint.pt) (48,58 MB).
   - Nạp vào model rỗng và so sánh tensor đầu ra trên cùng batch dữ liệu: Sai lệch tuyệt đối lớn nhất $\text{Max Diff} = \mathbf{0.00000000\text{e+}00}$. Model phục hồi tất định hoàn hảo.

---

## 4. Quyết định Cổng Dữ liệu và Kế hoạch Hành động Tiếp theo

### 4.1. Trạng thái các Cổng Kỹ thuật:
- **Cổng Môi trường & Mã nguồn (Code & Hardware Gate):** `PASSED` (GPU RTX 2050 hoạt động ổn định, PyTorch CUDA cu121 chuẩn, pipeline nạp dữ liệu - train - val - lưu - nạp lại checkpoint chạy thông suốt).
- **Cổng Dữ liệu (Data Gate):** `DATA_GATE_FLAGGED_WITH_LEAKAGE` (Phát hiện 15 cặp rò rỉ burst-shot xuyên split trên tập vật lý cũ).

### 4.2. Kế hoạch Hành động cho Task tiếp theo (Task 02 — Data Reconciliation & Official Training):
1. **Khắc phục 15 cặp rò rỉ burst-shot:** Viết script tự động gom 15 cặp ảnh này về cùng split Train (hoặc loại bỏ khỏi Val/Test) để đưa tập dữ liệu vật lý về trạng thái `ZERO_LEAKAGE` tuyệt đối.
2. **Chạy Huấn luyện Chính thức (Official Benchmark):**
   - Huấn luyện đầy đủ 30 epochs (Phase 1: 5 epochs head warmup; Phase 2: 25 epochs full fine-tuning) trên tập Train sạch.
   - Chọn best model dựa trên Validation Macro-F1.
3. **Mở Cổng Test Set Duy nhất 1 Lần:** Đánh giá mô hình đã đóng băng trên tập Test 2.225 ảnh và xuất báo cáo kết quả cuối cùng cho Giai đoạn A.
