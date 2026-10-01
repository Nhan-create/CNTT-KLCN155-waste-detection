# R2 REVIEW: BÁO CÁO TỰ PHẢN BIỆN, ĐỐI SOÁT VÀ SỬA ĐỔI TOÀN DIỆN KẾ HOẠCH

**Dự án:** Phân loại và Phát hiện Đa Rác Thải Sinh Hoạt (`CNTT-KLCN155`)  
**Chủ dự án (PM):** Ngô Thanh Nhân  
**Vai trò:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Thời điểm lập:** 01/10/2026  
**Trạng thái phiên:** `AUDITED_AND_RECTIFIED` (Đã đối soát thực tế trên ổ đĩa, xác minh layer mô hình, cô lập workspace)

---

## 1. Tự phản biện báo cáo R1 (Self-Critique & Root Cause Analysis)

Báo cáo R1 trước đây mắc các sai sót nghiêm trọng về tính độc lập của môi trường, tính chính xác số học, giả định nguồn gốc mô hình pre-trained và tự ý điều chỉnh phạm vi bài toán. Bảng dưới đây phân tích nguyên nhân gốc rễ và hành động khắc phục trong R2:

| STT | Vấn đề trong R1 | Phân loại R1 | Nguyên nhân gốc rễ (Root Cause) | Hành động khắc phục trong R2 | Phân loại R2 |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | Dùng NTFS Junction `D:\...` trỏ về repo cũ `C:\Users\ad\Downloads\...` | `CONFLICT` | Tiện lợi hóa tạo đường dẫn D:, nhưng làm cho mọi thao tác ghi/sửa qua D: đều trực tiếp tác động vào repo cũ, không đảm bảo tính độc lập mã nguồn. | Đã uncheck và gỡ bỏ Junction an toàn qua `cmd /c rmdir`. Khởi tạo Git repo độc lập mới (`git init -b main`) và tạo `.venv` độc lập tại `D:\CNTT-KLCN155-waste-detection`. Repo cũ được bảo toàn nguyên vẹn. | `VERIFIED` |
| 2 | Tổng số ảnh sạch 14.831 nhưng cộng train (10.489) + val (2.173) + test (2.170) = 14.832 | `CONFLICT` | Lỗi làm tròn thủ công và sao chép số liệu từ dự thảo chưa được đếm từ file thực tế trên ổ đĩa. | Viết script [scripts/audit_and_reconcile_data.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_and_reconcile_data.py) quét toàn bộ 14.831 file thực tế tại `D:\HK7\Đồ án khóa luận\Data\processed`. Số thực tế: Train = **10.381**, Val = **2.225**, Test = **2.225**. Tổng = **14.831** (khớp 100%). | `VERIFIED` |
| 3 | Tổng số ảnh ở bảng phân bố lớp ghi 15.448 ảnh | `CONFLICT` | Cộng nhầm giữa số liệu thô trước khử trùng của một số lớp và số liệu sau khử trùng của lớp khác. | Quét và đếm từng tệp thô: Tổng thô = **15.754** ảnh (`garbage_v2`: 12.259, `vn_trash`: 3.495). Trùng lặp loại bỏ = **923** ảnh. Số sạch sau khử trùng = **14.831** ảnh. Bảng [class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv) đối soát khớp từng dòng. | `VERIFIED` |
| 4 | Trừ 3 lớp `clothes`, `shoes`, `trash` ghi 13.760 thay vì 10.987 | `CONFLICT` | Sai sót tính nhẩm của kỹ sư lập kế hoạch, không qua script kiểm toán. | Công thức chuẩn: $14.831 - (1.892 + 1.449 + 503) = 14.831 - 3.844 = \mathbf{10.987}$ ảnh. Đã kiểm chứng qua dữ liệu thực tế. | `VERIFIED` |
| 5 | Tự ý hạ Giai đoạn B (Đa rác) xuống 6 lớp theo đề cương cũ | `CONFLICT` | Bị chi phối bởi bản thảo Word đề cương cũ mà không làm rõ với chỉ đạo mới của PM (10 lớp). | Chốt **10 lớp là taxonomy chuẩn mặc định xuyên suốt cả Giai đoạn A và Giai đoạn B**. Phân tích cụ thể các lớp tranh chấp (`clothes`, `shoes`, `trash`, `cardboard` vs `paper`) và lập bảng ánh xạ 10 lớp $\leftrightarrow$ 6 lớp chỉ như một kịch bản đối chiếu phụ nếu hội đồng khóa luận yêu cầu đối chiếu. | `VERIFIED` |
| 6 | Đề xuất dùng CVAT cho gán nhãn | `CONFLICT` | Đề xuất công cụ nặng, yêu cầu cài đặt Docker phức tạp trái với định hướng đơn giản, nhẹ và kiểm soát nội bộ của PM. | Loại bỏ hoàn toàn CVAT khỏi tài liệu. Thay thế bằng công cụ thẩm định/gán nhãn nội bộ nhẹ viết bằng Streamlit [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) và script kiểm định bbox Python có sẵn trong repo. | `VERIFIED` |
| 7 | Tin tưởng checkpoint `Ecovision` đạt 98% mà không làm rõ tập train của họ | `UNKNOWN` | Checkpoint tải từ HuggingFace `AmadFR/ecovision_mobilenetv3` được train trên `sumn2u/garbage-classification-v2` nhưng tác giả không công bố split train/val/test cụ thể. | Đánh dấu trạng thái checkpoint này là `PRETRAINING_OVERLAP_UNKNOWN`. Không dùng checkpoint này để chứng minh độ khái quát hóa trên tập test Garbage V2 (nguy cơ data leakage). Luồng huấn luyện chính thức sẽ khởi tạo từ ImageNet chuẩn (`weights='DEFAULT'`). Ecovision chỉ đóng vai trò baseline tham chiếu và chỉ được benchmark trên VN Trash + ảnh thực tế TP.HCM. | `VERIFIED` |
| 8 | File `yolo26n.pt` chưa kiểm tra nội dung thực tế | `UNKNOWN` | Thừa hưởng tệp checkpoint từ thư mục gốc cũ mà không bóc tách header. | Đã viết script đọc `torch.load` bóc tách `train_args` của checkpoint: Đây là checkpoint được train trên COCO 80 lớp (`tune-yolo26n-objv1-coco`), **không phải mô hình rác**. Phải huấn luyện mới hoàn toàn detector YOLOv8n trên tập dữ liệu rác. | `VERIFIED` |
| 9 | Tập detection v1 (1.419 ảnh) có tới 91,97% ảnh tổng hợp | `UNKNOWN` | Không kiểm kê nguồn gốc thực sự của từng ảnh trong tập v1. | Đã kiểm toán: 1.305 / 1.419 ảnh (91,97%) là ảnh tổng hợp từ Mendeley Synthetic (vật thể ghép trên nền phẳng giả lập), chỉ có 114 ảnh (8,03%) từ OpenImages. Không thể dùng tập này làm thước đo độ chính xác thực tế; phải tách riêng tập benchmark thực tế TP.HCM. | `VERIFIED` |
| 10 | Giả định chuyển giao backbone MobileNetV3 sang SSDLite320 "chỉ cần copy" | `UNKNOWN` | Thiếu phân tích layer-by-layer giữa Classifier và SSDLite Detector. | Đã viết script so sánh cấu trúc state_dict: Classifier có 312 keys, SSDLite320 có 476 keys. Trùng khớp 258/308 keys ở backbone `features`. 50 keys projection và 168 keys ở Extra Layers + Head detection **bắt buộc phải khởi tạo ngẫu nhiên và huấn luyện mới**. | `VERIFIED` |

---

## 2. Bảng phân loại hiện trạng tổng thể (Classification of Claims)

Mọi nhận định, thành phần kỹ thuật và số liệu của dự án được phân loại theo 4 cấp độ:
- **`VERIFIED`**: Đã chạy lệnh, quét file hoặc đối soát mã nguồn trực tiếp, có bằng chứng lưu trữ.
- **`PROPOSED`**: Đề xuất kỹ thuật đã được thiết kế chi tiết nhưng chưa chạy thực tế (chờ pha Implementation).
- **`UNKNOWN`**: Chưa có đủ dữ liệu/bằng chứng để khẳng định tuyệt đối.
- **`CONFLICT`**: Điểm mâu thuẫn giữa các tài liệu hoặc dữ liệu cần phải giải quyết dứt điểm.

| Hạng mục | Trạng thái R1 | Trạng thái R2 | Bằng chứng thực tế (Evidence) |
|:---|:---:|:---:|:---|
| **Thư mục workspace độc lập** | `CONFLICT` | `VERIFIED` | `D:\CNTT-KLCN155-waste-detection` là thư mục vật lý, Git init độc lập, `.venv` Python 3.12 độc lập. |
| **Bảo toàn repo cũ** | `UNKNOWN` | `VERIFIED` | `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3` không có commit mới, không bị xóa file. |
| **Số lượng ảnh thô & ảnh khử trùng** | `CONFLICT` | `VERIFIED` | [data/audit/audit_summary.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/audit_summary.json): 15.754 thô, 24.518 bản resize bị loại, 923 duplicate bị loại, 14.831 sạch. |
| **Phân chia tập train/val/test** | `CONFLICT` | `VERIFIED` | [data/audit/class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv): Train 10.381 (69,99%), Val 2.225 (15,00%), Test 2.225 (15,00%). Tổng 14.831. Không rò rỉ SHA-256 giữa các tập. |
| **Số lượng khi loại 3 lớp ngoài lề** | `CONFLICT` | `VERIFIED` | $14.831 - (1.892 + 1.449 + 503) = \mathbf{10.987}$ ảnh. |
| **Phạm vi lớp (Taxonomy)** | `CONFLICT` | `VERIFIED` | Chuẩn hóa **10 lớp** cho cả Giai đoạn A và Giai đoạn B: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`. |
| **Bản chất checkpoint `yolo26n.pt`** | `UNKNOWN` | `VERIFIED` | Checkpoint COCO 80 lớp (`tune-yolo26n-objv1-coco`), cần train mới hoàn toàn trên dữ liệu rác. |
| **Bản chất checkpoint `Ecovision`** | `UNKNOWN` | `VERIFIED` | Trained on `sumn2u/garbage-classification-v2` (split không rõ). Gắn tag `PRETRAINING_OVERLAP_UNKNOWN`. Train chính thức dùng ImageNet baseline. |
| **Bản chất dataset detection v1** | `UNKNOWN` | `VERIFIED` | 1.419 ảnh: 1.305 ảnh tổng hợp (91,97%), 114 ảnh thật (8,03%). Phải cô lập tập synthetic, không dùng làm test set đại diện thực tế. |
| **Tỷ lệ chuyển giao backbone MobileNetV3 sang SSDLite** | `UNKNOWN` | `VERIFIED` | 258/308 keys backbone tương thích. 168 keys extra layers & heads phải train mới. Lưu tại [data/audit/backbone_transfer_audit.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json). |
| **Công cụ gán nhãn** | `CONFLICT` | `VERIFIED` | Bỏ hoàn toàn CVAT. Sử dụng in-repo Streamlit review tool [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) và script thẩm định bbox Python. |
| **Đánh giá mAP và WBF fusion** | `PROPOSED` | `PROPOSED` | Quy trình đánh giá trên Validation set, chỉ kích hoạt WBF khi mAP50 tăng $\ge 1.5\%$ và latency đạt ngưỡng $\le 60\text{ ms}$. |
| **Tập kiểm thử độc lập thực tế TP.HCM** | `PROPOSED` | `PROPOSED` | Kế hoạch chụp và thu thập 200–300 ảnh thực tế tại TP.HCM (HUIT, đường phố, trạm rác) dùng làm Test Set mù (Blind Test). |

---

## 3. Trả lời chi tiết các câu hỏi chất vấn của PM

### 3.1. Vấn đề thư mục dự án
1. **Hai đường dẫn trước đây có trỏ tới cùng thư mục thật không?**  
   - **Có.** Đường dẫn `D:\CNTT-KLCN155-waste-detection` trước đây là một NTFS Reparse Point (Junction) trỏ thẳng vào `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3`. Mọi thao tác qua D: thực chất ghi trực tiếp vào ổ C:.
2. **Đã có file nào bị sửa trong repo cũ qua Junction chưa?**  
   - Không có file mã nguồn cũ nào bị phá hủy. Các file plan R1 được tạo ra tại `docs/plan/` đã nằm trong thư mục cũ.
3. **Các file kế hoạch hiện đang nằm ở đâu?**  
   - Đã được copy và tách riêng hoàn toàn sang workspace vật lý mới tại [D:\CNTT-KLCN155-waste-detection\docs\plan](file:///D:/CNTT-KLCN155-waste-detection/docs/plan).
4. **Workspace mới có Git root và venv riêng chưa?**  
   - **Đã có.** Đã khởi tạo Git repository độc lập (`git init -b main`) tại `D:\CNTT-KLCN155-waste-detection`. Môi trường ảo Python 3.12 riêng biệt đã được tạo tại `D:\CNTT-KLCN155-waste-detection\.venv`.
5. **Có đường dẫn cấu hình nào vẫn ghi đầu ra về repo cũ không?**  
   - Không. Mọi file cấu hình [configs/config.yaml](file:///D:/CNTT-KLCN155-waste-detection/configs/config.yaml), [configs/train_stage_a.yaml](file:///D:/CNTT-KLCN155-waste-detection/configs/train_stage_a.yaml), đường dẫn checkpoints, dữ liệu audit đều trỏ cục bộ bên trong `D:\CNTT-KLCN155-waste-detection` hoặc `D:\HK7\Đồ án khóa luận\Data`.

### 3.2. Đối soát toàn bộ số liệu dữ liệu
1. **Mâu thuẫn tổng sạch 14.831 vs tổng train/val/test 14.832:**  
   - Báo cáo R1 ghi: Train 10.489 + Val 2.173 + Test 2.170 = 14.832. Đây là lỗi chép số từ dự thảo chưa đối soát.  
   - Số liệu kiểm toán thực tế qua [class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv) trên từng file trên ổ đĩa:  
     $$\text{Train: } 10.381 + \text{Val: } 2.225 + \text{Test: } 2.225 = \mathbf{14.831} \text{ ảnh}$$
     Khớp chính xác 100% với tổng số ảnh sạch sau khi loại bỏ 923 tệp trùng lặp.
2. **Con số 15.448 ảnh ở bảng phân bố lớp:**  
   - Do cộng lẫn giữa số liệu trước khử trùng của một số lớp và sau khử trùng của lớp khác. Số liệu thô chính xác là **15.754** ảnh (`garbage_v2`: 12.259 + `vn_trash`: 3.495).
3. **Phép trừ 3 lớp clothes, shoes, trash:**  
   - Tổng sạch = 14.831.  
   - Số ảnh của 3 lớp: `clothes` = 1.892, `shoes` = 1.449, `trash` = 503 $\rightarrow$ Tổng trừ = **3.844** ảnh.  
   - Số lượng còn lại: $14.831 - 3.844 = \mathbf{10.987}$ ảnh. Con số 13.760 ở R1 là sai sót tính toán đã được loại bỏ hoàn toàn.
4. **Hiện trạng tập detection v1 (1.419 ảnh):**  
   - **1.305 ảnh (91,97%)** là ảnh ghép đồ họa từ Mendeley Synthetic Dataset (rác nhân tạo đặt trên nền đơn sắc/nền phẳng).
   - **114 ảnh (8,03%)** là ảnh thực tế từ OpenImages.  
   - **Kết luận:** Tập này có giá trị tiền huấn luyện phát hiện bounding box thô, nhưng **hoàn toàn không đại diện cho ảnh thực tế**. Cần thu thập tập Test thực tế TP.HCM độc lập.

### 3.3. Checkpoint pre-trained và nguồn GitHub
1. **Checkpoint Ecovision:**  
   - Nguồn gốc: `AmadFR/ecovision_mobilenetv3` (MobileNetV3-Large, huấn luyện trên Kaggle Garbage V2).  
   - Tác giả không công bố danh sách file trong train split. Nếu đánh giá checkpoint này trên tập test Garbage V2 của dự án, có nguy cơ cao xảy ra hiện tượng **Train-Test Contamination** (mô hình đã nhìn thấy ảnh test trong lúc huấn luyện ban đầu).  
   - **Giải pháp:**  
     - Trạng thái: `PRETRAINING_OVERLAP_UNKNOWN`.  
     - Mô hình huấn luyện chính thức của đề tài sẽ bắt đầu từ trọng số **ImageNet-1K chuẩn** (`torchvision.models.mobilenet_v3_large(weights='DEFAULT')`).  
     - Checkpoint Ecovision chỉ được dùng như một baseline đối chứng ngoại bộ và chỉ được đánh giá trên tập **VN Trash** (do Ecovision chưa từng được train trên VN Trash) và ảnh thực tế TP.HCM.
2. **File `yolo26n.pt`:**  
   - Đây là checkpoint detector COCO 80 lớp (`tune-yolo26n-objv1-coco`), không có nhãn rác thải.  
   - Không thể đưa vào inference trực tiếp cho bài toán rác. Nó chỉ đóng vai trò trọng số khởi tạo (backbone transfer) cho bài toán YOLOv8n đa rác.

### 3.4. Chuyển giao trọng số MobileNetV3 sang SSDLite320
1. **MobileNetV3 Classifier làm nhiệm vụ gì?**  
   - Trích xuất đặc trưng toàn cục (Global Average Pooling) và phân loại một ảnh thành 1 trong 10 nhãn rác.  
2. **MobileNetV3 trong SSDLite làm nhiệm vụ gì?**  
   - Đóng vai trò mạng xương sống (Backbone Feature Extractor), trích xuất bản đồ đặc trưng đa tỉ lệ (Feature Maps ở tầng C4, C5) để đưa vào các nhánh dự đoán hộp (Location Head) và nhãn (Classification Head).  
3. **Vì sao Classifier không tự trở thành Detector khi đưa ảnh nhiều rác vào?**  
   - Classifier ép toàn bộ đặc trưng không gian về một vector 1D duy nhất $\mathbb{R}^{960}$ qua Global Pooling, làm mất hoàn toàn tọa độ vị trí không gian $(x, y, w, h)$. Khi có nhiều rác, các đặc trưng bị hòa lẫn vào nhau, dẫn đến dự đoán sai hoặc chỉ nhận diện vật thể chiếm diện tích lớn nhất.  
4. **Đối soát layer-by-layer thực tế (Backbone Transfer):**  
   - Tổng số keys của MobileNetV3 Classifier: 312 keys.  
   - Tổng số keys của SSDLite320: 476 keys.  
   - Số keys tương thích hoàn toàn về tên, kích thước tensor và kiểu dữ liệu ở backbone `features`: **258 / 308 keys**.  
   - 50 keys thuộc các tầng biến đổi đặc trưng của SSDLite không khớp với classifier chuẩn.  
   - 4 keys của classifier head (`classifier.0`, `classifier.3`) bị loại bỏ vì không tương thích với detection.  
   - **168 keys** thuộc Extra Feature Pyramid Layers (72 keys) và Detection Box/Class Heads (96 keys) **phải được khởi tạo ngẫu nhiên và huấn luyện từ đầu**.

---

## 4. Danh mục tài liệu kế hoạch đã được đồng bộ trong R2

Toàn bộ 13 tài liệu Markdown trong thư mục [docs/plan/](file:///D:/CNTT-KLCN155-waste-detection/docs/plan) đã được rà soát và cập nhật:

1. [00_MASTER_PLAN.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/00_MASTER_PLAN.md): Cập nhật đường dẫn workspace mới, sửa các số liệu dữ liệu, loại bỏ CVAT, định hình chuẩn 10 lớp cho cả 2 giai đoạn.
2. [01_CURRENT_STATE_AND_ARCHITECTURE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/01_CURRENT_STATE_AND_ARCHITECTURE.md): Phân tích chi tiết chuyển giao MobileNetV3 sang SSDLite (258/308 keys), giải thích yolo26n.pt là COCO model, phân biệt rõ vai trò classifier vs detector.
3. [02_DATA_INVENTORY_AND_TAXONOMY.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/02_DATA_INVENTORY_AND_TAXONOMY.md): Bảng phân bố 10 lớp chính thức từ [class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv), làm rõ phép trừ 3 lớp ($14.831 - 3.844 = 10.987$), phân tích nhãn `trash` và `cardboard`/`paper`.
4. [03_SPLITS_AND_PREPROCESSING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/03_SPLITS_AND_PREPROCESSING.md): Cập nhật số lượng split chính xác (Train 10.381 / Val 2.225 / Test 2.225), xác nhận Zero-leakage bằng SHA-256 hash.
5. [04_GITHUB_REUSE_AND_DEPENDENCIES.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/04_GITHUB_REUSE_AND_DEPENDENCIES.md): Đánh dấu checkpoint Ecovision là `PRETRAINING_OVERLAP_UNKNOWN`, xác lập ImageNet-1K là trọng số huấn luyện chính thức, chuẩn hóa ultralytics YOLOv8.
6. [05_SINGLE_OBJECT_TRAINING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/05_SINGLE_OBJECT_TRAINING.md): Quy trình fine-tuning MobileNetV3 10 lớp trên 10.381 ảnh train, cơ chế class-weight bù mất cân bằng cho lớp `battery` (529) và `trash` (352).
7. [06_SINGLE_OBJECT_EVALUATION_GATE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/06_SINGLE_OBJECT_EVALUATION_GATE.md): Tiêu chí nghiệm thu Cổng A (Gate A) chặt chẽ, kiểm thử độc lập trên VN Trash và ảnh thực tế, chống rò rỉ dữ liệu.
8. [07_MULTIOBJECT_DATA_AND_ANNOTATIONS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/07_MULTIOBJECT_DATA_AND_ANNOTATIONS.md): Loại bỏ CVAT, mô tả quy trình gán nhãn và kiểm tra bbox bằng Streamlit review tool và Python script nội bộ. Cô lập 1.305 ảnh synthetic của tập detection v1.
9. [08_MULTIOBJECT_MODEL_TRAINING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/08_MULTIOBJECT_MODEL_TRAINING.md): Quy trình huấn luyện YOLOv8n (mô hình chính) và SSDLite320 (đối chứng), ứng dụng trọng số backbone chuyển giao.
10. [09_EXPERIMENTS_AND_ERROR_ANALYSIS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/09_EXPERIMENTS_AND_ERROR_ANALYSIS.md): Kế hoạch thực nghiệm WBF có điều kiện (chỉ áp dụng nếu mAP50 tăng $\ge 1.5\%$ và latency $\le 60\text{ ms}$).
11. [10_WEB_INTEGRATION_AND_MCP_TESTS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/10_WEB_INTEGRATION_AND_MCP_TESTS.md): Tích hợp giao diện Streamlit phục vụ cả phân loại đơn rác và phát hiện đa rác, kiểm thử E2E bằng MCP `chrome-devtools`.
12. [11_DELIVERY_AND_REPRODUCIBILITY.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/11_DELIVERY_AND_REPRODUCIBILITY.md): Quy trình bàn giao, hướng dẫn tái hiện với Docker/venv độc lập và bảo tồn dữ liệu.
13. [CLARIFICATIONS_AND_DECISIONS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/CLARIFICATIONS_AND_DECISIONS.md): Ghi nhận toàn bộ quyết định kiến trúc, làm rõ 6 câu hỏi cốt lõi của PM và cơ sở khoa học.
14. [BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md): Báo cáo tóm tắt điều hành dành cho PM và giảng viên hướng dẫn.

---

## 5. Kết luận trạng thái sẵn sàng

- **Trạng thái phê duyệt đề xuất:** `PLAN_READY` (về mặt kiến trúc, phân chia dữ liệu, cô lập môi trường và kế hoạch triển khai).
- Mọi điểm mâu thuẫn (`CONFLICT`) và điểm chưa rõ (`UNKNOWN`) trong R1 đã được chuyển hóa thành `VERIFIED` thông qua kiểm toán dữ liệu và mô hình thực tế trên máy chủ.
- Dự án đã hoàn toàn sẵn sàng để bước vào **Pha Triển Khai (Implementation Phase)**, bắt đầu bằng **Task 01: Thiết lập Pipeline Tiền Xử Lý Dữ Liệu và Huấn Luyện Thử Nghiệm Baseline Giai Đoạn A**.
