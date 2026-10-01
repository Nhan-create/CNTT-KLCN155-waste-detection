# BÁO CÁO TỔNG KẾT KIỂM KÊ HIỆN TRẠNG, ĐỐI SOÁT VÀ KẾ HOẠCH TRIỂN KHAI DỰ ÁN PHÂN LOẠI RÁC (CNTT-KLCN155 — R2)

**Kính gửi:** Chủ dự án — Anh Ngô Thanh Nhân  
**Người thực hiện:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Thời điểm:** 01/10/2026  
**Trạng thái kế hoạch:** **`PLAN_READY`** (Đã kiểm chứng dữ liệu thực tế trên ổ đĩa, xác minh layer mô hình, cô lập workspace)

---

## 1. Tóm tắt kiến trúc hệ thống

Theo định hướng của Chủ dự án và các ràng buộc học thuật của đề tài, hệ thống được phân rã thành **hai giai đoạn khoa học độc lập nhưng có tính kế thừa sâu sắc**:

```
+-----------------------------------------------------------------------------------+
|                        GIAI ĐOẠN A: Phân loại Đơn rác (10 lớp)                    |
|  Ảnh đơn vật thể (224x224) ---> MobileNetV3-Large Classifier ---> 1 Nhãn + Conf  |
+-----------------------------------------------------------------------------------+
                                         │
                 Tái sử dụng 258/308 keys Backbone & Copy-Paste Cutouts
                                         ▼
+-----------------------------------------------------------------------------------+
|                        GIAI ĐOẠN B: Phát hiện Đa rác (10 lớp / 6 nhóm)            |
|  Ảnh đa rác (640x640) ---> [YOLOv8n (Chính)] & [SSDLite320 (Đối chứng)]           |
|                                         │                                         |
|                                         ▼                                         |
|                           Weighted Boxes Fusion (WBF có điều kiện)                |
|                                         │                                         |
|                                         ▼                                         |
|                       Bounding Boxes + Nhãn + Đếm số lượng theo lớp               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
                 Giao diện Web Streamlit (streamlit_app.py) & Review Tool
```

### Bản chất kỹ thuật của các thành phần kiến trúc:
1. **MobileNetV3 Classifier (Giai đoạn A):** Mạng tích chập phân loại hình ảnh (5.4M tham số, ~17 MB). Sử dụng phép gộp trung bình toàn cục (Global Average Pooling - GAP) để nén toàn bộ không gian ảnh thành 1 vector duy nhất ($1 \times 960$) trước khi đi qua lớp Linear để phân loại 1 trong 10 lớp. Chuyên xử lý ảnh cận cảnh chứa **đúng một vật thể rác**. Khởi tạo huấn luyện chính thức từ **ImageNet-1K chuẩn** để đảm bảo tính khách quan cho tập test.
2. **MobileNetV3 trong SSDLite320 (Giai đoạn B):** Đóng vai trò là **Mạng xương sống trích xuất đặc trưng (Backbone Feature Extractor)**. Toàn bộ phần classifier head và GAP bị lược bỏ; mô hình kế thừa chính xác **258 / 308 keys** tương thích từ MobileNetV3 Classifier, cung cấp các Feature Maps đa tầng tỷ lệ ($C_4, C_5$) cho các đầu dò detection. Toàn bộ **168 keys** thuộc Extra Feature Pyramid Layers và Detection Heads bắt buộc khởi tạo và huấn luyện mới từ đầu.
3. **YOLOv8n Detector (Giai đoạn B):** Bộ phát hiện đối tượng đơn giai đoạn không dùng điểm neo (One-Stage Anchor-Free Detector) hiện đại (3.2M tham số). Kết hợp Backbone CSPDarknet, Neck PANet (C2f), và Decoupled Head để cùng lúc dự đoán tọa độ ($x, y, w, h$) và xác suất 10 nhóm vật liệu cho **nhiều vật thể rác hỗn hợp** ở các kích thước và vị trí ngẫu nhiên trên một ảnh duy nhất ($640 \times 640$).
4. **Weighted Boxes Fusion (WBF):** Thuật toán kết hợp có điều kiện. WBF không chọn lọc loại trừ như NMS mà tính trung bình có trọng số tọa độ và độ tin cậy từ dự đoán của cả hai mô hình ($w_{\text{YOLO}}, w_{\text{SSD}}$), được tối ưu hóa tham số hoàn toàn trên tập Validation. **Chỉ kích hoạt làm chế độ mặc định nếu tăng $\ge 1.5\%$ mAP50 và độ trễ $\le 60\text{ ms}$**.
5. **Vì sao Classifier không tự trở thành Detector khi đưa ảnh nhiều rác vào?**
   - **Mất mát tọa độ do GAP:** Lớp Global Average Pooling trung bình hóa toàn bộ không gian $H \times W$, xóa sạch thông tin vị trí không gian của các vật thể.
   - **Nhiễu loạn vector đặc trưng (Feature Dilution):** Khi ảnh có nhiều vật thể và nền lộn xộn, các đặc trưng bị hòa trộn hỗn độn; classifier chỉ xuất ra 1 phân bố xác suất duy nhất (thường bị chi phối bởi vật thể to nhất hoặc đoán sai hoàn toàn).
   - **Không có đầu ra hồi quy tọa độ (Regression Head):** Về mặt toán học, classifier không có các nhánh tích chập dự đoán hộp bao ($dx, dy, dw, dh$) nên không thể phân tách hay đếm số lượng vật thể.

---

## 2. Kết quả kiểm toán và đối soát hiện trạng (`VERIFIED`)

Bảng đối soát toàn diện các phát hiện kỹ thuật với bằng chứng thực tế trên ổ đĩa:

| Hạng mục | Phát hiện & Kết luận thực tế | Phân loại | Bằng chứng kiểm chứng (Evidence) |
|:---|:---|:---:|:---|
| **Thư mục dự án** | Đã tháo dỡ Junction an toàn. Khởi tạo Git repo mới và `.venv` độc lập tại `D:\CNTT-KLCN155-waste-detection`. Repo cũ tại `C:\Users\ad\Downloads\Do-an-deeplearning\...` được bảo toàn nguyên vẹn. | `VERIFIED` | Đường dẫn vật lý độc lập, lệnh PowerShell `Get-Item` kiểm tra không còn ReparsePoint. |
| **Dữ liệu thô** | 15.754 ảnh thô (`garbage_v2`: 12.259, `vn_trash`: 3.495). Đã loại trừ 24.518 ảnh bản sao resize (`standardized_256`, `standardized_384`). | `VERIFIED` | [data/audit/audit_summary.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/audit_summary.json), [raw_inventory.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/raw_inventory.csv) |
| **Trùng lặp dữ liệu** | Đúng 923 tệp trùng lặp (897 MD5 exact match, 26 pHash near match). Toàn bộ 923 tệp trùng nằm trong `vn_trash`. Trùng lặp giữa 2 nguồn: 0 ảnh. | `VERIFIED` | [duplicate_groups.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/duplicate_groups.csv), [excluded_samples.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/excluded_samples.csv) |
| **Dữ liệu sạch & Split** | 14.831 ảnh sạch: Train = **10.381** ($69,99\%$), Val = **2.225** ($15,00\%$), Test = **2.225** ($15,00\%$). Tổng khớp 100% bằng 14.831. Không rò rỉ SHA-256. | `VERIFIED` | [class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv), [split_manifest.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/split_manifest.csv) |
| **Phép trừ 3 lớp** | $14.831 - (1.892 + 1.449 + 503) = 14.831 - 3.844 = \mathbf{10.987\text{ ảnh}}$ (sửa dứt điểm lỗi 13.760 ở R1). | `VERIFIED` | Số liệu đối soát từ `class_counts.csv`. |
| **Detection v1** | 1.419 ảnh, 4.602 boxes. Trong đó 1.305 ảnh (91,97%) là ảnh tổng hợp (Mendeley Synthetic), chỉ 114 ảnh (8,03%) là ảnh thật (OpenImages). Phải cô lập tập synthetic khỏi tập Test. | `VERIFIED` | Quét thư mục ảnh và phân loại nguồn gốc trong `audit_summary.json`. |
| **Trọng số Ecovision** | Checkpoint `AmadFR/ecovision_mobilenetv3` được train trên Garbage V2 nhưng không rõ train split. Trạng thái: `PRETRAINING_OVERLAP_UNKNOWN`. Không dùng làm checkpoint chính thức để tránh ô nhiễm tập test; chuyển sang dùng **ImageNet-1K chuẩn**. | `VERIFIED` | HuggingFace model card metadata và đối soát mã nguồn Torchvision. |
| **Trọng số yolo26n.pt** | Bóc tách metadata xác nhận đây là mô hình COCO 80 lớp (`tune-yolo26n-objv1-coco`), không phải mô hình rác. Huấn luyện đa rác phải chạy mới trên dữ liệu rác. | `VERIFIED` | Trích xuất `torch.load` bóc tách `train_args` lưu trong nhật ký Task 260. |
| **Backbone Transfer** | So sánh state_dict: Khớp chính xác **258 / 308 keys** backbone features. 168 keys extra layers & heads bắt buộc phải train mới. | `VERIFIED` | [backbone_transfer_audit.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json) |
| **Quy trình gán nhãn** | Loại bỏ hoàn toàn CVAT. Sử dụng in-repo Streamlit review tool [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) và script Python nội bộ. | `VERIFIED` | Mã nguồn trực tiếp trong repo, không phụ thuộc bên ngoài. |

---

## 3. Danh mục 15 tài liệu kế hoạch đã đồng bộ hoàn chỉnh

Toàn bộ tài liệu nằm tại [docs/plan/](file:///D:/CNTT-KLCN155-waste-detection/docs/plan):

1. [R2_REVIEW.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/R2_REVIEW.md): Báo cáo tự phản biện, phân tích nguyên nhân gốc rễ và bằng chứng khắc phục.
2. [00_MASTER_PLAN.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/00_MASTER_PLAN.md): Kế hoạch tổng thể 5 chặng, tiêu chí nghiệm thu Gate A và Gate B.
3. [01_CURRENT_STATE_AND_ARCHITECTURE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/01_CURRENT_STATE_AND_ARCHITECTURE.md): Phân tích layer-by-layer chuyển giao backbone (258/308 keys), giải thích 6 câu hỏi cốt lõi.
4. [02_DATA_INVENTORY_AND_TAXONOMY.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/02_DATA_INVENTORY_AND_TAXONOMY.md): Bảng phân bố 10 lớp chính thức, khử trùng lặp và phân tích tập synthetic.
5. [03_SPLITS_AND_PREPROCESSING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/03_SPLITS_AND_PREPROCESSING.md): Phân chia tập 10.381/2.225/2.225, xác nhận Zero-leakage SHA-256.
6. [04_GITHUB_REUSE_AND_DEPENDENCIES.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/04_GITHUB_REUSE_AND_DEPENDENCIES.md): Phân tích bản quyền, kiểm toán nguy cơ rò rỉ Ecovision, chuẩn hóa YOLOv8n.
7. [05_SINGLE_OBJECT_TRAINING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/05_SINGLE_OBJECT_TRAINING.md): Quy trình fine-tuning MobileNetV3 10 lớp khởi tạo từ ImageNet-1K.
8. [06_SINGLE_OBJECT_EVALUATION_GATE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/06_SINGLE_OBJECT_EVALUATION_GATE.md): Tiêu chí nghiệm thu Cổng Gate A trên 2.225 ảnh test độc lập.
9. [07_MULTIOBJECT_DATA_AND_ANNOTATIONS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/07_MULTIOBJECT_DATA_AND_ANNOTATIONS.md): Quy trình gán nhãn bằng in-repo review tool, loại bỏ CVAT.
10. [08_MULTIOBJECT_MODEL_TRAINING.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/08_MULTIOBJECT_MODEL_TRAINING.md): Huấn luyện YOLOv8n và SSDLite320 với backbone chuyển giao.
11. [09_EXPERIMENTS_AND_ERROR_ANALYSIS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/09_EXPERIMENTS_AND_ERROR_ANALYSIS.md): Thực nghiệm WBF có điều kiện và phân tích sai số đếm.
12. [10_WEB_INTEGRATION_AND_MCP_TESTS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/10_WEB_INTEGRATION_AND_MCP_TESTS.md): Tích hợp Web đa chế độ và kiểm thử tự động MCP chrome-devtools.
13. [11_DELIVERY_AND_REPRODUCIBILITY.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/11_DELIVERY_AND_REPRODUCIBILITY.md): Quy chuẩn đóng gói bàn giao và các lệnh tái hiện.
14. [CLARIFICATIONS_AND_DECISIONS.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/CLARIFICATIONS_AND_DECISIONS.md): Ghi nhận toàn bộ quyết định kiến trúc và giải đáp thấu đáo 32 câu hỏi.
15. [BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md): Báo cáo tóm tắt điều hành này.

---

## 4. Quyết định trạng thái phê duyệt

- **Trạng thái đề xuất:** **`PLAN_READY`**
- **Căn cứ khoa học:** Toàn bộ các mâu thuẫn số học, rủi ro rò rỉ dữ liệu, bẫy ảnh tổng hợp, sự cố Junction và công cụ gán nhãn đã được giải quyết dứt điểm và lưu bằng chứng kiểm chứng trực tiếp trên máy chủ.
- **Hành động tiếp theo:** Khởi động ngay **Task 01: Thiết lập Pipeline Tiền Xử Lý Dữ Liệu và Huấn Luyện Thử Nghiệm Baseline Giai Đoạn A**.
