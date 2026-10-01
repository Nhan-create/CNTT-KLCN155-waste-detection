# 06 — SINGLE OBJECT EVALUATION GATE: CỔNG ĐÁNH GIÁ CHUYỂN GIAI ĐOẠN ĐƠN RÁC (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã cập nhật chỉ số test set thực tế 2.225 ảnh)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Thiết lập một **Cổng kiểm soát kỹ thuật nghiêm ngặt (Quality Gate A)** nhằm đánh giá toàn diện mô hình phân loại đơn rác MobileNetV3-Large trên tập Test độc lập (**2.225 ảnh sạch**); đo lường các chỉ số học máy đa chiều (Top-1 Accuracy, Macro-F1, Precision, Recall từng lớp, Confusion Matrix); phân tích sâu các sai số theo chi phí bất đối xứng (Asymmetric Misclassification Cost); và đưa ra quyết định có điều kiện cho phép chuyển giao sang Giai đoạn B (Phát hiện đa rác).

### 1.2. Vấn đề cần giải quyết
1. **Khắc phục sai lệch số lượng tập Test:** Sửa con số nháp 2.170 thành con số vật lý chính xác trên ổ đĩa: **2.225 ảnh** (đã kiểm toán trong [data/audit/class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv)).
2. **Chi phí sai lệch bất đối xứng (Asymmetric Cost):** Nhầm một chai nhựa thành tờ giấy chỉ gây phiền toái nhỏ trong khâu tái chế; nhưng bỏ sót hoặc nhầm một cục pin / rác nguy hại thành rác hữu cơ có thể dẫn đến nguy cơ cháy nổ lò ép rác. Lớp `battery` bắt buộc phải có ngưỡng Recall kiểm soát riêng ($\ge 0.880$).
3. **Đánh giá trên tập ngoại bộ (Out-of-distribution Test):** Tiến hành benchmark đối chứng trên tập VN Trash và ảnh thực tế TP.HCM để xác minh tính khái quát hóa độc lập.

---

## 2. Hiện trạng tập Test độc lập đã xác minh (`VERIFIED`)

Thư mục vật lý: [D:\HK7\Đồ án khóa luận\Data\processed\test](file:///D:/HK7/Đồ%20án%20khóa%20luận/Data/processed/test) gồm đúng **2.225 ảnh**, phân bố 10 lớp:
- `battery`: **114** ảnh
- `biological`: **105** ảnh
- `cardboard`: **309** ảnh
- `clothes`: **284** ảnh
- `glass`: **261** ảnh
- `metal`: **338** ảnh
- `paper`: **223** ảnh
- `plastic`: **298** ảnh
- `shoes`: **218** ảnh
- `trash`: **75** ảnh
- **Tổng cộng:** $114 + 105 + 309 + 284 + 261 + 338 + 223 + 298 + 218 + 75 = \mathbf{2.225\text{ ảnh}}$ (khớp 100%).

---

## 3. Bảng Tiêu chí Nghiệm thu Cổng Chuyển giai đoạn (Gate A Criteria)

Mô hình phân loại MobileNetV3-Large chỉ được phê duyệt hoàn thành Giai đoạn A khi đạt đầy đủ tất cả các điều kiện sau trên tập Test:

| Chỉ số kiểm soát | Ngưỡng bắt buộc (Threshold) | Ý nghĩa kỹ thuật & Ràng buộc học thuật | Trạng thái kiểm tra |
|:---|:---:|:---|:---:|
| **Top-1 Accuracy tổng** | **$\ge 88.0\%$** | Đảm bảo năng lực nhận dạng vượt trội trên toàn tập | Bắt buộc |
| **Macro-Averaged F1 Score** | **$\ge 0.850$** | Đảm bảo chất lượng đồng đều giữa các lớp | Bắt buộc |
| **Recall lớp `battery` (Nguy hại)** | **$\ge 0.880$** | Chặn đứng nguy cơ bỏ sót pin / rác thải nguy hại | Bắt buộc |
| **Recall tối thiểu trên mọi lớp** | **$\ge 0.800$** | Không cho phép bất kỳ lớp nào bị bỏ rơi dưới 80% | Bắt buộc |
| **Data Leakage Check** | **Zero Duplicate** | Xác nhận 0 tệp trùng băm SHA-256 hay pHash giữa Train và Test | Bắt buộc |
| **Độ trễ suy luận (Inference Latency)** | **$< 20\text{ ms/ảnh}$ (CPU)** | Đảm bảo chạy mượt trên CPU máy tính cá nhân | Bắt buộc |

---

## 4. Biên bản nghiệm thu và Quyết định chuyển tiếp

- Nếu **PASS Gate A**:
  - Trích xuất 258 keys backbone features từ `artifacts/run-001/best.pt` để chuyển giao sang mạng SSDLite320 ở Giai đoạn B.
  - Phê duyệt khởi động quy trình gán nhãn đa rác bằng in-repo review tool [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py).
- Nếu **FAIL Gate A**:
  - Dừng lại phân tích ma trận nhầm lẫn `confusion_matrix_test.png`.
  - Điều chỉnh hàm mất mát (Focal Loss / Tăng class-weight) và huấn luyện lại trước khi sang Giai đoạn B.
