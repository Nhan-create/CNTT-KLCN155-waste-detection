# 11 — DELIVERY AND REPRODUCIBILITY: ĐÓNG GÓI BÀN GIAO VÀ HƯỚNG DẪN TÁI HIỆN (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã đồng bộ quy chuẩn workspace độc lập)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Quy định chi tiết gói sản phẩm bàn giao cuối cùng (Delivery Package) của dự án khóa luận CNTT-KLCN155; thiết lập tài liệu hướng dẫn vận hành và các câu lệnh PowerShell chuẩn xác để tái hiện toàn bộ quy trình từ khâu chuẩn bị dữ liệu, huấn luyện mô hình, đánh giá benchmark đến khởi chạy ứng dụng Web; phân định rạch ròi giữa trạng thái hoàn thiện kỹ thuật phần mềm (`ENGINEERING_COMPLETE`) và mức độ kiểm chứng khoa học của mô hình (`MODEL_VALIDATION`); tạo tệp kê khai toàn vẹn `DELIVERY_MANIFEST.json` kèm mã băm SHA-256.

### 1.2. Vấn đề cần giải quyết
1. **Độc lập workspace hoàn toàn (`VERIFIED`):** Workspace tại [D:\CNTT-KLCN155-waste-detection](file:///D:/CNTT-KLCN155-waste-detection) là kho mã nguồn vật lý độc lập, có Git root riêng và `.venv` riêng. Repo cũ tại `C:\Users\ad\Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3` được bảo toàn nguyên trạng.
2. **Khả năng tái hiện (Reproducibility):** Khóa phiên bản cụ thể của các thư viện trong `requirements.txt` và cung cấp lệnh PowerShell kiểm chứng từng bước.
3. **Phân định ranh giới học thuật:** Code chạy không lỗi (`ENGINEERING_COMPLETE`) không tự động đồng nghĩa với mô hình hoàn hảo 100% ngoài thực tế (`MODEL_VALIDATION`). Báo cáo phản ánh trung thực nguồn gốc dữ liệu, cô lập dữ liệu tổng hợp và đánh giá trên tập kiểm thử thực tế TP.HCM độc lập.

---

## 2. Danh mục Sản phẩm Bàn giao Cuối cùng (Delivery Package)

1. **Mã nguồn và Lịch sử Git:**
   - Workspace độc lập [D:\CNTT-KLCN155-waste-detection](file:///D:/CNTT-KLCN155-waste-detection), Git nhánh `main`.
2. **Bộ Trọng số Mô hình Đã Huấn luyện:**
   - Giai đoạn A: `artifacts/run-001/best.pt` (MobileNetV3 10 lớp, khởi tạo ImageNet-1K).
   - Giai đoạn B (Mô hình chính): `artifacts/detection/yolov8n-waste/weights/best.pt` (YOLOv8n).
   - Giai đoạn B (Mô hình đối chứng): `artifacts/detection/ssdlite320-waste/weights/best.pth` (SSDLite320 với 258 keys backbone chuyển giao).
3. **Bộ Tài liệu Kế hoạch và Báo cáo Kiểm toán (15 tệp MD):**
   - Lưu trữ tại [docs/plan/](file:///D:/CNTT-KLCN155-waste-detection/docs/plan), kèm báo cáo tự phản biện [R2_REVIEW.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/R2_REVIEW.md) và biên bản kiểm kê dữ liệu [data/audit/](file:///D:/CNTT-KLCN155-waste-detection/data/audit).
4. **Bộ Dữ liệu Kiểm toán và Phân chia:**
   - Bảng kê 14.831 ảnh sạch, không trùng lặp, chia tập 10.381 Train / 2.225 Val / 2.225 Test có chứng thực Zero-leakage SHA-256.
5. **Giao diện Người dùng và Công cụ Thẩm định:**
   - [streamlit_app.py](file:///D:/CNTT-KLCN155-waste-detection/streamlit_app.py) và [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py).

---

## 3. Lệnh Tái hiện Quy trình từng bước (Reproducibility Commands)

Mọi câu lệnh được thực thi trong PowerShell tại thư mục `D:\CNTT-KLCN155-waste-detection`:

```powershell
# Bước 1: Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Bước 2: Kiểm tra tính toàn vẹn dữ liệu và kiểm toán
python scripts/audit_and_reconcile_data.py

# Bước 3: Kiểm tra tính tương thích layer chuyển giao backbone
python scripts/inspect_backbone_transfer.py

# Bước 4: Khởi chạy công cụ thẩm định nhãn nội bộ (Review Tool)
streamlit run src/ui/review_tool.py

# Bước 5: Khởi chạy ứng dụng Web chính thức
streamlit run streamlit_app.py
```
