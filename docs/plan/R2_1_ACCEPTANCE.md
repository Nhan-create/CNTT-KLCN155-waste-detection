# BIÊN BẢN NGHIỆM THU KỸ THUẬT TASK R2.1 (TECHNICAL ACCEPTANCE)

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Chủ dự án (PM):** ThS. Ngô Thanh Nhân  
**Tech Lead ML:** Antigravity Engineering Team  
**Ngày nghiệm thu:** 01/10/2026  
**Trạng thái nghiệm thu:** `CONDITIONALLY_ACCEPTED_FOR_SMOKE_TEST` | `DATA_GATE_FLAGGED_WITH_LEAKAGE`

---

## 1. Bảng Đối chiếu Yêu cầu Nghiệm thu Task R2.1

| Yêu cầu của PM trong TASK R2.1 | Kết quả thực hiện & Bằng chứng thực tế | Đánh giá |
|:---|:---|:---:|
| **1. Workspace mới độc lập** | Thư mục vật lý độc lập `D:\CNTT-KLCN155-waste-detection`, Git root riêng biệt, virtualenv `.venv` riêng biệt, log và artifacts ghi cục bộ. Repo cũ tại Downloads được giữ nguyên 100%. | **ĐẠT (VERIFIED)** |
| **2. Báo cáo bằng chứng thực tế cho 12 câu hỏi** | Trả lời đầy đủ 12 câu hỏi kèm trích dẫn file code, đường dẫn dữ liệu, trạng thái VERIFIED/CONFLICT và phân tích tác động. | **ĐẠT (VERIFIED)** |
| **3. Sửa lỗi kiểm toán dữ liệu và script** | Đã sửa bug gán cứng `cross_source_duplicates = 0` trong `scripts/audit_and_reconcile_data.py`; xác nhận 890 cặp trùng liên nguồn (879 cặp trùng exact MD5). | **ĐẠT (VERIFIED)** |
| **4. Tái hiện và kiểm định 18 cặp pHash $\le 4$** | Tái hiện chính xác 100% 18 cặp ứng viên; kiểm định trực quan phát hiện 15 cặp rò rỉ dữ liệu thật (10 burst-shot, 5 xoay góc) và 3 cặp trùng ngẫu nhiên khác lớp. Xuất ảnh plate trực quan tại `data/audit/visual_phash_inspection/`. | **ĐẠT (VERIFIED)** |
| **5. Đính chính nhận định VN Trash** | Bác bỏ hoàn toàn việc coi VN Trash là external test set; chỉ rõ 1.801 ảnh VN Trash nằm trong Train và 879 ảnh trùng với Garbage V2. | **ĐẠT (VERIFIED)** |
| **6. Loại bỏ vi phạm quy trình Test Set** | Đã xóa triệt để luồng phản hồi từ Test set về Train. Test set được đóng băng và chỉ mở đúng 1 lần sau khi mô hình đã hoàn tất. | **ĐẠT (VERIFIED)** |
| **7. Kiểm toán chuyển giao Backbone sang SSDLite** | Xuất bảng ánh xạ 258/308 keys khớp (83,8%); thực nghiệm nạp trọng số và chạy forward pass thành công trên tensor `(1, 3, 320, 320)`. Khẳng định rõ tỷ lệ khớp keys không chứng minh được mAP của detector. | **ĐẠT (VERIFIED)** |
| **8. Kích hoạt GPU và PyTorch CUDA** | Cài đặt thành công `torch==2.5.1+cu121` trên card NVIDIA GeForce RTX 2050 Laptop GPU (4.095,5 MB VRAM). | **ĐẠT (VERIFIED)** |
| **9. Chạy thử nghiệm Smoke Test (3–5 Epochs)** | Chạy thành công 3 epochs trên Train/Val bằng GPU RTX 2050 (AMP fp16, đỉnh VRAM 1.511,2 MB); Train loss giảm từ 0.8835 xuống 0.5889; Val Acc đạt 94,56%, Val Macro-F1 đạt 0.9430; checkpoint lưu 48,58 MB; kiểm tra reload test sai lệch bằng 0.0. Test set hoàn toàn không đụng tới. | **ĐẠT (VERIFIED)** |
| **10. Đóng gói bàn giao đầy đủ mã nguồn** | Đóng gói toàn bộ `docs/plan/`, `scripts/`, `data/audit/`, `artifacts/smoke_test/` vào tệp ZIP duy nhất kèm mã băm SHA-256. | **ĐẠT (VERIFIED)** |

---

## 2. Kết luận và Điều kiện Chuyển tiếp

1. **Về mặt Kỹ thuật & Hạ tầng (Code & Environment):** ĐÃ HOÀN TOÀN SẴN SÀNG. GPU RTX 2050 hoạt động ổn định, pipeline huấn luyện, kiểm thử, lưu trữ và nạp lại mô hình chạy thông suốt, không có lỗi cấu hình hay bộ nhớ.
2. **Về mặt Dữ liệu (Data Gate):** ĐẶT CẢNH BÁO `DATA_GATE_FLAGGED_WITH_LEAKAGE`. Tập dữ liệu vật lý cũ `Data/processed/` còn 15 cặp burst-shot xuyên split.
3. **Điều kiện để chạy Huấn luyện Chính thức (Official Benchmark Training):**
   - Chạy script xử lý gom cụm 15 cặp burst-shot về cùng split Train hoặc loại bỏ khỏi Val/Test.
   - Sau khi Cổng Dữ liệu xác nhận `ZERO_LEAKAGE`, tiến hành chạy huấn luyện đầy đủ 30 epochs (Phase 1 + Phase 2) để nộp khóa luận.
