# BÁO CÁO THỐNG KÊ TÍNH MỚI VÀ HẠCH TOÁN DỮ LIỆU PILOT (DATA NOVELTY REPORT)
**Thời điểm kiểm toán:** 2026-10-12T02:00:00Z  
**Nguyên tắc:** Báo cáo trung thực, tuyệt đối không thổi phồng tính mới. Phân biệt rõ giữa tái cấu trúc và dữ liệu mới thực tế.  

---

## 1. BẢNG TỔNG HỢP HẠCH TOÁN DỮ LIỆU PILOT

| Chỉ số hạch toán | Giá trị | Tỷ lệ (%) | Giải thích kỹ thuật |
|---|:---:|:---:|---|
| **Tổng số ảnh trong Pilot** | **15 ảnh** | 100.0% | Tập hợp khảo sát kỹ thuật trong `data/detection_pilot_v3/` |
| **Số ảnh thực sự mới từ ngoài (Genuinely New)** | **0 ảnh** | 0.0% | Không có ảnh nào tải mới từ internet hay chụp mới trong Phase 4C |
| **Số ảnh tái sử dụng từ repo cũ (Reused)** | **15 ảnh** | 100.0% | Toàn bộ 15 ảnh đều trích xuất từ kho ảnh TACO sẵn có trên đĩa |
| **Số box hoàn toàn mới (New from scratch)** | **0 box** | 0.0% | Không tạo box mới trên ảnh chưa có nhãn |
| **Số box chỉ được sửa tọa độ (Refined Coordinates)** | **3 boxes** | 15.8% | 2 box trên `taco_0072` (1 bio, 1 plastic) và 1 box trên `taco_0092` (1 bio) |
| **Số box giữ nguyên tọa độ gốc** | **16 boxes** | 84.2% | Lấy nguyên trạng từ nhãn gốc để kiểm thử loader |
| **Số ảnh CÓ THỂ THÊM VÀO TRAIN hợp lệ** | **2 ảnh** | 13.3% | Chỉ duy nhất `taco_0072` và `taco_0092` (sau khi người dùng duyệt) |

---

## 2. CHI TIẾT 15 ẢNH PILOT VÀ KHẢ NĂNG NẠP VÀO TRAIN

1. **3 ảnh ĐÃ NẰM SẴN TRONG TẬP TRAIN V2/V3 (0 ảnh mới):**
   - `taco_0456.jpg` (1 battery)
   - `taco_0841.jpg` (1 biological)
   - `taco_0924.jpg` (1 biological)
   - *Kết luận:* Đã ở trong tập Train từ trước, không làm tăng kích thước tập train.

2. **2 ảnh ĐANG THUỘC TẬP TEST V2/V3 (BẢO VỆ CHỐNG RÒ RỈ):**
   - `taco_0068.jpg` (1 biological, 1 metal)
   - `taco_0082.jpg` (1 battery, 2 metal, 4 plastic)
   - *Cảnh báo nguy hiểm:* Nếu merge vào train sẽ gây **Test Contamination / Data Leakage**. Bắt buộc giữ nguyên ở Test.

3. **8 ảnh 'Negative' BỊ TỪ CHỐI (0 ảnh được nạp):**
   - `taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`
   - *Nguyên nhân:* Có nhãn 0-byte vì lớp `Rope & strings` (Category 51) chưa được gán map_id. Ảnh thực chất có rác, không được coi là negative.

4. **2 ảnh cách ly ĐƯỢC ĐỀ XUẤT NẠP VÀO TRAIN (Tăng thực tế: +2 ảnh):**
   - `taco_0072.jpg` (+1 biological, +1 plastic)
   - `taco_0092.jpg` (+1 biological)
   - *Trạng thái:* **PROPOSED_PENDING_USER_APPROVAL**.
   - *Kết luận:* Sau khi người dùng duyệt, tập Train sẽ tăng thực tế **chính xác 2 ảnh và 2 box biological** mà không gây bất kỳ rò rỉ dữ liệu nào.
