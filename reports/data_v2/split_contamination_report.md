# BÁO CÁO ĐỐI CHIẾU SPLIT & KIỂM SOÁT DATA LEAKAGE (PHASE 4C)
**Thời điểm kiểm toán:** 2026-10-12T02:00:00Z  
**Chính sách bất biến:** TUYỆT ĐỐI GIỮ PILOT Ở TRẠNG THÁI REVIEW. KHÔNG TỰ ĐỘNG MERGE VÀO TRAIN.  

---

## 1. BẢNG PHÂN LOẠI 15 ẢNH PILOT THEO NGUỒN GỐC VÀ SPLIT

| STT | Tên ảnh | Phân loại trạng thái | Split v2 | Split v3 | Thuộc Holdout? | Phán quyết an toàn (QA Ruling) |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 01 | `taco_0072.jpg` | **QUARANTINED** | `NONE` | `NONE` | FALSE | PENDING_APPROVAL: Excluded from official splits. Re-annotated in pilot; requires user approval before train ingestion. |
| 02 | `taco_0092.jpg` | **QUARANTINED** | `NONE` | `NONE` | FALSE | PENDING_APPROVAL: Excluded from official splits. Re-annotated in pilot; requires user approval before train ingestion. |
| 03 | `taco_0068.jpg` | **EXISTING_TEST** | `test` | `test` | FALSE | CRITICAL_RISK: Image already belongs to TEST split. Merging to TRAIN would cause test-set contamination! |
| 04 | `taco_0841.jpg` | **EXISTING_TRAIN** | `train` | `train` | FALSE | REDUNDANT: Image is already part of official train split. Does not add new data. |
| 05 | `taco_0924.jpg` | **EXISTING_TRAIN** | `train` | `train` | FALSE | REDUNDANT: Image is already part of official train split. Does not add new data. |
| 06 | `taco_0456.jpg` | **EXISTING_TRAIN** | `train` | `train` | FALSE | REDUNDANT: Image is already part of official train split. Does not add new data. |
| 07 | `taco_0082.jpg` | **EXISTING_TEST** | `test` | `test` | FALSE | CRITICAL_RISK: Image already belongs to TEST split. Merging to TRAIN would cause test-set contamination! |
| 08 | `taco_0170.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 09 | `taco_0431.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 10 | `taco_0556.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 11 | `taco_0909.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 12 | `taco_0912.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 13 | `taco_0916.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 14 | `taco_0917.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |
| 15 | `taco_0925.jpg` | **EMPTY_LABEL_UNMAPPED_WASTE** | `NONE` | `NONE` | FALSE | REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste. |

---

## 2. PHÂN TÍCH RỦI RO CONTAMINATION NẾU TỰ Ý MERGE VÀO TRAIN

### A. Nguy cơ rò rỉ tập Test (Test Contamination):
- `taco_0068.jpg` (chứa 1 biological, 1 metal) và `taco_0082.jpg` (chứa 1 battery, 2 metal, 4 plastic) **ĐANG THUỘC TẬP TEST CHÍNH THỨC** trong cả v2 và v3 candidate!
- **HẬU QUẢ NẾU MERGE:** Nếu vô tình đưa 2 ảnh này vào train split chỉ vì chúng có mặt trong Pilot dataset, mô hình sẽ học thuộc ảnh của tập Test, gây ra **rò rỉ dữ liệu kiểm thử (Test Data Leakage)** nghiêm trọng và làm sai lệch chỉ số đánh giá độc lập.
- **PHÁN QUYẾT:** `taco_0068.jpg` và `taco_0082.jpg` **BẮT BUỘC PHẢI GIỮ NGUYÊN Ở TẬP TEST**, tuyệt đối không được chuyển sang train.

### B. Ảnh đã có sẵn trong tập Train (Redundant Train):
- `taco_0456.jpg` (1 battery), `taco_0841.jpg` (1 biological), `taco_0924.jpg` (1 biological) **VỐN ĐÃ NẰM SẴN TRONG TẬP TRAIN V2 VÀ V3**.
- Việc đưa chúng vào Pilot chỉ để làm mẫu tham chiếu kiểm thử loader/gates; chúng không bổ sung thêm bất kỳ ảnh train mới nào.

### C. 8 ảnh 'Hard-Negative' bị từ chối:
- 8 ảnh có nhãn 0-byte (`taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`) thực chất chứa rác thải chưa gán nhãn (Category 51: Rope & strings).
- Nếu nạp chúng vào train dưới dạng negative (0-object), mô hình sẽ bị dạy sai rằng dây thừng phế thải là background.

### D. Ảnh duy nhất có thể bổ sung hợp lệ vào Train sau khi được duyệt:
- Chỉ có **2 ảnh cách ly** được đề xuất sửa tọa độ: `taco_0072.jpg` (+1 biological, +1 plastic) và `taco_0092.jpg` (+1 biological).
- 2 ảnh này không thuộc train/val/test hiện tại và không trùng holdout, có thể nạp vào Train **sau khi người dùng phê duyệt**.
