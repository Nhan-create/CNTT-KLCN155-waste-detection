# BÁO CÁO MỞ RỘNG DỮ LIỆU RÁC THỰC TẾ CHO 8 LỚP HIẾM — TASK 8F (ĐÃ HIỆU CHỈNH SEMANTIC QA)

- **Repository:** `.`
- **Thư mục dữ liệu Wave 8F:** `data/acquisition_staging/task8f_scarce_classes/`
- **Manifest QA (Post-Correction):** [`task8f_qa_manifest.json`](../../data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.json) & [`.csv`](../../data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.csv)
- **Manifest QA (Pre-Correction Snapshot):** [`task8f_qa_manifest_pre_correction.json`](../../data/acquisition_staging/task8f_scarce_classes/qa_correction_before_after/task8f_qa_manifest_pre_correction.json)
- **Trạng thái phê duyệt huấn luyện:** `APPROVED_FOR_TRAIN = False` (toàn bộ ảnh mới giữ ở staging chờ PM nghiệm thu độc lập).

---

## 1. ĐỐI CHIẾU TRƯỚC VÀ SAU KHI HIỆU CHỈNH SEMANTIC QA (TASK 8F QUALITY CORRECTION)

| Chỉ số | Trước hiệu chỉnh (`V1`) | Sau hiệu chỉnh (`V2`) | Thay đổi |
|---|---:|---:|---|
| Tổng ứng viên tải về (đã khử trùng) | `748` | **`748`** | Giữ nguyên 748 ứng viên gốc |
| `REAL_WASTE_PASS` (Tổng ảnh) | `203` | **`86`** | `-117` ảnh (chuyển `69` sang `KEEP_AUXILIARY`, `41` sang `NEEDS_REVIEW`, `7` sang `REJECT`) |
| ├── `TRAIN_REAL_WASTE_W8F` | `161` | **`71`** | `-90` ảnh train |
| └── `VAL_REAL_WASTE_W8F` | `42` | **`15`** | `-27` ảnh val |
| Tổng Bounding Boxes (`REAL_WASTE_PASS`) | `407` | **`120`** | `-287` boxes (khử trùng và loại bỏ ảnh sai ngữ cảnh) |
| Cặp bbox cùng lớp trùng lặp (`IoU > 0.70`) | `33` cặp (`23` ảnh) | **`0` cặp (`0` ảnh)** | Đã xử lý triệt để 100% |
| `KEEP_AUXILIARY` | `310` | **`379`** | `+69` ảnh đúng vật thể nhưng chưa có bằng chứng vứt bỏ (`IN_USE` / `PRODUCT_STUDIO`) |
| `NEEDS_REVIEW` | `85` | **`126`** | `+41` ảnh đống rác dày / bao bì phức hợp / cần xác nhận thủ công |
| `REJECT` | `150` | **`157`** | `+7` ảnh không thuộc taxonomy / ảnh người trưng bày thực phẩm (`0690`, ...) |

---

## 2. KẾT QUẢ THU THẬP & KIỂM ĐỊNH QA WAVE 8F THEO NGUỒN (SAU HIỆU CHỈNH)

| Mã Nguồn | Đã tải (Khử trùng) | `REAL_WASTE_PASS` (Tổng) | `TRAIN_REAL_WASTE_W8F` | `VAL_REAL_WASTE_W8F` | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` | Hiệu suất `REAL_WASTE_PASS` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SRC_8F_02_WIKIMEDIA_SCARCE_REAL_WASTE` | 481 | **86** | **71** | **15** | 146 | 94 | 155 | **17.88%** |
| `SRC_8F_03_OPENIMAGES_SCARCE_CLASSES_TRAIN_VAL` | 267 | **0** | **0** | **0** | 233 | 32 | 2 | **0.0%** |
| **TỔNG WAVE 8F** | **748** | **86** | **71** | **15** | **379** | **126** | **157** | **11.5%** |

---

## 3. PHÂN BỐ `REAL_WASTE_PASS` CỦA TASK 8F THEO 10 LỚP (SAU HIỆU CHỈNH)

| Class ID | Tên lớp | Nhóm ưu tiên | Ảnh `REAL_WASTE_PASS` (Tổng) | Ảnh Train (`TRAIN_REAL_WASTE_W8F`) | Ảnh Val (`VAL_REAL_WASTE_W8F`) | Tổng Boxes (`REAL_WASTE_PASS`) | Boxes Val (`VAL_REAL_WASTE_W8F`) |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `battery` | 8 LỚP HIẾM TRỌNG TÂM | **8** | **7** | **1** | **16** | **1** |
| 1 | `biological` | 8 LỚP HIẾM TRỌNG TÂM | **11** | **8** | **3** | **18** | **3** |
| 2 | `cardboard` | 8 LỚP HIẾM TRỌNG TÂM | **4** | **3** | **1** | **4** | **1** |
| 3 | `clothes` | 8 LỚP HIẾM TRỌNG TÂM | **4** | **3** | **1** | **5** | **2** |
| 4 | `glass` | 8 LỚP HIẾM TRỌNG TÂM | **7** | **5** | **2** | **7** | **2** |
| 5 | `metal` | 8 LỚP HIẾM TRỌNG TÂM | **17** | **13** | **4** | **19** | **6** |
| 6 | `paper` | 8 LỚP HIẾM TRỌNG TÂM | **13** | **12** | **1** | **17** | **1** |
| 7 | `plastic` | Gán nhãn đúng chất liệu thực tế | **9** | **8** | **1** | **9** | **1** |
| 8 | `shoes` | 8 LỚP HIẾM TRỌNG TÂM | **13** | **12** | **1** | **24** | **2** |
| 9 | `trash` | Gán nhãn đúng chất liệu thực tế | **1** | **1** | **0** | **1** | **0** |
