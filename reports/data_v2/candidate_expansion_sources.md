# KẾ HOẠCH MỞ RỘNG DỮ LIỆU THẬT & KIỂM CHỨNG NGUỒN DỮ LIỆU (PHASE 4C)
**Phiên bản:** 3.0-audited-reality  
**Thời điểm cập nhật:** 2026-10-12T02:00:00Z  
**Chính sách tuân thủ:** TUYỆT ĐỐI KHÔNG TỰ Ý DOWNLOAD DỮ LIỆU NGOÀI LỚN KHI CHƯA ĐƯỢC PHÊ DUYỆT.  

---

## 1. KẾT QUẢ KIỂM CHỨNG NGUỒN TACO COCO RAW (P0 DISCOVERY)

Trong Phase 4C, nhóm kỹ thuật đã trực tiếp phân tích cấu trúc nhãn thô từ file COCO chuẩn `data/audit/taco_annotations_raw.json` (tổng cộng 1.500 ảnh, 4.784 annotations, 60 categories từ ID 0 đến 59):

### Sai lệch nghiêm trọng của các giả định trước đây:
1. **Category 60 (Batteries) và Category 63/64 (Organic/Food waste) KHÔNG HỀ TỒN TẠI** trong dataset TACO COCO chính thức. TACO chỉ có tối đa 60 categories (đánh số 0 đến 59).
2. **Thực tế về Pin (Battery - Category ID 1):** Toàn bộ dataset TACO chỉ có đúng **2 annotations** trên **2 ảnh duy nhất** (`taco_0456.jpg` trong Train và `taco_0082.jpg` trong Test). Cả 2 ảnh này **đều đã có sẵn trong repository CNTT-KLCN155** từ trước (`taco_0499` thực chất là hộp nhựa dùng một lần, nhầm lẫn do lỗi gõ phím ở bản thảo cũ)!
3. **Thực tế về Rác hữu cơ (Food waste - Category ID 25):** Toàn bộ dataset TACO chỉ có đúng **8 annotations** trên **7 ảnh duy nhất** (`taco_0072.jpg`, `taco_0092.jpg`, `taco_0109.jpg`, `taco_0423.jpg`, `taco_0472.jpg`, `taco_0485.jpg`, `taco_0493.jpg`). Toàn bộ 7 ảnh này **đều đã có trong repository**, trong đó 2 ảnh (`taco_0072`, `taco_0092`) đang nằm trong diện cách ly để re-annotate.
4. **Kết luận bất biến:** Ý tưởng "tải thêm 40 ảnh battery và 50 ảnh biological từ TACO" là **HOÀN TOÀN BẤT KHẢ THI (INFEASIBLE)** vì TACO đã cạn kiệt (0 ảnh còn lại chưa khai thác).

---

## 2. NGUỒN THAY THẾ KHẢ THI THỰC SỰ CHO BATTERY & BIOLOGICAL

| Lớp mục tiêu | Nguồn thay thế khả thi thực tế | Đặc điểm dữ liệu & Bounding Box | Giấy phép | Kế hoạch hành động |
|:---:|---|---|:---:|---|
| **`biological`** | **1. Re-annotate 2 ảnh TACO cách ly** (`taco_0072`, `taco_0092`) | Bounding box lõi táo và vỏ trái cây thật trên mặt đường (0 byte tải ngoài). | CC BY 4.0 | Chỉnh lại viền nhãn QA để phục hồi ngay 2 instance chất lượng cao. |
| **`biological`** | **2. Open Images V7 Validation Pool** (`D:\waste-training\openimages-validation-boxes.csv`) | 3.344 bounding box thật cho Fruit (`/m/02xwb`), Vegetable (`/m/0f4s2w`), Apple (`/m/014j1m`), Banana (`/m/09qck`). | CC BY 4.0 | Lọc theo ngữ cảnh rơi vãi (vỏ chuối, quả hỏng rơi trên sàn/đất), loại bỏ bàn ăn/siêu thị sạch. |
| **`battery`** | **1. Chụp thực tế có chủ đích tại điểm thu gom pin ở TP.HCM** | Ảnh chụp pin AA/AAA, pin cúc áo, pin lithium phồng rộp trên nền bê tông/khay thu gom. | CC0 / Nội bộ | Thu thập 40–60 ảnh với 2-stage QA signoff, bảo đảm 100% không trùng lặp và không rò rỉ. |
| **`battery`** | **2. Roboflow Universe CC BY 4.0 Battery Waste Subsets** | Bounding box format YOLO cho pin thải thực tế trong bối cảnh phân loại rác điện tử. | CC BY 4.0 | Rà soát loại bỏ ảnh render 3D/studio, giữ lại ảnh pin thải thực tế. |
| **`hard_negative`**| **Tự chụp tủ điện xanh, vỉa hè, tường gạch, thùng rác sạch** | File text rỗng 0-byte (chuẩn âm bản của object detector). | CC0 / Nội bộ | Cung cấp 75–100 ảnh để triệt tiêu false positives (đặc biệt lỗi tủ điện ở `rac_online_012.jpg`). |

---

## 3. LỘ TRÌNH MỞ RỘNG DATASET AN TOÀN

1. **Giai đoạn P0 (Nội bộ, 0 MB tải):**
   - Re-annotate `taco_0072.jpg` và `taco_0092.jpg` bổ sung vào pilot dataset.
   - Chuẩn bị 10 ảnh hard-negative mẫu (tủ điện, vỉa hè, tường sạch) với nhãn 0-byte.
2. **Giai đoạn P1 (Cần Stakeholder phê duyệt):**
   - Triển khai chụp 50 ảnh pin thải thực tế tại các trạm thu hồi pin cũ (HCMC).
   - Chọn lọc và trích xuất 60 ảnh rác hữu cơ ngoài trời từ pool Open Images đã lập chỉ mục cục bộ.
