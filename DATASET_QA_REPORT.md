# BÁO CÁO KIỂM KÊ VÀ ĐÁNH GIÁ CHẤT LƯỢNG DATASET TRƯỚC TRAINING (TASK 2)
**Sprint:** SPRINT AI-UPGRADE-01 — Cải Thiện Chất Lượng Nhận Diện Rác Thực Tế  
**Dự án:** Hệ thống Phát hiện & Phân loại 10 Nhóm Rác thải Đa Vật thể (CNTT-KLCN155)  
**Vai trò:** AI Engineer (Anti) | Kính gửi PM / Reviewer (ChatGPT)  
**Ngày thực hiện:** 12/10/2026  

---

## I. MỤC TIÊU KIỂM KÊ VÀ NGUYÊN TẮC BẢO TOÀN DỮ LIỆU

Nhằm chuẩn bị cơ sở dữ liệu tin cậy cho đợt fine-tune YOLOv8n, bước kiểm kê này rà soát toàn bộ các phân tập dữ liệu hiện hữu trên ổ đĩa, xác định chính xác số lượng ảnh, số lượng bounding boxes, tỷ lệ mất cân bằng và độ nhiễm chéo dữ liệu (data leakage).

### Nguyên tắc bất biến bắt buộc tuân thủ:
1. **Phân định rõ ràng giữa Train và Holdout:** Toàn bộ dữ liệu nằm trong `data/holdout` (gồm 27 ảnh holdout gốc, 20 ảnh kho và 100 ảnh online mới) là **tập đánh giá độc lập bên ngoài**, **TUYỆT ĐỐI KHÔNG ĐƯA VÀO TẬP HUẤN LUYỆN (TRAINING SET)** dưới bất kỳ hình thức nào.
2. **Không phá hủy dữ liệu thô:** Bảo toàn nguyên trạng các tệp ảnh và nhãn hiện có, không chạy các lệnh ghi đè hoặc xóa sạch dữ liệu.
3. **Bảo tồn Taxonomy 10 lớp chuẩn:**
   `0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash`. Không tự ý thay đổi số lượng lớp hay chỉ số index.

---

## II. KIỂM KÊ CHI TIẾT TỪNG PHÂN TẬP DỮ LIỆU

### 1. Phân tập 1: Dataset v2 Hiện hành (`configs/detection_dataset_real_v2.yaml`)
* **Đường dẫn dữ liệu:** `data/detection/images/real/` (2.143 ảnh thật) và `data/detection/labels/real/` (2.143 tệp nhãn YOLO).
* **Nguồn gốc:** Kết hợp giữa TACO Dataset (Flickr CC) và Open Images V7 (Validation real subset đã lọc sạch). Đã cách ly hoàn toàn dữ liệu giả lập Mendeley Synthetic.
* **Tỷ lệ phân chia:** Train 70% (1.398 ảnh), Val 15% (299 ảnh), Test 15% (299 ảnh). Tổng cộng: 1.996 ảnh sạch đã phân chia.

#### Bảng thống kê chi tiết từng lớp trong Dataset v2:

| Class ID | Tên lớp (Taxonomy) | Train Boxes | Train Imgs | % Train Box | Val Boxes | Val Imgs | Test Boxes | Test Imgs | Tổng Boxes (v2) |
|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | **battery** (Pin / Ắc quy) | **1** | 1 | **0.03%** | **0** | 0 | 1 | 1 | **2** |
| **1** | **biological** (Rác hữu cơ) | **3** | 3 | **0.08%** | **2** | 1 | 1 | 1 | **6** |
| **2** | **cardboard** (Bìa carton) | 169 | 126 | 4.26% | 39 | 31 | 36 | 26 | 244 |
| **3** | **clothes** (Quần áo cũ) | 368 | 212 | 9.29% | 68 | 44 | 91 | 45 | 527 |
| **4** | **glass** (Thủy tinh) | 213 | 101 | 5.37% | 82 | 22 | 32 | 22 | 327 |
| **5** | **metal** (Kim loại / Lon) | 420 | 214 | 10.60% | 74 | 45 | 73 | 43 | 567 |
| **6** | **paper** (Giấy / Báo) | 160 | 123 | 4.04% | 43 | 27 | 34 | 28 | 237 |
| **7** | **plastic** (Nhựa / Chai túi) | **1,560** | 791 | **39.36%** | **382** | 173 | **274** | 161 | **2,216** |
| **8** | **shoes** (Giày dép) | 215 | 117 | 5.43% | 41 | 23 | 39 | 23 | 295 |
| **9** | **trash** (Rác vô cơ khác) | **854** | 294 | **21.55%** | **170** | 64 | **139** | 63 | **1,163** |
| **TỔNG** | *(10 lớp)* | **3,963** | **1,398** | **100.0%** | **901** | **299** | **720** | **299** | **5,584** |

> [!WARNING]
> **Phát hiện nghiêm trọng trong Dataset v2:**
> 1. **Số lượng ảnh nền âm (Hard-negative background images) = ĐÚNG 0 ẢNH!** Toàn bộ 1.398 ảnh train đều có ít nhất 1 bounding box rác. Không có bất kỳ ảnh mặt đường sạch, mảng tường hay tủ điện nào có nhãn rỗng (0 bytes).
> 2. **Lớp Pin (`battery`) bị bỏ đói tuyệt đối:** Chỉ có đúng **1 bounding box** trong tập train và **0 box** trong tập val.
> 3. **Lớp Hữu cơ (`biological`) bị bỏ đói tuyệt đối:** Chỉ có đúng **3 bounding boxes** trong tập train và **2 boxes** trong tập val.
> 4. **Độ lệch mất cân bằng:** `plastic` (1.560 boxes) gấp **1.560 lần** `battery` và gấp **520 lần** `biological`!

---

### 2. Phân tập 2: Dataset v3 Candidate (`configs/detection_dataset_real_v3_candidate.yaml`)
* **Đường dẫn:** `data/detection_v3_candidate/yolo/` và `data/detection/splits_real_v3/`.
* **Thống kê:**
  - Split Train: 1.277 ảnh, 3.052 boxes.
  - Split Val: 269 ảnh, 684 boxes.
  - Split Test: 274 ảnh, 595 boxes.
  - Tổng cộng: 1.820 ảnh, 4.331 boxes.
* **Bản chất kỹ thuật:** Tập `v3_candidate` này được tạo ra trong các pha audit trước nhằm lọc bỏ 176 ảnh bị nghi ngờ nhiễu nhãn từ v2.
* **Đánh giá:** Tập ứng viên này **chưa giải quyết được vấn đề thiếu mẫu**:
  - `battery` trong train: vẫn chỉ có **1 box**!
  - `biological` trong train: vẫn chỉ có **3 boxes**!
  - Ảnh hard-negative (empty): **vẫn là 0 ảnh**!
  $\implies$ **Chưa đạt điều kiện huấn luyện nâng cấp.** Cần một đợt bổ sung ảnh thật thực sự (Task 3).

---

### 3. Phân tập 3: Các ảnh Wikimedia đã tải
* **Ảnh trong staging chuẩn bị nâng cấp (`data/acquisition_staging/phase4e_staging/images/`):**
  - Gồm 4 ảnh có nguồn gốc bản quyền Creative Commons minh bạch từ Wikimedia Commons:
    1. `wiki_bio_banana_peel_ground.jpg` (Vỏ chuối trên mặt đất - Rác hữu cơ `biological`, tác giả Vladiator360, CC BY-SA 4.0).
    2. `wiki_bio_street_food_remains.jpg` (Rác thức ăn thừa đường phố - Rác hữu cơ `biological`, tác giả Andrew Currie, CC BY-SA 3.0).
    3. `wiki_neg_junction_boxes.jpg` (Tủ điện và tường gạch sạch - Mẫu âm `hard_negatives`, tác giả Thomas Nugent, CC BY-SA 2.0).
    4. `wiki_neg_utility_box_leeds.jpg` (Tủ điện đường phố Leeds sạch - Mẫu âm `hard_negatives`, tác giả Mtaylor848, CC BY-SA 4.0).
  - Trạng thái: Đã có ảnh thật, đã kiểm tra pixel, có overlay kiểm chứng. Cần được đưa vào lô bổ sung chính thức sau khi được PM phê duyệt.

---

### 4. Phân tập 4: Toàn bộ ảnh trong thư mục `data/holdout`
* **Tổng số tệp ảnh quét được:** Đúng **147 tệp ảnh** (.jpg, .jpeg, .png, .webp).
* **Phân rã thành 3 nhóm phục vụ kiểm định độc lập:**
  1. **Bộ 27 ảnh External Holdout đóng băng gốc (`reports/qa_acceptance/manifest_hinh_anh_rac.json`):**
     - Chứa 27 ảnh rác thực tế đa dạng, được gán Ground Truth độc lập (57 vật thể rác).
     - Đã đóng băng từ đầu dự án, làm thước đo baseline chuẩn mực.
  2. **Bộ 20 ảnh kho chuyên biệt (`reports/qa_acceptance/manifest_20_anh_kho.json`):**
     - Đặt tên từ `kho_01_don_loai_pin.jpg` đến `kho_20_da_loai_goc_pho.jpg`.
     - Phục vụ kiểm tra định tính từng đơn loại rác và các cảnh rác phức tạp (bờ kênh, tham cỏ, góc phố).
  3. **Bộ 100 ảnh rác thực tế mới (`reports/qa_acceptance/manifest_100_anh_online.json`):**
     - Đặt tên từ `rac_online_001.jpg` đến `rac_online_100.jpg`.
     - Chứa ảnh rác đường phố, bãi rác, vỉa hè đa vật thể được tải về nhằm mục đích làm **External Test Set 2** phục vụ demo so sánh mô hình cũ/mới.
  4. **Thư mục con `demo_multiobject_100_20261012`:**
     - Chứa kết quả chạy demo YOLOv8n trên 100 ảnh, bao gồm ảnh vẽ box (overlays), contact sheets và báo cáo latency.
* **Kết quả quét rò rỉ dữ liệu (Cross-Contamination & Leakage Audit):**
  - Đã tính mã băm SHA-256 của toàn bộ 147 ảnh trong `data/holdout` và đối chiếu từng tệp với 2.143 ảnh trong `data/detection/images/real`.
  - **Kết quả:** **0 tệp trùng lặp SHA-256 (0 byte collisions).**
  - Toàn bộ 147 ảnh này hoàn toàn độc lập với tập huấn luyện.
  - **Kết luận:** Giữ nguyên trạng thái **ĐÓNG BĂNG NGOẠI KIỂM** cho toàn bộ 147 ảnh này. Tuyệt đối không lấy ảnh từ `data/holdout` đưa vào tập train.

---

## III. ĐÁNH GIÁ 8 TIÊU CHÍ CHẤT LƯỢNG DATASET

### 1. Số lượng ảnh thật mỗi lớp
- Hiện tại các lớp `plastic`, `trash`, `metal`, `clothes` có số lượng ảnh dồi dào (> 200–700 ảnh).
- Hai lớp `battery` (1 ảnh) và `biological` (3 ảnh) thiếu hụt nghiêm trọng, không thể đáp ứng yêu cầu huấn luyện một detector 10 lớp.

### 2. Số lượng bounding box mỗi lớp
- Cần bổ sung để đạt tối thiểu:
  * `battery`: $\ge 50 - 70$ boxes thật.
  * `biological`: $\ge 40 - 60$ boxes thật.
- Các lớp còn lại (`plastic`, `metal`, `paper`, `cardboard`, `glass`, `shoes`) đã có từ 200 đến 1.500 boxes, đủ để học các đặc trưng cơ bản.

### 3. Tỷ lệ mất cân bằng (Class Imbalance)
- Hiện tại tỷ lệ mất cân bằng cao nhất là $1.560 : 1$.
- Trong đợt fine-tune tới, cần áp dụng focal loss hoặc loss weighting trong hàm mất mát phân loại để giảm sức nặng của lớp `plastic` và nâng trọng số phạt cho các lớp hiếm.

### 4. Ảnh trùng lặp và gần trùng lặp (Near-Duplicates)
- Báo cáo pHash (`data/audit/leakage_audit_report.json`) tại ngưỡng khoảng cách Hamming $\le 4$ đã kiểm tra các cặp ảnh nghi vấn.
- Toàn bộ các cặp nghi vấn (như can kim loại vs hộp sữa giấy trên nền trắng) đã được kiểm chứng trực quan bằng mắt người và xác nhận là các vật thể vật lý riêng biệt (visual coincidences), không phải ảnh biến thể (augmented duplicates).

### 5. Chất lượng Bounding Box (Box Geometry & Tightness)
- Trong tập TACO: Một số box bị kéo rộng do polygon bao gồm cả bóng đổ. Đã được chuẩn hóa ở các pha trước.
- Cần loại bỏ các box chiếm $> 95\%$ diện tích ảnh trên các ảnh không phải đại cảnh để tránh sinh box ảo toàn khung hình.

### 6. Độ chính xác của ánh xạ nhãn nguồn (Source Label Mapping)
- Tệp `configs/detection_source_mapping.yaml` đã xác lập quy tắc ánh xạ rõ ràng:
  * Đã khóa vĩnh viễn việc map toàn bộ lớp từ Mendeley Synthetic.
  * Trong Open Images V7, đã loại trừ người mặc quần áo (`exclude_in_use_person_clothing: true`).
  * Các loại hộp phức hợp (nhựa pha carton, túi giấy ép nilon) được đánh dấu `REQUIRES_REVIEW` hoặc loại bỏ để không gây mâu thuẫn nhãn.

### 7. Bối cảnh camera cố định và camera di động
- Dữ liệu hiện tại chủ yếu là ảnh chụp từ điện thoại di động (Mobile Camera) ở cự ly gần ($0.5\text{m} - 2\text{m}$).
- Thiếu các góc nhìn từ trên cao (Camera giám sát cố định / CCTV) ở cự ly trung bình ($3\text{m} - 8\text{m}$).
- Đợt bổ sung dữ liệu mới cần kết hợp cả hai góc chụp để phục vụ lộ trình triển khai tương lai.

### 8. Ranh giới giữa "Vật thể bình thường" và "Rác thải"
- Đây là nguyên nhân gốc rễ gây ra dương tính giả trên vỉa hè, tường gạch và tủ điện.
- Bắt buộc phải bổ sung tối thiểu **30–40 ảnh Hard-Negative** (tủ điện, mảng tường, vỉa hè sạch không rác) với tệp nhãn `.txt` rỗng (0 bytes). Khi mô hình nhìn thấy các vật thể này mà không có bounding box, hàm loss sẽ phạt nặng mọi proposal sinh ra trên chúng, giúp mô hình học cách **ức chế ngoại cảnh đô thị**.

---

## IV. BẢNG TỔNG KẾT KIỂM KÊ VÀ TRẠNG THÁI TỪNG PHÂN TẬP

| Phân tập dữ liệu | Vị trí lưu trữ | Số ảnh | Số Bounding Boxes | Tình trạng kiểm định | Quyền sử dụng trong Huấn luyện |
|---|---|---:|---:|:---:|:---:|
| **Dataset v2 Train** | `data/detection/images/real/` | 1.398 | 3.963 | Đã kiểm định sạch | **ĐƯỢC PHÉP (Baseline Data)** |
| **Dataset v2 Val** | `data/detection/images/real/` | 299 | 901 | Đã kiểm định sạch | **ĐƯỢC PHÉP (Validation)** |
| **Dataset v2 Test** | `data/detection/images/real/` | 299 | 720 | Đã kiểm định sạch | **ĐƯỢC PHÉP (Internal Test)** |
| **Dataset v3 Candidate** | `data/detection_v3_candidate/` | 1.820 | 4.331 | Đã lọc 176 ảnh nghi vấn | **TẠM DỪNG (Vẫn thiếu Battery/Bio)** |
| **Wikimedia Staged (4 ảnh)** | `data/acquisition_staging/` | 4 | 2 Bio + 2 Neg | Đã kiểm định pixel/license | **CHỜ DUYỆT (Task 3 Batch)** |
| **Holdout 27 ảnh gốc** | `data/holdout/` | 27 | 57 GT | Đã đóng băng tuyệt đối | **TUYỆT ĐỐI CẤM (External Test 1)** |
| **Kho 20 ảnh chuyên biệt** | `data/holdout/` | 20 | Đơn loại / Đa cảnh | Đã đóng băng | **TUYỆT ĐỐI CẤM (Audit Set)** |
| **Bộ 100 ảnh online mới** | `data/holdout/` | 100 | Rác online đa vật thể | Đã đóng băng | **TUYỆT ĐỐI CẤM (External Test 2)** |

---

## V. KẾT LUẬN CHO TASK 3

Dữ liệu huấn luyện hiện tại (`splits_real_v2`) hoàn toàn sạch về mặt rò rỉ dữ liệu, nhưng **bị khiếm khuyết nghiêm trọng về cấu trúc**:
1. Thiếu hẳn ảnh âm Hard-Negative ($0\%$).
2. Thiếu hụt trầm trọng hai lớp `battery` (1 box) và `biological` (3 boxes).

**Khuyến nghị:** Bước vào Task 3, cần thiết lập danh mục nguồn thu thập (Source Whitelist) đạt chuẩn 150–250 ảnh mới, tập trung giải quyết triệt để 2 lớp thiếu và ảnh âm ngoại cảnh, trình PM phê duyệt trước khi tải và dán nhãn.
