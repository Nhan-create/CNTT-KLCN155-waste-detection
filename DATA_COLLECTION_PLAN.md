# KẾ HOẠCH THU THẬP VÀ XỬ LÝ DỮ LIỆU THỰC TẾ (DATA COLLECTION PLAN)
**Lộ trình Thu thập Dữ liệu Nhận diện Rác 10 Lớp Ngoài Thực địa**  
**Dự án:** `CNTT-KLCN155-waste-detection`  
**Giai đoạn:** Sprint 0 — Quy hoạch Tập dữ liệu & Kế hoạch Thực thi Đợt 1  
**Tác giả:** Antigravity Integrator (Multi-Agent Team — Chuyên trách: Agent A Data & Agent D QA)  
**Ngày lập:** 2026-10-12  

---

## 1. MỤC TIÊU VÀ QUY MÔ DỮ LIỆU DÀI HẠN

### 1.1 Khảo sát thiếu hụt thực tế (Gap Analysis)
Kết quả đánh giá trên tập kiểm thử độc lập ngoài thực tế (`D:\Hinh_Anh_Rac`, 27 ảnh, 57 vật thể) từ [PRACTICAL_METRICS_REPORT.md](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/PRACTICAL_METRICS_REPORT.md) đã chỉ ra các điểm yếu cốt tử của mô hình hiện tại:
- **Pin / Ắc quy (Battery):** Recall = 0.00% (0/2 vật thể phát hiện được trên tập holdout). Số lượng mẫu train hiện có trong `splits_real_v2` quá ít và chủ yếu là ảnh đồ họa/studio, thiếu pin bị gỉ, pin cúc áo nằm trên đất.
- **Rác sinh học / Hữu cơ (Biological):** Recall = 0.00% (0/3 vật thể phát hiện được). Mô hình chưa được học vỏ chuối nát, thức ăn thừa mục rữa trên mặt đường thật.
- **Dương tính giả ngoại cảnh (False Positives):** Trên các cảnh đường phố, gạch lát, hộp kỹ thuật điện, mô hình nhận diện nhầm thiết bị công cộng thành `cardboard` hoặc `metal`.

### 1.2 Mục tiêu quy mô dài hạn: 1.900 – 2.800 ảnh mới độc lập
Tập dữ liệu mới được mở rộng có giấy phép rõ ràng, phân bổ đồng đều giữa bối cảnh **Camera cố định** (nhìn từ trên xuống, góc cố định thùng rác/băng chuyền) và **Camera di động** (góc nghiêng, rung lắc, camera điện thoại):

| Nhóm mục tiêu | Quy mô dài hạn (Ảnh) | Phân bổ Camera Cố định | Phân bổ Camera Di động | Trọng tâm chất lượng |
|---|:---:|:---:|:---:|---|
| **Pin / Ắc quy (Battery)** | 350 – 500 | ~175 | ~225 | Pin tiểu AA/AAA, pin vuông 9V, pin lithium phồng, ắc quy cũ, pin vứt lẫn trong rác. |
| **Rác sinh học (Biological)** | 350 – 500 | ~200 | ~200 | Vỏ trái cây (chuối, cam, dưa), rau củ dập nát, bã cà phê, thức ăn thừa thực địa. |
| **8 Lớp rác còn lại** | 600 – 900 | ~350 | ~400 | Bìa carton ẩm, lon bia bẹp, chai nhựa trong suốt, túi nilon rách, vải vụn, giày cũ. |
| **Cảnh âm tính khó (Hard-Negative)** | 250 – 400 | ~150 | ~150 | Tủ điện lề đường, nắp cống kim loại, mặt đường bê tông sạch, sàn nhà, thùng rác rỗng. |
| **Đa rác phức tạp (Multi-Waste)** | 350 – 500 | ~200 | ~200 | Đống rác hỗn hợp chứa từ 3 đến 8 loại rác cùng xuất hiện trong một khung hình. |
| **TỔNG CỘNG** | **1.900 – 2.800** | **~1.075** | **~1.175** | **Đảm bảo cân bằng bối cảnh vận hành thực tế.** |

*Quy tắc đếm:* Mỗi ảnh chỉ tính một lần vào nhóm mục tiêu chính. Riêng ảnh Multi-Waste phải được gắn nhãn đầy đủ bounding box cho mọi vật thể rác thuộc 10 lớp xuất hiện trong ảnh.

---

## 2. QUY TRÌNH QUẢN LÝ DỮ LIỆU CHẶT CHẼ (6 CẤP ĐỘ LIFECYCLE)

Để ngăn ngừa triệt để dữ liệu rác, nhãn sai và rò rỉ tập kiểm thử, mọi ảnh thu thập phải tuân thủ nghiêm ngặt 6 trạng thái:

```
[1. SOURCE_IDENTIFIED]
       |  (Kiểm tra giấy phép bản quyền CC BY/CC0/Public Domain & Danh mục URL whitelist)
       v
[2. DOWNLOADED]
       |  (Tải về theo ngân sách dung lượng, tính mã băm SHA-256 và pHash)
       v
[3. VISUALLY_REVIEWED]
       |  (Soi pixel 100% full-frame: loại bỏ ảnh hỏng, xung đột taxonomy, phát hiện vật thể ẩn)
       v
[4. ANNOTATED]
       |  (Vẽ bounding box snug pixel-level, kiểm tra va chạm biên x=W, y=H)
       v
[5. USER_APPROVED]  <--- CỔNG PHÊ DUYỆT BẮT BUỘC CỦA NGƯỜI DÙNG (STRICT ZERO AUTO-APPROVAL)
       |  (Chỉ merge khi người dùng xác nhận nghiệm thu visual overlay và manifest)
       v
[6. TRAIN_READY]
          (Merge vào train split manifest, kiểm tra chống rò rỉ holdout lần cuối)
```

### 2.1 Các quy tắc chất lượng cốt lõi
1. **Không tạo Image ID giả mạo:** Mọi tệp ảnh phải bắt nguồn từ file ảnh thật, có tên định danh chuẩn mực theo nguồn (`wiki_bio_...`, `real_cam_...`).
2. **Không gán nhãn ước lượng trước khi xem ảnh:** Tuyệt đối không sinh file nhãn `.txt` tạm thời hoặc rỗng trước khi con người soi ảnh thực tế.
3. **Phân biệt ngữ cảnh rác:** Quả táo tươi trên đĩa ăn không phải là rác sinh học; chỉ dán nhãn `1: biological` khi quả táo đã bị cắn dở, vứt vào thùng rác hoặc ném trên mặt đất.
4. **Kiểm tra Hard-Negative tuyệt đối:** Một ảnh được gán nhãn Hard-Negative (nhãn trống 0-byte) bắt buộc phải sạch 100% rác; nếu phát hiện dù chỉ một mẩu đầu lọc thuốc lá hay mẩu giấy vụn thì phải chuyển sang dán nhãn đầy đủ hoặc loại bỏ.
5. **Duyệt trùng lặp 2 lớp:**
   - Trùng lặp tuyệt đối: SHA-256 bit-for-bit.
   - Trùng lặp thị giác: pHash với khoảng cách Hamming $\le 8$.
   - Đối chiếu chéo tự động với **3.842 mã băm đóng băng** của `splits_real_v2` và `holdout_27` (`D:\Hinh_Anh_Rac`).
6. **Chia split theo Session / Video / Source:** Không bao giờ cắt các frame từ cùng một đoạn video ngắn vào cả tập Train và tập Test (Data Leakage nghiêm trọng). Mọi frame từ cùng một nguồn/buổi quay phải nằm trọn trong một tập duy nhất.

---

## 3. LỘ TRÌNH THU THẬP THEO 3 ĐỢT (BATCH ROADMAP)

### 3.1 Đợt A (Batch A — Pilot 100 – 200 ảnh) — *Chuẩn bị ngay cho Sprint 1*
- **Quy mô:** 100 – 200 ảnh chất lượng cao có nguồn gốc thật.
- **Trọng tâm phân bổ:**
  - Pin / Ắc quy (Battery): 40 – 60 ảnh (ưu tiên pin vứt lẫn thực tế, pin ngoài trời).
  - Rác sinh học (Biological): 40 – 60 ảnh (vỏ chuối, bã mía, cuống rau thực địa).
  - Hard-Negative sạch: 30 – 50 ảnh (tủ điện hè phố, vỉa hè sạch, nắp cống, tường gạch).
  - Đa rác thực tế (Multi-waste): 20 – 30 ảnh (chụp từ webcam và điện thoại).
- **Phân bổ bối cảnh:** 50% Camera cố định (góc cao, thùng rác), 50% Camera di động (cầm tay, góc nghiêng).
- **Đầu ra đợt A:** Bộ ảnh qua 100% Visual QA $\to$ Trình người dùng phê duyệt $\to$ Đề xuất đợt huấn luyện Pilot ngắn (20–40 epochs) để chứng minh cải thiện thực chất.

### 3.2 Đợt B (Batch B — 300 – 500 ảnh tích lũy)
- **Quy mô:** Mở rộng thêm 200 – 300 ảnh (tổng tích lũy 300 – 500 ảnh).
- **Trọng tâm:** Các trường hợp khó phát hiện ngoài thực tế:
  - Rác bị che khuất một phần (occlusion $\ge 30\%$).
  - Điều kiện ánh sáng phức tạp: ngược sáng (backlight), thiếu sáng ban đêm (low light), bóng đổ gắt.
  - Rác bị biến dạng: lon nước bị xe cán bẹp, chai nhựa vỡ vụn, giấy thấm ướt.
- **Đầu ra đợt B:** Huấn luyện lại mô hình, đối soát và so sánh với baseline.

### 3.3 Đợt C (Batch C — Mở rộng quy mô 1.900 – 2.800 ảnh)
- **Nguyên tắc triển khai:** Không tải dữ liệu hàng loạt vô tội vạ. Chỉ kích hoạt đợt C dựa trên **phân tích lỗi thực địa (Failure Analysis)** của mô hình sau Đợt B:
  - Lớp nào thường xuyên bị nhận nhầm? $\to$ Bổ sung lớp đó.
  - Vị trí camera nào hay sinh False Positive? $\to$ Bổ sung Hard-Negative tương ứng.
- Mọi gói tải về đều có whitelist URL, dung lượng giới hạn và phê duyệt trước của người dùng.

---

## 4. HIỆN TRẠNG 4 ẢNH WIKIMEDIA GIAI ĐOẠN 4E (INVENTORY CHECK)

Bốn ảnh thực tế đã được tải về trong giai đoạn 4E với tổng dung lượng **10.22 MiB** (nằm trong hạn mức an toàn $\le 20\text{ MiB}$):

| ID Ứng viên | Tệp gốc Wikimedia | Kích thước | SHA-256 | Kết quả QA Kỹ thuật | Trạng thái Nghiệp vụ Hiện tại |
|---|---|:---:|---|:---:|---|
| `wiki_bio_banana_peel_ground` | `File:Banana_peel_on_the_ground.jpg` | $5184\times 3456$ | `086eae0e94991...` | **QA PASS** | **PROPOSED_PENDING_USER_APPROVAL** (Chờ người dùng duyệt box `[2100, 1280, 5184, 2300]`) |
| `wiki_neg_junction_boxes` | `File:Junction_boxes_..._580156.jpg` | $480\times 640$ | `bbf480384cc91...` | **QA PASS** | **CLEAN_NEGATIVE_CANDIDATE** (Chờ người dùng duyệt làm Hard-Negative) |
| `wiki_bio_street_food_remains` | `File:Street_food_remains.jpg` | $2048\times 1365$ | `05a0cf4609a96...` | **QA FAIL** | **REJECTED_TAXONOMY_CONFLICT** (Hộp xốp styrofoam chiếm đa số, xung đột với `biological`) |
| `wiki_neg_utility_box_leeds` | `File:Utility_box_leeds.jpg` | $5152\times 3864$ | `c3af6456ce23c...` | **QA FAIL** | **PENDING_OBJECT_IDENTITY_REVIEW** (Nghi vấn có lon kim loại ở chân tủ cỏ) |

```text
======================================================================
CHỐT TRẠNG THÁI HIỆN TẠI (PHASE 4E):
- REAL NEW IMAGES DOWNLOADED = 4 (10.22 MiB)
- VISUALLY QA PASSED         = 2 (wiki_bio_banana_peel_ground, wiki_neg_junction_boxes)
- APPROVED FOR TRAIN         = 0 (TUYỆT ĐỐI KHÔNG TỰ AUTO-APPROVE; GIỮ NGUYÊN CHỜ DUYỆT)
======================================================================
```
*Ghi chú quan trọng:* Hai ảnh QA PASS trên hiện đang được lưu an toàn tại [train_expansion_candidate_manifest.json](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/manifests/train_expansion_candidate_manifest.json). Chưa có bất kỳ file nhãn hay dòng dữ liệu nào bị chèn vào tập huấn luyện chính thức khi người dùng chưa bấm duyệt.

---

## 5. DỰ TOÁN THỜI GIAN VÀ DUNG LƯỢNG LƯU TRỮ

| Hạng mục / Giai đoạn | Số lượng ảnh dự kiến | Dung lượng lưu trữ ước tính | Thời gian thực hiện ước tính | Người phụ trách chính |
|---|:---:|:---:|:---:|:---:|
| **Hiện trạng Phase 4E** | 4 ảnh (2 ứng viên) | ~10.2 MiB | Đã hoàn tất | Agent A / Integrator |
| **Đợt A (Batch A Pilot)** | 100 – 200 ảnh | ~250 – 500 MiB | 2 – 3 ngày làm việc | Agent A (Thu thập) + Agent D (QA) |
| **Đợt B (Batch B)** | 300 – 500 ảnh | ~800 MiB – 1.5 GiB | 4 – 5 ngày làm việc | Agent A + Agent D |
| **Đợt C (Mở rộng dài hạn)** | 1.900 – 2.800 ảnh | ~4.5 – 7.5 GiB | 2 – 3 tuần (chia đợt) | Multi-Agent Team |

*Khuyến nghị an toàn ổ cứng:* Dung lượng còn trống trên ổ đĩa `D:` cần duy trì tối thiểu $\ge 20\text{ GiB}$ để đảm bảo không gian cho checkpoint huấn luyện và video kiểm thử thực địa.
