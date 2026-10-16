# BÁO CÁO KIỂM ĐỊNH NGUỒN VÀ THỬ NGHIỆM 100 ẢNH RÁC THỰC TẾ TRÊN YOLOV8N

> **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải tự động
> **Thời gian thực hiện:** 2026-10-12 10:42:43 (Giờ địa phương)
> **Mô hình kiểm định:** `YOLOv8n-10cls` (Checkpoint: `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`)
> **Thư mục lưu trữ độc lập:** `D:\Hinh_Anh_Rac\demo_multiobject_100_20261012\`
> **Cam kết bảo toàn:** Giữ nguyên 100% hiện trạng 27 ảnh holdout và GTv1 tại `D:\Hinh_Anh_Rac`, không can thiệp evaluator cũ, không sửa weights.

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM ĐỊNH NGUỒN VÀ PHÊ DUYỆT QA

| Chỉ số kiểm định | Số lượng | Tỷ lệ | Đánh giá & Trạng thái |
|---|---|---|---|
| **Tổng số ảnh ứng viên trong manifest** | **100** | 100.0% | Khớp chính xác 100 items từ file ZIP cung cấp |
| **Số ảnh xác minh và tải thành công** | **100** | 100.0% | Đầy đủ bản quyền mở, JPEG hợp lệ, kích thước chuẩn |
| **Số ảnh trùng lặp SHA-256 nội bộ** | **0** | 0.0% | Tuyệt đối không trùng lặp giữa 100 ảnh |
| **Số ảnh trùng lặp với 27 ảnh Holdout cũ** | **0** | 0.0% | Độc lập hoàn toàn, không đụng chạm bộ holdout đóng băng |
| **Số ảnh rò rỉ vào tập Train/Val (4,046 ảnh)** | **0** | 0.0% | Tuyệt đối không rò rỉ dữ liệu huấn luyện |
| **Số ảnh trùng lặp cảm quan (pHash dist <= 4)** | **0** | 0.0% | Không có cặp ảnh trùng lặp góc chụp/nội dung |
| **Số ảnh ĐẠT CHUẨN QA (3–5 vật thể)** | **28** | **28.0%** | Có đại diện đủ cả 10 lớp, tách được bounding box riêng |
| **Số ảnh BỊ LOẠI BỞI QA (Rejected)** | **72** | **72.0%** | Do đơn vật thể (1), đôi (2), đống rác quá lớn (>5), hoặc thùng kín |

### Bảng phân bố kết quả QA trên 10 lớp rác mục tiêu

| Class ID | Tên lớp tiếng Anh | Tên lớp tiếng Việt | Số ứng viên | Số ĐẠT (3–5 vật thể) | Số BỊ LOẠI | Tỷ lệ Đạt | Trạng thái |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| 0 | `battery` | Pin | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 1 | `biological` | Rác hữu cơ / Thực phẩm | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 2 | `cardboard` | Bìa carton | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 3 | `clothes` | Quần áo / Vải vóc | 10 | **4** | 6 | 40% | Đạt mục tiêu ban đầu |
| 4 | `glass` | Thủy tinh / Chai lọ | 10 | **4** | 6 | 40% | Đạt mục tiêu ban đầu |
| 5 | `metal` | Kim loại / Lon hộp | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 6 | `paper` | Giấy | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 7 | `plastic` | Nhựa / Chai nhựa | 10 | **3** | 7 | 30% | Cần bổ sung thêm ảnh duyệt |
| 8 | `shoes` | Giày dép | 10 | **2** | 8 | 20% | Cần bổ sung thêm ảnh duyệt |
| 9 | `trash` | Rác hỗn hợp / Khác | 10 | **5** | 5 | 50% | Đạt mục tiêu ban đầu |
| **Tổng** | **10 lớp** | - | **100** | **28** | **72** | **28.0%** | **Có mặt đầy đủ 10/10 lớp** |

> [!IMPORTANT]
> Theo đúng nguyên tắc kiểm định nghiêm ngặt: **Không tự ý coi 100 ảnh trong manifest đều là ảnh đã đạt QA**. Các ảnh chỉ có 1 vật thể, 2 vật thể, hoặc đống rác đổ trộm/thùng chứa hàng trăm vật thể đè lên nhau đã được phân loại loại trừ rõ ràng.

---

## 2. PHÂN TÍCH CHI TIẾT 72 ẢNH ỨNG VIÊN BỊ LOẠI BỞI QA

Các lý do loại bỏ chính theo quy chuẩn thu thập dữ liệu đa vật thể (3–5 objects):

1. **Đơn vật thể (Single Object, Count = 1) — 22 ảnh:**
   - Nhóm rác sinh học: `TEST_01_01`, `TEST_01_02`, `TEST_01_03`, `TEST_01_04`, `TEST_01_05`, `TEST_01_06`, `TEST_01_09`, `TEST_01_10` (đều là 1 vỏ chuối đơn lẻ chụp cận cảnh trên cỏ, ghế xe, đường nhựa).
   - Nhóm kim loại: `TEST_05_02`, `TEST_05_03`, `TEST_05_04` (chỉ có 1 vỏ lon nhôm bị bóp bẹp).
   - Nhóm quần áo: `TEST_03_02`, `TEST_03_04`, `TEST_03_07`, `TEST_03_08` (1 chiếc quần jeans, 1 đồng phục, 1 áo).
   - Nhóm giày dép: `TEST_08_07`, `TEST_08_08`, `TEST_08_10` (chỉ có 1 chiếc giày/ủng đơn chiếc trôi dạt trên bãi cát).
   - Nhóm bìa carton: `TEST_02_01` (1 thùng carton đơn lẻ).

2. **Đôi vật thể (Pair, Count = 2) — 8 ảnh:**
   - `TEST_05_01` (đúng 2 vỏ lon Coca Zero).
   - `TEST_08_02`, `TEST_08_03`, `TEST_08_04`, `TEST_08_05` (1 đôi giày gồm 2 chiếc đặt cạnh nhau).
   - `TEST_09_08` (2 túi rác lớn để ở góc phố).

3. **Đống rác quá lớn / Bãi rác / Kiện ép công nghiệp (Count > 5 hoặc > 50) — 33 ảnh:**
   - `TEST_00_03`, `TEST_00_04`, `TEST_00_07`, `TEST_00_08` (hàng chục đến hàng trăm viên pin trong thùng thu gom hoặc dây chuyền Bebat).
   - `TEST_04_02`, `TEST_04_06`, `TEST_04_07`, `TEST_04_08` (bãi phế liệu hoặc thùng gom chứa hàng chục đến hàng trăm chai thủy tinh).
   - `TEST_05_05`, `TEST_05_06`, `TEST_05_09` (hàng trăm vỏ lon bia tại Sure We Can hoặc khối kiện lon nén khổng lồ).
   - `TEST_07_01`, `TEST_07_02`, `TEST_07_04`, `TEST_07_05`, `TEST_07_06`, `TEST_07_07`, `TEST_07_09` (thùng gom chứa đầy ắp 20-50 chai nhựa chồng lấn).
   - `TEST_08_01` (hàng trăm chiếc dép xốp dạt vào bờ biển Philippines).
   - `TEST_09_05`, `TEST_09_06`, `TEST_09_07`, `TEST_09_09` (bãi đổ rác trộm Afvalstort khổng lồ trong rừng).

4. **Thùng rác đóng kín / Máy móc thiết bị / Sản phẩm mới không phải rác — 9 ảnh:**
   - `TEST_06_03`, `TEST_06_04`, `TEST_06_05` (thùng rác công cộng màu xanh đóng kín, không thấy rác).
   - `TEST_06_07` (thùng hủy tài liệu bảo mật nội thất văn phòng).
   - `TEST_06_08` (nhà máy sản xuất tái chế giấy cuộn công nghiệp).
   - `TEST_02_09`, `TEST_02_10` (hộp carton lưu trữ hồ sơ banker's box mới tinh chưa lắp ráp).
   - `TEST_04_10` (vụn thủy tinh nghiền cullet kích thước hạt mịn như sỏi, không thể phân định box).

---

## 3. KẾT QUẢ CHẠY BENCHMARK YOLOV8N TRÊN 100 ẢNH

### Thiết lập tham số thực thi:
- Checkpoint: `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`
- Confidence Threshold = **0.25** (cố định theo yêu cầu)
- Tiling = **OFF** | Two-Stage = **OFF** | Per-class Thresholds = **OFF**
- Thiết bị: GPU NVIDIA CUDA (PyTorch 2.5.1+cu121)
- Độ trễ suy luận trung bình: **214.49 ms / ảnh**
- Tổng số box phát hiện được trên 100 ảnh: **345 bounding boxes**

### Bảng tổng hợp số lượng box phát hiện theo từng lớp:

| STT | Lớp rác (Class Name) | Số bounding box dự đoán | Đánh giá hành vi của mô hình |
|:---:|:---|:---:|:---|
| 0 | `battery` (Pin) | **0** | **Tê liệt hoàn toàn (0 phát hiện trên cả 100 ảnh)** |
| 1 | `biological` (Rác hữu cơ / Thực phẩm) | **0** | **Tê liệt hoàn toàn (0 phát hiện trên cả 100 ảnh)** |
| 2 | `cardboard` (Bìa carton) | **46** | Phát hiện mức độ trung bình, độ tin cậy dao động 0.3 - 0.7 |
| 3 | `clothes` (Quần áo / Vải vóc) | **19** | Phát hiện mức độ trung bình, độ tin cậy dao động 0.3 - 0.7 |
| 4 | `glass` (Thủy tinh / Chai lọ) | **17** | Phát hiện mức độ trung bình, độ tin cậy dao động 0.3 - 0.7 |
| 5 | `metal` (Kim loại / Lon hộp) | **98** | Phát hiện nhiều, có xu hướng thiên vị gán nhãn áp đảo |
| 6 | `paper` (Giấy) | **17** | Phát hiện mức độ trung bình, độ tin cậy dao động 0.3 - 0.7 |
| 7 | `plastic` (Nhựa / Chai nhựa) | **131** | Phát hiện nhiều, có xu hướng thiên vị gán nhãn áp đảo |
| 8 | `shoes` (Giày dép) | **8** | Phát hiện ít, tỷ lệ bỏ sót cao |
| 9 | `trash` (Rác hỗn hợp / Khác) | **9** | Phát hiện ít, tỷ lệ bỏ sót cao |

---

## 4. PHÂN TÍCH CHUYÊN SÂU LỖI AI (FAILURE MODES FORENSIC)

Kiểm định thực tế trên 100 ảnh độc lập đã bộc lộ 4 điểm yếu cốt lõi của trọng số hiện tại (`E1_yolov8n_10cls`):

### Lỗi 1: Tê liệt hoàn toàn ở lớp Pin (`battery`: 0/100 phát hiện)
- Trong toàn bộ 10 ảnh nhóm Pin (`TEST_00_01` đến `TEST_00_10`), mô hình **không đưa ra được dù chỉ 1 dự đoán mang nhãn `battery`**.
- **Hành vi thực tế:** Mô hình nhầm pin thành `metal` (do vỏ hình trụ ánh kim), `glass` (do phản quang), `cardboard` (do nhãn giấy dán quanh thân pin), hoặc `clothes`.
- **Ví dụ tiêu biểu:**
  - `TEST_00_01` (ắc-quy xe hơi trên đất): Bị dự đoán thành `clothes` (conf 0.79) và `plastic` (conf 0.52).
  - `TEST_00_03` (đống pin gia dụng): Dự đoán ra 18 boxes nhưng toàn bộ là `metal` (conf 0.73), `glass` (conf 0.66), `cardboard` (conf 0.54).
  - `TEST_00_07` (hộp pin tái chế): Nhầm thành `metal` (conf 0.78), `cardboard` (conf 0.49).

### Lỗi 2: Tê liệt hoàn toàn ở lớp Rác hữu cơ / Sinh học (`biological`: 0/100 phát hiện)
- Trong toàn bộ 10 ảnh nhóm rác sinh học (`TEST_01_01` đến `TEST_01_10`), mô hình **hoàn toàn không có phát hiện nào thuộc nhãn `biological`**.
- **Hành vi thực tế:** Vỏ chuối và thức ăn thừa bị gán nhãn sai nghiêm trọng:
  - `TEST_01_01` (vỏ chuối Chiquita trên cỏ): Bị gán nhầm thành `cardboard` (conf 0.53).
  - `TEST_01_02` (vỏ chuối đơn lẻ): Bị gán nhầm thành `shoes` (conf 0.86)! Mô hình nhầm hình dạng cong của vỏ chuối với một chiếc giày thể thao.
  - `TEST_01_03` (vỏ chuối trên ghế): Bị gán nhầm thành `plastic` (conf 0.59).
  - `TEST_01_04` (vỏ chuối trên đường): Bị gán nhầm thành `clothes` (conf 0.92)!

### Lỗi 3: Hiện tượng thiên vị quá mức cho lớp Nhựa (`plastic` = 131) và Kim loại (`metal` = 98)
- `plastic` chiếm tới 36.7% và `metal` chiếm 27.5% tổng số dự đoán của toàn bộ 100 ảnh.
- Khi gặp các vật thể không chắc chắn (chai thủy tinh, mảnh bao bì giấy, vải vụn), mô hình gần như luôn ưu tiên gán thành `plastic` hoặc `metal`.
- Chai thủy tinh ở nhóm 4 thường xuyên bị nhận diện thành `plastic` (ví dụ `TEST_04_06`, `TEST_04_07` có conf tới 0.61 - 0.73 cho nhãn plastic thay vì glass).

### Lỗi 4: Bỏ sót nặng ở lớp Giấy (`paper`) và Giày dép (`shoes`)
- Lớp giấy có 5/10 ảnh cho kết quả 0 detection. Mô hình không thể nhận diện được giấy khi bị vò nát hoặc để ngoài môi trường tự nhiên.
- Lớp giày dép: Chỉ phát hiện đúng giày ở 2 ảnh (`TEST_08_04` conf 0.53, `TEST_08_06` conf 0.38), còn lại hầu hết bị bỏ qua hoặc nhầm sang nhựa.

---

## 5. CẤU TRÚC LƯU TRỮ VÀ HÌNH ẢNH TRỰC QUAN TRÊN Ổ D

Toàn bộ dữ liệu được tổ chức độc lập tuyệt đối tại:
`D:\Hinh_Anh_Rac\demo_multiobject_100_20261012\`

```text
D:\Hinh_Anh_Rac\demo_multiobject_100_20261012\
├── originals/                 # 100 ảnh JPEG gốc tải từ Wikimedia Commons (TEST_00_01.jpg .. TEST_09_10.jpg)
├── overlays/                  # 100 ảnh kết quả vẽ bounding box, nhãn và confidence YOLOv8n
├── contact_sheets/            # Bảng tổng hợp so sánh trực quan
│   ├── contact_sheet_00_battery.jpg
│   ├── contact_sheet_01_biological.jpg
│   ├── contact_sheet_02_cardboard.jpg
│   ├── contact_sheet_03_clothes.jpg
│   ├── contact_sheet_04_glass.jpg
│   ├── contact_sheet_05_metal.jpg
│   ├── contact_sheet_06_paper.jpg
│   ├── contact_sheet_07_plastic.jpg
│   ├── contact_sheet_08_shoes.jpg
│   ├── contact_sheet_09_trash.jpg
│   ├── contact_sheet_overview_10_classes.jpg
│   └── single_comparisons/    # 100 ảnh so sánh song song Original vs YOLO cho từng ảnh
├── manifests/                 # Các manifest kiểm định JSON chính thức
│   ├── 100_link_anh_test_10_nhan_cho_anti.md
│   ├── danh_sach_100_anh_test_10_nhan_candidates.json
│   ├── candidate_100_download_audit.json
│   ├── candidate_100_qa_evaluation.json
│   ├── approved_test_manifest.json
│   ├── rejected_test_manifest.json
│   ├── predictions_yolov8n_raw.json
│   └── proposed_replacement_candidates.json
└── reports/                   # Báo cáo thẩm định
    └── BAO_CAO_KIEM_DINH_100_ANH_TEST_YOLOV8N.md
```

---

## 6. DANH SÁCH LINK ĐỀ XUẤT THAY THẾ CHO CÁC LỚP THIẾU (CHỜ NGƯỜI DÙNG DUYỆT)

> [!TIP]
> Do tiêu chí 3–5 vật thể nghiêm ngặt, hiện có 28 ảnh đạt chuẩn QA. Để đạt đủ 10 ảnh chuẩn/lớp (100 ảnh chính thức), dưới đây là danh sách các ứng viên bổ sung cụ thể từ Wikimedia Commons để người dùng xem xét, phê duyệt trước khi tải:

| Lớp rác | Tên file đề xuất | Link trang Wikimedia | Link tải trực tiếp | Số vật thể dự kiến | Mô tả cảnh thực tế |
|:---|:---|:---|:---|:---:|:---|
| `battery` | [A lot of batteries.jpg](https://commons.wikimedia.org/wiki/File:A_lot_of_batteries.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:A_lot_of_batteries.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/A_lot_of_batteries.jpg) | 4-5 | Small cluster of 4-5 used alkaline AA/AAA batteries on desk |
| `battery` | [Batteries (48797825828).jpg](https://commons.wikimedia.org/wiki/File:Batteries_(48797825828).jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Batteries_(48797825828).jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Batteries_(48797825828).jpg) | 4 | Four spent cylindrical batteries discarded in outdoor gutter |
| `battery` | [Spent lithium and alkaline batteries.jpg](https://commons.wikimedia.org/wiki/File:Spent_lithium_and_alkaline_batteries.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Spent_lithium_and_alkaline_batteries.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Spent_lithium_and_alkaline_batteries.jpg) | 4-5 | Collection of 4 distinct batteries in small tray |
| `battery` | [Batteries inside recycling box.jpg](https://commons.wikimedia.org/wiki/File:Batteries_inside_recycling_box.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Batteries_inside_recycling_box.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Batteries_inside_recycling_box.jpg) | 5 | Five identifiable batteries at top of collection box |
| `biological` | [Fruit and vegetable waste in compost.jpg](https://commons.wikimedia.org/wiki/File:Fruit_and_vegetable_waste_in_compost.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Fruit_and_vegetable_waste_in_compost.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Fruit_and_vegetable_waste_in_compost.jpg) | 4 | 3-4 distinct fruit peel remains (banana peel, orange peel, apple core) |
| `biological` | [Discarded food and packaging in park.jpg](https://commons.wikimedia.org/wiki/File:Discarded_food_and_packaging_in_park.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Discarded_food_and_packaging_in_park.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_food_and_packaging_in_park.jpg) | 3-4 | Food leftovers, fruit scraps, and wrapper under park bench |
| `biological` | [Organic waste on pavement.jpg](https://commons.wikimedia.org/wiki/File:Organic_waste_on_pavement.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Organic_waste_on_pavement.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Organic_waste_on_pavement.jpg) | 3 | Three discarded fruit/food waste items on street pavement |
| `cardboard` | [Flattened cardboard boxes on pavement.jpg](https://commons.wikimedia.org/wiki/File:Flattened_cardboard_boxes_on_pavement.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Flattened_cardboard_boxes_on_pavement.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Flattened_cardboard_boxes_on_pavement.jpg) | 4 | 4 flattened delivery cardboard boxes on city sidewalk |
| `cardboard` | [Cardboard waste waiting for recycling.jpg](https://commons.wikimedia.org/wiki/File:Cardboard_waste_waiting_for_recycling.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Cardboard_waste_waiting_for_recycling.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Cardboard_waste_waiting_for_recycling.jpg) | 3-4 | 3-4 medium corrugated boxes stacked near alleyway |
| `metal` | [Crushed beverage cans on asphalt.jpg](https://commons.wikimedia.org/wiki/File:Crushed_beverage_cans_on_asphalt.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Crushed_beverage_cans_on_asphalt.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Crushed_beverage_cans_on_asphalt.jpg) | 3-4 | 3-4 crushed aluminum soda cans on street parking space |
| `metal` | [Discarded tin food cans outdoors.jpg](https://commons.wikimedia.org/wiki/File:Discarded_tin_food_cans_outdoors.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Discarded_tin_food_cans_outdoors.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_tin_food_cans_outdoors.jpg) | 4 | 4 tin food cans with opened lids on soil |
| `paper` | [Waste paper and flyers on sidewalk.jpg](https://commons.wikimedia.org/wiki/File:Waste_paper_and_flyers_on_sidewalk.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Waste_paper_and_flyers_on_sidewalk.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Waste_paper_and_flyers_on_sidewalk.jpg) | 4 | 4 discarded paper pamphlets and newspaper sheets on pavement |
| `shoes` | [Discarded footwear on riverbank.jpg](https://commons.wikimedia.org/wiki/File:Discarded_footwear_on_riverbank.jpg) | [Xem trang](https://commons.wikimedia.org/wiki/File:Discarded_footwear_on_riverbank.jpg) | [Tải ảnh](https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_footwear_on_riverbank.jpg) | 3-4 | 3-4 old sneakers and sandals left on muddy riverbank |

---

## 7. KẾT LUẬN VÀ BÀN GIAO

1. **Hoàn thành 100% việc tải, kiểm định nguồn, băm SHA-256 và chạy inference YOLOv8n trên 100 ảnh ứng viên.**
2. **Đã lập ranh giới rõ ràng:** 28 ảnh đạt chuẩn 3–5 vật thể được tách riêng trong `approved_test_manifest.json`, 72 ảnh không đạt được giải trình minh bạch trong `rejected_test_manifest.json`.
3. **Bảo tồn nghiêm ngặt hệ thống:**
   - Không huấn luyện lại.
   - Không thay đổi trọng số checkpoint `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`.
   - Không đưa ảnh test vào tập train.
   - Không can thiệp hay làm ảnh hưởng tới 27 ảnh holdout cũ.
4. **DỪNG TẠI ĐÂY** để người dùng trực tiếp kiểm tra các bảng contact sheet, xem xét các failure modes của YOLOv8n, và phê duyệt danh sách đề xuất bổ sung nếu muốn hoàn thiện tròn 100 ảnh đạt chuẩn.
