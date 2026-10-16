# BÁO CÁO NGHIỆM THU: REAL IMAGE PREVIEW & VISUAL QA (OPEN IMAGES 3 SAMPLES)

**Dự án:** Hệ thống nhận diện rác thải sinh hoạt KLCN155  
**Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`  
**Thời điểm thực thi:** 2026-10-12  
**Trạng thái kiểm duyệt:** 2 REJECTED, 1 PENDING_NEGATIVE_REVIEW, 0 APPROVED  
**Tình trạng dữ liệu:** 0 ảnh được merge vào train, 0 ảnh được huấn luyện  

---

## 1. TỔNG QUAN THỰC THI & CƠ CHẾ AN TOÀN

Thực hiện theo chỉ đạo phê duyệt của người dùng với script [`scripts/stage_whitelisted_pilot_images.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/stage_whitelisted_pilot_images.py):

1. **Cơ chế kiểm soát tải ảnh (Fail-Closed):**
   - **Whitelist cứng 3 ảnh duy nhất:** `078821d86db99fc9` (Apple), `74a44994e3d6b962` (Banana), `e29db1c1963628e7` (Building).
   - **Đọc HTTP dạng luồng (Streaming Read):** Đọc từng khối `8192 bytes`, kiểm tra trực tiếp giới hạn dung lượng trong vòng lặp đọc. Tuyệt đối không dùng `resp.read()` không giới hạn.
   - **Khống chế trần an toàn:** Trần dung lượng tối đa 1200 KB, kiểm soát tải từng ảnh và toàn bộ phiên.
   - **Bắt buộc cờ phê duyệt:** Mọi lệnh chạy không có `--approve-preview-download` đều bị từ chối với exit code 1.

2. **Dữ liệu đệm độc lập:**
   - Lưu trữ biệt lập tại: [`data/acquisition_staging/pilot_preview_3/`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/pilot_preview_3/)
   - Tuyệt đối không can thiệp vào `data/detection_v2`, `data/detection_v3_candidate` hay bất kỳ split huấn luyện/kiểm thử nào.

---

## 2. BẢNG KÊ CHI TIẾT 3 ẢNH THẬT ĐÃ TẢI & MÃ BĂM MẬT MÃ SHA-256

| # | ImageID | Filename | Phân loại đề xuất | Độ phân giải | Dung lượng | SHA-256 Ảnh Thật (Disk) | SHA-256 Nhãn Đề Xuất | Kết quả Visual QA |
|---|---|---|---|---|---|---|---|---|
| **1** | `078821d86db99fc9` | `oi_078821d86db99fc9.jpg` | `biological` | 1024x683 | 333,549 B | `6c83db71cd99f14713b71e7f9ba1877447e52dea559266c7f300430dc30b58ec` | `43ebd322bb3e905a8bce927ae9bf7a74a76bd7aa64723122721dd8151a900a1b` | **REJECT** (Quả tươi trên cây) |
| **2** | `74a44994e3d6b962` | `oi_74a44994e3d6b962.jpg` | `biological` | 768x1024 | 633,651 B | `f9a1a25e273653539e37f1bff8564779d61de90b314e7cbf0988b9570f05175d` | `695b39ca81bd84455a82e8f7651a8fd995529224641b84bb89413ff805f89015` | **REJECT** (Món ăn phục vụ trên đĩa) |
| **3** | `e29db1c1963628e7` | `oi_e29db1c1963628e7.jpg` | `hard_negatives` | 1024x679 | 132,274 B | `b769f9e141936b18ec82bab2f51bd55d5fb67d0ed02d600d91220eb0fc232d3f` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 byte) | **PENDING_NEGATIVE_REVIEW** (Chờ người dùng QA) |

**Tổng dung lượng tải thực tế:** 1,099,474 bytes (~1.05 MB).

---

## 3. ĐÁNH GIÁ TRỰC QUAN THEO NGỮ CẢNH (PHYSICAL VISUAL QA)

### 3.1 Mẫu 1: `078821d86db99fc9` (Apple / biological)
- **Tiêu đề gốc Flickr:** *"Omenapuu Åkerö"* (Cây táo giống Åkerö) của tác giả Matleena Laakso.
- **Quan sát trực quan từ ảnh overlay ([`overlay_oi_078821d86db99fc9.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/pilot_preview_3/overlays/overlay_oi_078821d86db99fc9.jpg)):**
  - Hai quả táo tươi đang treo trực tiếp trên cành cây ăn quả, lá cây xanh tươi bao quanh.
  - Đây là nông sản / cây trồng tự nhiên trong vườn cây ăn quả.
- **Tiêu chí loại trừ:** *"Trái cây còn sống trên cây hoặc đang được trồng -> REJECT."*
- **Quyết định QA:** **REJECT**. Tuyệt đối không đưa vào tập dữ liệu nhận diện rác thải sinh hoạt vì sẽ gây dương tính giả trầm trọng (mô hình sẽ nhận diện cây cối ngoài đường/vườn là rác).

### 3.2 Mẫu 2: `74a44994e3d6b962` (Banana / biological)
- **Tiêu đề gốc Flickr:** *"Vegan Lemon Cornmeal Pancakes"* của tác giả tomatoes and friends.
- **Quan sát trực quan từ ảnh overlay ([`overlay_oi_74a44994e3d6b962.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/pilot_preview_3/overlays/overlay_oi_74a44994e3d6b962.jpg)):**
  - 4 lát chuối tươi được xếp trang trí làm topping bên trên đĩa bánh pancake nóng hổi, đặt trên đĩa sứ hoa văn đặt trên bàn ăn gia đình.
  - Đây là món ăn ẩm thực đang được phục vụ thưởng thức.
- **Tiêu chí loại trừ:** *"Chuối trên đĩa thức ăn bình thường -> REJECT."*
- **Quyết định QA:** **REJECT**. Không phải thức ăn thừa hỏng hay rác thải hữu cơ bị vứt bỏ. Nếu huấn luyện ảnh này, mô hình sẽ học sai rằng bữa ăn trên bàn của người dùng là rác.

### 3.3 Mẫu 3: `e29db1c1963628e7` (Building / hard_negatives)
- **Tiêu đề gốc Flickr:** *"Ronda"* của tác giả Manuela Aguilera.
- **Quan sát trực quan từ ảnh overlay ([`overlay_oi_e29db1c1963628e7.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/pilot_preview_3/overlays/overlay_oi_e29db1c1963628e7.jpg)):**
  - Toàn cảnh cầu đá lịch sử Puente Nuevo tại thị trấn Ronda (Tây Ban Nha), với hẻm núi đá vôi hùng vĩ và bầu trời xanh.
  - Rà soát toàn bộ khung hình không thấy bất kỳ mảnh rác sinh hoạt nào (không vỏ chai, túi ni-lông, cốc giấy).
- **Tiêu chí kiểm định:** Không tự động đánh dấu `VERIFIED_TRUE_NEGATIVE` khi chưa có quyết định của người dùng.
- **Quyết định QA:** **PENDING_NEGATIVE_REVIEW**. Ảnh sạch, là ứng viên Hard-Negative tiềm năng tốt, tạm giữ nguyên trạng thái chờ người dùng xác nhận nghiệm thu.

---

## 4. TỔNG KẾT SỐ LIỆU PHÂN LOẠI & CONTACT SHEET

- **Số ảnh bị loại (REJECT):** 2 ảnh (`078821d86db99fc9`, `74a44994e3d6b962`).
- **Số ảnh chờ duyệt (PENDING):** 1 ảnh (`e29db1c1963628e7`).
- **Số ảnh được phê duyệt huấn luyện (APPROVED):** 0 ảnh.
- **Số ảnh được merge vào tập train:** 0 ảnh.
- **Contact Sheet hợp nhất:** Đã xuất tại [`contact_sheet_preview_3.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/pilot_preview_3/contact_sheet/contact_sheet_preview_3.jpg) hiển thị trực quan cả 3 mẫu cùng thông tin bbox, nguồn và thẻ đánh giá.

---

## 5. BÀI HỌC KIỂM ĐỊNH & ĐỊNH HƯỚNG CHIẾN LƯỢC TIẾP THEO

Kiểm tra trực quan 3 ảnh thật đã chứng minh lập luận của chuyên gia đánh giá độc lập là hoàn toàn chính xác:
1. **Rủi ro lớn từ Open Images lớp chung:**
   - Việc chỉ truy vấn nhãn `/m/014j1m` (Apple) hoặc `/m/09qck` (Banana) từ Open Images sẽ thu nạp phần lớn là cây ăn quả trong vườn hoặc món ăn tươi ngon trong nhà hàng. Cả hai trường hợp này đều vi phạm trực tiếp định nghĩa rác thải sinh hoạt (municipal waste).
2. **Chấm dứt việc lấy mẫu ngẫu nhiên Apple/Banana:**
   - Không tiếp tục cào/lấy ngẫu nhiên các lớp thực phẩm thông thường từ Open Images.
3. **Chuyển hướng thu thập dữ liệu rác thật sự:**
   - **Đối với nhóm Pin (Battery):** Tiếp nhận dữ liệu pin thải thật từ Roboflow `waste-battery-only/1` hoặc các bộ dữ liệu e-waste chuyên dụng.
   - **Đối với nhóm Hữu cơ (Biological):** Chỉ thu thập ảnh rác hữu cơ thực tế (vỏ hoa quả đã bóc bỏ, phế phẩm thực phẩm trong thùng rác, rác thải hữu cơ tại bãi tập kết).
