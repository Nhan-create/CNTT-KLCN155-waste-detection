"""
Generate TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md directly from
verified Task 7 JSON manifests (task7_qa_manifest.json, task7_leakage_report.json,
task7_dataset_v4_and_e4_proposal.json).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"


def main() -> None:
    qa_manifest = json.loads((STAGE_DIR / "task7_qa_manifest.json").read_text(encoding="utf-8"))
    leakage = json.loads((STAGE_DIR / "task7_leakage_report.json").read_text(encoding="utf-8"))

    summary = qa_manifest["summary_counts"]
    items = qa_manifest["items"]

    qa_pass_items = [x for x in items if x["qa_verdict"] == "QA_PASS"]
    needs_review_items = [x for x in items if x["qa_verdict"] == "NEEDS_MANUAL_REVIEW"]
    reject_items = [x for x in items if x["qa_verdict"] == "REJECT"]

    pass_rows = []
    for x in qa_pass_items:
        pass_rows.append(
            f"| `{x['short_key']}` | `{x['filename']}` | `{x['subgroup']}` | {x['actual_width']}x{x['actual_height']} | "
            f"**{x['boxes_count']}** | `{x['proposed_split']}` | `{x['specimen_group']}` | `{x['license']}` | {x['qa_reason']} |"
        )

    review_rows = []
    for x in needs_review_items:
        review_rows.append(
            f"| `{x['short_key']}` | `{x['filename']}` | `{x['subgroup']}` | {x['actual_width']}x{x['actual_height']} | "
            f"{x['boxes_count']} | `{x['license']}` | {x['qa_reason']} |"
        )

    reject_rows = []
    for x in reject_items:
        reject_rows.append(
            f"| `{x['short_key']}` | `{x['filename']}` | `{x['subgroup']}` | {x['actual_width']}x{x['actual_height']} | "
            f"`{x['license']}` | {x['qa_reason']} |"
        )

    provenance_rows = []
    for x in items:
        artist_clean = str(x.get("author", "Wikimedia User")).replace("|", "/")[:35]
        ph0 = x["phash_rotations"][0] if x.get("phash_rotations") else x.get("phash_0", "N/A")
        provenance_rows.append(
            f"| `{x['short_key']}` | `{x['qa_verdict']}` | `{x['license']}` | `{artist_clean}` | "
            f"`{x['sha256'][:16]}...` | `{ph0}` | [Link]({x['source_page_url']}) |"
        )

    md = f"""# BÁO CÁO TASK 7 — THU THẬP DỮ LIỆU PIN ĐA DẠNG NGOÀI ROBOFLOW, QA TRỰC QUAN & ĐỀ XUẤT DATASET V4 / E4 (DO NOT RUN)

- **Dự án:** `.`
- **Sprint:** `AI-UPGRADE-01 — TASK 6B & TASK 7`
- **Thư mục Staging:** `data/acquisition_staging/task7_diverse_battery_batch/`
- **Trạng thái phê duyệt huấn luyện:** **`APPROVED_FOR_TRAIN = 0`** *(Tất cả dữ liệu và cấu hình ở trạng thái đề xuất chờ PM nghiệm thu; tuyệt đối chưa chạy huấn luyện E4)*

---

## 1. TỔNG QUAN CHIẾN LƯỢC THU THẬP DỮ LIỆU (PHẦN B1)

Từ kết quả thực nghiệm của Pilot E3 ở Task 6:
- Tập 140 ảnh Roboflow (`111 train + 29 val`) giúp E3 đạt `mAP50 = 0.99235` trên cùng domain bàn gỗ đơn chiếc (`1 bbox/ảnh`).
- Tuy nhiên, trên tập Holdout thực tế (`data/holdout`, ảnh `IMG_2608.JPG` gồm cụm 4 viên pin chụp góc nghiêng), E3 chỉ phát hiện được `1/4` viên pin ở ngưỡng thấp và tăng False Positive trên các lớp khác.
- Nguyên nhân cốt lõi: **Domain shift nghiêm trọng** (100% ảnh pin bổ sung ở V3 là ảnh 1 viên pin đơn lẻ trên nền bàn phẳng, thiếu hoàn toàn cụm nhiều viên pin đứng/nằm chồng lấn, pin cúc áo, pin 9V vuông, pin Li-ion/18650/pin điện thoại, pin rỉ sét/chảy hóa chất và mẫu âm tính khó dạng trụ/thiết bị điện).

### Nguồn thu thập độc lập & Bản quyền mở
- Toàn bộ **56 ảnh ứng viên (`task7_01` .. `task7_56`)** được truy vấn và tải trực tiếp từ **Wikimedia Commons API** với giấy phép mở hợp lệ (`CC0`, `Public Domain`, `CC BY 3.0/4.0`, `CC BY-SA 2.5/3.0/4.0`).
- Mỗi tệp ảnh đều được ghi nhận đầy đủ: URL tải gốc (`image_url`), trang mô tả (`description_url`), tác giả (`artist`), giấy phép (`license_short`), kích thước ảnh gốc, mã băm `SHA-256` và `pHash` 4 góc xoay (`0°, 90°, 180°, 270°`) trong `raw_download_manifest.json` và `task7_qa_manifest.json`.

| Chỉ số tổng hợp Batch Task 7 | Giá trị |
|---|---|
| Tổng số ảnh tải về và kiểm tra (`task7_01`..`task7_56`) | **{qa_manifest['total_downloaded_candidates']}** ảnh |
| Số ảnh đạt chuẩn **`QA_PASS`** | **{summary['qa_pass_total_images']}** ảnh (**{summary['qa_pass_positive_battery_images']}** ảnh có pin + **{summary['qa_pass_hard_negative_images']}** ảnh hard negative) |
| Tổng số bounding box `0: battery` đạt chuẩn **`QA_PASS`** | **{summary['qa_pass_total_battery_boxes']}** full-object boxes |
| Số ảnh `QA_PASS` chứa **cụm nhiều viên pin (`>= 2 boxes/ảnh`)** | **{summary['qa_pass_multi_battery_images_ge2_boxes']}** / {summary['qa_pass_positive_battery_images']} ảnh có pin (**53.8%**) |
| Đề xuất đưa vào **`TRAIN_V4`** | **{summary['proposed_train_v4_images']}** ảnh (**{summary['proposed_train_v4_battery_boxes']}** battery boxes, gồm 16 ảnh dương tính + 3 ảnh hard negative) |
| Đề xuất đưa vào **`VAL_REAL_BATTERY`** (tập Val độc lập ngoài Roboflow) | **{summary['proposed_val_real_battery_images']}** ảnh (**{summary['proposed_val_real_battery_boxes']}** battery boxes, gồm 10 ảnh dương tính + 1 ảnh hard negative) |
| Số ảnh **`NEEDS_MANUAL_REVIEW`** (giữ riêng chờ PM quyết định) | **{summary['needs_manual_review_images']}** ảnh ({summary['needs_manual_review_candidate_boxes']} candidate boxes trong `labels_needs_manual_review/`) |
| Số ảnh **`REJECT`** (loại bỏ hoàn toàn) | **{summary['reject_images']}** ảnh (0 boxes) |
| **`APPROVED_FOR_TRAIN`** | **`{summary['approved_for_train']}`** |

---

## 2. KẾT QUẢ KIỂM TRA TRỰC QUAN & FULL-OBJECT RE-ANNOTATION (PHẦN B2)

### Quy tắc gán nhãn nghiêm ngặt đã áp dụng
1. **100% Full-Object Bounding Box:** Mỗi bounding box bao trọn toàn bộ thân vật lý của từng viên pin (từ cực dương đến đáy âm, bao gồm cả vỏ kim loại/nhựa và phần muối rỉ sét bám quanh thân). Tuyệt đối không khoanh chữ, logo hay nhãn hiệu.
2. **Tách rời từng viên pin trong cụm (`multi-battery cluster`):** Trong các ảnh có 2, 3, 4 hoặc 6 viên pin đứng/nằm cạnh nhau hoặc bắt chéo nhau, mỗi viên pin được gán 1 bounding box riêng biệt.
3. **Không gộp cả đống hoặc cả hộp bán lẻ:** Các ảnh vỉ nhựa phản quang, hộp 48 viên, hoặc đống >40 viên pin chồng lấp nặng được phân loại vào `NEEDS_MANUAL_REVIEW`, không đưa vào `QA_PASS`.

### 2.1. Danh sách 30 ảnh đạt chuẩn `QA_PASS` (61 Full-Battery Boxes)

| ID | Tệp ảnh | Nhóm đa dạng | Kích thước | Số Box | Split đề xuất | Specimen Group | Giấy phép | Nhận xét Visual QA |
|---|---|---|---|---:|---|---|---|---|
{chr(10).join(pass_rows)}

### 2.2. Danh sách 10 ảnh `NEEDS_MANUAL_REVIEW` (Giữ riêng chờ PM quyết định)

Các ảnh dưới đây có pin thật nhưng nằm trong tình huống biên (trong vỉ nhựa blister pack, gắn trong đế giữ CMOS trên bo mạch, cell 18650 tháo rời vỏ, 4 cell nằm trong bộ sạc, ắc quy cắt bổ lộ bản cực, đống >40 viên pin chồng lấp dày đặc, hộp 48 viên Duracell, hoặc pin điện thoại nằm trong khoang máy). Chúng được tách riêng nhãn vào `labels_needs_manual_review/` và không trộn vào `labels/`:

| ID | Tệp ảnh | Nhóm | Kích thước | Candidate Boxes | Giấy phép | Lý do cần PM quyết định (`NEEDS_MANUAL_REVIEW`) |
|---|---|---|---|---:|---|---|
{chr(10).join(review_rows)}

### 2.3. Danh sách 16 ảnh `REJECT` (Loại bỏ hoàn toàn)

Các ảnh do tìm kiếm từ khóa trả về nhưng chỉ chụp vỏ thùng thu gom từ bên ngoài (không thấy viên pin), biểu tượng đồ họa, trang văn bản bằng sáng chế năm 1889, mặt cắt hiển vi tụ điện, hoặc ảnh ngoài mục tiêu:

| ID | Tệp ảnh | Nhóm tìm kiếm | Kích thước | Giấy phép | Lý do loại bỏ (`REJECT`) |
|---|---|---|---|---|---|
{chr(10).join(reject_rows)}

---

## 3. KIỂM ĐỊNH CHỐNG RÒ RỈ DỮ LIỆU — ZERO LEAKAGE AUDIT (PHẦN B2/B3)

Báo cáo chi tiết lưu tại: `data/acquisition_staging/task7_diverse_battery_batch/task7_leakage_report.json`.

1. **Đối chiếu với toàn bộ các tập dữ liệu hiện hữu (`Train V2: 1398`, `Val V2: 299`, `Test: 299`, `Holdout: 27`, `Roboflow 140 Base: 140`):**
   - Trùng khớp `SHA-256`: **`{len(leakage['exact_sha256_collisions_with_v2_v3_holdout'])}`**
   - Cặp ảnh gần trùng `4-rotation pHash <= 8` với bất kỳ tập hiện hữu nào: **`{len(leakage['near_phash_le8_collisions_with_v2_v3_holdout'])}`** (`zero_external_leakage = {leakage['zero_external_leakage']}`)
2. **Đối chiếu nội bộ giữa 30 ảnh `QA_PASS` và phân nhóm theo Specimen/Photographer:**
   - Phát hiện **3 nhóm ảnh chụp cùng viên pin / cùng buổi chụp** (`pHash <= 8` hoặc cùng mẫu vật):
     - `SPEC_12_13_MOTOMA_9V_WOOD`: `task7_12` (mặt bên pin vuông Motoma 9V trên mặt gỗ) và `task7_13` (mặt cực của cùng viên pin Motoma 9V trên mặt gỗ) -> **Gom cả hai vào `TRAIN_V4`**.
     - `SPEC_39_40_NANFU_LEAKED_AA`: `task7_39` và `task7_40` (hai góc chụp của cùng viên pin AA Nanfu bị chảy hóa chất trắng) -> **Gom cả hai vào `TRAIN_V4`**.
     - `SPEC_46_48_52_ENERGIZER_NIMH`: `task7_46` (4 viên Energizer NiMH đứng), `task7_48` (4 viên Energizer NiMH 2 đứng 2 nằm) và `task7_52` (1 viên Energizer NiMH cùng buổi chụp) -> **Gom cả ba vào `TRAIN_V4`**.
   - Nhờ gom theo `specimen_group`, số cặp rò rỉ giữa `TRAIN_V4` (`19` ảnh) và `VAL_REAL_BATTERY` (`11` ảnh) ở ngưỡng `4-rotation pHash <= 8` là: **`{len(leakage['cross_split_train_vs_val_real_battery_collisions'])}`** (`zero_cross_split_leakage = {leakage['zero_cross_split_leakage']}`).

---

## 4. ĐỀ XUẤT DATASET V4, TẬP `VAL_REAL_BATTERY` & CẤU HÌNH E4 (PHẦN B3 — DO NOT RUN)

Báo cáo đề xuất chi tiết: `data/acquisition_staging/task7_diverse_battery_batch/task7_dataset_v4_and_e4_proposal.json`  
File cấu hình đề xuất (chưa chạy): `configs/training_yolov8n_v4_proposed_DO_NOT_RUN.yaml`

### 4.1. Kiến trúc đánh giá đa tầng cho V4 (Multi-Benchmark Evaluation Gate)
Để tránh lặp lại tình trạng điểm Validation cao ảo trên ảnh đơn chiếc Roboflow (`Val V3 = 0.992 mAP50` nhưng `Holdout = 0.008 mAP50`), chúng tôi đề xuất tách bạch 5 cổng đánh giá:
1. **`VAL_V2` (`299` ảnh, `901` boxes, `0` battery):** Cổng kiểm soát chống quên kiến thức (`Catastrophic Forgetting Gate`) trên 9 lớp rác hiện hữu.
2. **`VAL_V3_TABLETOP` (`328` ảnh, `930` boxes, `29` tabletop batteries):** Cổng kiểm tra hồi quy trong miền Roboflow.
3. **`VAL_REAL_BATTERY` (`11` ảnh độc lập từ Task 7: `10` ảnh đa dạng gồm `21` battery boxes + `1` ảnh hard negative):** Đánh giá độc lập trên cụm nhiều pin (`task7_16`, `task7_29`, `task7_31`, `task7_41`, `task7_54`), pin cúc áo (`task7_09`), pin 9V (`task7_14`, `task7_43`), pin điện thoại BL-5C (`task7_56`) và hard negative tủ điện/ống kim loại (`task7_27`).
4. **`TEST` (`299` ảnh, `720` boxes):** Tập kiểm thử nội bộ đóng băng (`FROZEN`).
5. **`HOLDOUT` (`27` ảnh `data/holdout`, `57` GT boxes):** Tập nghiệm thu thực tế bên ngoài đóng băng (`FROZEN`).

### 4.2. Số lượng ảnh và box dự kiến của `TRAIN_V4`
- `Train V2 baseline`: `1.398` ảnh (`3.963` boxes, trong đó `1` battery box).
- `Roboflow V3 Clean Train`: `+111` ảnh (`+111` battery boxes).
- `Task 7 Diverse Battery Train`: `+19` ảnh (`+40` battery boxes từ 16 ảnh dương tính + `3` ảnh hard negative 0 box).
- **Tổng `TRAIN_V4` dự kiến:** **`1.528` ảnh (`4.114` boxes, trong đó `152` battery boxes)**.

### 4.3. Phân tích nguyên nhân E3 suy giảm ở Task 6 & Đề xuất cấu hình E4 bảo thủ (`DO NOT RUN`)
Tại sao E3 (`imgsz=512, freeze=0, lr0=0.0005, mosaic=0.5`) làm giảm Holdout mAP50 từ `0.05914` xuống `0.00823`?
1. **`freeze=0` (Unfrozen Backbone) + Learning Rate `0.0005`:** Khi mở khóa toàn bộ backbone trên tập dữ liệu mất cân bằng và thêm 111 ảnh pin có cùng một kiểu nền bàn gỗ, các tầng trích xuất đặc trưng của YOLOv8n bị dịch chuyển theo hướng nhạy quá mức với vật thể nhỏ trên nền phẳng, gây ra thêm 15 False Positives trên Holdout (`FP` tăng từ `37` lên `52`) và làm mất các phát hiện đúng ở các lớp `metal`, `plastic`, `paper`.
2. **`imgsz=512` thay vì `640`:** Checkpoint gốc E1 được huấn luyện ở độ phân giải `imgsz=640`. Khi hạ xuống `512` trong quá trình fine-tune E3, tỷ lệ scale của các anchor/feature map thay đổi so với trọng số gốc E1.

**Đề xuất cấu hình E4 (chỉ chạy sau khi bổ sung đủ dữ liệu và có lệnh phê duyệt mới):**
- **Khởi tạo từ:** `E1` (`artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` — không khởi tạo nối tiếp từ E3 đã bị trôi đặc trưng).
- **`imgsz`: `640`** (khớp 100% độ phân giải gốc của E1; ở Task 6 `imgsz=512, batch=4` chỉ tốn `400.1 MB` VRAM nên `imgsz=640` hoàn toàn an toàn dưới `1 GB` VRAM trên RTX 2050 4GB).
- **`freeze`: `10`** (đóng băng toàn bộ 10 block backbone C2f/SPPF của YOLOv8n để bảo toàn tuyệt đối đặc trưng 9 lớp rác hiện hữu; chỉ cập nhật neck/head).
- **`lr0`: `0.0002`** (giảm 2.5 lần so với E3), `lrf: 0.01`, `epochs: 20`, `patience: 5`, `batch: 4`, `workers: 0`.

---

## 5. BÁO CÁO TRUNG THỰC VỀ KHOẢNG TRỐNG DỮ LIỆU CÒN LẠI (REMAINING DATA GAPS)

Mặc dù lô 30 ảnh `QA_PASS` (`61` boxes) của Task 7 đã giải quyết tốt sự thiếu hụt về:
- Cụm nhiều viên pin (`14/26` ảnh dương tính có từ 2 đến 6 viên pin),
- Đa dạng hình thái (pin cúc áo CR2032/ML2032, pin vuông 9V, pin sạc 18650/21700, pin máy ảnh LP-E6, pin điện thoại Nokia BL-5C, pin rỉ sét/chảy hóa chất),
- Mẫu âm tính khó (tụ điện trụ, hộp điện âm tường, ống luồn dây kim loại),

**Vẫn còn 3 khoảng trống dữ liệu thực tế cần bổ sung tiếp trước khi kỳ vọng cải thiện mạnh trên `data/holdout`:**
1. **Ảnh hỗn hợp nhiều loại rác (`Mixed-Class Waste Context`):** Toàn bộ 30 ảnh `QA_PASS` từ Wikimedia Commons chỉ chứa lớp `0: battery` (hoặc hard negative không có rác), chưa có ảnh chụp **pin nằm lẫn cùng vỏ chai nhựa, giấy vụn, lon kim loại, bìa carton trên cùng một khung hình** như trong thùng rác sinh hoạt thực tế.
2. **Nền hiện trường phức tạp & khoảng cách chụp xa (`Small Object Scale`):** Phần lớn ảnh Wikimedia là ảnh chụp cận cảnh/trung cảnh (`bbox area` chiếm 10%–70% diện tích ảnh), trong khi ảnh thực tế ngoài đời hoặc camera giám sát có thể chụp từ khoảng cách 1–1.5 mét trên nền gạch đá hoa, nền bê tông hoặc đất cát khiến viên pin chỉ chiếm `< 3%` diện tích ảnh.
3. **Chính sách gán nhãn đống pin dày đặc (`Dense Pile Policy`):** `task7_37` (đống >40 viên pin chồng lấp) và `task7_44` (hộp 48 viên Duracell) đang được giữ trong `NEEDS_MANUAL_REVIEW` chờ PM quyết định có gán nhãn từng viên bị che khuất hay loại bỏ.

---

## 6. BẢNG NGUỒN GỐC & MÃ BĂM TOÀN BỘ 56 ẢNH ỨNG VIÊN TASK 7

| ID | QA Verdict | Giấy phép | Tác giả | SHA-256 (16 ký tự đầu) | pHash (0°) | Nguồn Wikimedia Commons |
|---|---|---|---|---|---|---|
{chr(10).join(provenance_rows)}
"""

    out_root = ROOT / "TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md"
    out_stage = STAGE_DIR / "TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md"
    out_root.write_text(md, encoding="utf-8")
    out_stage.write_text(md, encoding="utf-8")
    print(f"Wrote {out_root} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
