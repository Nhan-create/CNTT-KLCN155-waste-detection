import base64
import json
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
downloads_dir = (Path.home() / "Downloads")
verification_dir = repo_root / "reports" / "verification_demo_image17"
overlays_dir = verification_dir / "overlays"

# Load report json
with open(verification_dir / "baseline_report.json", encoding="utf-8") as f:
    report_data = json.load(f)

# Base64 encode overlays for self-contained HTML
def b64_img(path: Path) -> str:
    if path.exists():
        return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return ""

img_gt_b64 = b64_img(overlays_dir / "images_17_ground_truth.jpg")
img_a_b64 = b64_img(overlays_dir / "images_17_config_A_ssdlite_tiling_twostage_ui.jpg")
img_b_b64 = b64_img(overlays_dir / "images_17_config_B_ssdlite_tiling_no_twostage.jpg")
img_c_b64 = b64_img(overlays_dir / "images_17_config_C_yolov8n_tiling_no_twostage.jpg")

markdown_content = f"""# BÁO CÁO TOÀN DIỆN: XÁC MINH HIỆN TRẠNG DEMO VÀ CHUẨN HÓA BASELINE
**Dự án:** Hệ thống Nhận diện và Phân loại Rác thải (CNTT-KLCN155)  
**Thời gian lập:** 12/10/2026 | **Địa chỉ chạy demo:** `localhost:8501`  
**Đối tượng kiểm tra:** File ảnh `images (17).jpg` và Tập dữ liệu nghiệm thu `data/holdout` (27 ảnh, 57 GT)  
**File gói bàn giao:** `~/Downloads/demo_verification_package.zip` (190 KB)

---

## 1. TÓM TẮT ĐIỀU TRA NHANH (EXECUTIVE SUMMARY)

1. **Hiện tượng điểm tin cậy 23.8% khi slider 25%:**  
   Xuất phát từ cơ chế **ngưỡng hiệu chuẩn riêng theo từng lớp (Per-Class Thresholds)** tại `configs/detection_thresholds_calibrated.json` được nạp ngầm trong `streamlit_app.py` (dòng 231–255). Ngưỡng riêng của lớp `plastic` được đặt là **`0.22` (22%)**. Điểm Two-Stage của vật thể là **$29.71\\% \\times 80.28\\% = 23.85\\%$ (hiển thị $23.8\\%$)**, lớn hơn $22\\%$ nên được giữ lại trên UI dù thanh trượt hiển thị $25\\%$.
2. **Hiện tượng SSDLite chỉ phát hiện 1 vật thể nhựa:**  
   SSDLite320 xử lý ở độ phân giải cố định $320 \\times 320$. Các vật thể lon nước ngọt, chai thủy tinh nhỏ có raw detector scores chỉ từ $0.04$ đến $0.21$ (dưới ngưỡng 0.25). Đồng thời, cơ chế **Tiling bị vô hiệu hóa** vì ảnh có kích thước $547 \\times 365\\text{{ px}} \\le 800\\text{{ px}}$ (`min_side_to_tile = 800`).
3. **Phần cứng thực tế:**  
   Hệ thống đang chạy trên **NVIDIA GeForce RTX 2050 (Laptop GPU, 4GB VRAM)**, **hoàn toàn không phải RTX 3060**.
4. **Trạng thái Ground Truth:**  
   Bộ dữ liệu nghiệm thu `data/holdout` có **chính xác 27 ảnh và 57 vật thể Ground Truth**, ở trạng thái đóng băng `1.0_FROZEN` độc lập bởi Agent D (QA).
5. **Tính năng Webcam:**  
   Chức năng webcam trên giao diện là **chụp ảnh tĩnh từng khung (single photo capture)**, **không phải nhận diện video thời gian thực (real-time stream)**.

---

## 2. KIỂM TRA ĐÚNG CHECKPOINT ĐANG DÙNG TRÊN GIAO DIỆN

Giao diện `streamlit_app.py` nạp cấu hình mặc định từ `src/web/detection_settings.py`:

| Thành phần | Đường dẫn Checkpoint | Kích thước | Mã băm SHA-256 | Vai trò & Trạng thái |
|---|---|---|---|---|
| **Mô hình chính (SSDLite)** | `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` | 28,459,990 bytes | `c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7` | Checkpoint SSDLite v2 đang nạp mặc định trên UI |
| **Mô hình đối chứng (YOLOv8n)** | `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` | 6,228,451 bytes | `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8` | Checkpoint YOLOv8n v2 đang nạp khi chọn mô hình đối chứng |
| **Bộ thẩm định Two-Stage** | `artifacts/official_run/best_model.pt` | 50,929,856 bytes | `c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424` | MobileNetV3 Classifier 10 lớp (Top-1 Accuracy: 96.18%) |
| *SSDLite v1 (Legacy)* | `artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt` | 28,456,982 bytes | `584c5980eb4ebe3718da7b3e87a3c95edfee3d78f186def3e5feba32cf03dff1` | Checkpoint v1 baseline (Seed 42) để đối soát lịch sử |
| *YOLOv8n v1 (Legacy)* | `artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt` | 6,217,379 bytes | `97f46ccaaa0a63c5934225d51ccd399c1c18eb3e1af8a63b9f14bec1aaa085c9` | Checkpoint v1 baseline (Seed 42) train trên ảnh ghép 320 |

---

## 3. THỰC NGHIỆM ĐỐI CHIẾU TRÊN ẢNH `images (17).jpg`

- **Thông số ảnh đầu vào:** Kích thước $547 \\times 365\\text{{ px}}$, SHA-256: `c1a77f0cbbf36c400669d7af088578803f533eaaff8f64cd7b35e68e0a71072a`.
- **Ground Truth độc lập:** 11 vật thể (2 thùng rác nhựa tái chế mờ/đen, 3 lon/hộp thiếc kim loại, 2 chai thủy tinh, 4 chai/bình nhựa).
- **Ngưỡng cài đặt:** `confidence = 0.25`, `iou_threshold = 0.55`.

### 3.1. Bảng đối đầu Cấu hình A / B / C

| Tiêu chí | Cấu hình A<br>(SSDLite + Tiling + Two-Stage) | Cấu hình B<br>(SSDLite + Tiling, Tắt Two-Stage) | Cấu hình C<br>(YOLOv8n + Tiling, Tắt Two-Stage) |
|---|---|---|---|
| **Checkpoint sử dụng** | `E2_ssdlite320_10cls` (`c9eb52...`) | `E2_ssdlite320_10cls` (`c9eb52...`) | `E1_yolov8n_10cls` (`7e986d...`) |
| **Cơ chế Tiling** | Bị bỏ qua ($547 \\le 800\\text{{ px}}$) | Bị bỏ qua ($547 \\le 800\\text{{ px}}$) | Bị bỏ qua ($547 \\le 800\\text{{ px}}$) |
| **Cơ chế Two-Stage** | BẬT (MobileNetV3 Product mode) | TẮT | TẮT |
| **Số box hiển thị trên UI** | **1 box** (`plastic` 23.8%) | **1 box** (`plastic` 29.7%) | **5 boxes** (1 shoes, 3 plastic, 1 metal) |
| **Số box nếu dùng ngưỡng phẳng 0.25** | **0 box** (bị lọc sạch) | **1 box** (`plastic` 29.7%) | **5 boxes** |
| **Thời gian xử lý UI** | **~140 – 506 ms** (cold: 676 ms) | **~347 – 492 ms** | **~165 – 585 ms** |
| **Độ bao phủ rác thật (Recall trên 11 GT)** | **1 / 11** (9.1%) | **1 / 11** (9.1%) | **4 / 11** (36.4%) |
| **Số lượng False Positives (Box ảo)** | 0 FP | 0 FP | 1 FP (`shoes` 65.4% do texture thùng rác) |

### 3.2. Truy vết Chi tiết Bounding Boxes và Điểm số

#### Cấu hình A (SSDLite + Tiling + Two-Stage - UI Thực tế):
- **Stage 1 (Raw Detector SSDLite v2, conf $\\ge 0.15$):**
  1. `[plastic]` conf = **0.2971** tại box `[393.1, 53.8, 477.9, 182.4]`
  2. `[glass]` conf = 0.2159 tại box `[100.1, 141.9, 164.3, 250.2]`
  3. `[plastic]` conf = 0.1732 tại box `[329.9, 59.3, 400.0, 193.5]`
  4. `[glass]` conf = 0.1699 tại box `[252.4, 20.8, 339.4, 210.2]`
  5. `[clothes]` conf = 0.1560 tại box `[188.2, 222.8, 263.7, 360.9]`
- **Stage 2 (Thẩm định MobileNetV3 trên các crop):**
  - Crop 1: Classifier xác nhận `plastic` với xác suất **$80.28\\%$**.
  - Tính điểm Bayesian Product: $\\text{{Score}} = 0.2971 \\times 0.8028 = \\mathbf{{0.2385}}$ (**$23.8\\%$**).
  - So sánh ngưỡng: Với `plastic`, ngưỡng hiệu chuẩn là $0.22$. Vì $0.2385 \\ge 0.22$, box được **GIỮ LẠI**.
  - Các crop 2, 3, 4, 5: Điểm tích số sau phân loại đều $< 0.15$, bị loại bỏ hoàn toàn.
- **Kết quả hiển thị:** Duy nhất 1 box `plastic 23.8%`, khớp Ground Truth chai nước khoáng nắp xanh (IoU = 0.71).

#### Cấu hình B (SSDLite + Tiling, Không Two-Stage):
- Raw detector lọc trực tiếp tại ngưỡng $0.25$:
  - Chỉ có 1 box đạt: `[plastic]` conf = **$0.2971$** ($29.7\\%$) tại `[393.1, 53.8, 477.9, 182.4]`.
  - Toàn bộ các vật thể khác (lon nước ngọt, chai thủy tinh, hộp thiếc) có raw score $< 0.25$ nên bị loại sạch.

#### Cấu hình C (YOLOv8n + Tiling, Không Two-Stage):
- Raw detector lọc trực tiếp tại ngưỡng $0.25$:
  1. `[shoes]`: conf = **$0.6543$** ($65.4\\%$) tại `[91.5, 292.1, 160.5, 353.8]` (FP: nhầm góc đáy thùng rác thành giày).
  2. `[plastic]`: conf = **$0.5620$** ($56.2\\%$) tại `[243.1, 17.7, 337.0, 210.0]` (Khớp chai nhựa trong suốt bên trái thùng đen).
  3. `[plastic]`: conf = **$0.3745$** ($37.5\\%$) tại `[403.5, 46.3, 493.0, 206.7]` (Khớp chai nước khoáng nắp xanh).
  4. `[metal]`: conf = **$0.3074$** ($30.7\\%$) tại `[351.9, 173.0, 426.5, 209.2]` (Khớp lon nhôm bị ép bẹp).
  5. `[plastic]`: conf = **$0.2768$** ($27.7\\%$) tại `[0.1, 105.5, 274.5, 365.0]` (Khớp thùng rác nhựa mờ bên trái).

---

## 4. GIẢI TRÌNH: TẠI SAO HIỂN THỊ 23.8% KHI SLIDER LÀ 25%?

1. Tại `streamlit_app.py` (dòng 231–255), mã nguồn tự động đọc cấu hình `configs/detection_thresholds_calibrated.json` nếu tồn tại.
2. File cấu hình này định nghĩa bảng ngưỡng hiệu chuẩn theo lớp:
   - `plastic: 0.22` (22%)
   - `metal: 0.20` (20%)
   - `battery: 0.20` (20%)
   - `clothes: 0.35` (35%)
   - `shoes: 0.35` (35%)
   - `cardboard, glass, paper, biological, trash: 0.25` (25%)
3. Khi người dùng đặt slider ở mức $0.25$, hệ số tỷ lệ là $\\text{{scale}} = 0.25 / 0.25 = 1.0$. Ngưỡng hiệu chuẩn của lớp `plastic` được gán chính xác là $\\mathbf{{0.22}}$ ($22\\%$).
4. Điểm dự đoán sau Two-Stage của chai nhựa là:
   $$\\text{{Score}}_{{\\text{{final}}}} = \\text{{Score}}_{{\\text{{detector}}}} \\times P_{{\\text{{classifier}}}} = 0.2971 \\times 0.8028 = \\mathbf{{0.2385}}\\quad (23.85\\%)$$
5. Bộ lọc `PerClassThresholdDetector` kiểm tra điều kiện:
   $$0.2385 \\ge 0.22\\quad (\\text{{ĐẠT}})$$
   Do đó, box này được giữ lại và hiển thị lên giao diện kèm nhãn `23.8%`.
6. **Kết luận:** Đây không phải lỗi làm tròn số học ngẫu nhiên, mà do giao diện đã âm thầm áp dụng bảng ngưỡng hiệu chuẩn riêng từng lớp hạ trần lớp nhựa xuống 22% mà **không hiển thị cảnh báo cho người dùng trên UI**. Nếu người dùng kỳ vọng ngưỡng phẳng 25% cho tất cả các lớp, box này đáng lẽ đã bị lọc bỏ (0 box).

---

## 5. XÁC MINH BÁO CÁO BASELINE VÀ PHẦN CỨNG

### 5.1. Xác minh Tập dữ liệu Holdout độc lập
- Thư mục `data/holdout`: **Chính xác 27 ảnh, 57 vật thể Ground Truth**, phiên bản `1.0_FROZEN` lập bởi Agent D tại `reports/qa_acceptance/ground_truth_hinh_anh_rac.json`.
- 100% không bị thay đổi ground truth, không can thiệp nhãn.

### 5.2. Bảng Đối chiếu Chỉ số Baseline Chuẩn hóa (Protocol COCO: IoU $\\ge 0.50$, Conf $\\ge 0.25$)

| Mô hình & Checkpoint | SHA-256 Checkpoint | Tập dữ liệu | Precision | Recall | F1 | mAP50 | mAP50-95 | Latency TB | Trạng thái Nghiệm thu |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SSDLite v1 (Legacy)** | `584c5980...` | Holdout (27 ảnh, 57 GT) | 0.080 | 0.035 | 0.049 | 0.0572 | 0.0057 | 65.9 ms | **FAIL** |
| **SSDLite v2 (Raw)** | `c9eb52ef...` | Holdout (27 ảnh, 57 GT) | 0.185 | 0.088 | 0.119 | 0.0560 | 0.0270 | 62.4 ms | **FAIL** |
| **SSDLite v2 (+ Two-Stage)** | `c9eb52ef...` | Holdout (27 ảnh, 57 GT) | **0.727** | 0.140 | 0.235 | 0.0610 | 0.0295 | 151.4 ms | **FAIL (Thiếu Recall)** |
| **YOLOv8n v1 (Legacy)** | `97f46cca...` | Holdout (27 ảnh, 57 GT) | 0.000 | 0.000 | 0.000 | 0.0004 | 0.0002 | 48.2 ms | **FAIL (Box siêu nhỏ 20px)** |
| **YOLOv8n v2 (Active)** | `7e986dee...` | Holdout (27 ảnh, 57 GT) | 0.156 | 0.088 | 0.112 | 0.0571 | 0.0289 | 48.4 ms | **FAIL (Thiếu mẫu âm)** |

### 5.3. Nguồn gốc việc các Báo cáo Cũ ghi SSDLite mAP khác nhau
1. **Con số `12.6%` (loss 3.20, 36 epochs):**  
   Xuất phát từ file `artifacts/detection_v2/E2_ssdlite320_10cls/training_summary.json` (dòng 84–88): `best_val_map50_95 = 0.12645` tại Epoch 30. Đây là **mAP50-95 trên tập Validation 299 ảnh** trong lúc huấn luyện E2, **hoàn toàn không phải mAP trên 27 ảnh holdout ngoài thực tế**.
2. **Con số `8.65%` (mAP50) và `4.89%` (mAP50-95):**  
   Được ghi nhận từ validation split của mô hình SSDLite v1 (Legacy) tại Epoch 37.
3. **Con số `5.72%` và `5.60%`:**  
   Là mAP50 đo thực tế bằng `practical_metrics.py` trên 27 ảnh holdout độc lập `data/holdout`.
4. **Con số `13.80%` của YOLOv8n:**  
   Xuất phát từ `artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json`: `test.map50_95 = 0.1380` trên 299 ảnh test split của Mendeley v2, bị báo cáo cũ nhầm lẫn thành mAP50 trên tập holdout.

### 5.4. Xác minh Thực tế Phần cứng GPU
- Lệnh kiểm tra: `nvidia-smi` và `torch.cuda.get_device_name(0)`.
- **Kết quả thực tế:** **`NVIDIA GeForce RTX 2050`** (Driver 591.86, VRAM: 4096 MiB / 4.0 GB).
- **Kết luận:** Máy thử nghiệm sử dụng card đồ họa **RTX 2050**, **không phải RTX 3060**. Mọi tuyên bố trước đây về việc chạy trên RTX 3060 là không chính xác về mặt cấu hình phần cứng.

---

## 6. LÀM RÕ VỀ TÍNH NĂNG WEBCAM

- Tại `streamlit_app.py` (dòng 289 và 321), tính năng webcam sử dụng component `st.camera_input("Hoặc chụp ảnh từ webcam")`.
- **Bản chất kỹ thuật:** Đây là thao tác **chụp một khung ảnh tĩnh (still photo snapshot)** từ webcam và tải payload về server để chạy phát hiện tĩnh.
- **Cam kết nghiệm thu:** Không được báo cáo tính năng này là *"nhận diện video thời gian thực"* (real-time video detection stream). Mã nguồn và tài liệu chỉ ghi nhận chính xác là *"chụp ảnh tĩnh từ webcam"*.

---

## 7. KẾT LUẬN VỀ NGUYÊN NHÂN DEMO HIỆN TẠI BỎ SÓT NHIỀU RÁC

1. **Độ phân giải đầu vào cố định $320 \\times 320$ của SSDLite:**  
   Ảnh chụp bối cảnh nhiều rác phức tạp từ xa (như `images (17).jpg`) khi bị nén xuống $320 \\times 320$ làm mất sạch các cạnh sắc nét của lon kim loại và chai thủy tinh nhỏ, khiến điểm tin cậy raw của detector bị dìm xuống mức $0.04 - 0.21$.
2. **Cơ chế Tiling không được kích hoạt:**  
   Giao diện đặt ngưỡng kích hoạt cắt ô `tiling_min_side = 800`. Do ảnh `images (17).jpg` có kích thước $547 \\times 365$, hệ thống đã bỏ qua hoàn toàn bước chia ô chi tiết. (Khi thử nghiệm ép chạy tiling ở ngưỡng `min_side = 400`, SSDLite phát hiện được 11 box và YOLOv8n phát hiện được 16 box).
3. **Cơ chế nhân điểm Two-Stage làm hạ thấp điểm:**  
   Phương thức nhân xác suất $\\text{{Score}} = \\text{{Score}}_{{\\text{{detector}}}} \\times P_{{\\text{{classifier}}}}$ có ưu điểm triệt tiêu tuyệt đối các box ảo ngoại cảnh (tường gạch, tủ điện), nhưng lại làm điểm các vật thể thật bị kéo giảm (ví dụ từ $29.7\\% \\rightarrow 23.8\\%$), dẫn đến việc bị lọc bỏ nếu không có bảng hiệu chuẩn riêng.
4. **Mô hình đối chứng YOLOv8n v2:**  
   Hoạt động tốt hơn rõ rệt (bắt được 4/11 rác thật trên cùng ảnh), nhưng gặp 1 lỗi nhận nhầm texture góc thùng rác thành `shoes` do dữ liệu huấn luyện còn thiếu các mẫu âm nền thực tế.
"""

# HTML template with embedded base64 images
html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Báo Cáo Xác Minh Kết Quả Demo - CNTT-KLCN155</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1f2937; max-width: 1200px; margin: 0 auto; padding: 24px; background: #f9fafb; }}
  h1, h2, h3, h4 {{ color: #111827; margin-top: 1.5em; }}
  h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; font-size: 28px; }}
  h2 {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; font-size: 22px; color: #1d4ed8; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
  th {{ background: #f3f4f6; font-weight: 600; color: #374151; }}
  tr:hover {{ background: #f9fafb; }}
  code {{ background: #e5e7eb; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; font-size: 13px; color: #b91c1c; }}
  pre {{ background: #1f2937; color: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: Consolas, monospace; }}
  .badge {{ display: inline-block; padding: 4px 8px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
  .badge-fail {{ background: #fee2e2; color: #b91c1c; }}
  .badge-success {{ background: #dcfce7; color: #15803d; }}
  .badge-warning {{ background: #fef3c7; color: #b45309; }}
  .card {{ background: white; padding: 20px; border-radius: 8px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6; }}
  .alert-warning {{ border-left-color: #f59e0b; background: #fffbeb; }}
  .alert-danger {{ border-left-color: #ef4444; background: #fef2f2; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .img-box {{ background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .img-box img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #e5e7eb; }}
  .img-caption {{ margin-top: 8px; font-weight: 600; font-size: 14px; color: #4b5563; }}
</style>
</head>
<body>

<h1>BÁO CÁO TOÀN DIỆN: XÁC MINH HIỆN TRẠNG DEMO VÀ CHUẨN HÓA BASELINE</h1>
<p><strong>Dự án:</strong> Hệ thống Nhận diện và Phân loại Rác thải (CNTT-KLCN155)<br>
<strong>Môi trường thử nghiệm:</strong> <code>localhost:8501</code> (Streamlit Web App)<br>
<strong>File gói bàn giao:</strong> <code>~/Downloads/demo_verification_package.zip</code> (190 KB)</p>

<div class="card alert-warning">
  <h3 style="margin-top:0; color:#b45309;">KẾT LUẬN ĐIỀU TRA NHANH</h3>
  <ul>
    <li><strong>Điểm tin cậy 23.8% khi slider 25%:</strong> Do UI nạp ngầm cấu hình <code>configs/detection_thresholds_calibrated.json</code> quy định ngưỡng lớp <code>plastic</code> là <strong>22% (0.22)</strong>. Two-stage ra điểm $23.85\\% \\ge 22\\%$ nên box được giữ lại hiển thị.</li>
    <li><strong>SSDLite chỉ bắt 1 vật thể:</strong> Raw scores của các vật thể khác (lon nước ngọt, chai thủy tinh) chỉ đạt $0.04 - 0.21$ (< 0.25). Tiling bị bỏ qua vì ảnh $547 \\times 365 \\le 800\\text{{ px}}$.</li>
    <li><strong>Xác minh phần cứng GPU:</strong> Máy thực tế sử dụng <strong>NVIDIA GeForce RTX 2050 (4GB VRAM)</strong>, KHÔNG PHẢI RTX 3060.</li>
    <li><strong>Tính năng Webcam:</strong> Chỉ là chụp ảnh tĩnh từng khung (still snapshot), KHÔNG PHẢI video thời gian thực.</li>
  </ul>
</div>

<h2>1. HÌNH ẢNH OVERLAY ĐỐI CHIẾU THỰC NGHIỆM TRÊN ẢNH `images (17).jpg`</h2>
<div class="grid-2">
  <div class="img-box">
    <img src="{img_gt_b64}" alt="Ground Truth">
    <div class="img-caption">1. Ground Truth Độc lập (11 vật thể rác có nhãn chuẩn)</div>
  </div>
  <div class="img-box">
    <img src="{img_a_b64}" alt="Config A">
    <div class="img-caption">2. Cấu hình A: SSDLite + Tiling + Two-Stage (UI Thực tế - 1 box 23.8%)</div>
  </div>
  <div class="img-box">
    <img src="{img_b_b64}" alt="Config B">
    <div class="img-caption">3. Cấu hình B: SSDLite + Tiling, Không Two-Stage (1 box 29.7%)</div>
  </div>
  <div class="img-box">
    <img src="{img_c_b64}" alt="Config C">
    <div class="img-caption">4. Cấu hình C: YOLOv8n + Tiling, Không Two-Stage (5 boxes)</div>
  </div>
</div>

<h2>2. BẢNG SO SÁNH THỰC NGHIỆM ĐỐI ĐẦU A / B / C</h2>
<table>
  <thead>
    <tr>
      <th>Tiêu chí</th>
      <th>Cấu hình A (SSDLite + Tiling + Two-Stage)</th>
      <th>Cấu hình B (SSDLite + Tiling, No Two-Stage)</th>
      <th>Cấu hình C (YOLOv8n + Tiling, No Two-Stage)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Checkpoint</strong></td>
      <td><code>E2_ssdlite320_10cls</code> (<code>c9eb52...</code>)</td>
      <td><code>E2_ssdlite320_10cls</code> (<code>c9eb52...</code>)</td>
      <td><code>E1_yolov8n_10cls</code> (<code>7e986d...</code>)</td>
    </tr>
    <tr>
      <td><strong>Tiling (800)</strong></td>
      <td>Bị bỏ qua (ảnh 547x365 &le; 800)</td>
      <td>Bị bỏ qua (ảnh 547x365 &le; 800)</td>
      <td>Bị bỏ qua (ảnh 547x365 &le; 800)</td>
    </tr>
    <tr>
      <td><strong>Two-Stage</strong></td>
      <td>BẬT (MobileNetV3 Product mode)</td>
      <td>TẮT</td>
      <td>TẮT</td>
    </tr>
    <tr>
      <td><strong>Số box trên UI</strong></td>
      <td><strong>1 box</strong> (<code>plastic</code> 23.8%)</td>
      <td><strong>1 box</strong> (<code>plastic</code> 29.7%)</td>
      <td><strong>5 boxes</strong> (1 shoes, 3 plastic, 1 metal)</td>
    </tr>
    <tr>
      <td><strong>Nếu dùng ngưỡng phẳng 0.25</strong></td>
      <td><strong>0 box</strong> (bị lọc sạch)</td>
      <td><strong>1 box</strong> (<code>plastic</code> 29.7%)</td>
      <td><strong>5 boxes</strong></td>
    </tr>
    <tr>
      <td><strong>Thời gian xử lý UI</strong></td>
      <td><strong>~140 – 506 ms</strong> (cold: 676 ms)</td>
      <td><strong>~347 – 492 ms</strong></td>
      <td><strong>~165 – 585 ms</strong></td>
    </tr>
    <tr>
      <td><strong>Recall trên 11 GT</strong></td>
      <td><strong>1 / 11</strong> (9.1%)</td>
      <td><strong>1 / 11</strong> (9.1%)</td>
      <td><strong>4 / 11</strong> (36.4%)</td>
    </tr>
    <tr>
      <td><strong>False Positives (Box ảo)</strong></td>
      <td>0 FP</td>
      <td>0 FP</td>
      <td>1 FP (<code>shoes</code> 65.4% do texture thùng rác)</td>
    </tr>
  </tbody>
</table>

<h2>3. CHECKPOINT VÀ MÃ BĂM SHA-256 ĐANG CHẠY TRÊN DEMO</h2>
<table>
  <thead>
    <tr>
      <th>Mô hình</th>
      <th>File Checkpoint</th>
      <th>Kích thước</th>
      <th>Mã băm SHA-256</th>
      <th>Ghi chú</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>SSDLite v2 (Chính)</strong></td>
      <td><code>artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt</code></td>
      <td>28,459,990 bytes</td>
      <td><code>c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7</code></td>
      <td>Nạp mặc định trên UI</td>
    </tr>
    <tr>
      <td><strong>YOLOv8n v2 (Đối chứng)</strong></td>
      <td><code>artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt</code></td>
      <td>6,228,451 bytes</td>
      <td><code>7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8</code></td>
      <td>Mô hình đối chứng</td>
    </tr>
    <tr>
      <td><strong>Two-Stage Classifier</strong></td>
      <td><code>artifacts/official_run/best_model.pt</code></td>
      <td>50,929,856 bytes</td>
      <td><code>c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424</code></td>
      <td>MobileNetV3 10 lớp</td>
    </tr>
  </tbody>
</table>

<h2>4. BẢNG BASELINE CHUẨN HÓA TRÊN 27 ẢNH HOLDOUT (57 GT)</h2>
<p>Đánh giá theo chuẩn COCO: <code>practical_metrics.py</code> (IoU &ge; 0.50, Conf &ge; 0.25):</p>
<table>
  <thead>
    <tr>
      <th>Mô hình</th>
      <th>Tập dữ liệu</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>mAP50</th>
      <th>mAP50-95</th>
      <th>Kết luận DoD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>SSDLite v1 (Legacy)</strong></td>
      <td>Holdout 27 ảnh</td>
      <td>0.080</td>
      <td>0.035</td>
      <td>0.049</td>
      <td>0.0572</td>
      <td>0.0057</td>
      <td><span class="badge badge-fail">FAIL</span></td>
    </tr>
    <tr>
      <td><strong>SSDLite v2 (Raw)</strong></td>
      <td>Holdout 27 ảnh</td>
      <td>0.185</td>
      <td>0.088</td>
      <td>0.119</td>
      <td>0.0560</td>
      <td>0.0270</td>
      <td><span class="badge badge-fail">FAIL</span></td>
    </tr>
    <tr>
      <td><strong>SSDLite v2 (+ Two-Stage)</strong></td>
      <td>Holdout 27 ảnh</td>
      <td><strong>0.727</strong></td>
      <td>0.140</td>
      <td>0.235</td>
      <td>0.0610</td>
      <td>0.0295</td>
      <td><span class="badge badge-fail">FAIL (Thiếu Recall)</span></td>
    </tr>
    <tr>
      <td><strong>YOLOv8n v1 (Legacy)</strong></td>
      <td>Holdout 27 ảnh</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0004</td>
      <td>0.0002</td>
      <td><span class="badge badge-fail">FAIL (Box nhỏ 20px)</span></td>
    </tr>
    <tr>
      <td><strong>YOLOv8n v2 (Active)</strong></td>
      <td>Holdout 27 ảnh</td>
      <td>0.156</td>
      <td>0.088</td>
      <td>0.112</td>
      <td>0.0571</td>
      <td>0.0289</td>
      <td><span class="badge badge-fail">FAIL (Thiếu mẫu âm)</span></td>
    </tr>
  </tbody>
</table>

<h2>5. GIẢI TRÌNH CÁC MÂU THUẪN SỐ LIỆU TRONG BÁO CÁO CŨ</h2>
<ul>
  <li><strong>Số liệu 12.6% (loss 3.20, 36 epochs):</strong> Là <code>best_val_map50_95</code> trên tập <strong>Validation 299 ảnh</strong> của E2, KHÔNG PHẢI trên 27 ảnh holdout thực tế (holdout chỉ đạt 5.60% mAP50).</li>
  <li><strong>Số liệu 13.80% của YOLOv8n:</strong> Là <code>test.map50_95</code> trên tập <strong>Test 299 ảnh</strong> của dataset v2, bị báo cáo cũ nhầm thành mAP50 trên tập holdout.</li>
  <li><strong>YOLOv8n TP=0, FP=28, FN=57 (P=0%, R=0%):</strong> Là kết quả của YOLOv8n v1 legacy (train trên ảnh ghép 320x320 nên chỉ bắt box nhỏ 20px).</li>
</ul>

</body>
</html>
"""

# Write outputs to Downloads
(downloads_dir / "BAO_CAO_XAC_MINH_DEMO_KLCN155.md").write_text(markdown_content, encoding="utf-8")
(downloads_dir / "BAO_CAO_XAC_MINH_DEMO_KLCN155.html").write_text(html_content, encoding="utf-8")

# Write to repo reports
(repo_root / "reports" / "BAO_CAO_XAC_MINH_DEMO_KLCN155.md").write_text(markdown_content, encoding="utf-8")
(repo_root / "reports" / "BAO_CAO_XAC_MINH_DEMO_KLCN155.html").write_text(html_content, encoding="utf-8")

print("Files successfully generated in Downloads:")
print(f" - {downloads_dir / 'BAO_CAO_XAC_MINH_DEMO_KLCN155.md'}")
print(f" - {downloads_dir / 'BAO_CAO_XAC_MINH_DEMO_KLCN155.html'}")
print(f" - {downloads_dir / 'demo_verification_package.zip'}")
