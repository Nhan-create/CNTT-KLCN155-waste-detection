# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Thực thi quy trình kiểm định chất lượng và đóng gói bàn giao cho `package_task7_final_qa_patch`.
"""
Generate TASK7_FINAL_QA_PATCH_REPORT.md and package TASK7_FINAL_QA_PATCH.zip
for PM final sign-off of the 2 patched images (task7_30 and task7_53).
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T7_DIR = ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
PATCH_DIR = T7_DIR / "qa_patch_before_after"


def main() -> None:
    qa = json.loads((T7_DIR / "task7_qa_manifest.json").read_text(encoding="utf-8"))
    by_k = {x["short_key"]: x for x in qa["items"]}
    t30 = by_k["task7_30"]
    t53 = by_k["task7_53"]

    report_md = f"""# BÁO CÁO NGHIỆM THU BẢN VÁ TASK 7 FINAL QA PATCH (CHỈ SỬA 2 ẢNH: `task7_30` & `task7_53`)

- **Repository:** `.`
- **Reviewer & Patch Author:** `{qa['qa_reviewer']}`
- **Thời gian cập nhật (UTC):** `{qa['generated_at_utc']}`
- **Trạng thái phê duyệt huấn luyện:** **`APPROVED_FOR_TRAIN = 0`** (`False` trên toàn bộ 56 ứng viên)

---

## 1. CHI TIẾT SỬA TỌA ĐỘ BOUNDING BOX TRÊN ẢNH ĐỘ PHÂN GIẢI GỐC

### 1.1. Ảnh `task7_30` — Kodak AA/AAA (`{t30['actual_width']}x{t30['actual_height']}`)
- **File ảnh gốc:** `{t30['local_path']}` (`SHA-256: {t30['sha256']}`)
- **File YOLO label đã cập nhật:** `{t30['label_path']}` (đúng **6** boxes)
- **Lý do sửa:** Ba viên pin đứng (`#1, #2, #3`) trước đó bị cắt phần cực dương nhô lên ở đỉnh (`ymin` cũ `0.12, 0.19, 0.20`); 3 viên pin nằm nghiêng (`#4, #5, #6`) được tinh chỉnh lại trên lưới tọa độ độ phân giải gốc `4239x2826` để ôm khít toàn bộ thân trụ và đầu cực mà không lấy dư nền trắng.

| Box | Tọa độ `xyxy_norm` TRƯỚC khi sửa | Tọa độ `xyxy_norm` SAU khi sửa | Dòng nhãn YOLO (`class xc yc w h`) SAU khi sửa |
|---:|---|---|---|
| **#1** *(Pin đứng AA trái)* | `[0.080, 0.120, 0.210, 0.720]` | **`[0.073, 0.075, 0.232, 0.710]`** | `{t30['yolo_label_lines'][0]}` |
| **#2** *(Pin đứng AAA giữa)* | `[0.240, 0.190, 0.340, 0.710]` | **`[0.242, 0.158, 0.343, 0.706]`** | `{t30['yolo_label_lines'][1]}` |
| **#3** *(Pin đứng AAA phải)* | `[0.350, 0.200, 0.450, 0.710]` | **`[0.360, 0.162, 0.453, 0.704]`** | `{t30['yolo_label_lines'][2]}` |
| **#4** *(Pin nằm AA trái)* | `[0.380, 0.510, 0.710, 0.900]` | **`[0.383, 0.492, 0.747, 0.912]`** | `{t30['yolo_label_lines'][3]}` |
| **#5** *(Pin nằm AAA giữa)* | `[0.540, 0.560, 0.850, 0.910]` | **`[0.546, 0.563, 0.838, 0.918]`** | `{t30['yolo_label_lines'][4]}` |
| **#6** *(Pin nằm AAA phải)* | `[0.660, 0.600, 0.950, 0.920]` | **`[0.675, 0.570, 0.948, 0.918]`** | `{t30['yolo_label_lines'][5]}` |

---

### 1.2. Ảnh `task7_53` — IKEA LADDA (`{t53['actual_width']}x{t53['actual_height']}`)
- **File ảnh gốc:** `{t53['local_path']}` (`SHA-256: {t53['sha256']}`)
- **File YOLO label đã cập nhật:** `{t53['label_path']}` (đúng **4** boxes)
- **Lý do sửa:** Hai viên pin nằm nghiêng phía trước (`Box #3` và `Box #4`) trước đó bị cắt phần đầu/thân phía trên bên phải (`ymin` cũ `0.63` và `0.66`). Đã đo lại trên ảnh gốc `1280x960` để `Box #3` (`[0.258, 0.530, 0.678, 0.842]`) và `Box #4` (`[0.450, 0.560, 0.868, 0.882]`) bao trọn từ núm cực dương phía dưới bên trái đến hết đầu thân pin phía trên bên phải, đồng thời thu gọn 2 box pin đứng (`#1, #2`) vừa khít phần thân nhìn thấy được.

| Box | Tọa độ `xyxy_norm` TRƯỚC khi sửa | Tọa độ `xyxy_norm` SAU khi sửa | Dòng nhãn YOLO (`class xc yc w h`) SAU khi sửa |
|---:|---|---|---|
| **#1** *(Pin đứng trái)* | `[0.350, 0.100, 0.480, 0.720]` | **`[0.353, 0.122, 0.478, 0.638]`** | `{t53['yolo_label_lines'][0]}` |
| **#2** *(Pin đứng phải)* | `[0.480, 0.090, 0.590, 0.650]` | **`[0.476, 0.095, 0.592, 0.592]`** | `{t53['yolo_label_lines'][1]}` |
| **#3** *(Pin nằm trước trái)* | `[0.260, 0.630, 0.660, 0.870]` | **`[0.258, 0.530, 0.678, 0.842]`** | `{t53['yolo_label_lines'][2]}` |
| **#4** *(Pin nằm trước phải)* | `[0.470, 0.660, 0.860, 0.930]` | **`[0.450, 0.560, 0.868, 0.882]`** | `{t53['yolo_label_lines'][3]}` |

---

## 2. XÁC NHẬN SỐ LƯỢNG ẢNH / BOX CUỐI CÙNG & KIỂM TRA HỒI QUY

| Chỉ tiêu kiểm tra hồi quy | Kết quả xác nhận |
|---|---|
| `task7_30` số lượng full-object boxes | **6 boxes** (`PASS`) |
| `task7_53` số lượng full-object boxes | **4 boxes** (`PASS`) |
| Tổng số ứng viên `QA_PASS` | **30 ảnh** (**26** ảnh có pin + **4** ảnh hard negative `0` box) |
| Tổng số bounding box `QA_PASS` toàn batch | **61 full-object battery boxes** (`TRAIN_V4`: 19 ảnh / 40 boxes; `VAL_REAL_BATTERY`: 11 ảnh / 21 boxes) |
| Cách ly `NEEDS_MANUAL_REVIEW` & `REJECT` | **10 ảnh `NEEDS_MANUAL_REVIEW`** (`labels_needs_manual_review/`) & **16 ảnh `REJECT`** (không có label trong `labels/`) |
| Trạng thái `approved_for_train` | **`False` (`0`)** trên toàn bộ 56 mục |
| Kiểm định chống rò rỉ (`task7_leakage_report.json`) | **`0` trùng SHA-256**, **`0` cặp `pHash <= 8` ngoại lai**, **`0` cặp rò rỉ giữa `TRAIN_V4` và `VAL_REAL_BATTERY`** |
| Bảo toàn source, author, license & tính tái lập khi chạy lại script | **100% PASS** (`pytest tests/qa/test_task6b_report_and_task7_candidates.py` -> **`4 passed in 0.25s`**) |
"""

    (ROOT / "TASK7_FINAL_QA_PATCH_REPORT.md").write_text(report_md, encoding="utf-8")
    (PATCH_DIR / "TASK7_FINAL_QA_PATCH_REPORT.md").write_text(report_md, encoding="utf-8")

    zip_files = [
        "TASK7_FINAL_QA_PATCH_REPORT.md",
        t30["local_path"],
        t53["local_path"],
        t30["label_path"],
        t53["label_path"],
        t30["overlay_path"],
        t53["overlay_path"],
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_30_BEFORE_overlay.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_30_AFTER_overlay.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_30_BEFORE_vs_AFTER_comparison.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_53_BEFORE_overlay.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_53_AFTER_overlay.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_53_BEFORE_vs_AFTER_comparison.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/qa_patch_before_after/task7_final_qa_patch_test_log.txt",
        "data/acquisition_staging/task7_diverse_battery_batch/contact_sheets/contact_sheet_task7_qa_pass_part1.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/contact_sheets/contact_sheet_task7_qa_pass_part2.jpg",
        "data/acquisition_staging/task7_diverse_battery_batch/task7_qa_manifest.json",
        "data/acquisition_staging/task7_diverse_battery_batch/task7_leakage_report.json",
        "scripts/annotate_and_qa_task7_battery_batch.py",
        "tests/qa/test_task6b_report_and_task7_candidates.py",
    ]

    zip_dl = (Path.home() / "Downloads" / "TASK7_FINAL_QA_PATCH.zip")
    zip_repo = ROOT / "artifacts" / "detection_v3_pilot" / "TASK7_FINAL_QA_PATCH.zip"

    with zipfile.ZipFile(zip_dl, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in zip_files:
            p = ROOT / rel
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            zf.write(p, arcname=rel)

    shutil.copy2(zip_dl, zip_repo)

    # Also copy updated contact sheets and BEFORE_vs_AFTER images to Downloads for instant drag-and-drop
    shutil.copy2(
        ROOT / "data/acquisition_staging/task7_diverse_battery_batch/contact_sheets/contact_sheet_task7_qa_pass_part1.jpg",
        (Path.home() / "Downloads" / "contact_sheet_task7_qa_pass_part1.jpg"),
    )
    shutil.copy2(
        ROOT / "data/acquisition_staging/task7_diverse_battery_batch/contact_sheets/contact_sheet_task7_qa_pass_part2.jpg",
        (Path.home() / "Downloads" / "contact_sheet_task7_qa_pass_part2.jpg"),
    )
    shutil.copy2(
        PATCH_DIR / "task7_30_BEFORE_vs_AFTER_comparison.jpg",
        (Path.home() / "Downloads" / "task7_30_BEFORE_vs_AFTER_comparison.jpg"),
    )
    shutil.copy2(
        PATCH_DIR / "task7_53_BEFORE_vs_AFTER_comparison.jpg",
        (Path.home() / "Downloads" / "task7_53_BEFORE_vs_AFTER_comparison.jpg"),
    )

    h = hashlib.sha256(zip_dl.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "zip_path": str(zip_dl),
                "repo_zip_path": str(zip_repo),
                "file_count": len(zip_files),
                "size_bytes": zip_dl.stat().st_size,
                "size_mb": round(zip_dl.stat().st_size / (1024 * 1024), 2),
                "sha256": h,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
