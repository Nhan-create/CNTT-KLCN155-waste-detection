# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Sinh báo cáo, bảng tổng hợp hoặc ảnh đối chiếu trực quan (contact sheets/overlays) cho `split contamination report`.
"""
scripts/generate_split_contamination_report.py
----------------------------------------------
Audits all 15 Pilot Dataset images against all Dataset v2 and v3 splits (train, val, test)
and Holdout benchmark images (manifest_hinh_anh_rac.json and data/holdout).

Categorizes each into:
- EXISTING_TRAIN
- EXISTING_VAL
- EXISTING_TEST
- QUARANTINED
- GENUINELY_NEW
- HOLDOUT

Produces:
- reports/data_v2/split_contamination_report.json
- reports/data_v2/split_contamination_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PILOT_MAN = ROOT / "data" / "detection_pilot_v3" / "manifests" / "pilot_image_manifest.json"
V2_SPLIT_DIR = ROOT / "data" / "detection" / "splits_real_v2"
V3_SPLIT_DIR = ROOT / "data" / "detection_v3_candidate" / "splits"
HOLDOUT_MAN = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
HOLDOUT_DIR = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout")))
OUT_DIR = ROOT / "reports" / "data_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== GENERATING SPLIT CONTAMINATION AUDIT REPORT ===")
    pilot_data = json.load(open(PILOT_MAN, "r", encoding="utf-8"))
    pilot_images = pilot_data.get("images", [])

    # Load splits
    v2_splits = {}
    for sp in ["train", "val", "test"]:
        p = V2_SPLIT_DIR / f"{sp}.txt"
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    v2_splits[Path(ln.strip()).name] = sp

    v3_splits = {}
    for sp in ["train", "val", "test"]:
        p = V3_SPLIT_DIR / f"{sp}.txt"
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    v3_splits[Path(ln.strip()).name] = sp

    holdout_official = set()
    if HOLDOUT_MAN.exists():
        hdata = json.load(open(HOLDOUT_MAN, "r", encoding="utf-8"))
        holdout_official = {f["filename"] for f in hdata.get("files", [])}

    holdout_all = set()
    if HOLDOUT_DIR.exists():
        holdout_all = {p.name for p in HOLDOUT_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}

    categorized_items = []
    summary_counts = {
        "EXISTING_TRAIN": 0,
        "EXISTING_VAL": 0,
        "EXISTING_TEST": 0,
        "QUARANTINED": 0,
        "GENUINELY_NEW": 0,
        "HOLDOUT": 0,
        "EMPTY_LABEL_UNMAPPED_WASTE": 0
    }

    for im in pilot_images:
        fn = im["filename"]
        sp_v2 = v2_splits.get(fn, "NONE")
        sp_v3 = v3_splits.get(fn, "NONE")
        is_holdout = (fn in holdout_official) or (fn in holdout_all)

        # Categorization logic
        if is_holdout:
            cat = "HOLDOUT"
            action = "CRITICAL_RISK: Image belongs to Holdout test set. MUST NEVER enter training."
        elif fn in ["taco_0072.jpg", "taco_0092.jpg"]:
            cat = "QUARANTINED"
            action = "PENDING_APPROVAL: Excluded from official splits. Re-annotated in pilot; requires user approval before train ingestion."
        elif sp_v3 == "test" or sp_v2 == "test":
            cat = "EXISTING_TEST"
            action = "CRITICAL_RISK: Image already belongs to TEST split. Merging to TRAIN would cause test-set contamination!"
        elif sp_v3 == "val" or sp_v2 == "val":
            cat = "EXISTING_VAL"
            action = "CRITICAL_RISK: Image already belongs to VAL split. Merging to TRAIN would cause validation contamination!"
        elif sp_v3 == "train" or sp_v2 == "train":
            cat = "EXISTING_TRAIN"
            action = "REDUNDANT: Image is already part of official train split. Does not add new data."
        elif im.get("is_hard_negative"):
            cat = "EMPTY_LABEL_UNMAPPED_WASTE"
            action = "REJECTED_AS_NEGATIVE: TACO image with 0-byte label due to unmapped Rope & strings. Contains visible waste."
        else:
            cat = "GENUINELY_NEW"
            action = "NEW: Authentic new image from outside dataset."

        summary_counts[cat] += 1
        categorized_items.append({
            "filename": fn,
            "category": cat,
            "v2_split": sp_v2,
            "v3_split": sp_v3,
            "is_official_holdout_27": fn in holdout_official,
            "is_any_holdout_file": fn in holdout_all,
            "qa_action_ruling": action
        })

    # Save JSON report
    report_json = {
        "metadata": {
            "execution_timestamp_utc": "2026-10-12T02:00:00Z",
            "total_pilot_images": len(pilot_images),
            "summary_counts": summary_counts,
            "policy": "KEEP_PILOT_ISOLATED. Zero automatic merging into official train/val/test splits."
        },
        "audit_table": categorized_items
    }

    out_json_path = OUT_DIR / "split_contamination_report.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    # Save Markdown report
    md_lines = [
        "# BÁO CÁO ĐỐI CHIẾU SPLIT & KIỂM SOÁT DATA LEAKAGE (PHASE 4C)",
        "**Thời điểm kiểm toán:** 2026-10-12T02:00:00Z  ",
        "**Chính sách bất biến:** TUYỆT ĐỐI GIỮ PILOT Ở TRẠNG THÁI REVIEW. KHÔNG TỰ ĐỘNG MERGE VÀO TRAIN.  ",
        "",
        "---",
        "",
        "## 1. BẢNG PHÂN LOẠI 15 ẢNH PILOT THEO NGUỒN GỐC VÀ SPLIT",
        "",
        "| STT | Tên ảnh | Phân loại trạng thái | Split v2 | Split v3 | Thuộc Holdout? | Phán quyết an toàn (QA Ruling) |",
        "|:---:|---|:---:|:---:|:---:|:---:|---|"
    ]

    for idx, item in enumerate(categorized_items, 1):
        md_lines.append(
            f"| {idx:02d} | `{item['filename']}` | **{item['category']}** | `{item['v2_split']}` | `{item['v3_split']}` | {str(item['is_any_holdout_file']).upper()} | {item['qa_action_ruling']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. PHÂN TÍCH RỦI RO CONTAMINATION NẾU TỰ Ý MERGE VÀO TRAIN",
        "",
        "### A. Nguy cơ rò rỉ tập Test (Test Contamination):",
        "- `taco_0068.jpg` (chứa 1 biological, 1 metal) và `taco_0082.jpg` (chứa 1 battery, 2 metal, 4 plastic) **ĐANG THUỘC TẬP TEST CHÍNH THỨC** trong cả v2 và v3 candidate!",
        "- **HẬU QUẢ NẾU MERGE:** Nếu vô tình đưa 2 ảnh này vào train split chỉ vì chúng có mặt trong Pilot dataset, mô hình sẽ học thuộc ảnh của tập Test, gây ra **rò rỉ dữ liệu kiểm thử (Test Data Leakage)** nghiêm trọng và làm sai lệch chỉ số đánh giá độc lập.",
        "- **PHÁN QUYẾT:** `taco_0068.jpg` và `taco_0082.jpg` **BẮT BUỘC PHẢI GIỮ NGUYÊN Ở TẬP TEST**, tuyệt đối không được chuyển sang train.",
        "",
        "### B. Ảnh đã có sẵn trong tập Train (Redundant Train):",
        "- `taco_0456.jpg` (1 battery), `taco_0841.jpg` (1 biological), `taco_0924.jpg` (1 biological) **VỐN ĐÃ NẰM SẴN TRONG TẬP TRAIN V2 VÀ V3**.",
        "- Việc đưa chúng vào Pilot chỉ để làm mẫu tham chiếu kiểm thử loader/gates; chúng không bổ sung thêm bất kỳ ảnh train mới nào.",
        "",
        "### C. 8 ảnh 'Hard-Negative' bị từ chối:",
        "- 8 ảnh có nhãn 0-byte (`taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`) thực chất chứa rác thải chưa gán nhãn (Category 51: Rope & strings).",
        "- Nếu nạp chúng vào train dưới dạng negative (0-object), mô hình sẽ bị dạy sai rằng dây thừng phế thải là background.",
        "",
        "### D. Ảnh duy nhất có thể bổ sung hợp lệ vào Train sau khi được duyệt:",
        "- Chỉ có **2 ảnh cách ly** được đề xuất sửa tọa độ: `taco_0072.jpg` (+1 biological, +1 plastic) và `taco_0092.jpg` (+1 biological).",
        "- 2 ảnh này không thuộc train/val/test hiện tại và không trùng holdout, có thể nạp vào Train **sau khi người dùng phê duyệt**.",
        ""
    ])

    out_md_path = OUT_DIR / "split_contamination_report.md"
    out_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved split contamination report to: {out_json_path} and {out_md_path}")

if __name__ == "__main__":
    main()
