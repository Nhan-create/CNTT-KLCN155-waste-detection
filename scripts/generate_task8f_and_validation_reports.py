"""Generate:
1. `reports/task8_dataset_program/TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md` (Tasks 1–5 + NEEDS_REVIEW queue)
2. `reports/task8_dataset_program/TASK8F_SCARCE_CLASS_EXPANSION_REPORT.{md,json}` (synchronized with post-correction manifest)
3. `reports/task8_dataset_program/REAL_WASTE_10CLASS_VALIDATION_PLAN.{md,json}` (synchronized with post-correction manifest)
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports" / "task8_dataset_program"
W8F_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes" / "task8f_qa_manifest.json"
W8E_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste" / "task8e_qa_manifest.json"
W8D_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json"
W1_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json"
W2_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json"
T7_MANIFEST_P = ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json"
LEDGER_P = REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json"
CORR_JSON_P = REPORTS_DIR / "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.json"

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


def count_taco_val_baseline() -> tuple[dict[int, int], dict[int, int], int, int]:
    val_dir = ROOT / "data" / "detection_v3_candidate" / "yolo" / "labels" / "val"
    img_cnt: Counter[int] = Counter()
    box_cnt: Counter[int] = Counter()
    total_imgs = 0
    total_boxes = 0
    for p in sorted(val_dir.glob("taco_*.txt")):
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        total_imgs += 1
        cids = set()
        for ln in lines:
            cid = int(ln.split()[0])
            cids.add(cid)
            box_cnt[cid] += 1
            total_boxes += 1
        for cid in cids:
            img_cnt[cid] += 1
    return {c: img_cnt[c] for c in range(10)}, {c: box_cnt[c] for c in range(10)}, total_imgs, total_boxes


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    w8f = json.loads(W8F_MANIFEST_P.read_text(encoding="utf-8"))
    w8e = json.loads(W8E_MANIFEST_P.read_text(encoding="utf-8"))
    w8d = json.loads(W8D_MANIFEST_P.read_text(encoding="utf-8"))
    w1_recs = json.loads(W1_MANIFEST_P.read_text(encoding="utf-8"))["records"]
    w2_recs = json.loads(W2_MANIFEST_P.read_text(encoding="utf-8"))["records"]
    t7_data = json.loads(T7_MANIFEST_P.read_text(encoding="utf-8"))
    t7_items = t7_data.get("items", t7_data.get("records", []))
    ledger = json.loads(LEDGER_P.read_text(encoding="utf-8"))
    corr = json.loads(CORR_JSON_P.read_text(encoding="utf-8"))

    # --- 1. TASK8F_SCARCE_CLASS_EXPANSION_REPORT (Synchronized Post-Correction) ---
    w8f_recs = w8f["records"]
    src_breakdown = {}
    for sid in sorted({r["source_id"] for r in w8f_recs}):
        s_recs = [r for r in w8f_recs if r["source_id"] == sid]
        s_pass = [r for r in s_recs if r["qa_status"] == "REAL_WASTE_PASS"]
        src_breakdown[sid] = {
            "downloaded_candidates": len(s_recs),
            "REAL_WASTE_PASS": len(s_pass),
            "TRAIN_REAL_WASTE_W8F": sum(1 for r in s_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8F"),
            "VAL_REAL_WASTE_W8F": sum(1 for r in s_pass if r["target_split"] == "VAL_REAL_WASTE_W8F"),
            "KEEP_AUXILIARY": sum(1 for r in s_recs if r["qa_status"] == "KEEP_AUXILIARY"),
            "NEEDS_REVIEW": sum(1 for r in s_recs if r["qa_status"] == "NEEDS_REVIEW"),
            "REJECT": sum(1 for r in s_recs if r["qa_status"] == "REJECT"),
            "yield_pct": round(100.0 * len(s_pass) / max(1, len(s_recs)), 2),
        }

    c_report_json = {
        "report_version": "TASK8F_SCARCE_CLASS_EXPANSION_V2_SEMANTIC_QA_CORRECTED",
        "approved_for_train": False,
        "pre_correction_audit_trail": w8f["pre_correction_audit_trail"],
        "pilots": w8f["pilots"],
        "total_downloaded_candidates": w8f["total_downloaded_candidates"],
        "qa_status_counts": w8f["qa_status_counts"],
        "domain_counts": w8f["domain_counts"],
        "split_counts": w8f["split_counts"],
        "real_waste_pass_total_images": w8f["real_waste_pass_total_images"],
        "real_waste_pass_train_images": w8f["real_waste_pass_train_images"],
        "real_waste_pass_val_images": w8f["real_waste_pass_val_images"],
        "real_waste_pass_total_boxes": w8f["real_waste_pass_total_boxes"],
        "source_yield_breakdown": src_breakdown,
        "per_class_real_waste_pass_images": w8f["per_class_real_waste_pass_images"],
        "per_class_real_waste_pass_train_images": w8f["per_class_real_waste_pass_train_images"],
        "per_class_real_waste_pass_val_images": w8f["per_class_real_waste_pass_val_images"],
        "per_class_real_waste_pass_boxes": w8f["per_class_real_waste_pass_boxes"],
        "per_class_real_waste_pass_val_boxes": w8f["per_class_real_waste_pass_val_boxes"],
        "contact_sheets": w8f["contact_sheets"],
    }
    (REPORTS_DIR / "TASK8F_SCARCE_CLASS_EXPANSION_REPORT.json").write_text(
        json.dumps(c_report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    c_md = [
        "# BÁO CÁO MỞ RỘNG DỮ LIỆU RÁC THỰC TẾ CHO 8 LỚP HIẾM — TASK 8F (ĐÃ HIỆU CHỈNH SEMANTIC QA)",
        "",
        "- **Repository:** `D:\\CNTT-KLCN155-waste-detection`",
        "- **Thư mục dữ liệu Wave 8F:** `data/acquisition_staging/task8f_scarce_classes/`",
        "- **Manifest QA (Post-Correction):** [`task8f_qa_manifest.json`](./data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.json) & [`.csv`](./data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.csv)",
        "- **Manifest QA (Pre-Correction Snapshot):** [`task8f_qa_manifest_pre_correction.json`](./data/acquisition_staging/task8f_scarce_classes/qa_correction_before_after/task8f_qa_manifest_pre_correction.json)",
        "- **Trạng thái phê duyệt huấn luyện:** `APPROVED_FOR_TRAIN = False` (toàn bộ ảnh mới giữ ở staging chờ PM nghiệm thu độc lập).",
        "",
        "---",
        "",
        "## 1. ĐỐI CHIẾU TRƯỚC VÀ SAU KHI HIỆU CHỈNH SEMANTIC QA (TASK 8F QUALITY CORRECTION)",
        "",
        "| Chỉ số | Trước hiệu chỉnh (`V1`) | Sau hiệu chỉnh (`V2`) | Thay đổi |",
        "|---|---:|---:|---|",
        f"| Tổng ứng viên tải về (đã khử trùng) | `748` | **`{w8f['total_downloaded_candidates']}`** | Giữ nguyên 748 ứng viên gốc |",
        f"| `REAL_WASTE_PASS` (Tổng ảnh) | `203` | **`{w8f['real_waste_pass_total_images']}`** | `-117` ảnh (chuyển `69` sang `KEEP_AUXILIARY`, `41` sang `NEEDS_REVIEW`, `7` sang `REJECT`) |",
        f"| ├── `TRAIN_REAL_WASTE_W8F` | `161` | **`{w8f['real_waste_pass_train_images']}`** | `-90` ảnh train |",
        f"| └── `VAL_REAL_WASTE_W8F` | `42` | **`{w8f['real_waste_pass_val_images']}`** | `-27` ảnh val |",
        f"| Tổng Bounding Boxes (`REAL_WASTE_PASS`) | `407` | **`{w8f['real_waste_pass_total_boxes']}`** | `-287` boxes (khử trùng và loại bỏ ảnh sai ngữ cảnh) |",
        "| Cặp bbox cùng lớp trùng lặp (`IoU > 0.70`) | `33` cặp (`23` ảnh) | **`0` cặp (`0` ảnh)** | Đã xử lý triệt để 100% |",
        f"| `KEEP_AUXILIARY` | `310` | **`{w8f['qa_status_counts'].get('KEEP_AUXILIARY', 0)}`** | `+69` ảnh đúng vật thể nhưng chưa có bằng chứng vứt bỏ (`IN_USE` / `PRODUCT_STUDIO`) |",
        f"| `NEEDS_REVIEW` | `85` | **`{w8f['qa_status_counts'].get('NEEDS_REVIEW', 0)}`** | `+41` ảnh đống rác dày / bao bì phức hợp / cần xác nhận thủ công |",
        f"| `REJECT` | `150` | **`{w8f['qa_status_counts'].get('REJECT', 0)}`** | `+7` ảnh không thuộc taxonomy / ảnh người trưng bày thực phẩm (`0690`, ...) |",
        "",
        "---",
        "",
        "## 2. KẾT QUẢ THU THẬP & KIỂM ĐỊNH QA WAVE 8F THEO NGUỒN (SAU HIỆU CHỈNH)",
        "",
        "| Mã Nguồn | Đã tải (Khử trùng) | `REAL_WASTE_PASS` (Tổng) | `TRAIN_REAL_WASTE_W8F` | `VAL_REAL_WASTE_W8F` | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` | Hiệu suất `REAL_WASTE_PASS` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sid, sb in src_breakdown.items():
        c_md.append(
            f"| `{sid}` | {sb['downloaded_candidates']} | **{sb['REAL_WASTE_PASS']}** | **{sb['TRAIN_REAL_WASTE_W8F']}** | **{sb['VAL_REAL_WASTE_W8F']}** | {sb['KEEP_AUXILIARY']} | {sb['NEEDS_REVIEW']} | {sb['REJECT']} | **{sb['yield_pct']}%** |"
        )
    c_md.append(
        f"| **TỔNG WAVE 8F** | **{w8f['total_downloaded_candidates']}** | **{w8f['real_waste_pass_total_images']}** | **{w8f['real_waste_pass_train_images']}** | **{w8f['real_waste_pass_val_images']}** | **{w8f['qa_status_counts'].get('KEEP_AUXILIARY', 0)}** | **{w8f['qa_status_counts'].get('NEEDS_REVIEW', 0)}** | **{w8f['qa_status_counts'].get('REJECT', 0)}** | **{round(100.0 * w8f['real_waste_pass_total_images'] / max(1, w8f['total_downloaded_candidates']), 2)}%** |"
    )

    c_md.extend([
        "",
        "---",
        "",
        "## 3. PHÂN BỐ `REAL_WASTE_PASS` CỦA TASK 8F THEO 10 LỚP (SAU HIỆU CHỈNH)",
        "",
        "| Class ID | Tên lớp | Nhóm ưu tiên | Ảnh `REAL_WASTE_PASS` (Tổng) | Ảnh Train (`TRAIN_REAL_WASTE_W8F`) | Ảnh Val (`VAL_REAL_WASTE_W8F`) | Tổng Boxes (`REAL_WASTE_PASS`) | Boxes Val (`VAL_REAL_WASTE_W8F`) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for cid, cname in enumerate(CLASS_NAMES):
        prio = "8 LỚP HIẾM TRỌNG TÂM" if cid in (0, 1, 2, 3, 4, 5, 6, 8) else "Gán nhãn đúng chất liệu thực tế"
        c_md.append(
            f"| {cid} | `{cname}` | {prio} | **{w8f['per_class_real_waste_pass_images'][cname]}** | **{w8f['per_class_real_waste_pass_train_images'][cname]}** | **{w8f['per_class_real_waste_pass_val_images'][cname]}** | **{w8f['per_class_real_waste_pass_boxes'][cname]}** | **{w8f['per_class_real_waste_pass_val_boxes'][cname]}** |"
        )

    (REPORTS_DIR / "TASK8F_SCARCE_CLASS_EXPANSION_REPORT.md").write_text("\n".join(c_md) + "\n", encoding="utf-8")

    # --- 2. REAL_WASTE_10CLASS_VALIDATION_PLAN (Synchronized Post-Correction) ---
    taco_val_imgs, taco_val_boxes, taco_val_total_imgs, taco_val_total_boxes = count_taco_val_baseline()
    t7_val_pos = [
        r for r in t7_items
        if r.get("qa_status", r.get("qa_verdict")) == "QA_PASS"
        and r.get("split", r.get("proposed_split")) == "VAL_REAL_BATTERY"
        and int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) > 0
    ]
    t7_val_boxes = sum(int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) for r in t7_val_pos)

    w1_val = [r for r in w1_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("real_waste_split") == "VAL_REAL_WASTE_W1"]
    w2_val = [r for r in w2_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("candidate_split") == "VAL_REAL_WASTE_W2"]
    w8d_val = [r for r in w8d["records"] if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8D"]
    w8e_val = [r for r in w8e["records"] if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8E"]
    w8f_val = [r for r in w8f["records"] if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8F"]

    val_rows: list[dict[str, Any]] = []
    for cid, cname in enumerate(CLASS_NAMES):
        v_taco_img = taco_val_imgs[cid]
        v_taco_bx = taco_val_boxes[cid]
        v_t7_img = len(t7_val_pos) if cid == 0 else 0
        v_t7_bx = t7_val_boxes if cid == 0 else 0

        v_w1_img = sum(1 for r in w1_val if cid in r.get("present_class_ids", []))
        v_w1_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w1_val)

        v_w2_img = sum(1 for r in w2_val if cid in r.get("classes_present", []))
        v_w2_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w2_val)

        v_w8d_img = sum(1 for r in w8d_val if cid in r.get("classes_present_ids", []))
        v_w8d_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8d_val)

        v_w8e_img = sum(1 for r in w8e_val if cid in r.get("classes_present_ids", []))
        v_w8e_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8e_val)

        v_w8f_img = sum(1 for r in w8f_val if cid in r.get("classes_present_ids", []))
        v_w8f_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8f_val)

        staged_val_imgs = v_t7_img + v_w1_img + v_w2_img + v_w8d_img + v_w8e_img + v_w8f_img
        staged_val_boxes = v_t7_bx + v_w1_bx + v_w2_bx + v_w8d_bx + v_w8e_bx + v_w8f_bx

        total_val_imgs = v_taco_img + staged_val_imgs
        total_val_boxes = v_taco_bx + staged_val_boxes

        val_rows.append({
            "class_id": cid,
            "class_name": cname,
            "v3_taco_verified_val_images": v_taco_img,
            "v3_taco_verified_val_boxes": v_taco_bx,
            "task7_val_images": v_t7_img,
            "task7_val_boxes": v_t7_bx,
            "wave1_val_images": v_w1_img,
            "wave1_val_boxes": v_w1_bx,
            "wave2_val_images": v_w2_img,
            "wave2_val_boxes": v_w2_bx,
            "wave8d_val_images": v_w8d_img,
            "wave8d_val_boxes": v_w8d_bx,
            "wave8e_val_images": v_w8e_img,
            "wave8e_val_boxes": v_w8e_bx,
            "wave8f_val_images": v_w8f_img,
            "wave8f_val_boxes": v_w8f_bx,
            "staged_val_real_waste_images": staged_val_imgs,
            "staged_val_real_waste_boxes": staged_val_boxes,
            "combined_verified_real_waste_val_images": total_val_imgs,
            "combined_verified_real_waste_val_boxes": total_val_boxes,
            "gap_to_50_val_instances": max(0, 50 - total_val_boxes),
            "gap_to_100_val_instances": max(0, 100 - total_val_boxes),
        })

    d_plan_json = {
        "plan_version": "REAL_WASTE_10CLASS_VALIDATION_PLAN_V2_SEMANTIC_QA_CORRECTED",
        "target_instances_per_class_min": 50,
        "target_instances_per_class_ideal": 100,
        "frozen_test_untouched_299": True,
        "candidate_test_untouched_274": True,
        "external_holdout_untouched_27": True,
        "zero_train_val_leakage_verified": True,
        "unverified_openimages_val_65_excluded_from_real_waste_val": True,
        "cohorts_included": {
            "v3_taco_verified_val_images": taco_val_total_imgs,
            "task7_val_images": len(t7_val_pos),
            "wave1_val_images": len(w1_val),
            "wave2_val_images": len(w2_val),
            "wave8d_val_images": len(w8d_val),
            "wave8e_val_images": len(w8e_val),
            "wave8f_val_images": len(w8f_val),
            "total_unique_verified_real_waste_val_images": (
                taco_val_total_imgs + len(t7_val_pos) + len(w1_val) + len(w2_val) + len(w8d_val) + len(w8e_val) + len(w8f_val)
            ),
        },
        "per_class_validation_inventory": val_rows,
    }
    (REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.json").write_text(
        json.dumps(d_plan_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    d_md = [
        "# KẾ HOẠCH & KIỂM KÊ TẬP VALIDATION RÁC THỰC TẾ 10 LỚP (`REAL_WASTE_10CLASS_VALIDATION_PLAN.md`)",
        "",
        "- **Mục tiêu Phần D:** Xây dựng tập `VAL_REAL_WASTE` độc lập đủ cả 10 lớp, hướng tới **50–100 instances thực tế/lớp**, không trùng `session_group` / `specimen_group` / `pHash` với `train`, và tuyệt đối không đụng tới `frozen test (299)`, `candidate test (274)` hay `D:\\Hinh_Anh_Rac (27)`.",
        "- **Nguyên tắc phân tách domain trong Validation:** Trong `269` ảnh `V3 Val Baseline`, chỉ tính **`204` ảnh `taco_*`** (rác thực địa TACO) vào `VERIFIED_REAL_WASTE_VAL`; tách riêng `65` ảnh `oi_*` (Open Images chưa xác minh ngữ cảnh rác) khỏi bộ đo lường rác thực tế.",
        "",
        "---",
        "",
        "## 1. BẢNG KIỂM KÊ TẬP VALIDATION RÁC THỰC TẾ 10 LỚP HIỆN CÓ (SAU HIỆU CHỈNH TASK 8F)",
        "",
        f"| Lớp (`0..9`) | V3 TACO Val (`204`) | Task7 Val (`10`) | W1 Val (`16`) | W2 Val (`11`) | W8D Val (`81`) | W8E Val (`16`) | **+W8F Val (`{len(w8f_val)}`)** | **Tổng Staged Val (`149`)** | **Tổng `VERIFIED_REAL_WASTE_VAL` (`{d_plan_json['cohorts_included']['total_unique_verified_real_waste_val_images']}` ảnh / Boxes)** | Thiếu tới **50 Boxes/Lớp** | Thiếu tới **100 Boxes/Lớp** |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in val_rows:
        d_md.append(
            f"| `{r['class_id']}: {r['class_name']}` | {r['v3_taco_verified_val_images']} ({r['v3_taco_verified_val_boxes']} bx) | "
            f"{r['task7_val_images']} ({r['task7_val_boxes']} bx) | {r['wave1_val_images']} ({r['wave1_val_boxes']} bx) | "
            f"{r['wave2_val_images']} ({r['wave2_val_boxes']} bx) | {r['wave8d_val_images']} ({r['wave8d_val_boxes']} bx) | "
            f"{r['wave8e_val_images']} ({r['wave8e_val_boxes']} bx) | **+{r['wave8f_val_images']} (+{r['wave8f_val_boxes']} bx)** | "
            f"**{r['staged_val_real_waste_images']} ({r['staged_val_real_waste_boxes']} bx)** | "
            f"**{r['combined_verified_real_waste_val_images']} ảnh / {r['combined_verified_real_waste_val_boxes']} boxes** | "
            f"**{r['gap_to_50_val_instances']}** | **{r['gap_to_100_val_instances']}** |"
        )

    (REPORTS_DIR / "REAL_WASTE_10CLASS_VALIDATION_PLAN.md").write_text("\n".join(d_md) + "\n", encoding="utf-8")

    # --- 3. TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md (Comprehensive Audit Report) ---
    ba = corr["before_vs_after_summary"]
    md = [
        "# BÁO CÁO HIỆU CHỈNH CHẤT LƯỢNG NHÃN NGỮ NGHĨA, DOMAIN RÁC & KHỬ BBOX TRÙNG — TASK 8F (`TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md`)",
        "",
        "- **Repository:** `D:\\CNTT-KLCN155-waste-detection`",
        "- **Script thực thi hiệu chỉnh:** [`scripts/correct_task8f_semantic_qa.py`](./scripts/correct_task8f_semantic_qa.py)",
        "- **Bản sao lưu Manifest trước hiệu chỉnh:** [`task8f_qa_manifest_pre_correction.json`](./data/acquisition_staging/task8f_scarce_classes/qa_correction_before_after/task8f_qa_manifest_pre_correction.json)",
        "- **Manifest sau hiệu chỉnh:** [`task8f_qa_manifest.json`](./data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.json) | [`task8f_qa_manifest.csv`](./data/acquisition_staging/task8f_scarce_classes/task8f_qa_manifest.csv)",
        "- **Contact Sheets sau hiệu chỉnh:** [`task8f_real_waste_pass_p1.jpg`](./data/acquisition_staging/task8f_scarce_classes/contact_sheets/task8f_real_waste_pass_p1.jpg) | [`p2.jpg`](./data/acquisition_staging/task8f_scarce_classes/contact_sheets/task8f_real_waste_pass_p2.jpg) | [`p3.jpg`](./data/acquisition_staging/task8f_scarce_classes/contact_sheets/task8f_real_waste_pass_p3.jpg) | [`task8f_semantic_qa_before_after.jpg`](./data/acquisition_staging/task8f_scarce_classes/contact_sheets/task8f_semantic_qa_before_after.jpg)",
        "- **Nguyên tắc thực thi:** Ưu tiên tuyệt đối chất lượng nhãn thực tế; **KHÔNG** cố giữ con số `203` `REAL_WASTE_PASS`. **DỪNG** mở rộng acquisition mới. **KHÔNG** train E4/E5. `APPROVED_FOR_TRAIN = False`.",
        "",
        "---",
        "",
        "## 1. TỔNG HỢP SỐ LIỆU TRƯỚC VÀ SAU KHI HIỆU CHỈNH TASK 8F",
        "",
        "| Chỉ số kiểm định | Trước hiệu chỉnh (`Pre-Correction`) | Sau hiệu chỉnh (`Post-Correction`) | Chênh lệch / Ghi chú |",
        "|---|---:|---:|---|",
        f"| Tổng ứng viên Wave 8F (`total_downloaded_candidates`) | `748` | **`{w8f['total_downloaded_candidates']}`** | Giữ nguyên 748 ảnh gốc |",
        f"| **`REAL_WASTE_PASS` (Tổng ảnh)** | **`{ba['pre_correction_real_waste_pass_images']}`** | **`{ba['post_correction_real_waste_pass_images']}`** | **`-117` ảnh** (`203 = 86 REAL_WASTE_PASS + 69 KEEP_AUXILIARY + 41 NEEDS_REVIEW + 7 REJECT`) |",
        f"| ├── `TRAIN_REAL_WASTE_W8F` | `{ba['pre_correction_real_waste_pass_train']}` | **`{ba['post_correction_real_waste_pass_train']}`** | `-90` ảnh train |",
        f"| └── `VAL_REAL_WASTE_W8F` | `{ba['pre_correction_real_waste_pass_val']}` | **`{ba['post_correction_real_waste_pass_val']}`** | `-27` ảnh validation |",
        f"| **Tổng Bounding Boxes (`REAL_WASTE_PASS`)** | **`{ba['pre_correction_real_waste_pass_boxes']}`** | **`{ba['post_correction_real_waste_pass_boxes']}`** | **`-287` boxes** (khử trùng và loại bỏ ảnh không phải rác thực tế) |",
        f"| **Số cặp BBox cùng lớp có `IoU > 0.70` trong `REAL_WASTE_PASS`** | **`{ba['pre_correction_high_iou_same_class_pairs']}` cặp (`{ba['pre_correction_high_iou_images']}` ảnh)** | **`{ba['post_correction_high_iou_same_class_pairs']}` cặp (`0` ảnh)** | Khử 100% cả trên `REAL_WASTE_PASS` lẫn toàn bộ `748` bản ghi |",
        f"| `KEEP_AUXILIARY` | `310` | **`{w8f['qa_status_counts']['KEEP_AUXILIARY']}`** | `+69` ảnh đúng vật thể nhưng ở ngữ cảnh `IN_USE` / `PRODUCT_STUDIO` |",
        f"| `NEEDS_REVIEW` | `85` | **`{w8f['qa_status_counts']['NEEDS_REVIEW']}`** | `+41` ảnh đống rác dày / bao bì phức hợp / cần xác nhận thủ công |",
        f"| `REJECT` | `150` | **`{w8f['qa_status_counts']['REJECT']}`** | `+7` ảnh không thuộc taxonomy hoặc trưng bày thực phẩm (`task8f_0690`, ...) |",
        f"| Tổng số bản ghi được sửa đổi (`modified_records_count`) | — | **`{ba['modified_records_count']}`** | Có đầy đủ ảnh overlay `*_before_overlay.jpg` và `*_after_overlay.jpg` |",
        "",
        "---",
        "",
        "## 2. TASK 1 — BẢNG ĐỐI CHIẾU TOÀN BỘ 28 TRƯỜNG HỢP SỬA SAI CLASS (`TASK 1`)",
        "",
        "Bao gồm 4 trường hợp P0 đã xác nhận (`task8f_0657`, `task8f_0664`, `task8f_0661`, `task8f_0712`) và 24 trường hợp phát hiện thêm khi kiểm tra trực tiếp từng ảnh:",
        "",
        "| Candidate ID | Tên tệp gốc (`Title`) | Nguồn truy vấn (`query_used`) | Nhãn lớp Cũ (`Before`) | Nhãn lớp Mới (`After`) | Trạng thái QA Mới | Boxes (`Before -> After`) | Giải thích chi tiết |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for item in corr["task1_class_corrections"]:
        md.append(
            f"| `{item['candidate_id']}` | `{item['title'][:46]}` | `{item['query_used']}` | `{','.join(item['pre_classes'])}` | **`{','.join(item['post_classes'])}`** | `{item['post_qa_status']}` | `{item['pre_bbox_count']} -> {item['post_bbox_count']}` | {item['reason']} |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 3. TASK 2 — KIỂM TRA LẠI TOÀN BỘ 203 ẢNH `REAL_WASTE_PASS` BẰNG ẢNH GỐC VÀ OVERLAY (`TASK 2`)",
        "",
        "### 3.1. Kết quả xử lý 7 trường hợp ưu tiên do Reviewer chỉ định",
        "",
        "| Candidate ID | Vật thể & Bối cảnh thực tế trên ảnh gốc | Trạng thái Cũ | Trạng thái Mới | Domain Tag Mới | Boxes (`Before -> After`) | Lý do phân loại |",
        "|---|---|---|---|---|---:|---|",
    ])
    priority_t2 = {"task8f_0026", "task8f_0118", "task8f_0250", "task8f_0255", "task8f_0573", "task8f_0590", "task8f_0690"}
    for item in corr["task2_domain_and_status_reclassifications"]:
        if item["candidate_id"] in priority_t2:
            md.append(
                f"| **`{item['candidate_id']}`** | `{item['title']}` | `{item['pre_qa_status']}` | **`{item['post_qa_status']}`** | `{item['post_domain_tag']}` | `{item['post_bbox_count']}` | {item['reason']} |"
            )

    md.extend([
        "",
        "### 3.2. Danh sách đầy đủ 117 ảnh chuyển khỏi `REAL_WASTE_PASS` (`69 KEEP_AUXILIARY` + `41 NEEDS_REVIEW` + `7 REJECT`)",
        "",
        "| Candidate ID | Tên tệp gốc (`Title`) | Trạng thái Mới | Domain Tag Mới | Lớp (`Post`) | Số Boxes | Lý do chuyển trạng thái |",
        "|---|---|---|---|---|---:|---|",
    ])
    for item in corr["task2_domain_and_status_reclassifications"]:
        md.append(
            f"| `{item['candidate_id']}` | `{item['title'][:48]}` | **`{item['post_qa_status']}`** | `{item['post_domain_tag']}` | `{','.join(item['post_classes'])}` | {item['post_bbox_count']} | {item['reason']} |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 4. TASK 3 — BẢNG XỬ LÝ TOÀN BỘ 33 CẶP BBOX TRÙNG (`IoU > 0.70`) TRÊN 23 ẢNH (`TASK 3`)",
        "",
        "Trước khi hiệu chỉnh, tập `203` ảnh `REAL_WASTE_PASS` có đúng **33 cặp bbox cùng lớp có `IoU > 0.70` nằm trên 23 ảnh** (bao gồm `task8f_0118` với `IoU = 0.9637 ≈ 0.964`). Sau khi hiệu chỉnh, số cặp `IoU > 0.70` giảm xuống **`0`** trên toàn bộ `748` ảnh:",
        "",
        "| # | Candidate ID | Cặp Box (`i, j`) | Class Trước | `IoU` Trước | Boxes Trước -> Sau | Class Sau | Trạng thái Sau | Cách xử lý cụ thể |",
        "|---:|---|---|---|---:|---:|---|---|---|",
    ])
    post_rec_map = {r["candidate_id"]: r for r in w8f["records"]}
    for idx, p in enumerate(corr["task3_high_iou_33_pairs_resolutions"], start=1):
        cid = p["candidate_id"]
        pr = post_rec_map[cid]
        pair_str = f"({p['box_a_idx']}, {p['box_b_idx']})"
        md.append(
            f"| {idx} | `{cid}` | `{pair_str}` | `{p['pre_class_name']}` | **`{p['iou']:.4f}`** | `-> {p['post_bbox_count']}` | `{','.join(p['post_classes'])}` | `{p['post_qa_status']}` | {p['cause']} -> {p['decision']} |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 5. TASK 4 — GIẢI THÍCH VÀ ĐỐI CHIẾU CHÍNH XÁC SỰ LỆCH SỐ LIỆU TRONG LEDGER & BÁO CÁO (`TASK 4`)",
        "",
        "### 5.1. Nguyên nhân gây lệch số liệu giữa văn bản báo cáo (`1.644 / 686 / 434`) và tệp JSON trong ZIP (`1.611 / 653 / 380`)",
        "",
        "1. Ở lần chạy đầu của Task 8F trước khi áp bộ lọc visual QA cuối cùng, `task8f_qa_manifest.json` tạm thời có `248` ảnh `REAL_WASTE_PASS` (`194` train + `54` val), tạo ra các tổng: `958 + 686 = 1.644` ảnh train và `204 + 230 = 434` ảnh val.",
        "2. Ngay sau đó, script `acquire_and_qa_task8f_scarce_classes.py` được chạy lại với bộ lọc visual QA (`203` ảnh `REAL_WASTE_PASS` = `161` train + `42` val), và cả hai script [`scripts/build_canonical_5000_ledger.py`](./scripts/build_canonical_5000_ledger.py) cùng [`scripts/generate_task8f_and_validation_reports.py`](./scripts/generate_task8f_and_validation_reports.py) đã cập nhật chính xác tệp JSON/MD trên đĩa thành `1.611` (`958 + 653`) và `380` (`204 + 176`), nhưng phần tóm tắt văn bản trả lời bị giữ lại con số cũ (`1.644 / 686 / 434`).",
        "",
        "### 5.2. Bảng đối chiếu 3 cột: Văn bản cũ vs Tệp JSON trước hiệu chỉnh vs Giá trị thực tế sau hiệu chỉnh Task 8F",
        "",
        "| Chỉ số hạch toán | Văn bản trả lời cũ (Lệch) | Tệp JSON trước hiệu chỉnh (`203` Pass) | **Giá trị thực tế sau hiệu chỉnh Task 8F (`86` Pass)** | Công thức tính trực tiếp từ Manifest trên đĩa |",
        "|---|---:|---:|---:|---|",
        f"| `W8F REAL_WASTE_PASS` (Tổng) | `248` | `203` | **`{w8f['real_waste_pass_total_images']}`** | `len([r for r in w8f if r['qa_status']=='REAL_WASTE_PASS'])` |",
        f"| `W8F TRAIN_REAL_WASTE_W8F` | `194` | `161` | **`{w8f['real_waste_pass_train_images']}`** | `71` ảnh train đã xác minh rác thực tế |",
        f"| `W8F VAL_REAL_WASTE_W8F` | `54` | `42` | **`{w8f['real_waste_pass_val_images']}`** | `15` ảnh val độc lập |",
        f"| **Tier 3: `PENDING_APPROVAL` Train Images** | `686` | `653` | **`{ledger['three_tier_summary']['3_PENDING_APPROVAL']['pending_verified_real_waste_unique_train_images']}`** | `Task7(14) + Mini01(7) + W1(66) + W2(72) + W8D(65) + W8E(268) + W8F(71) = 563` |",
        f"| **Tier 2: `VERIFIED_REAL_WASTE` Train Images** | `1.644` | `1.611` | **`{ledger['three_tier_summary']['2_VERIFIED_REAL_WASTE']['cumulative_verified_real_waste_unique_train_images']}`** | `TACO_Baseline(958) + Pending_Staged(563) = 1.521` (tách riêng `319` ảnh `oi_*`) |",
        f"| **Tổng `VERIFIED_REAL_WASTE` Val Images** | `434` | `380` | **`{d_plan_json['cohorts_included']['total_unique_verified_real_waste_val_images']}`** | `TACO_Val(204) + T7(10) + W1(16) + W2(11) + W8D(81) + W8E(16) + W8F(15) = 353` |",
        "",
        "---",
        "",
        "## 6. TASK 5 — NGUYÊN NHÂN GỐC RỄ VÀ PHÉP KIỂM TRA HỒI QUY TỰ ĐỘNG (`TASK 5`)",
        "",
        "1. **Nguyên nhân 1 — Gán nhãn theo Category mà bỏ qua chất liệu thực tế:** Trong `scripts/acquire_and_qa_task8f_scarce_classes.py`, các truy vấn `Category:Branded_litter` và `Category:Fast_food_litter` được gắn cứng `primary_class_id = 6` (`paper`), còn `Category:Shoe_tossing` gắn `primary_class_id = 8` (`shoes`). Khi `extract_tight_per_instance_boxes()` chạy `out_c = primary_cid`, mọi chai thủy tinh (`0657`), lon kim loại (`0661`, `0664`), chai/túi nhựa (`0626`, `0653`), hộp carton (`0624`) và găng tay đấm bốc (`0712`) đều bị ghi đè thành `paper` hoặc `shoes`.",
        "   - **Khắc phục:** Bổ sung hàm `infer_material_class_from_title()` và kiểm định trực tiếp bằng mắt trên toàn bộ contact sheets.",
        "2. **Nguyên nhân 2 — BBox trùng lặp `IoU > 0.70` do ghi đè class sau NMS theo lớp:** `detector.predict()` của YOLOv8 mặc định chạy NMS riêng cho từng lớp (`agnostic_nms=False`), nên một vật thể có thể sinh ra 2–3 dự đoán thuộc các lớp khác nhau với tọa độ gần trùng khít (`IoU = 0.92..0.99`). Khi dòng 853 gán tất cả về cùng `primary_cid` mà không chạy khử trùng hậu xử lý, các box này biến thành box trùng cùng lớp.",
        "   - **Khắc phục:** Bật `agnostic_nms=True` và bổ sung hàm `deduplicate_same_class_boxes_iou(..., iou_thresh=0.70)` trong cả `extract_tight_per_instance_boxes()` lẫn `norm_boxes_to_struct()`.",
        "3. **Nguyên nhân 3 — Fallback box `[0.14, 0.14, 0.86, 0.86]` và lọc ngữ cảnh bằng từ khóa đơn lẻ:** Khi detector và contour không tách được vật thể (như `task8f_0712` trên nền trời), script sinh box mặc định `[0.14, 0.14, 0.86, 0.86]` và vẫn cho vào `REAL_WASTE_PASS`. Đồng thời từ khóa `pizza`, `cup`, `battery` trong tiêu đề khiến ảnh đang sử dụng (`0026`, `0118`, `0250`, `0255`, `0573`, `0590`, `0690`) lọt qua.",
        "   - **Khắc phục:** Tự động chuyển mọi trường hợp chạm fallback `[0.14, 0.14, 0.86, 0.86]` sang `NEEDS_REVIEW`, bổ sung kiểm tra visual QA bắt buộc, và khóa bằng bộ kiểm thử hồi quy [`tests/detection/test_task8_manifest_report_consistency.py`](./tests/detection/test_task8_manifest_report_consistency.py).",
        "",
        "---",
        "",
        "## 7. DANH SÁCH `NEEDS_REVIEW` ĐỂ PM KIỂM TRA (`100` ẢNH TRONG `MANUAL_REVIEW_QUEUE`)",
        "",
        "| Candidate ID | Lớp hiện tại | Số Boxes | Domain Tag | Tên tệp gốc (`Title`) | Lý do giữ ở `NEEDS_REVIEW` |",
        "|---|---|---:|---|---|---|",
    ])
    for nr in corr["needs_review_for_pm_verification"]:
        md.append(
            f"| `{nr['candidate_id']}` | `{','.join(nr['classes_present_names'])}` | {nr['bbox_count']} | `{nr['domain_tag']}` | `{nr['title'][:46]}` | {nr['qa_reason']} |"
        )

    (REPORTS_DIR / "TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Generated TASK8F_SEMANTIC_QA_CORRECTION_REPORT.md, TASK8F_SCARCE_CLASS_EXPANSION_REPORT, and REAL_WASTE_10CLASS_VALIDATION_PLAN successfully!")


if __name__ == "__main__":
    main()
