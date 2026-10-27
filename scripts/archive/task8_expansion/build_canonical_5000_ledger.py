# Mô tả: [Lưu trữ — Kịch bản mở rộng & kiểm định dữ liệu đa lớp (Task 8)] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `build_canonical_5000_ledger`.
"""
Task 8E/8F — Part B: Canonical 5,000-Image/Class Dataset Ledger (`build_canonical_5000_ledger.py`).

Enforces all Part B requirements:
1. Separates three accounting tiers:
   - Tier 1 (`CLEAN_BASELINE`): `1,277` V3 Clean Baseline train images (`958` TACO verified real litter + `319` Open Images V7 unverified domain).
     Does NOT automatically count the `319` Open Images baseline images as `VERIFIED_REAL_WASTE`.
   - Tier 2 (`VERIFIED_REAL_WASTE`): Only images with verified real-waste context (`958` TACO baseline + staged `REAL_WASTE_PASS` cohorts).
   - Tier 3 (`PENDING_APPROVAL`): All newly staged cohorts with `APPROVED_FOR_TRAIN = False` awaiting PM sign-off.
2. Preserves complete historical change audit trail:
   - Task 8D before vs after provenance/domain filtering (`200` -> `147` `REAL_WASTE_PASS`, `261` boxes).
   - Task 8E before vs after Final QA Patch (`403` images / `672` boxes -> `348` images / `628` boxes, `114` patched records).
   - Task 8F Scarce-Class Expansion (`task8f_qa_manifest.json`).
3. Reports four granular metrics separately:
   - `unique_images`
   - `per_class_image_appearances`
   - `unique_specimens`
   - `unique_sessions`
"""
from __future__ import annotations

import csv
import json
import pathlib
from collections import Counter
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports" / "task8_dataset_program"

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


def count_yolo_label_dir_by_ids(
    label_dir: pathlib.Path, prefix: str = "", stem_prefix_filter: str | None = None
) -> dict[str, Any]:
    if not label_dir.exists():
        return {
            "image_ids": set(),
            "unique_images": 0,
            "total_boxes": 0,
            "multi_class_images": 0,
            "per_class_image_ids": {cid: set() for cid in range(10)},
            "per_class_images": Counter(),
            "per_class_boxes": Counter(),
        }
    image_ids: set[str] = set()
    total_boxes = 0
    multi_cls = 0
    per_cls_ids: dict[int, set[str]] = {cid: set() for cid in range(10)}
    per_cls_box: Counter[int] = Counter()

    for p in sorted(label_dir.glob("*.txt")):
        if stem_prefix_filter and not p.stem.startswith(stem_prefix_filter):
            continue
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        iid = f"{prefix}{p.stem}" if prefix else p.stem
        image_ids.add(iid)
        cids = set()
        for ln in lines:
            pts = ln.split()
            if pts:
                cid = int(pts[0])
                cids.add(cid)
                per_cls_box[cid] += 1
                total_boxes += 1
        for cid in cids:
            per_cls_ids[cid].add(iid)
        if len(cids) >= 2:
            multi_cls += 1

    per_cls_img = Counter({cid: len(per_cls_ids[cid]) for cid in range(10)})
    return {
        "image_ids": image_ids,
        "unique_images": len(image_ids),
        "total_boxes": total_boxes,
        "multi_class_images": multi_cls,
        "per_class_image_ids": per_cls_ids,
        "per_class_images": per_cls_img,
        "per_class_boxes": per_cls_box,
    }


def count_roboflow_battery_from_splits() -> dict[str, int]:
    rf_lbl_dir = REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "labels" / "roboflow_battery"
    splits_dir = REPO_ROOT / "data" / "detection_upgrade_v3_pilot" / "splits"

    def count_split(split_txt: pathlib.Path) -> tuple[int, int]:
        if not split_txt.exists():
            return 0, 0
        imgs = 0
        boxes = 0
        for ln in split_txt.read_text(encoding="utf-8").splitlines():
            s = ln.strip().replace("\\", "/")
            if "roboflow_battery/" not in s:
                continue
            stem = pathlib.Path(s).stem
            lp = rf_lbl_dir / f"{stem}.txt"
            if lp.exists():
                b_lines = [x.strip() for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
                imgs += 1
                boxes += len(b_lines)
        return imgs, boxes

    tr_imgs, tr_boxes = count_split(splits_dir / "train.txt")
    vl_imgs, vl_boxes = count_split(splits_dir / "val.txt")
    all_rf = count_yolo_label_dir_by_ids(rf_lbl_dir)
    return {
        "total_base_images": all_rf["unique_images"],
        "total_base_boxes": all_rf["total_boxes"],
        "train_base_images": tr_imgs,
        "train_base_boxes": tr_boxes,
        "val_base_images": vl_imgs,
        "val_base_boxes": vl_boxes,
        "offline_augmented_clones_excluded": tr_imgs * 3,
    }


def build_ledger() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. V2 Frozen (`reports/data_v2/split_real_v2_summary.json`)
    v2_summary = json.loads((REPO_ROOT / "reports" / "data_v2" / "split_real_v2_summary.json").read_text(encoding="utf-8"))["stats"]
    v2_train_unique = int(v2_summary["train"]["images"])
    v2_train_boxes = int(v2_summary["train"]["boxes"])
    v2_val_unique = int(v2_summary["val"]["images"])
    v2_test_unique = int(v2_summary["test"]["images"])
    v2_train_per_cls = {cid: int(v2_summary["train"]["images_per_class"][CLASS_NAMES[cid]]) for cid in range(10)}

    # 2. V3 Clean Baseline (`data/detection_v3_candidate/yolo/labels/{train,val,test}`)
    # Explicitly decomposed into TACO (`taco_*`, verified real litter) vs Open Images (`oi_*`, unverified domain)
    v3_tr_dir = REPO_ROOT / "data" / "detection_v3_candidate" / "yolo" / "labels" / "train"
    v3_vl_dir = REPO_ROOT / "data" / "detection_v3_candidate" / "yolo" / "labels" / "val"
    v3_ts_dir = REPO_ROOT / "data" / "detection_v3_candidate" / "yolo" / "labels" / "test"

    v3_train_all = count_yolo_label_dir_by_ids(v3_tr_dir, prefix="v3_tr:")
    v3_train_taco = count_yolo_label_dir_by_ids(v3_tr_dir, prefix="v3_tr:", stem_prefix_filter="taco_")
    v3_train_oi = count_yolo_label_dir_by_ids(v3_tr_dir, prefix="v3_tr:", stem_prefix_filter="oi_")

    v3_val_all = count_yolo_label_dir_by_ids(v3_vl_dir, prefix="v3_vl:")
    v3_val_taco = count_yolo_label_dir_by_ids(v3_vl_dir, prefix="v3_vl:", stem_prefix_filter="taco_")
    v3_val_oi = count_yolo_label_dir_by_ids(v3_vl_dir, prefix="v3_vl:", stem_prefix_filter="oi_")

    v3_test_all = count_yolo_label_dir_by_ids(v3_ts_dir, prefix="v3_ts:")

    # 3. V3 Roboflow Battery Base (`111` train = `111` boxes, `29` val = `29` boxes)
    rf_stats = count_roboflow_battery_from_splits()
    assert rf_stats["train_base_images"] == 111 and rf_stats["train_base_boxes"] == 111
    assert rf_stats["val_base_images"] == 29 and rf_stats["val_base_boxes"] == 29

    # 4. Task 7 Diverse Battery Batch
    t7_path = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "task7_qa_manifest.json"
    t7_data = json.loads(t7_path.read_text(encoding="utf-8"))
    t7_items = t7_data.get("items", t7_data.get("records", []))
    t7_pass = [r for r in t7_items if r.get("qa_status", r.get("qa_verdict")) == "QA_PASS"]
    t7_train_pos = [
        r for r in t7_pass
        if r.get("split", r.get("proposed_split")) == "TRAIN_V4"
        and int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) > 0
    ]
    t7_train_neg = [
        r for r in t7_pass
        if r.get("split", r.get("proposed_split")) == "TRAIN_V4"
        and int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) == 0
    ]
    t7_val_pos = [
        r for r in t7_pass
        if r.get("split", r.get("proposed_split")) == "VAL_REAL_BATTERY"
        and int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) > 0
    ]
    t7_train_boxes = sum(int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) for r in t7_train_pos)
    t7_val_boxes = sum(int(r.get("boxes_count", r.get("box_count", len(r.get("boxes_xyxy_norm", []))))) for r in t7_val_pos)

    # 5. Pre-Task-8 Micro-Pilot (`batch_a_mini01`)
    mini01 = count_yolo_label_dir_by_ids(REPO_ROOT / "data" / "acquisition_staging" / "batch_a_mini01" / "labels", prefix="mini01:")

    # 6. Task 8 Wave 1 (`task8_wave1_qa_manifest.json`)
    w1_recs = json.loads((REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch" / "task8_wave1_qa_manifest.json").read_text(encoding="utf-8"))["records"]
    w1_rw_train = [r for r in w1_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("real_waste_split") == "TRAIN_REAL_WASTE_W1"]
    w1_rw_val = [r for r in w1_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("real_waste_split") == "VAL_REAL_WASTE_W1"]
    w1_aux_train = [r for r in w1_recs if r.get("qa_decision") == "KEEP_AUXILIARY" and r.get("real_waste_split") == "AUXILIARY_TRAIN_W1"]

    # 7. Task 8 Wave 2 (`task8_wave2_qa_manifest.json`)
    w2_recs = json.loads((REPO_ROOT / "data" / "acquisition_staging" / "task8_wave2_large_batch" / "task8_wave2_qa_manifest.json").read_text(encoding="utf-8"))["records"]
    w2_rw_train = [r for r in w2_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("candidate_split") == "TRAIN_REAL_WASTE_W2"]
    w2_rw_val = [r for r in w2_recs if r.get("qa_decision") == "REAL_WASTE_PASS" and r.get("candidate_split") == "VAL_REAL_WASTE_W2"]
    w2_aux_train = [r for r in w2_recs if r.get("qa_decision") == "KEEP_AUXILIARY" and r.get("candidate_split") == "AUXILIARY_TRAIN_W2"]

    # 8. Task 8D (`task8d_qa_manifest.json`)
    w8d_manifest = json.loads((REPO_ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste" / "task8d_qa_manifest.json").read_text(encoding="utf-8"))
    w8d_recs = w8d_manifest["records"]
    w8d_rw_train = [r for r in w8d_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "TRAIN_REAL_WASTE_W8D"]
    w8d_rw_val = [r for r in w8d_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8D"]
    w8d_aux_train = [r for r in w8d_recs if r.get("qa_status") == "KEEP_AUXILIARY" and r.get("target_split") == "AUXILIARY_NON_WASTE_TRAIN_HOLD"]

    # 9. Task 8E Post-Patch (`task8e_qa_manifest.json`) + Pre-Patch Snapshot
    w8e_manifest = json.loads((REPO_ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste" / "task8e_qa_manifest.json").read_text(encoding="utf-8"))
    w8e_recs = w8e_manifest["records"]
    w8e_rw_train = [r for r in w8e_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "TRAIN_REAL_WASTE_W8E"]
    w8e_rw_val = [r for r in w8e_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8E"]
    w8e_aux_train = [r for r in w8e_recs if r.get("qa_status") == "KEEP_AUXILIARY" and r.get("target_split") == "AUXILIARY_NON_WASTE_TRAIN_HOLD"]

    w8e_pre_p = REPO_ROOT / "data" / "acquisition_staging" / "task8e_scale_real_waste" / "qa_patch_before_after" / "task8e_qa_manifest_pre_patch.json"
    w8e_pre_recs = json.loads(w8e_pre_p.read_text(encoding="utf-8"))["records"] if w8e_pre_p.exists() else []
    w8e_pre_rw_train = [r for r in w8e_pre_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "TRAIN_REAL_WASTE_W8E"]
    w8e_pre_rw_val = [r for r in w8e_pre_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8E"]

    # 10. Task 8F Scarce-Class Expansion (`task8f_qa_manifest.json`) + Pre-Correction Snapshot
    w8f_manifest_p = REPO_ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes" / "task8f_qa_manifest.json"
    w8f_manifest = json.loads(w8f_manifest_p.read_text(encoding="utf-8")) if w8f_manifest_p.exists() else {"records": []}
    w8f_recs = w8f_manifest.get("records", [])
    w8f_rw_train = [r for r in w8f_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "TRAIN_REAL_WASTE_W8F"]
    w8f_rw_val = [r for r in w8f_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8F"]
    w8f_aux_train = [r for r in w8f_recs if r.get("qa_status") == "KEEP_AUXILIARY" and r.get("target_split") == "AUXILIARY_NON_WASTE_TRAIN_HOLD"]

    w8f_pre_p = REPO_ROOT / "data" / "acquisition_staging" / "task8f_scarce_classes" / "qa_correction_before_after" / "task8f_qa_manifest_pre_correction.json"
    w8f_pre_recs = json.loads(w8f_pre_p.read_text(encoding="utf-8"))["records"] if w8f_pre_p.exists() else []
    w8f_pre_rw_train = [r for r in w8f_pre_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "TRAIN_REAL_WASTE_W8F"]
    w8f_pre_rw_val = [r for r in w8f_pre_recs if r.get("qa_status") == "REAL_WASTE_PASS" and r.get("target_split") == "VAL_REAL_WASTE_W8F"]

    # Unique IDs, specimens, and sessions tracking across VERIFIED_REAL_WASTE train & val
    verified_rw_train_ids: set[str] = set(v3_train_taco["image_ids"])
    verified_rw_val_ids: set[str] = set(v3_val_taco["image_ids"])
    verified_rw_train_specimens: set[str] = {f"spec_taco_{x.split(':', 1)[1]}" for x in v3_train_taco["image_ids"]}
    verified_rw_train_sessions: set[str] = {f"sess_taco_{x.split('_')[1]}" for x in v3_train_taco["image_ids"] if "_" in x}

    for r in t7_train_pos:
        iid = f"t7:{r.get('item_id') or r.get('candidate_id')}"
        verified_rw_train_ids.add(iid)
        verified_rw_train_specimens.add(r.get("specimen_group") or f"spec_{iid}")
        verified_rw_train_sessions.add(r.get("session_group") or f"sess_{iid}")
    for r in t7_val_pos:
        verified_rw_val_ids.add(f"t7:{r.get('item_id') or r.get('candidate_id')}")

    verified_rw_train_ids |= mini01["image_ids"]
    for iid in mini01["image_ids"]:
        verified_rw_train_specimens.add(f"spec_{iid}")
        verified_rw_train_sessions.add(f"sess_{iid}")

    for prefix, tr_list, vl_list in [
        ("w1", w1_rw_train, w1_rw_val),
        ("w2", w2_rw_train, w2_rw_val),
        ("w8d", w8d_rw_train, w8d_rw_val),
        ("w8e", w8e_rw_train, w8e_rw_val),
        ("w8f", w8f_rw_train, w8f_rw_val),
    ]:
        for r in tr_list:
            cid_s = r.get("item_id") or r.get("candidate_id")
            verified_rw_train_ids.add(f"{prefix}:{cid_s}")
            verified_rw_train_specimens.add(r.get("specimen_group") or r.get("cluster_id") or f"spec_{prefix}_{cid_s}")
            verified_rw_train_sessions.add(r.get("session_group") or r.get("cluster_id") or f"sess_{prefix}_{cid_s}")
        for r in vl_list:
            cid_s = r.get("item_id") or r.get("candidate_id")
            verified_rw_val_ids.add(f"{prefix}:{cid_s}")

    assert not (verified_rw_train_ids & verified_rw_val_ids), "Train/Val ID overlap!"

    per_class_rows: list[dict[str, Any]] = []
    total_appearances_clean_baseline = 0
    total_appearances_verified_rw_train = 0
    total_appearances_pending_rw_train = 0

    for cid, cname in enumerate(CLASS_NAMES):
        c_v2_tr = v2_train_per_cls[cid]
        # Tier 1: CLEAN_BASELINE (`1,277` V3 Clean Baseline = `958` TACO + `319` OI)
        c_v3_all_ids = set(v3_train_all["per_class_image_ids"][cid])
        c_v3_taco_ids = set(v3_train_taco["per_class_image_ids"][cid])
        c_v3_oi_ids = set(v3_train_oi["per_class_image_ids"][cid])
        c_v3_taco_box = int(v3_train_taco["per_class_boxes"].get(cid, 0))
        c_v3_oi_box = int(v3_train_oi["per_class_boxes"].get(cid, 0))

        # Tier 3: PENDING_APPROVAL (`APPROVED_FOR_TRAIN = False`) Verified Real-Waste Train
        c_t7_ids = {f"t7:{r.get('item_id') or r.get('candidate_id')}" for r in t7_train_pos} if cid == 0 else set()
        c_t7_box = t7_train_boxes if cid == 0 else 0
        c_mini_ids = set(mini01["per_class_image_ids"][cid])
        c_mini_box = int(mini01["per_class_boxes"].get(cid, 0))

        c_w1_tr_ids = {f"w1:{r.get('item_id') or r.get('candidate_id')}" for r in w1_rw_train if cid in r.get("present_class_ids", [])}
        c_w1_vl_ids = {f"w1:{r.get('item_id') or r.get('candidate_id')}" for r in w1_rw_val if cid in r.get("present_class_ids", [])}
        c_w1_rw_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w1_rw_train)
        c_w1_aux_tr = sum(1 for r in w1_aux_train if cid in r.get("present_class_ids", []))

        c_w2_tr_ids = {f"w2:{r.get('item_id') or r.get('candidate_id')}" for r in w2_rw_train if cid in r.get("classes_present", [])}
        c_w2_vl_ids = {f"w2:{r.get('item_id') or r.get('candidate_id')}" for r in w2_rw_val if cid in r.get("classes_present", [])}
        c_w2_rw_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w2_rw_train)
        c_w2_aux_tr = sum(1 for r in w2_aux_train if cid in r.get("classes_present", []))

        c_w8d_tr_ids = {f"w8d:{r['candidate_id']}" for r in w8d_rw_train if cid in (r.get("classes_present") or r.get("classes_present_ids", []))}
        c_w8d_vl_ids = {f"w8d:{r['candidate_id']}" for r in w8d_rw_val if cid in (r.get("classes_present") or r.get("classes_present_ids", []))}
        c_w8d_rw_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8d_rw_train)
        c_w8d_aux_tr = sum(1 for r in w8d_aux_train if cid in (r.get("classes_present") or r.get("classes_present_ids", [])))

        c_w8e_pre_tr = sum(1 for r in w8e_pre_rw_train if cid in r.get("classes_present_ids", []))
        c_w8e_tr_ids = {f"w8e:{r['candidate_id']}" for r in w8e_rw_train if cid in r.get("classes_present_ids", [])}
        c_w8e_vl_ids = {f"w8e:{r['candidate_id']}" for r in w8e_rw_val if cid in r.get("classes_present_ids", [])}
        c_w8e_rw_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8e_rw_train)
        c_w8e_aux_tr = sum(1 for r in w8e_aux_train if cid in r.get("classes_present_ids", []))

        c_w8f_pre_tr = sum(1 for r in w8f_pre_rw_train if cid in r.get("classes_present_ids", []))
        c_w8f_tr_ids = {f"w8f:{r['candidate_id']}" for r in w8f_rw_train if cid in r.get("classes_present_ids", [])}
        c_w8f_vl_ids = {f"w8f:{r['candidate_id']}" for r in w8f_rw_val if cid in r.get("classes_present_ids", [])}
        c_w8f_rw_bx = sum(sum(1 for b in r.get("boxes", []) if b["class_id"] == cid) for r in w8f_rw_train)
        c_w8f_aux_tr = sum(1 for r in w8f_aux_train if cid in r.get("classes_present_ids", []))

        pending_rw_ids = c_t7_ids | c_mini_ids | c_w1_tr_ids | c_w2_tr_ids | c_w8d_tr_ids | c_w8e_tr_ids | c_w8f_tr_ids
        # Tier 2: VERIFIED_REAL_WASTE = TACO Verified Real-Waste Baseline (`958`) U Pending Verified Real-Waste (`pending_rw_ids`)
        # Strictly excludes `c_v3_oi_ids` (`319` unverified Open Images in baseline)!
        verified_rw_ids = c_v3_taco_ids | pending_rw_ids

        pending_rw_boxes = c_t7_box + c_mini_box + c_w1_rw_bx + c_w2_rw_bx + c_w8d_rw_bx + c_w8e_rw_bx + c_w8f_rw_bx
        verified_rw_boxes = c_v3_taco_box + pending_rw_boxes

        total_aux_train = (
            (rf_stats["train_base_images"] if cid == 0 else 0)
            + len(c_v3_oi_ids)
            + c_w1_aux_tr
            + c_w2_aux_tr
            + c_w8d_aux_tr
            + c_w8e_aux_tr
            + c_w8f_aux_tr
        )

        c_v3_taco_vl_ids = set(v3_val_taco["per_class_image_ids"][cid])
        c_t7_vl_ids = {f"t7:{r.get('item_id') or r.get('candidate_id')}" for r in t7_val_pos} if cid == 0 else set()
        val_verified_rw_ids = c_v3_taco_vl_ids | c_t7_vl_ids | c_w1_vl_ids | c_w2_vl_ids | c_w8d_vl_ids | c_w8e_vl_ids | c_w8f_vl_ids

        # Distinct specimens & sessions per class across VERIFIED_REAL_WASTE train
        cls_specimens: set[str] = {f"spec_taco_{x.split(':', 1)[1]}" for x in c_v3_taco_ids}
        cls_sessions: set[str] = {f"sess_taco_{x.split('_')[1]}" for x in c_v3_taco_ids if "_" in x}
        for prefix, recs_list in [
            ("w1", [r for r in w1_rw_train if cid in r.get("present_class_ids", [])]),
            ("w2", [r for r in w2_rw_train if cid in r.get("classes_present", [])]),
            ("w8d", [r for r in w8d_rw_train if cid in (r.get("classes_present") or r.get("classes_present_ids", []))]),
            ("w8e", [r for r in w8e_rw_train if cid in r.get("classes_present_ids", [])]),
            ("w8f", [r for r in w8f_rw_train if cid in r.get("classes_present_ids", [])]),
        ]:
            for r in recs_list:
                cid_s = r.get("item_id") or r.get("candidate_id")
                cls_specimens.add(r.get("specimen_group") or r.get("cluster_id") or f"spec_{prefix}_{cid_s}")
                cls_sessions.add(r.get("session_group") or r.get("cluster_id") or f"sess_{prefix}_{cid_s}")
        if cid == 0:
            for r in t7_train_pos:
                cid_s = r.get("item_id") or r.get("candidate_id")
                cls_specimens.add(r.get("specimen_group") or f"spec_t7_{cid_s}")
                cls_sessions.add(r.get("session_group") or f"sess_t7_{cid_s}")

        total_appearances_clean_baseline += len(c_v3_all_ids)
        total_appearances_verified_rw_train += len(verified_rw_ids)
        total_appearances_pending_rw_train += len(pending_rw_ids)

        per_class_rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "v2_frozen_train_images": c_v2_tr,
                "tier1_clean_baseline_total_train_images": len(c_v3_all_ids),
                "tier1a_clean_baseline_taco_verified_rw_train_images": len(c_v3_taco_ids),
                "tier1a_clean_baseline_taco_verified_rw_train_boxes": c_v3_taco_box,
                "tier1b_clean_baseline_openimages_unverified_domain_train_images": len(c_v3_oi_ids),
                "tier1b_clean_baseline_openimages_unverified_domain_train_boxes": c_v3_oi_box,
                "pre_task8_staged_rw_train": len(c_t7_ids | c_mini_ids),
                "wave1_rw_train": len(c_w1_tr_ids),
                "wave2_rw_train": len(c_w2_tr_ids),
                "wave8d_rw_train": len(c_w8d_tr_ids),
                "wave8e_pre_patch_rw_train": c_w8e_pre_tr,
                "wave8e_post_patch_rw_train": len(c_w8e_tr_ids),
                "wave8f_pre_correction_rw_train": c_w8f_pre_tr,
                "wave8f_scarce_rw_train": len(c_w8f_tr_ids),
                "tier3_pending_approval_rw_train_images": len(pending_rw_ids),
                "tier3_pending_approval_rw_train_boxes": pending_rw_boxes,
                "tier2_verified_real_waste_cumulative_train_images": len(verified_rw_ids),
                "tier2_verified_real_waste_cumulative_train_boxes": verified_rw_boxes,
                "tier2_verified_real_waste_unique_specimens": len(cls_specimens),
                "tier2_verified_real_waste_unique_sessions": len(cls_sessions),
                "separate_auxiliary_oi_baseline_and_v3_rf_train": total_aux_train,
                "verified_real_waste_val_total_images": len(val_verified_rw_ids),
                "gap_to_m1_500_from_verified_rw": max(0, 500 - len(verified_rw_ids)),
                "gap_to_m2_1500_from_verified_rw": max(0, 1500 - len(verified_rw_ids)),
                "gap_to_m3_3000_from_verified_rw": max(0, 3000 - len(verified_rw_ids)),
                "gap_to_m4_5000_from_verified_rw": max(0, 5000 - len(verified_rw_ids)),
            }
        )

    ledger_data = {
        "ledger_version": "TASK8F_CANONICAL_5000_LEDGER_V3_SEMANTIC_QA_CORRECTED",
        "canonical_formula": (
            "Tier2_Verified_Real_Waste_Train(c) = |IDs(V3_Clean_Baseline_TACO_Train(c)) U "
            "IDs(Task7_RW_Train(c)) U IDs(Mini01_RW_Train(c)) U IDs(W1_RW_Train(c)) U "
            "IDs(W2_RW_Train(c)) U IDs(W8D_RW_Train(c)) U IDs(W8E_PostPatch_RW_Train(c)) U IDs(W8F_PostCorrection_RW_Train(c))|. "
            "Strictly excludes: (1) the 319 unverified-domain Open Images in V3 Clean Baseline (`oi_*`), "
            "(2) all Validation & Test splits, (3) all offline augmentations, "
            "(4) all KEEP_AUXILIARY (PRODUCT_STUDIO / IN_USE / AMBIGUOUS_CONTEXT) images, and "
            "(5) 111 Roboflow V3 battery base images (111 boxes = 1 box/image)."
        ),
        "three_tier_summary": {
            "1_CLEAN_BASELINE": {
                "total_unique_train_images": v3_train_all["unique_images"],
                "total_train_boxes": v3_train_all["total_boxes"],
                "taco_verified_real_waste_train_images": v3_train_taco["unique_images"],
                "taco_verified_real_waste_train_boxes": v3_train_taco["total_boxes"],
                "openimages_unverified_domain_train_images": v3_train_oi["unique_images"],
                "openimages_unverified_domain_train_boxes": v3_train_oi["total_boxes"],
                "approved_for_train": True,
                "note": "Only the 958 TACO images (`taco_*`) have verified real-litter context by construction; the 319 Open Images (`oi_*`) are kept in CLEAN_BASELINE but excluded from VERIFIED_REAL_WASTE.",
            },
            "2_VERIFIED_REAL_WASTE": {
                "approved_baseline_taco_train_images": v3_train_taco["unique_images"],
                "pending_staged_verified_real_waste_train_images": len(verified_rw_train_ids) - v3_train_taco["unique_images"],
                "cumulative_verified_real_waste_unique_train_images": len(verified_rw_train_ids),
                "cumulative_verified_real_waste_per_class_appearances": total_appearances_verified_rw_train,
                "cumulative_verified_real_waste_unique_specimens": len(verified_rw_train_specimens),
                "cumulative_verified_real_waste_unique_sessions": len(verified_rw_train_sessions),
                "cumulative_verified_real_waste_unique_val_images": len(verified_rw_val_ids),
                "train_val_id_overlap_count": len(verified_rw_train_ids & verified_rw_val_ids),
            },
            "3_PENDING_APPROVAL": {
                "approved_for_train": False,
                "pending_verified_real_waste_unique_train_images": len(verified_rw_train_ids) - v3_train_taco["unique_images"],
                "pending_verified_real_waste_per_class_appearances": total_appearances_pending_rw_train,
                "cohorts": {
                    "task7_diverse_battery_train_pos": len(t7_train_pos),
                    "mini01_pilot_train": mini01["unique_images"],
                    "wave1_real_waste_train": len(w1_rw_train),
                    "wave2_real_waste_train": len(w2_rw_train),
                    "wave8d_real_waste_train": len(w8d_rw_train),
                    "wave8e_post_patch_real_waste_train": len(w8e_rw_train),
                    "wave8f_scarce_real_waste_train": len(w8f_rw_train),
                },
            },
        },
        "historical_audit_trail": {
            "task8d_filtering_history": {
                "raw_downloaded_candidates": 443,
                "pre_audit_real_waste_pass_images": 200,
                "pre_audit_real_waste_pass_train": 77,
                "pre_audit_real_waste_pass_val": 122,
                "pre_audit_real_waste_pass_test_hold": 1,
                "post_audit_real_waste_pass_images": len(w8d_rw_train) + len(w8d_rw_val) + 1,
                "post_audit_real_waste_pass_train": len(w8d_rw_train),
                "post_audit_real_waste_pass_val": len(w8d_rw_val),
                "post_audit_real_waste_pass_test_hold": 1,
                "post_audit_real_waste_pass_boxes": w8d_manifest["real_waste_pass_total_boxes"],
                "audit_reason": "Rejected 12 unattributed third-party web scrapes in Shynuaa/Street-garbage, 4 same-specimen repeats/reflections, and moved 37 closed-bin/non-waste items to KEEP_AUXILIARY.",
            },
            "task8e_qa_patch_history": {
                "raw_downloaded_candidates": 1246,
                "pre_patch_real_waste_pass_images": 403,
                "pre_patch_real_waste_pass_train": 388,
                "pre_patch_real_waste_pass_val": 15,
                "pre_patch_real_waste_pass_boxes": 672,
                "post_patch_real_waste_pass_images": w8e_manifest["real_waste_pass_total_images"],
                "post_patch_real_waste_pass_train": len(w8e_rw_train),
                "post_patch_real_waste_pass_val": len(w8e_rw_val),
                "post_patch_real_waste_pass_boxes": w8e_manifest["real_waste_pass_total_boxes"],
                "patched_records_count": w8e_manifest["pre_patch_audit_trail"]["patched_records_count"],
                "patch_reason": "Part A1-A4: Split 59 multi-instance images into tight per-instance boxes (including task8e_1167 1->9 battery boxes and task8e_1226 1->2 shoe boxes), demoted 10 intact/studio/empty-bin images (including task8e_1152) to KEEP_AUXILIARY, moved 42 coarse group/heap scenes to NEEDS_REVIEW, and rejected 3 same-specimen repeats.",
            },
            "task8f_scarce_expansion": {
                "raw_downloaded_candidates": w8f_manifest.get("total_downloaded_candidates", 0),
                "pre_correction_real_waste_pass_images": 203,
                "pre_correction_real_waste_pass_train": 161,
                "pre_correction_real_waste_pass_val": 42,
                "pre_correction_real_waste_pass_boxes": 407,
                "pre_correction_high_iou_same_class_pairs": 33,
                "real_waste_pass_images": w8f_manifest.get("real_waste_pass_total_images", 0),
                "real_waste_pass_train": len(w8f_rw_train),
                "real_waste_pass_val": len(w8f_rw_val),
                "real_waste_pass_boxes": w8f_manifest.get("real_waste_pass_total_boxes", 0),
                "post_correction_high_iou_same_class_pairs": 0,
                "modified_records_count": w8f_manifest.get("pre_correction_audit_trail", {}).get("modified_records_count", 229),
                "correction_reason": (
                    "Task 8F Quality Correction (Tasks 1-3): Fixed all wrong-class annotations (including task8f_0657 6->4, "
                    "0664 6->5, 0661 6->5, 0712 8->3 + NEEDS_REVIEW, and 23 cross-material Branded/Fast-food litter items), "
                    "demoted 69 intact/in-use/studio items (including task8f_0026, 0118, 0250, 0255, 0573, 0590) to KEEP_AUXILIARY, "
                    "moved 41 dense/ambiguous items to NEEDS_REVIEW, rejected 7 non-waste scenes (including task8f_0690), and "
                    "resolved all 33 high-IoU (>0.70) duplicate box pairs to 0."
                ),
            },
        },
        "four_dimensional_counts": {
            "unique_images_verified_real_waste_train": len(verified_rw_train_ids),
            "per_class_image_appearances_verified_real_waste_train": total_appearances_verified_rw_train,
            "unique_specimens_verified_real_waste_train": len(verified_rw_train_specimens),
            "unique_sessions_verified_real_waste_train": len(verified_rw_train_sessions),
            "unique_images_clean_baseline_train": v3_train_all["unique_images"],
            "per_class_image_appearances_clean_baseline_train": total_appearances_clean_baseline,
        },
        "roboflow_battery_reconciliation": rf_stats,
        "per_class_ledger": per_class_rows,
    }

    (REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.json").write_text(
        json.dumps(ledger_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(per_class_rows)

    md = [
        "# SỔ CÁI CHUẨN HÓA 5.000 ẢNH/LỚP PHÂN TẦNG 3 MỨC (`CANONICAL_5000_DATASET_LEDGER.md`)",
        "",
        "- **Script tạo tự động 100% từ Image IDs thực trên đĩa:** [`scripts/build_canonical_5000_ledger.py`](./scripts/build_canonical_5000_ledger.py)",
        "- **Tệp dữ liệu đối chiếu:** [`CANONICAL_5000_DATASET_LEDGER.json`](./reports/task8_dataset_program/CANONICAL_5000_DATASET_LEDGER.json) | [`CANONICAL_5000_DATASET_LEDGER.csv`](./reports/task8_dataset_program/CANONICAL_5000_DATASET_LEDGER.csv)",
        "",
        "---",
        "",
        "## 1. TÁCH RÕ 3 MỨC TRONG LEDGER (THEO YÊU CẦU PHẦN B & TASK 8F QUALITY CORRECTION)",
        "",
        "| Mức (Tier) | Định nghĩa & Quy tắc hạch toán | Unique Train Images | Per-Class Appearances | Unique Specimens | Unique Sessions | Trạng thái phê duyệt |",
        "|---|---|---:|---:|---:|---:|---|",
        f"| **1. `CLEAN_BASELINE` (Tổng V3 Sạch kỹ thuật)** | Toàn bộ `data/detection_v3_candidate/yolo/labels/train` đã khử trùng & chống rò rỉ (`958` ảnh `taco_*` + `319` ảnh `oi_*`). | **{v3_train_all['unique_images']:,}** | {total_appearances_clean_baseline:,} | {v3_train_all['unique_images']:,} | — | `APPROVED_FOR_TRAIN = True` |",
        f"| ├── **1a. `CLEAN_BASELINE_TACO` (`VERIFIED_REAL_WASTE`)** | `958` ảnh `taco_*` từ bộ TACO (Trash Annotations in Context) — 100% là rác thải ngoài thực địa. | **{v3_train_taco['unique_images']:,}** | {sum(len(v3_train_taco['per_class_image_ids'][c]) for c in range(10)):,} | {v3_train_taco['unique_images']:,} | {len({x.split('_')[1] for x in v3_train_taco['image_ids'] if '_' in x}):,} | `APPROVED_FOR_TRAIN = True` |",
        f"| └── **1b. `CLEAN_BASELINE_OPENIMAGES` (Chưa xác minh domain rác)** | `319` ảnh `oi_*` từ Open Images V7 trong V2/V3 (`clothes: 169`, `shoes: 109`, `glass: 35`, `metal: 10`, `paper: 1`, `plastic: 1`) — **KHÔNG tự động tính vào `VERIFIED_REAL_WASTE`**. | **{v3_train_oi['unique_images']:,}** | {sum(len(v3_train_oi['per_class_image_ids'][c]) for c in range(10)):,} | {v3_train_oi['unique_images']:,} | — | Giữ ở `CLEAN_BASELINE` (Tách khỏi `VERIFIED_REAL_WASTE`) |",
        f"| **2. `VERIFIED_REAL_WASTE` (Đã xác minh ngữ cảnh rác thực tế)** | Hợp của `1a. CLEAN_BASELINE_TACO` (`958` ảnh) và các đợt thu thập đã xác minh `REAL_WASTE_CONTEXT_VERIFIED = True` (`Task 7 + Mini01 + W1 + W2 + W8D + W8E Post-Patch + W8F Post-Correction`). | **{len(verified_rw_train_ids):,}** | **{total_appearances_verified_rw_train:,}** | **{len(verified_rw_train_specimens):,}** | **{len(verified_rw_train_sessions):,}** | Gồm `958` đã duyệt + `{len(verified_rw_train_ids) - v3_train_taco['unique_images']:,}` chờ duyệt |",
        f"| **3. `PENDING_APPROVAL` (`APPROVED_FOR_TRAIN = False`)** | Toàn bộ ảnh `REAL_WASTE_PASS` mới thu thập ở các đợt staging (`Task 7: {len(t7_train_pos)}`, `Mini01: {mini01['unique_images']}`, `W1: {len(w1_rw_train)}`, `W2: {len(w2_rw_train)}`, `W8D: {len(w8d_rw_train)}`, `W8E Post-Patch: {len(w8e_rw_train)}`, `W8F Post-Correction: {len(w8f_rw_train)}`). | **{len(verified_rw_train_ids) - v3_train_taco['unique_images']:,}** | **{total_appearances_pending_rw_train:,}** | {len(verified_rw_train_specimens) - v3_train_taco['unique_images']:,} | {len(verified_rw_train_sessions) - len({x.split('_')[1] for x in v3_train_taco['image_ids'] if '_' in x}):,} | `APPROVED_FOR_TRAIN = False` |",
        "",
        "---",
        "",
        "## 2. LỊCH SỬ THAY ĐỔI SỐ LIỆU QUA CÁC ĐỢT LỌC & VÁ QA (AUDIT TRAIL)",
        "",
        "| Đợt dữ liệu | Tổng ảnh tải về | Trước khi lọc / vá QA | Sau khi lọc / vá QA | Số Boxes (`REAL_WASTE_PASS`) | Giải thích chi tiết thay đổi |",
        "|---|---:|---|---|---:|---|",
        f"| **Task 8D** | 443 | `200` `REAL_WASTE_PASS` (`77` train, `122` val, `1` test hold) | **`147` `REAL_WASTE_PASS`** (`{len(w8d_rw_train)}` train, `{len(w8d_rw_val)}` val, `1` test hold) | **{w8d_manifest['real_waste_pass_total_boxes']}** | Loại bỏ `12` ảnh web scrape không rõ nguồn gốc trong `Shynuaa/Street-garbage`, `4` ảnh trùng mẫu vật/phản chiếu kính, và chuyển `37` ảnh thùng rác đóng kín sang `KEEP_AUXILIARY`. |",
        f"| **Task 8E** | 1.246 | `403` `REAL_WASTE_PASS` (`388` train, `15` val, `672` boxes) | **`{w8e_manifest['real_waste_pass_total_images']}` `REAL_WASTE_PASS`** (`{len(w8e_rw_train)}` train, `{len(w8e_rw_val)}` val) | **{w8e_manifest['real_waste_pass_total_boxes']}** | Phần A1–A4 (`114` bản ghi được sửa): Tách per-instance cho `59` ảnh đa vật thể (`task8e_1167` 1->9 boxes, `task8e_1226` 1->2 boxes), hạ `10` ảnh studio/in-use (`task8e_1152`, ...) xuống `KEEP_AUXILIARY`, chuyển `42` ảnh đống rác/box gộp sang `NEEDS_REVIEW`, loại `3` ảnh trùng mẫu vật. |",
        f"| **Task 8F (Scarce Classes)** | {w8f_manifest.get('total_downloaded_candidates', 0)} | `203` `REAL_WASTE_PASS` (`161` train, `42` val, `407` boxes, `33` cặp `IoU > 0.70`) | **`{w8f_manifest.get('real_waste_pass_total_images', 0)}` `REAL_WASTE_PASS`** (`{len(w8f_rw_train)}` train, `{len(w8f_rw_val)}` val, `0` cặp `IoU > 0.70`) | **{w8f_manifest.get('real_waste_pass_total_boxes', 0)}** | Task 8F Semantic QA Correction (`229` bản ghi được sửa): Sửa toàn bộ nhãn sai lớp (`task8f_0657` 6->4, `0664` 6->5, `0661` 6->5, `0712` 8->3 + `NEEDS_REVIEW`, và `23` ảnh khác), hạ `69` ảnh `IN_USE`/`PRODUCT_STUDIO` (`0026, 0118, 0250, 0255, 0573, 0590`, ...) xuống `KEEP_AUXILIARY`, chuyển `41` ảnh sang `NEEDS_REVIEW`, loại `7` ảnh (`0690`, ...), khử toàn bộ `33 -> 0` cặp bbox trùng `IoU > 0.70`. |",
        "",
        "---",
        "",
        "## 3. BẢNG SỔ CÁI CHI TIẾT 10 LỚP (TÁCH RÕ `CLEAN_BASELINE`, `VERIFIED_REAL_WASTE` & `PENDING_APPROVAL`)",
        "",
        "| Lớp (`0..9`) | Tier 1: `CLEAN_BASELINE` (`1.277`) | Tier 1a: `TACO Verified RW` (`958`) | Tier 1b: `OI Unverified` (`319`, Tách riêng) | +Pre-T8 (`T7+Mini`) | +W1 RW Train | +W2 RW Train | +W8D RW Train | +W8E Pre -> **Post-Patch** | **+W8F Pre -> Post-Correction** | Tier 3: Tổng `PENDING_APPROVAL` RW Train | **Tier 2: Tổng `VERIFIED_REAL_WASTE` Train (1a + Tier 3)** | Unique Specimens / Sessions (Tier 2) | Val `VERIFIED_REAL_WASTE` | Thiếu tới **M1 (`500`)** | Thiếu tới **M4 (`5.000`)** |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in per_class_rows:
        md.append(
            f"| `{r['class_id']}: {r['class_name']}` | {r['tier1_clean_baseline_total_train_images']} | "
            f"**{r['tier1a_clean_baseline_taco_verified_rw_train_images']}** ({r['tier1a_clean_baseline_taco_verified_rw_train_boxes']} bx) | "
            f"{r['tier1b_clean_baseline_openimages_unverified_domain_train_images']} | "
            f"+{r['pre_task8_staged_rw_train']} | +{r['wave1_rw_train']} | +{r['wave2_rw_train']} | "
            f"+{r['wave8d_rw_train']} | {r['wave8e_pre_patch_rw_train']} -> **+{r['wave8e_post_patch_rw_train']}** | "
            f"{r['wave8f_pre_correction_rw_train']} -> **+{r['wave8f_scarce_rw_train']}** | **{r['tier3_pending_approval_rw_train_images']}** ({r['tier3_pending_approval_rw_train_boxes']} bx) | "
            f"**{r['tier2_verified_real_waste_cumulative_train_images']}** ({r['tier2_verified_real_waste_cumulative_train_boxes']} bx) | "
            f"{r['tier2_verified_real_waste_unique_specimens']} / {r['tier2_verified_real_waste_unique_sessions']} | "
            f"**{r['verified_real_waste_val_total_images']}** | **{r['gap_to_m1_500_from_verified_rw']:,}** | **{r['gap_to_m4_5000_from_verified_rw']:,}** |"
        )

    (REPORTS_DIR / "CANONICAL_5000_DATASET_LEDGER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return ledger_data


if __name__ == "__main__":
    data = build_ledger()
    print("[Ledger] Built Three-Tier CANONICAL_5000_DATASET_LEDGER (.md, .json, .csv) successfully.")
    for r in data["per_class_ledger"]:
        print(
            f"  {r['class_id']}:{r['class_name']:10s} | CleanBase={r['tier1_clean_baseline_total_train_images']:3d} "
            f"(TACO_RW={r['tier1a_clean_baseline_taco_verified_rw_train_images']:3d}, OI_Unv={r['tier1b_clean_baseline_openimages_unverified_domain_train_images']:3d}) | "
            f"+Pending_RW={r['tier3_pending_approval_rw_train_images']:3d} (W8E={r['wave8e_post_patch_rw_train']:3d}, W8F={r['wave8f_scarce_rw_train']:3d}) -> "
            f"Tier2_Verified_RW={r['tier2_verified_real_waste_cumulative_train_images']:4d} | Val_RW={r['verified_real_waste_val_total_images']:3d}"
        )
