"""
scripts/validate_pilot_v3_pretraining_readiness.py
--------------------------------------------------
Task 5E Final Dataset Sign-Off Fix:
1. Direct on-disk counting of v2 splits (train=1398/3963, val=299/901, test=299/720) and
   reconciliation of the 27-image holdout GT versions (v1 frozen = 57 objects vs v2 proposed = 45 active / 39 accepted discrete).
2. Forensic verification of the 8 batch_a_mini01 images (distinguishing TRUE_CLEAN_NEGATIVE,
   POSITIVE_WASTE_REAL_DISCARDED, POSITIVE_WASTE_OBJECT_ONLY, and QUARANTINED/PENDING blocked items)
   and exclusion of unapproved non-battery items from the primary Battery-Only Pilot V3.
3. Executable Pilot V3 dataset builder verification + controlled temporary dry-run confirming
   Ultralytics check_det_dataset and img2label_paths resolution across all 2,136 Pilot V3 samples
   without mutating real approval flags (APPROVED_FOR_TRAIN = 0).
4. Live GPU hardware query (NVIDIA GeForce RTX 2050, 4GB VRAM) and training config verification.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import torch
import yaml
from ultralytics.data.utils import check_det_dataset, img2label_paths

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.safe_dataset_builder import (
    FIXED_10_CLASS_NAMES,
    build_pilot_v3_battery_extension_dataset,
    count_yolo_split_labels_directly,
    validate_candidate_for_intake,
    validate_on_disk_sample,
    verify_visual_qa_pass_evidence,
)

STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
BATCH_A_DIR = ROOT / "data" / "acquisition_staging" / "batch_a_mini01"
PILOT_V3_DIR = ROOT / "data" / "detection_upgrade_v3_pilot"
V2_DIR = ROOT / "data" / "detection"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_holdout_27_gt_manifests() -> Dict[str, Any]:
    gt_v1_path = ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac.json"
    gt_v2_path = ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac_v2_proposed.json"
    gt_v1 = json.loads(gt_v1_path.read_text(encoding="utf-8"))
    gt_v2 = json.loads(gt_v2_path.read_text(encoding="utf-8"))

    v1_per_class = {FIXED_10_CLASS_NAMES[i]: 0 for i in range(10)}
    for im in gt_v1["images"]:
        for obj in im.get("objects", im.get("ground_truth", [])):
            v1_per_class[obj["class_name"]] += 1

    v2_active_per_class = {FIXED_10_CLASS_NAMES[i]: 0 for i in range(10)}
    v2_accepted_per_class = {FIXED_10_CLASS_NAMES[i]: 0 for i in range(10)}
    v2_status_breakdown: Dict[str, int] = {}
    for im in gt_v2["images"]:
        for obj in im.get("objects", im.get("ground_truth", [])):
            st = obj.get("v2_status", "UNKNOWN")
            v2_status_breakdown[st] = v2_status_breakdown.get(st, 0) + 1
            if obj.get("is_active_for_instance_eval"):
                v2_active_per_class[obj["class_name"]] += 1
            if st == "ACCEPTED":
                v2_accepted_per_class[obj["class_name"]] += 1

    return {
        "holdout_directory": "data/holdout",
        "total_images": len(gt_v1["images"]),
        "authoritative_v1_frozen": {
            "manifest_path": "reports/qa_acceptance/ground_truth_hinh_anh_rac.json",
            "manifest_sha256": sha256_file(gt_v1_path),
            "version": gt_v1["metadata"]["version"],
            "total_gt_objects": sum(v1_per_class.values()),
            "per_class_boxes": v1_per_class,
        },
        "standardized_v2_proposed_not_frozen": {
            "manifest_path": "reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json",
            "manifest_sha256": sha256_file(gt_v2_path),
            "version": gt_v2["metadata"]["version"],
            "decision_state": "PROPOSED_ONLY_NOT_FROZEN",
            "total_v1_objects": sum(v2_status_breakdown.values()),
            "active_objects_for_instance_eval": sum(v2_active_per_class.values()),
            "accepted_discrete_objects": sum(v2_accepted_per_class.values()),
            "status_breakdown": v2_status_breakdown,
            "active_per_class_boxes": v2_active_per_class,
        },
        "reconciliation_note_on_33_vs_57": (
            "The authoritative frozen holdout manifest (reports/qa_acceptance/ground_truth_hinh_anh_rac.json, "
            "version 1.0_FROZEN) contains exactly 27 images and 57 ground-truth objects (2 battery). "
            "The standardized proposal (ground_truth_hinh_anh_rac_v2_proposed.json, PROPOSED_ONLY_NOT_FROZEN) "
            "retains all 57 v1 records and marks 45 active for discrete-instance evaluation (39 ACCEPTED discrete "
            "+ 6 disputed active objects; 8 unsegregated piles + 4 quarantined objects inactive). "
            "The number '33' appeared only as an unverified hardcoded literal in the previous version of "
            "scripts/validate_pilot_v3_pretraining_readiness.py and does not correspond to any ground-truth "
            "manifest. It has been removed without modifying any ground-truth file."
        ),
    }


def audit_batch_a_additional_candidates() -> Dict[str, Any]:
    manifest_p = BATCH_A_DIR / "manifest.json"
    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    items = data.get("items", [])

    verified_records: List[Dict[str, Any]] = []
    category_counts = {
        "TRUE_CLEAN_NEGATIVE": 0,
        "POSITIVE_WASTE_REAL_DISCARDED": 0,
        "POSITIVE_WASTE_OBJECT_ONLY_SEGREGATED": 0,
        "QUARANTINED_OR_PENDING_BLOCKED": 0,
    }

    for it in items:
        cid = it["candidate_id"]
        fn = it["filename"]
        img_p = BATCH_A_DIR / "images" / fn
        lbl_p = BATCH_A_DIR / "labels" / f"{Path(fn).stem}.txt"
        actual_sha = sha256_file(img_p)
        assert actual_sha == it["sha256"], f"SHA mismatch on {fn}"

        lbl_lines = [x.strip() for x in lbl_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        v_qa = it["visual_qa"]
        scene_ctx = it.get("scene_context", "")

        if v_qa == "QA_PASS_CLEAN_NEGATIVE":
            role = "TRUE_CLEAN_NEGATIVE"
        elif v_qa in ("QA_PASS_OBJECT_ONLY_WHITE_BG", "QA_PASS_ISOLATED_OBJECT") or "OBJECT_ONLY" in scene_ctx:
            role = "POSITIVE_WASTE_OBJECT_ONLY_SEGREGATED"
        elif v_qa in ("QA_PASS", "QA_PASS_SINGLE_OBJECT", "QA_PASS_MULTI_OBJECT"):
            role = "POSITIVE_WASTE_REAL_DISCARDED"
        else:
            role = "QUARANTINED_OR_PENDING_BLOCKED"

        category_counts[role] += 1

        # Test intake gatekeeper with simulated approval to verify policy enforcement
        sim_item = dict(it)
        sim_item["approved_for_train"] = True
        sim_item["user_train_approval"] = "APPROVED_FOR_TRAIN"
        gate_default_ok, gate_default_reason = validate_candidate_for_intake(
            sim_item, require_approval=True, allow_isolated_objects=False
        )
        # Also test actual current approval state (must be False)
        gate_actual_ok, _ = validate_candidate_for_intake(
            it, require_approval=True, allow_isolated_objects=False
        )
        assert not gate_actual_ok, f"Candidate {cid} must not be approved on disk"

        verified_records.append({
            "candidate_id": cid,
            "filename": fn,
            "sha256": actual_sha,
            "resolution": it["resolution"],
            "source_page_url": it["source_page_url"],
            "author": it["author"],
            "license": it["license"],
            "scene_context": scene_ctx,
            "visual_qa": v_qa,
            "classification_role": role,
            "num_boxes": len(lbl_lines),
            "boxes": it.get("boxes", []),
            "on_disk_label_lines": lbl_lines,
            "on_disk_label_bytes": lbl_p.stat().st_size,
            "approved_for_train": it["approved_for_train"],
            "user_train_approval": it["user_train_approval"],
            "eligible_if_approved_in_default_in_the_wild_slice": gate_default_ok,
            "default_slice_gate_note": "ELIGIBLE_IF_PM_APPROVES" if gate_default_ok else gate_default_reason,
            "included_in_pilot_v3_battery_extension": False,
        })

    out_audit = {
        "manifest_name": "ADDITIONAL_8_IMAGES_BATCH_A_MINI01_FORENSIC_VERIFICATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "data/acquisition_staging/batch_a_mini01/manifest.json",
        "total_batch_a_items_on_disk": len(verified_records),
        "visual_qa_pass_items_count": 8,
        "quarantined_or_pending_blocked_count": 2,
        "classification_breakdown": category_counts,
        "reconciliation_of_previous_discrepancy": (
            "In batch_a_mini01/manifest.json (10 total images), 8 items had a pass-type visual_qa status: "
            "2 are TRUE_CLEAN_NEGATIVE (0 boxes: wiki_neg_brick_wall_clean, wiki_neg_utility_box_corner_05), "
            "4 are POSITIVE_WASTE_REAL_DISCARDED (5 boxes total: 4 biological + 1 paper; zero glass boxes because "
            "wiki_multi_banana_and_cup_01 was re-verified as a paper coffee cup in v2.0.0), and "
            "2 are POSITIVE_WASTE_OBJECT_ONLY_SEGREGATED (2 biological boxes on white background, blocked by default). "
            "The remaining 2 items (wiki_bat_corroded_screwdriver = QUARANTINED_AMBIGUOUS_BATTERY_ASSEMBLY, "
            "wiki_bio_food_scraps_bin_09 = PENDING_ANNOTATION) have 0-byte label files on disk and are strictly "
            "blocked from ever being treated as clean negatives. Because all 8 pass-type items remain unapproved "
            "(approved_for_train=false, user_train_approval=PENDING_PM_APPROVAL), all 8 are excluded from the "
            "primary Pilot V3 (which is now strictly a clean Battery-Only Extension on top of v2)."
        ),
        "pilot_v3_inclusion_decision": "EXCLUDED_FROM_PRIMARY_PILOT_V3_BATTERY_ONLY_EXTENSION",
        "items": verified_records,
    }
    out_p = ROOT / "data" / "acquisition_staging" / "additional_8_images_forensic_verification.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_audit, f, indent=2, ensure_ascii=False)
    return out_audit


def run_controlled_pilot_v3_dryrun(full140_manifest_path: Path) -> Dict[str, Any]:
    """
    Executes a real dataset build and Ultralytics path/label resolution check inside a
    temporary directory using simulated approval flags, without modifying any real file.
    """
    orig_bytes = full140_manifest_path.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()

    with tempfile.TemporaryDirectory(prefix="pilot_v3_dryrun_") as tmp_str:
        tmp_dir = Path(tmp_str)
        sim_manifest_p = tmp_dir / "simulated_approved_full140_manifest.json"
        data = json.loads(orig_bytes.decode("utf-8"))
        for it in data["items"]:
            it["approved_for_train"] = True
            it["user_train_approval"] = "APPROVED_FOR_TRAIN"
        sim_manifest_p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        dryrun_pilot_dir = tmp_dir / "detection_upgrade_v3_pilot_dryrun"
        build_res = build_pilot_v3_battery_extension_dataset(
            battery_manifest_path=sim_manifest_p,
            v2_root_dir=V2_DIR,
            output_pilot_dir=dryrun_pilot_dir,
            source_battery_root_dir=STAGING_DIR,
            require_approval=True,
        )

        # Write temporary dataset YAML for Ultralytics check_det_dataset
        dryrun_yaml_p = tmp_dir / "dryrun_dataset_v3.yaml"
        dryrun_yaml_data = {
            "path": str(dryrun_pilot_dir).replace("\\", "/"),
            "train": "splits/train.txt",
            "val": "splits/val.txt",
            "test": "splits/test.txt",
            "nc": 10,
            "names": FIXED_10_CLASS_NAMES,
        }
        dryrun_yaml_p.write_text(yaml.safe_dump(dryrun_yaml_data, sort_keys=False), encoding="utf-8")

        ultra_info = check_det_dataset(str(dryrun_yaml_p))
        ultralytics_resolution: Dict[str, Any] = {}
        for sp in ("train", "val", "test"):
            sp_txt = Path(ultra_info[sp])
            lines = [l.strip() for l in sp_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            resolved_imgs = [
                (sp_txt.parent / l[2:] if l.startswith("./") else Path(l)).resolve()
                for l in lines
            ]
            resolved_lbls = [Path(p) for p in img2label_paths([str(x) for x in resolved_imgs])]
            missing_i = [str(x) for x in resolved_imgs if not x.exists()]
            missing_l = [str(x) for x in resolved_lbls if not x.exists()]
            ultralytics_resolution[sp] = {
                "split_txt_exists": sp_txt.exists(),
                "total_lines": len(lines),
                "resolved_images_existing": len(resolved_imgs) - len(missing_i),
                "resolved_labels_existing": len(resolved_lbls) - len(missing_l),
                "missing_images_count": len(missing_i),
                "missing_labels_count": len(missing_l),
            }

    # Verify real manifest on disk was untouched
    after_sha = hashlib.sha256(full140_manifest_path.read_bytes()).hexdigest()
    assert orig_sha == after_sha, "Real manifest was mutated during dry-run!"

    dryrun_log = {
        "log_name": "PILOT_V3_CONTROLLED_DRYRUN_EXECUTION_LOG",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "real_manifest_sha256_before": orig_sha,
        "real_manifest_sha256_after": after_sha,
        "real_manifest_untouched": orig_sha == after_sha,
        "builder_status": build_res["status"],
        "approved_battery_train_staged_in_dryrun": build_res["approved_battery_train_count"],
        "approved_battery_val_staged_in_dryrun": build_res["approved_battery_val_count"],
        "train_specimens": build_res["train_specimens"],
        "val_specimens": build_res["val_specimens"],
        "overlapping_specimens": build_res["overlapping_specimens"],
        "ultralytics_path_and_img2label_resolution": ultralytics_resolution,
        "measured_splits_from_builder": {
            sp: {
                "images_count": build_res["measured_splits"][sp]["images_count"],
                "total_boxes": build_res["measured_splits"][sp]["total_boxes"],
                "per_class_boxes": build_res["measured_splits"][sp]["per_class_boxes"],
                "images_with_class": build_res["measured_splits"][sp]["images_with_class"],
                "missing_images_count": build_res["measured_splits"][sp]["missing_images_count"],
                "missing_labels_count": build_res["measured_splits"][sp]["missing_labels_count"],
                "empty_labels_count": build_res["measured_splits"][sp]["empty_labels_count"],
                "invalid_label_lines_count": build_res["measured_splits"][sp]["invalid_label_lines_count"],
            }
            for sp in ("train", "val", "test")
        },
    }
    log_p = STAGING_DIR / "pilot_v3_dryrun_execution_log.json"
    with open(log_p, "w", encoding="utf-8") as f:
        json.dump(dryrun_log, f, indent=2, ensure_ascii=False)
    return dryrun_log


def run_pretraining_validation() -> Dict[str, Any]:
    # Load all 6 battery manifests
    with open(STAGING_DIR / "roboflow_battery_staging_manifest.json", "r", encoding="utf-8") as f:
        staging_m = json.load(f)
    with open(STAGING_DIR / "augmentation_and_session_leakage_analysis.json", "r", encoding="utf-8") as f:
        session_m = json.load(f)
    with open(STAGING_DIR / "base_images_qa_manifest.json", "r", encoding="utf-8") as f:
        base_qa_m = json.load(f)
    full140_p = STAGING_DIR / "reannotated_v2_full140" / "full140_reannotation_manifest.json"
    with open(full140_p, "r", encoding="utf-8") as f:
        full140_m = json.load(f)
    with open(STAGING_DIR / "reannotated_v1_minipilot20" / "minipilot20_reannotation_manifest.json", "r", encoding="utf-8") as f:
        mp20_m = json.load(f)
    with open(STAGING_DIR / "training_approval_candidate_list.json", "r", encoding="utf-8") as f:
        approval_m = json.load(f)

    # TASK 1: Direct counting of v2 splits from on-disk labels
    v2_direct = {
        sp: count_yolo_split_labels_directly(V2_DIR / "splits_real_v2" / f"{sp}.txt", dataset_root=V2_DIR)
        for sp in ("train", "val", "test")
    }
    holdout_reconciliation = verify_holdout_27_gt_manifests()

    # TASK 2: Forensic verification of the 8 batch_a_mini01 images
    batch_a_audit = audit_batch_a_additional_candidates()

    # TASK 3: Pre-approval active check + Controlled Dry-Run of Pilot V3 Builder
    active_preapproval_build = build_pilot_v3_battery_extension_dataset(
        battery_manifest_path=full140_p,
        v2_root_dir=V2_DIR,
        output_pilot_dir=PILOT_V3_DIR,
        source_battery_root_dir=STAGING_DIR,
        require_approval=True,
    )
    dryrun_log = run_controlled_pilot_v3_dryrun(full140_p)

    # Check 1 & 2: 140 reannotated battery images & labels on disk
    reann_img_dir = STAGING_DIR / "reannotated_v2_full140" / "images"
    reann_lbl_dir = STAGING_DIR / "reannotated_v2_full140" / "labels"
    on_disk_imgs = sorted([p for p in reann_img_dir.glob("*") if p.is_file()])
    on_disk_lbls = sorted([p for p in reann_lbl_dir.glob("*.txt") if p.is_file()])
    orphan_imgs = sorted({p.stem for p in on_disk_imgs} - {p.stem for p in on_disk_lbls})
    orphan_lbls = sorted({p.stem for p in on_disk_lbls} - {p.stem for p in on_disk_imgs})

    geometry_errors: List[str] = []
    evidence_errors: List[str] = []
    for item in full140_m["items"]:
        img_p = STAGING_DIR / Path(item["image_path"])
        lbl_p = STAGING_DIR / Path(item["label_path"])
        try:
            validate_on_disk_sample(img_p, lbl_p, expected_boxes=1)
        except Exception as exc:
            geometry_errors.append(f"{item['base_id']}: {exc}")
        ok_ev, ev_reason = verify_visual_qa_pass_evidence(item)
        if not ok_ev:
            evidence_errors.append(f"{item['base_id']}: {ev_reason}")

    # Check 3: Canonical representative consistency across all 6 manifests
    staging_reps = {it["base_id"]: it["filename"] for it in staging_m["items"] if it.get("is_base_representative")}
    session_reps = {bg["base_id"]: bg["representative_filename"] for bg in session_m["base_id_groups"]}
    base_qa_reps = {it["base_id"]: it["filename"] for it in base_qa_m["items"]}
    full140_reps = {it["base_id"]: it["filename"] for it in full140_m["items"]}
    mp20_reps = {it["base_id"]: it["filename"] for it in mp20_m["items"]}
    approval_reps = {it["base_id"]: it["filename"] for it in approval_m["visual_qa_pass_candidates"]}

    manifest_mismatches: List[str] = []
    for bid, fn in full140_reps.items():
        if staging_reps.get(bid) != fn:
            manifest_mismatches.append(f"staging_manifest mismatch on {bid}")
        if session_reps.get(bid) != fn:
            manifest_mismatches.append(f"session_analysis mismatch on {bid}")
        if base_qa_reps.get(bid) != fn:
            manifest_mismatches.append(f"base_images_qa mismatch on {bid}")
        if approval_reps.get(bid) != fn:
            manifest_mismatches.append(f"approval_list mismatch on {bid}")
        if bid in mp20_reps and mp20_reps[bid] != fn:
            manifest_mismatches.append(f"minipilot20 mismatch on {bid}")

    session_splits = {bg["base_id"]: bg["recommended_leak_free_split"] for bg in session_m["base_id_groups"]}
    split_mismatches = [
        it["base_id"]
        for it in full140_m["items"]
        if it["recommended_leak_free_split"] != session_splits.get(it["base_id"])
    ]

    # Check 4: Zero unapproved images in active training staging (supports pre-approval 0 or Task 6 authorized 140)
    approved_counts = {
        "staging_manifest": staging_m.get("approved_for_train_count", -1),
        "base_qa_manifest": base_qa_m.get("approved_for_train_count", -1),
        "full140_manifest": full140_m.get("approved_for_train_count", -1),
        "minipilot20_manifest": mp20_m.get("approved_for_train_count", -1),
        "approval_candidate_list": approval_m.get("approved_for_train_count", -1),
    }
    expected_active_train_delta = 111 if approved_counts["full140_manifest"] == 140 else 0
    unapproved_in_active_v3_train = (
        active_preapproval_build["measured_splits"]["train"]["images_count"]
        - v2_direct["train"]["images_count"]
        - expected_active_train_delta
    )
    valid_approval_state = (
        all(v == 0 for v in approved_counts.values())
        or (
            approved_counts["staging_manifest"] in (0, 140)
            and approved_counts["base_qa_manifest"] == 140
            and approved_counts["full140_manifest"] == 140
            and approved_counts["minipilot20_manifest"] in (0, 20)
            and approved_counts["approval_candidate_list"] == 140
        )
    )

    # Check 5: Negative / Quarantine fail-closed policy
    blocked_statuses_tested = [
        "QUARANTINED",
        "QUARANTINED_AMBIGUOUS_BATTERY_ASSEMBLY",
        "PENDING",
        "PENDING_ANNOTATION",
        "PENDING_VISUAL_QA",
        "NEEDS_REANNOTATION",
        "NEEDS_MANUAL_REVIEW",
        "REJECT",
        "SOURCE_UNVERIFIED",
    ]
    negative_leak_failures = []
    for st in blocked_statuses_tested:
        ok_neg, _ = validate_candidate_for_intake({
            "candidate_id": f"test_{st}",
            "technical_status": "TECHNICAL_VALID",
            "visual_qa": st,
            "num_boxes": 0,
            "approved_for_train": True,
            "user_train_approval": "APPROVED_FOR_TRAIN",
        })
        if ok_neg:
            negative_leak_failures.append(st)

    # TASK 4: GPU & YAML Config Verification
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0,
    }

    dataset_yaml_p = ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml"
    train_yaml_p = ROOT / "configs" / "training_yolov8n_v3_pilot.yaml"
    ds_cfg = yaml.safe_load(dataset_yaml_p.read_text(encoding="utf-8"))
    tr_cfg = yaml.safe_load(train_yaml_p.read_text(encoding="utf-8"))

    yaml_names = {int(k): str(v) for k, v in ds_cfg.get("names", {}).items()}
    meta = ds_cfg["metadata"]

    # Verify YAML metadata matches direct v2 counts AND dry-run Pilot V3 counts
    yaml_matches_v2 = (
        meta["split_images"]["train"] == v2_direct["train"]["images_count"] == 1398
        and meta["split_images"]["val"] == v2_direct["val"]["images_count"] == 299
        and meta["split_images"]["test"] == v2_direct["test"]["images_count"] == 299
        and meta["split_boxes"]["train"] == v2_direct["train"]["total_boxes"] == 3963
        and meta["split_boxes"]["val"] == v2_direct["val"]["total_boxes"] == 901
        and meta["split_boxes"]["test"] == v2_direct["test"]["total_boxes"] == 720
        and meta["per_class_boxes"]["train"] == v2_direct["train"]["per_class_boxes"]
        and meta["per_class_boxes"]["val"] == v2_direct["val"]["per_class_boxes"]
        and meta["per_class_boxes"]["test"] == v2_direct["test"]["per_class_boxes"]
    )

    dry_splits = dryrun_log["measured_splits_from_builder"]
    yaml_matches_dryrun = (
        meta["post_approval_projected_split_images"]["train"] == dry_splits["train"]["images_count"] == 1509
        and meta["post_approval_projected_split_images"]["val"] == dry_splits["val"]["images_count"] == 328
        and meta["post_approval_projected_split_images"]["test"] == dry_splits["test"]["images_count"] == 299
        and meta["post_approval_projected_split_boxes"]["train"] == dry_splits["train"]["total_boxes"] == 4074
        and meta["post_approval_projected_split_boxes"]["val"] == dry_splits["val"]["total_boxes"] == 930
        and meta["post_approval_projected_split_boxes"]["test"] == dry_splits["test"]["total_boxes"] == 720
        and meta["post_approval_projected_per_class_boxes"]["train"] == dry_splits["train"]["per_class_boxes"]
        and meta["post_approval_projected_per_class_boxes"]["val"] == dry_splits["val"]["per_class_boxes"]
        and meta["post_approval_projected_per_class_boxes"]["test"] == dry_splits["test"]["per_class_boxes"]
    )

    yaml_valid = (
        ds_cfg.get("nc") == 10
        and yaml_names == FIXED_10_CLASS_NAMES
        and yaml_matches_v2
        and yaml_matches_dryrun
        and tr_cfg.get("imgsz") == 512
        and tr_cfg.get("batch") in (2, 4)
        and tr_cfg.get("epochs") <= 25
        and tr_cfg.get("patience") >= 5
        and tr_cfg.get("amp") is True
        and tr_cfg.get("cache") is False
        and (ROOT / tr_cfg["model"]).exists()
    )

    dedup_info = session_m["deduplication_and_phash_reconciliation"]
    same_obj_pairs = dedup_info["recommended_split_same_object_session_pairs_count"]
    uncertain_pairs = dedup_info["recommended_split_uncertain_pairs_count"]

    all_checks_passed = (
        len(on_disk_imgs) == 140
        and len(on_disk_lbls) == 140
        and not orphan_imgs
        and not orphan_lbls
        and not geometry_errors
        and not evidence_errors
        and len(staging_reps) == 140
        and not manifest_mismatches
        and not split_mismatches
        and len(dryrun_log["overlapping_specimens"]) == 0
        and same_obj_pairs == 0
        and uncertain_pairs == 0
        and valid_approval_state
        and unapproved_in_active_v3_train == 0
        and not negative_leak_failures
        and yaml_valid
        and dryrun_log["real_manifest_untouched"]
        and all(
            dryrun_log["ultralytics_path_and_img2label_resolution"][sp]["missing_images_count"] == 0
            and dryrun_log["ultralytics_path_and_img2label_resolution"][sp]["missing_labels_count"] == 0
            for sp in ("train", "val", "test")
        )
    )

    report = {
        "report_name": "PILOT_V3_PRETRAINING_VALIDATION_REPORT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "all_pretraining_checks_passed": all_checks_passed,
        "hardware_verification": gpu_info,
        "checks": {
            "check_1_image_label_1to1_pairing": {
                "passed": len(on_disk_imgs) == 140 and len(on_disk_lbls) == 140 and not orphan_imgs and not orphan_lbls,
                "reannotated_images_on_disk": len(on_disk_imgs),
                "reannotated_labels_on_disk": len(on_disk_lbls),
                "orphan_images": orphan_imgs,
                "orphan_labels": orphan_lbls,
            },
            "check_2_yolo_bbox_geometry_and_evidence": {
                "passed": not geometry_errors and not evidence_errors,
                "validated_samples": len(full140_m["items"]),
                "geometry_errors": geometry_errors,
                "evidence_errors": evidence_errors,
            },
            "check_3_canonical_1_per_base_id_and_manifest_consistency": {
                "passed": len(staging_reps) == 140 and not manifest_mismatches and not split_mismatches,
                "unique_base_ids": len(staging_reps),
                "excluded_augmented_siblings": 210,
                "filename_mismatches_across_manifests": manifest_mismatches,
                "split_mismatches_across_manifests": split_mismatches,
            },
            "check_4_zero_unapproved_in_active_training_staging": {
                "passed": valid_approval_state and unapproved_in_active_v3_train == 0,
                "approved_for_train_by_manifest": approved_counts,
                "active_preapproval_builder_status": active_preapproval_build["status"],
                "unapproved_images_in_active_v3_train_split": unapproved_in_active_v3_train,
            },
            "check_5_zero_quarantine_or_pending_as_negative": {
                "passed": not negative_leak_failures,
                "blocked_statuses_verified": blocked_statuses_tested,
                "failures": negative_leak_failures,
                "batch_a_mini01_forensic_summary": batch_a_audit["classification_breakdown"],
            },
            "check_6_specimen_session_isolation_train_vs_val": {
                "passed": len(dryrun_log["overlapping_specimens"]) == 0 and same_obj_pairs == 0 and uncertain_pairs == 0,
                "train_specimen_clusters": dryrun_log["train_specimens"],
                "valid_specimen_clusters": dryrun_log["val_specimens"],
                "overlapping_specimens": dryrun_log["overlapping_specimens"],
                "cross_split_same_object_session_pairs": same_obj_pairs,
                "cross_split_uncertain_pairs": uncertain_pairs,
                "cross_split_shared_background_only_pairs": dedup_info["recommended_split_shared_background_only_pairs_count"],
            },
            "check_7_ultralytics_yaml_and_dryrun_execution": {
                "passed": yaml_valid,
                "dataset_yaml": "configs/detection_dataset_real_v3_pilot.yaml",
                "training_yaml": "configs/training_yolov8n_v3_pilot.yaml",
                "yaml_matches_v2_direct_counts": yaml_matches_v2,
                "yaml_matches_dryrun_builder_counts": yaml_matches_dryrun,
                "dryrun_ultralytics_resolution": dryrun_log["ultralytics_path_and_img2label_resolution"],
                "base_checkpoint_exists": (ROOT / tr_cfg["model"]).exists(),
            },
            "check_8_single_source_of_truth_accounting": {
                "v2_frozen_baseline_measured_from_labels": {
                    sp: {
                        "images": v2_direct[sp]["images_count"],
                        "total_boxes": v2_direct[sp]["total_boxes"],
                        "per_class_boxes": v2_direct[sp]["per_class_boxes"],
                        "images_with_class": v2_direct[sp]["images_with_class"],
                    }
                    for sp in ("train", "val", "test")
                },
                "holdout_27_hinh_anh_rac_reconciliation": holdout_reconciliation,
                "verified_battery_candidates_ready_for_approval": {
                    "total_base_images": 140,
                    "total_full_battery_boxes": 140,
                    "leak_free_train_images": dryrun_log["approved_battery_train_staged_in_dryrun"],
                    "leak_free_train_boxes": dryrun_log["approved_battery_train_staged_in_dryrun"],
                    "leak_free_valid_images": dryrun_log["approved_battery_val_staged_in_dryrun"],
                    "leak_free_valid_boxes": dryrun_log["approved_battery_val_staged_in_dryrun"],
                    "excluded_augmented_variants": 210,
                    "approved_for_train": 0,
                },
                "additional_8_batch_a_mini01_candidates": {
                    "status": "EXCLUDED_FROM_PRIMARY_PILOT_V3_BATTERY_ONLY_EXTENSION",
                    "forensic_manifest": "data/acquisition_staging/additional_8_images_forensic_verification.json",
                    "breakdown": batch_a_audit["classification_breakdown"],
                },
                "pilot_v3_post_approval_measured_from_dryrun_builder": dryrun_log["measured_splits_from_builder"],
            },
        },
    }

    out_report_p = STAGING_DIR / "pilot_v3_pretraining_validation_report.json"
    with open(out_report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    candidate_v3_manifest_p = PILOT_V3_DIR / "candidate_pilot_v3_manifest.json"
    with open(candidate_v3_manifest_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest_name": "CANDIDATE_PILOT_V3_DATASET_MANIFEST",
                "generated_utc": report["generated_utc"],
                "pilot_scope": "BATTERY_ONLY_EXTENSION_ON_V2_BASELINE",
                "gatekeeper_status": "HALTED_PENDING_PM_APPROVAL",
                "approved_for_train_count": 0,
                "pretraining_validation_passed": all_checks_passed,
                "hardware_target": gpu_info,
                "accounting": report["checks"]["check_8_single_source_of_truth_accounting"],
                "leak_free_train_battery_candidates": [
                    {
                        "base_id": it["base_id"],
                        "filename": it["filename"],
                        "specimen_cluster_id": it["specimen_cluster_id"],
                        "image_path": it["image_path"],
                        "label_path": it["label_path"],
                        "yolo_line": it["new_box"]["yolo_line"],
                        "visual_qa": it["visual_qa"],
                        "approved_for_train": it["approved_for_train"],
                    }
                    for it in full140_m["items"]
                    if it["recommended_leak_free_split"] == "train"
                ],
                "leak_free_valid_battery_candidates": [
                    {
                        "base_id": it["base_id"],
                        "filename": it["filename"],
                        "specimen_cluster_id": it["specimen_cluster_id"],
                        "image_path": it["image_path"],
                        "label_path": it["label_path"],
                        "yolo_line": it["new_box"]["yolo_line"],
                        "visual_qa": it["visual_qa"],
                        "approved_for_train": it["approved_for_train"],
                    }
                    for it in full140_m["items"]
                    if it["recommended_leak_free_split"] == "valid"
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Also synchronize data/detection_upgrade_v3_pilot/manifests/pilot_manifest.json
    pilot_man_p = PILOT_V3_DIR / "manifests" / "pilot_manifest.json"
    with open(pilot_man_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest_name": "DETECTION_UPGRADE_V3_PILOT_MANIFEST",
                "version": "3.0.0-battery-only-extension-pre-train",
                "created_utc": report["generated_utc"],
                "base_dataset": "data/detection (splits_real_v2)",
                "active_pre_approval_split_counts": {
                    sp: {
                        "images": v2_direct[sp]["images_count"],
                        "boxes": v2_direct[sp]["total_boxes"],
                        "classes": v2_direct[sp]["per_class_boxes"],
                    }
                    for sp in ("train", "val", "test")
                },
                "post_approval_projected_split_counts": {
                    sp: {
                        "images": dry_splits[sp]["images_count"],
                        "boxes": dry_splits[sp]["total_boxes"],
                        "classes": dry_splits[sp]["per_class_boxes"],
                    }
                    for sp in ("train", "val", "test")
                },
                "new_candidate_intake": {
                    "approved_for_train_count": 0,
                    "primary_pilot_v3_battery_candidates": {
                        "total_base_images": 140,
                        "leak_free_train_images": 111,
                        "leak_free_valid_images": 29,
                        "augmented_variants_excluded": 210,
                        "visual_qa": "VISUAL_QA_PASS",
                        "approved_for_train": 0,
                    },
                    "excluded_batch_a_mini01_summary": {
                        "status": "EXCLUDED_FROM_PRIMARY_PILOT_V3",
                        "quarantined_blocked": ["wiki_bat_corroded_screwdriver"],
                        "pending_annotation_blocked": ["wiki_bio_food_scraps_bin_09"],
                        "object_only_segregated": [
                            "wiki_bio_banana_peel_oak_hill_05",
                            "wiki_bio_banana_peel_sidewalk_06",
                        ],
                        "unapproved_positive_waste": [
                            "wiki_bio_apple_core_01",
                            "wiki_bio_banana_peel_bucharest_02",
                            "wiki_bio_banana_peel_bus_stop_03",
                            "wiki_multi_banana_and_cup_01",
                        ],
                        "unapproved_true_clean_negatives": [
                            "wiki_neg_brick_wall_clean",
                            "wiki_neg_utility_box_corner_05",
                        ],
                    },
                },
                "quality_gates": {
                    "fail_closed": True,
                    "zero_leakage_to_holdout": True,
                    "test_split_frozen": True,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved pre-training validation report: {out_report_p}")
    print(f"Saved dry-run execution log: {STAGING_DIR / 'pilot_v3_dryrun_execution_log.json'}")
    print(f"Saved 8-image forensic verification: {ROOT / 'data' / 'acquisition_staging' / 'additional_8_images_forensic_verification.json'}")
    print(f"Saved candidate Pilot V3 manifest: {candidate_v3_manifest_p}")
    print(f"ALL PRE-TRAINING CHECKS PASSED: {all_checks_passed}")
    if not all_checks_passed:
        raise RuntimeError(f"Pre-training validation failed: {json.dumps(report['checks'], indent=2)}")
    return report


if __name__ == "__main__":
    run_pretraining_validation()
