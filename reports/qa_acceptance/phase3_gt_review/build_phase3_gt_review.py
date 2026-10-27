# Mô tả: Đóng gói bộ hồ sơ rà soát nhãn chuẩn Ground Truth Giai đoạn 3 mà không làm thay đổi dữ liệu gốc.
"""Build the Phase 3 Ground Truth review package without modifying either GT file.

The package is evidence for human approval.  It copies the in-scope source images,
draws current/proposed overlays, records decisions in JSON, and independently
recomputes threshold metrics under explicit image exclusion semantics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
QA_DIR = ROOT / "reports" / "qa_acceptance"
OUT_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_DIR = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout")))
V1_PATH = QA_DIR / "ground_truth_hinh_anh_rac.json"
V2_PATH = QA_DIR / "ground_truth_hinh_anh_rac_v2_proposed.json"
PREDICTION_PATHS = {
    "ssdlite_before_quick_fix": QA_DIR / "raw_predictions_before_quick_fix.json",
    "ssdlite_after_quick_fix": QA_DIR / "raw_predictions_after_quick_fix.json",
    "yolov8n_e1": QA_DIR / "raw_predictions_yolov8n_e1.json",
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

ACTIVE_DECISIONS = {
    "img04_obj01": {
        "recommendation": "EXCLUDE_IMAGE_PENDING_INSTANCE_REANNOTATION",
        "proposed_class_name": "clothes",
        "proposed_class_id": 3,
        "proposed_bbox_xyxy": None,
        "rationale": (
            "Hai box cùng class mô tả các phần chồng lấn của một đống quần áo; "
            "không có ranh giới instance độc lập đủ chắc chắn để đề xuất tọa độ mới."
        ),
        "approval_effect": "Exclude the whole image from the main instance benchmark until exhaustive reannotation.",
    },
    "img06_obj01": {
        "recommendation": "KEEP_AND_MARK_ACCEPTED",
        "proposed_class_name": "cardboard",
        "proposed_class_id": 2,
        "proposed_bbox_xyxy": [32, 36, 535, 519],
        "rationale": (
            "Cardboard container and paper contents are visibly distinct target materials; "
            "cross-class containment is not itself a duplicate annotation."
        ),
        "approval_effect": "No object-count or class-count change.",
    },
    "img20_obj01": {
        "recommendation": "EXCLUDE_IMAGE_PENDING_INSTANCE_REANNOTATION",
        "proposed_class_name": "clothes",
        "proposed_class_id": 3,
        "proposed_bbox_xyxy": None,
        "rationale": (
            "Hai box cùng class mô tả các vùng chồng lấn của một bãi quần áo ở độ phân giải thấp; "
            "không thể xác nhận hai instance độc lập từ ảnh hiện tại."
        ),
        "approval_effect": "Exclude the whole image from the main instance benchmark until exhaustive reannotation.",
    },
    "img22_obj01": {
        "recommendation": "RELABEL_TO_TRASH",
        "proposed_class_name": "trash",
        "proposed_class_id": 9,
        "proposed_bbox_xyxy": [230, 220, 460, 550],
        "rationale": "Túi đang chứa rác hỗn hợp; quy tắc v2 hiện tại quy định bagged heterogeneous waste là trash.",
        "approval_effect": "Move one GT instance from plastic to trash; total GT count is unchanged.",
    },
    "img22_obj02": {
        "recommendation": "RELABEL_TO_TRASH",
        "proposed_class_name": "trash",
        "proposed_class_id": 9,
        "proposed_bbox_xyxy": [340, 445, 675, 705],
        "rationale": "Túi đang chứa rác hỗn hợp; quy tắc v2 hiện tại quy định bagged heterogeneous waste là trash.",
        "approval_effect": "Move one GT instance from plastic to trash; total GT count is unchanged.",
    },
    "img22_obj04": {
        "recommendation": "KEEP_CARDBOARD_AND_MARK_ACCEPTED",
        "proposed_class_name": "cardboard",
        "proposed_class_id": 2,
        "proposed_bbox_xyxy": [680, 570, 765, 650],
        "rationale": (
            "Với taxonomy 10 lớp cố định không có beverage_carton, cardboard là lớp gần nhất "
            "cho bao bì giấy/carton cứng đa lớp; không có bằng chứng để đổi sang class khác."
        ),
        "approval_effect": "No object-count or class-count change.",
    },
}

EXCLUDE_PENDING_REANNOTATION = {"images (10).jpg", "OIP (1).webp"}

MIXED_IMAGE_INACTIVE_DECISIONS = {
    "img01_obj07": {
        "recommendation": "REMOVE_INVALID_TEXTURE_CARPET_GT_ONLY",
        "rationale": (
            "The 85.38% region describes a glass-fragment surface and overlaps four accepted discrete glass objects. "
            "Keep the image and its discrete GT; do not treat the area box as another instance."
        ),
    },
    "img03_obj03": {
        "recommendation": "REMOVE_DUPLICATE_FULL_CANVAS_GT_ONLY",
        "rationale": (
            "The full-canvas plastic box duplicates two accepted plastic bottles. Keep the image and the two "
            "discrete GT objects; do not score the canvas as a third instance."
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stem(filename: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)


def compute_iou(box_a, box_b) -> float:
    xa = max(float(box_a[0]), float(box_b[0]))
    ya = max(float(box_a[1]), float(box_b[1]))
    xb = min(float(box_a[2]), float(box_b[2]))
    yb = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(prediction_data, gt_by_filename, excluded_images=frozenset()):
    total_gt = sum(len(objects) for name, objects in gt_by_filename.items() if name not in excluded_images)
    total_predictions = tp = fp = 0
    per_object = {}
    per_image = {}
    for image_record in prediction_data["images"]:
        filename = image_record["filename"]
        if filename in excluded_images:
            continue
        objects = gt_by_filename.get(filename, [])
        predictions = [
            prediction
            for prediction in image_record.get("predictions", [])
            if float(prediction["confidence"]) >= CONFIDENCE_THRESHOLD
        ]
        predictions.sort(key=lambda prediction: -float(prediction["confidence"]))
        matched = set()
        image_tp = image_fp = 0
        for prediction in predictions:
            candidates = []
            for index, obj in enumerate(objects):
                if index in matched or int(prediction["class_id"]) != int(obj["class_id"]):
                    continue
                candidates.append((compute_iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]), index))
            best_iou, best_index = max(candidates, default=(0.0, -1))
            total_predictions += 1
            if best_index >= 0 and best_iou >= IOU_THRESHOLD:
                matched.add(best_index)
                tp += 1
                image_tp += 1
                obj = objects[best_index]
                per_object[obj["object_id"]] = {
                    "status": "TP",
                    "pred_id": prediction.get("pred_id"),
                    "prediction_class": prediction.get("class_name"),
                    "confidence": prediction["confidence"],
                    "iou": round(best_iou, 6),
                }
            else:
                fp += 1
                image_fp += 1
        for index, obj in enumerate(objects):
            if index not in matched:
                per_object[obj["object_id"]] = {"status": "FN"}
        per_image[filename] = {
            "gt": len(objects),
            "predictions_ge_0_25": len(predictions),
            "tp": image_tp,
            "fp": image_fp,
            "fn": len(objects) - image_tp,
        }
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_pct": round(100 * precision, 2),
        "recall_pct": round(100 * recall, 2),
        "f1_pct": round(100 * f1, 2),
        "per_object": per_object,
        "per_image": per_image,
    }


def font(size=20):
    candidates = [Path(r"arial.ttf"), Path(r"segoeui.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_label(draw, xy, text, fill, image_width):
    label_font = font(18)
    x, y = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((x, y), text, font=label_font)
    label_width = min(bbox[2] - bbox[0] + 8, image_width - x)
    label_height = bbox[3] - bbox[1] + 8
    draw.rectangle((x, y, x + label_width, y + label_height), fill=fill)
    draw.text((x + 4, y + 3), text, font=label_font, fill="white")


def draw_current(image, record):
    draw = ImageDraw.Draw(image, "RGBA")
    for obj in record["objects"]:
        status = obj["v2_status"]
        if status == "ACCEPTED":
            color = (0, 150, 70, 255)
        elif obj["is_active_for_instance_eval"]:
            color = (230, 130, 0, 255)
        else:
            color = (185, 40, 120, 255)
        box = obj["bbox_xyxy"]
        draw.rectangle(tuple(box), outline=color, width=max(2, min(image.size) // 150))
        draw_label(draw, (box[0], max(0, box[1] - 28)), f"{obj['object_id']} {obj['class_name']} | {status}", color, image.width)
    return image


def draw_proposed(image, record, is_zero_active):
    draw = ImageDraw.Draw(image, "RGBA")
    filename = record["filename"]
    if filename in EXCLUDE_PENDING_REANNOTATION:
        draw.rectangle((0, 0, image.width, image.height), fill=(115, 65, 150, 75), outline=(115, 65, 150, 255), width=8)
        draw.line((0, 0, image.width, image.height), fill=(115, 65, 150, 255), width=8)
        draw.line((image.width, 0, 0, image.height), fill=(115, 65, 150, 255), width=8)
        draw_label(draw, (8, 8), "PROPOSED: EXCLUDE IMAGE pending exhaustive instance reannotation", (115, 65, 150, 255), image.width)
        return image

    if is_zero_active:
        statuses = {obj["v2_status"] for obj in record["objects"]}
        utility = statuses == {"QUARANTINED_UTILITY_RECEPTACLE"}
        if utility:
            text = "PROPOSED: QUARANTINE; negative only after exhaustive review"
            color = (190, 85, 20, 255)
        else:
            text = "PROPOSED: EXCLUDE from instance eval; pile is NOT a clean negative"
            color = (115, 65, 150, 255)
        draw.rectangle((0, 0, image.width, image.height), fill=(*color[:3], 75), outline=color, width=8)
        draw.line((0, 0, image.width, image.height), fill=color, width=8)
        draw.line((image.width, 0, 0, image.height), fill=color, width=8)
        draw_label(draw, (8, 8), text, color, image.width)
        for obj in record["objects"]:
            draw.rectangle(tuple(obj["bbox_xyxy"]), outline=color, width=max(2, min(image.size) // 150))
        return image

    for obj in record["objects"]:
        decision = ACTIVE_DECISIONS.get(obj["object_id"])
        inactive_decision = MIXED_IMAGE_INACTIVE_DECISIONS.get(obj["object_id"])
        if inactive_decision:
            proposed_box = obj["bbox_xyxy"]
            color = (190, 35, 45, 255)
            draw.rectangle(tuple(proposed_box), outline=color, width=max(2, min(image.size) // 150))
            draw.line((proposed_box[0], proposed_box[1], proposed_box[2], proposed_box[3]), fill=color, width=6)
            draw.line((proposed_box[2], proposed_box[1], proposed_box[0], proposed_box[3]), fill=color, width=6)
            draw_label(draw, (proposed_box[0], max(0, proposed_box[1] - 28)), f"{obj['object_id']} REMOVE INVALID GT", color, image.width)
            continue
        if decision:
            proposed_box = decision["proposed_bbox_xyxy"]
            if proposed_box is None:
                continue
            changed = decision["proposed_class_id"] != obj["class_id"]
            color = (0, 120, 200, 255) if changed else (0, 150, 70, 255)
            label = f"{obj['object_id']} {decision['proposed_class_name']} | {decision['recommendation']}"
        else:
            proposed_box = obj["bbox_xyxy"]
            color = (0, 150, 70, 255)
            label = f"{obj['object_id']} {obj['class_name']} | KEEP"
        draw.rectangle(tuple(proposed_box), outline=color, width=max(2, min(image.size) // 150))
        draw_label(draw, (proposed_box[0], max(0, proposed_box[1] - 28)), label, color, image.width)
    return image


def make_contact_sheet(image_paths, output_path, title):
    cell_width, cell_height, columns = 520, 400, 3
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, 54 + cell_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), title, font=font(26), fill="black")
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width - 20, cell_height - 54))
        x = (index % columns) * cell_width + 10
        y = 54 + (index // columns) * cell_height + 34
        sheet.paste(image, (x, y))
        draw.text((x, y - 26), path.stem, font=font(17), fill="black")
    sheet.save(output_path, quality=92)


def main():
    generated_at = datetime.now(timezone.utc).isoformat()
    v1 = load_json(V1_PATH)
    v2 = load_json(V2_PATH)
    predictions = {name: load_json(path) for name, path in PREDICTION_PATHS.items()}
    v1_by_filename = {image["filename"]: image for image in v1["images"]}
    v2_by_filename = {image["filename"]: image for image in v2["images"]}

    active_disputed = []
    zero_active = []
    mixed_image_inactive = []
    for image in v2["images"]:
        active = [obj for obj in image["objects"] if obj["is_active_for_instance_eval"]]
        disputed = [obj for obj in active if obj["v2_status"] != "ACCEPTED"]
        if disputed:
            active_disputed.extend((image, obj) for obj in disputed)
        if not active:
            zero_active.append(image)
        elif any(obj["object_id"] in MIXED_IMAGE_INACTIVE_DECISIONS for obj in image["objects"]):
            mixed_image_inactive.extend(
                (image, obj) for obj in image["objects"] if obj["object_id"] in MIXED_IMAGE_INACTIVE_DECISIONS
            )

    assert len(active_disputed) == 6, len(active_disputed)
    assert len(zero_active) == 10, len(zero_active)
    assert set(ACTIVE_DECISIONS) == {obj["object_id"] for _, obj in active_disputed}
    assert set(MIXED_IMAGE_INACTIVE_DECISIONS) == {obj["object_id"] for _, obj in mixed_image_inactive}
    assert v2["summary"]["active_objects_for_instance_detection"] == 45

    originals_dir = OUT_DIR / "originals"
    current_dir = OUT_DIR / "overlays_current"
    proposed_dir = OUT_DIR / "overlays_proposed"
    for directory in (originals_dir, current_dir, proposed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    review_filenames = []
    for image, _ in active_disputed:
        if image["filename"] not in review_filenames:
            review_filenames.append(image["filename"])
    for image in zero_active:
        if image["filename"] not in review_filenames:
            review_filenames.append(image["filename"])
    for image, _ in mixed_image_inactive:
        if image["filename"] not in review_filenames:
            review_filenames.append(image["filename"])

    visual_manifest = {}
    current_paths = []
    proposed_paths = []
    for ordinal, filename in enumerate(review_filenames, start=1):
        image_record = v2_by_filename[filename]
        source_path = SOURCE_IMAGE_DIR / filename
        assert source_path.exists(), source_path
        assert sha256(source_path) == image_record["sha256"], filename
        safe_name = f"{ordinal:02d}_{safe_stem(filename)}"
        original_path = originals_dir / safe_name
        shutil.copy2(source_path, original_path)
        current_path = current_dir / f"{safe_name}.png"
        proposed_path = proposed_dir / f"{safe_name}.png"
        with Image.open(source_path) as opened:
            base = opened.convert("RGB")
        draw_current(base.copy(), image_record).save(current_path)
        draw_proposed(base.copy(), image_record, image_record in zero_active).save(proposed_path)
        current_paths.append(current_path)
        proposed_paths.append(proposed_path)
        visual_manifest[filename] = {
            "original": original_path.relative_to(OUT_DIR).as_posix(),
            "current_overlay": current_path.relative_to(OUT_DIR).as_posix(),
            "proposed_overlay": proposed_path.relative_to(OUT_DIR).as_posix(),
        }

    make_contact_sheet(current_paths, OUT_DIR / "contact_sheet_current.jpg", "Phase 3 GT review - CURRENT v2 proposal")
    make_contact_sheet(proposed_paths, OUT_DIR / "contact_sheet_proposed.jpg", "Phase 3 GT review - RECOMMENDED decisions")

    current_active_gt = {
        image["filename"]: [obj.copy() for obj in image["objects"] if obj["is_active_for_instance_eval"]]
        for image in v2["images"]
    }
    proposed_gt = {filename: [obj.copy() for obj in objects] for filename, objects in current_active_gt.items()}
    for filename, objects in proposed_gt.items():
        for obj in objects:
            decision = ACTIVE_DECISIONS.get(obj["object_id"])
            if decision and decision["proposed_bbox_xyxy"] is not None:
                obj["class_name"] = decision["proposed_class_name"]
                obj["class_id"] = decision["proposed_class_id"]
                obj["bbox_xyxy"] = decision["proposed_bbox_xyxy"]

    zero_active_names = {image["filename"] for image in zero_active}
    metrics = {}
    object_outcomes = {}
    zero_active_predictions = {}
    for pipeline_name, prediction_data in predictions.items():
        legacy = evaluate(prediction_data, current_active_gt)
        current_explicit = evaluate(prediction_data, current_active_gt, zero_active_names)
        recommended = evaluate(
            prediction_data,
            proposed_gt,
            zero_active_names | EXCLUDE_PENDING_REANNOTATION,
        )
        metrics[pipeline_name] = {
            "legacy_v2_behavior_zero_active_as_negative": {k: v for k, v in legacy.items() if not k.startswith("per_")},
            "current_v2_with_explicit_zero_active_exclusion": {k: v for k, v in current_explicit.items() if not k.startswith("per_")},
            "recommended_if_all_decisions_approved": {k: v for k, v in recommended.items() if not k.startswith("per_")},
        }
        object_outcomes[pipeline_name] = {
            object_id: current_explicit["per_object"].get(object_id, {"status": "IMAGE_EXCLUDED"})
            for object_id in ACTIVE_DECISIONS
        }
        zero_active_predictions[pipeline_name] = {
            image["filename"]: legacy["per_image"].get(image["filename"], {}).get("predictions_ge_0_25", 0)
            for image in zero_active
        }

    active_records = []
    for image, obj in active_disputed:
        decision = ACTIVE_DECISIONS[obj["object_id"]]
        active_records.append(
            {
                "filename": image["filename"],
                "image_sha256": image["sha256"],
                "object_id": obj["object_id"],
                "current_gt": {
                    "class_name": obj["class_name"],
                    "class_id": obj["class_id"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "v2_status": obj["v2_status"],
                    "v2_action": obj["v2_action"],
                    "is_active_for_instance_eval": obj["is_active_for_instance_eval"],
                },
                "recommendation": decision,
                "visuals": visual_manifest[image["filename"]],
                "current_outcomes_at_conf_0_25_iou_0_50": {
                    pipeline: outcomes[obj["object_id"]] for pipeline, outcomes in object_outcomes.items()
                },
                "approval_required": True,
            }
        )

    zero_records = []
    for image in zero_active:
        statuses = sorted({obj["v2_status"] for obj in image["objects"]})
        utility = statuses == ["QUARANTINED_UTILITY_RECEPTACLE"]
        if utility:
            recommendation = "QUARANTINE_PENDING_EXHAUSTIVE_NEGATIVE_REVIEW"
            rationale = (
                "The annotated object is a receptacle. Do not count it as a target and do not count the image "
                "as a clean negative until a reviewer confirms that no target waste is present anywhere in the frame."
            )
        else:
            recommendation = "EXCLUDE_IMAGE_FROM_MAIN_INSTANCE_EVALUATION"
            rationale = (
                "The only v1 object is a pile/full-frame region. It is outside discrete-instance evaluation and "
                "the unannotated pile area makes the image an invalid hard negative."
            )
        zero_records.append(
            {
                "filename": image["filename"],
                "image_sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "inactive_objects": [
                    {
                        "object_id": obj["object_id"],
                        "class_name": obj["class_name"],
                        "class_id": obj["class_id"],
                        "bbox_xyxy": obj["bbox_xyxy"],
                        "v2_status": obj["v2_status"],
                    }
                    for obj in image["objects"]
                ],
                "recommendation": recommendation,
                "rationale": rationale,
                "predictions_ge_0_25_that_legacy_behavior_counts_as_fp": {
                    pipeline: values[image["filename"]] for pipeline, values in zero_active_predictions.items()
                },
                "visuals": visual_manifest[image["filename"]],
                "approval_required": True,
            }
        )

    duplicate_groups = []
    by_hash = {}
    for record in zero_records:
        by_hash.setdefault(record["image_sha256"], []).append(record["filename"])
    for digest, names in by_hash.items():
        if len(names) > 1:
            duplicate_groups.append({"sha256": digest, "filenames": names, "type": "EXACT_CONTENT_DUPLICATE"})

    mixed_inactive_records = []
    for image, obj in mixed_image_inactive:
        mixed_inactive_records.append(
            {
                "filename": image["filename"],
                "image_sha256": image["sha256"],
                "object_id": obj["object_id"],
                "current_gt": {
                    "class_name": obj["class_name"],
                    "class_id": obj["class_id"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "v2_status": obj["v2_status"],
                    "v2_action": obj["v2_action"],
                    "is_active_for_instance_eval": obj["is_active_for_instance_eval"],
                },
                **MIXED_IMAGE_INACTIVE_DECISIONS[obj["object_id"]],
                "visuals": visual_manifest[image["filename"]],
                "approval_required": True,
            }
        )

    manifest = {
        "schema_version": "phase3_gt_review_v1",
        "generated_at_utc": generated_at,
        "decision_state": "PROPOSED_ONLY_NOT_FROZEN",
        "scope": {
            "active_disputed_objects": 6,
            "zero_active_filenames": 10,
            "zero_active_unique_content": len({record["image_sha256"] for record in zero_records}),
            "mixed_image_inactive_objects": len(mixed_inactive_records),
        },
        "source_artifacts": {
            V1_PATH.relative_to(ROOT).as_posix(): sha256(V1_PATH),
            V2_PATH.relative_to(ROOT).as_posix(): sha256(V2_PATH),
            **{path.relative_to(ROOT).as_posix(): sha256(path) for path in PREDICTION_PATHS.values()},
        },
        "evaluation_protocol": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "matching": "one-to-one greedy by descending confidence, same class, maximum IoU",
            "metric_type": "operating-threshold precision/recall/F1; not AP or mAP",
        },
        "proposed_policy": {
            "main_task": "10-class discrete waste instance detection",
            "excluded_image": (
                "Remove both GT and predictions from the main metric. An excluded image contributes no TP, FP, or FN."
            ),
            "ignore_region": (
                "For a mixed image with accepted objects and an ambiguous pile, keep accepted GT; predictions whose "
                "center falls inside an approved ignore region are neither TP nor FP. Ignore regions are never GT objects."
            ),
            "zero_active_pile": (
                "Exclude from main instance evaluation unless exhaustive discrete-instance annotation is completed. "
                "Do not use as a hard negative."
            ),
            "utility_receptacle": (
                "Quarantine. It may become a hard negative only after exhaustive whole-image review confirms no target waste."
            ),
            "pile_task": (
                "If pile detection is a product goal, create a separate pile benchmark/task; do not silently map pile "
                "regions into the fixed 10-class instance benchmark."
            ),
        },
        "active_disputed_objects": active_records,
        "zero_active_images": zero_records,
        "mixed_image_inactive_objects": mixed_inactive_records,
        "exact_duplicate_groups": duplicate_groups,
        "metrics_simulation": metrics,
    }
    manifest_path = OUT_DIR / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validation = {
        "generated_at_utc": generated_at,
        "status": "PASS",
        "checks": {
            "v1_unchanged_sha256": sha256(V1_PATH),
            "v2_unchanged_sha256": sha256(V2_PATH),
            "active_disputed_count": len(active_records),
            "zero_active_filename_count": len(zero_records),
            "zero_active_unique_sha256_count": len({record["image_sha256"] for record in zero_records}),
            "mixed_image_inactive_object_count": len(mixed_inactive_records),
            "all_source_image_hashes_match_v2": True,
            "all_visual_artifacts_exist": all(
                (OUT_DIR / path).exists() for visuals in visual_manifest.values() for path in visuals.values()
            ),
            "gt_files_written": False,
        },
    }
    (OUT_DIR / "validation_results.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_log = {
        "generated_at_utc": generated_at,
        "status": "PASS",
        "output_directory": str(OUT_DIR),
        "active_disputed_objects": len(active_records),
        "zero_active_filenames": len(zero_records),
        "visual_review_filenames": len(review_filenames),
        "files_hashed": None,
        "notes": [
            "No source GT file was modified.",
            "Metrics simulation uses explicit image exclusion semantics and independent same-class matching.",
        ],
    }
    (OUT_DIR / "generation_log.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = sorted(
        path for path in OUT_DIR.rglob("*") if path.is_file() and path.name != "ARTIFACT_SHA256.txt"
    )
    run_log["files_hashed"] = len(artifacts)
    (OUT_DIR / "generation_log.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"{sha256(path)}  {path.relative_to(OUT_DIR).as_posix()}" for path in artifacts]
    (OUT_DIR / "ARTIFACT_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(run_log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
