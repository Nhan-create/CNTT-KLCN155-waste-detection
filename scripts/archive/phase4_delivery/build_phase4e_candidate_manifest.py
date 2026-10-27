# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Đóng gói hồ sơ minh chứng, manifest và tệp nén bàn giao cho `build_phase4e_candidate_manifest`.
"""
scripts/build_phase4e_candidate_manifest.py
--------------------------------------------
Generates the authoritative Phase 4E Real Waste Data Acquisition Manifest:
data/acquisition_staging/manifests/phase4e_real_waste_catalog.json

Strict Compliance:
- Zero fake IDs, zero auto-approval.
- Clear separation of lifecycle states:
  SOURCE_IDENTIFIED -> DOWNLOADED -> VISUALLY_REVIEWED -> ANNOTATED -> APPROVED_FOR_TRAIN.
- Covers 4 prioritized streams:
  1. BATTERY (Roboflow Universe waste-battery-only/1)
  2. BIOLOGICAL (Real discarded fruit/food waste from Wikimedia Commons)
  3. HARD-NEGATIVE (Utility cabinets & clean urban backgrounds addressing rac_online_012 error)
  4. MULTI-WASTE (Images containing multiple distinct waste objects)
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = ROOT / "data" / "acquisition_staging" / "manifests"
MANIFEST_FILE = MANIFEST_DIR / "phase4e_real_waste_catalog.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    print("=" * 75)
    print("PHASE 4E: BUILDING REAL WASTE DATA CATALOG MANIFEST")
    print("=" * 75)

    catalog: Dict[str, Any] = {
        "manifest_version": "1.0.0-phase4e-real-waste",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project": "CNTT-KLCN155 Waste Detection",
        "target_taxonomy_10_classes": [
            "0: battery",
            "1: biological",
            "2: cardboard",
            "3: clothes",
            "4: glass",
            "5: metal",
            "6: paper",
            "7: plastic",
            "8: shoes",
            "9: trash"
        ],
        "lifecycle_states": [
            "SOURCE_IDENTIFIED: Proven real source record with authentic URL, author, and license.",
            "DOWNLOADED: Image bytes transferred to acquisition_staging with verified SHA-256 and pHash.",
            "VISUALLY_REVIEWED: Evaluated by human/QA against discarded municipal waste criteria.",
            "ANNOTATED: Bounding boxes verified or newly drawn to match 10-class taxonomy.",
            "APPROVED_FOR_TRAIN: Formally accepted into training candidate set (STRICTLY 0 AT PRESENT)."
        ],
        "streams": {}
    }

    # -------------------------------------------------------------
    # STREAM 1: BATTERY
    # -------------------------------------------------------------
    catalog["streams"]["battery"] = {
        "description": "Real discarded batteries with bounding boxes from Roboflow Universe.",
        "source_name": "Roboflow Universe waste-battery-only/1",
        "source_url": "https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1",
        "access_status": "BLOCKED_REQUIRES_ACCESS",
        "block_reason": "HTTP 403 Forbidden due to Cloudflare anti-bot WAF. Automated scraping prohibited.",
        "export_instructions_for_user": [
            "1. Open https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1 in web browser.",
            "2. Click 'Download Dataset' button in top-right corner.",
            "3. Select export format: 'YOLOv8' or 'COCO'.",
            "4. Download the resulting ZIP file to your computer.",
            "5. Place the ZIP into 'data/acquisition_staging/roboflow_battery/' in this workspace.",
            "6. Run ingestion script 'scripts/ingest_roboflow_battery_export.py' to filter and process."
        ],
        "filtering_rules_ready": [
            "Discard synthetic augmentations (.rf.<hash> with extreme blur/shear/noise).",
            "Filter 3D CAD renders; keep only real physical discarded batteries.",
            "Map all internal battery sub-classes into project class 0 'battery'.",
            "Compute SHA-256 and pHash, cross-check against train/val/test/holdout splits."
        ],
        "target_quota": "10-15 verified images",
        "current_downloaded_count": 0,
        "current_approved_count": 0
    }

    # -------------------------------------------------------------
    # STREAM 2: BIOLOGICAL (Real Waste from Wikimedia Commons)
    # -------------------------------------------------------------
    biological_candidates: List[Dict[str, Any]] = [
        {
            "candidate_id": "wiki_bio_apple_core_01",
            "category": "biological",
            "title": "File:Apple_core.jpg",
            "filename": "wiki_bio_apple_core.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/4/48/Apple_core.jpg",
            "license": "Copyrighted free use",
            "author": "Wikimedia Commons Contributor",
            "resolution": "960x1280",
            "size_bytes": 130620,
            "waste_context": "Discarded eaten apple core. Authentic municipal biological waste.",
            "qa_inspection_criteria": "Verify object is genuine eaten/discarded core, not fresh whole apple.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_banana_peel_bucharest_02",
            "category": "biological",
            "title": "File:Banana_peel.jpg",
            "filename": "wiki_bio_banana_peel_bucharest.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Banana_peel.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Tiia Monto",
            "resolution": "3152x2856",
            "size_bytes": 4604017,
            "waste_context": "Banana peel discarded on street asphalt in Bucharest. Authentic litter waste.",
            "qa_inspection_criteria": "Verify discarded peel on asphalt pavement; distinct from edible fruit.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_banana_peel_bus_stop_03",
            "category": "biological",
            "title": "File:Banana_peel_on_seat.JPG",
            "filename": "wiki_bio_banana_peel_seat.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Banana_peel_on_seat.JPG",
            "license": "CC BY-SA 3.0",
            "author": "Lemur12",
            "resolution": "3264x2448",
            "size_bytes": 2293236,
            "waste_context": "Banana peel left on public bus stop bench in London. Authentic urban litter.",
            "qa_inspection_criteria": "Verify discarded skin on public seating; authentic discarded waste.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_banana_peel_ground_04",
            "category": "biological",
            "title": "File:Banana_peel_on_the_ground.jpg",
            "filename": "wiki_bio_banana_peel_ground.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/c/cf/Banana_peel_on_the_ground.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Vladiator360",
            "resolution": "5184x3456",
            "size_bytes": 4573954,
            "waste_context": "Banana peel lying flat on floor/ground. Authentic biological waste.",
            "qa_inspection_criteria": "Verify discarded status on ground level.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_banana_peel_oak_hill_05",
            "category": "biological",
            "title": "File:2019-02-07_13_32_24_A_single_Chiquita_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill,_Fairfax_County,_Virginia.jpg",
            "filename": "wiki_bio_banana_peel_outdoor_soil.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/4/40/2019-02-07_13_32_24_A_single_Chiquita_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Famartin",
            "resolution": "4032x3024",
            "size_bytes": 2933165,
            "waste_context": "Single discarded Chiquita banana peel lying outdoor on grass/soil.",
            "qa_inspection_criteria": "Verify outdoor discarded peel.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_banana_peel_sidewalk_06",
            "category": "biological",
            "title": "File:2020-03-15_01_55_05_A_single_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill,_Fairfax_County,_Virginia.jpg",
            "filename": "wiki_bio_banana_peel_sidewalk.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/2020-03-15_01_55_05_A_single_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Famartin",
            "resolution": "4032x3024",
            "size_bytes": 1208913,
            "waste_context": "Single banana peel discarded on concrete sidewalk edge.",
            "qa_inspection_criteria": "Verify discarded position on concrete sidewalk.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_rotting_banana_07",
            "category": "biological",
            "title": "File:Banano_en_descomposición.JPG",
            "filename": "wiki_bio_rotting_banana.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/5/51/Banano_en_descomposici%C3%B3n.JPG",
            "license": "CC BY-SA 3.0",
            "author": "Pabos95",
            "resolution": "1936x1296",
            "size_bytes": 1224155,
            "waste_context": "Decomposing brown banana fruit discarded in open environment.",
            "qa_inspection_criteria": "Verify decomposition state; discarded agricultural/food waste.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_street_food_remains_08",
            "category": "biological",
            "title": "File:Street_Food_Remains_(188107205).jpeg",
            "filename": "wiki_bio_street_food_remains.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Street_Food_Remains_%28188107205%29.jpeg",
            "license": "CC BY-SA 3.0",
            "author": "Andrew Currie",
            "resolution": "2048x1365",
            "size_bytes": 294623,
            "waste_context": "Discarded leftover street food remnants on pavement.",
            "qa_inspection_criteria": "Verify discarded street food leftovers, distinct from served restaurant meals.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_food_scraps_bin_09",
            "category": "biological",
            "title": "File:Food_Scraps_and_Yard_Debris_Collection_in_Portland_2010_by_Tim_Jewett.JPG",
            "filename": "wiki_bio_food_scraps_bin.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Food_Scraps_and_Yard_Debris_Collection_in_Portland_2010_by_Tim_Jewett.JPG",
            "license": "CC BY-SA 2.5",
            "author": "Tim Jewett",
            "resolution": "2000x3000",
            "size_bytes": 4057650,
            "waste_context": "Kitchen food scraps and organic debris placed into waste collection bin.",
            "qa_inspection_criteria": "Verify municipal organic collection context.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        },
        {
            "candidate_id": "wiki_bio_trash_bucket_remains_10",
            "category": "biological",
            "title": "File:Müll_im_Eimer.JPG",
            "filename": "wiki_bio_trash_bucket_remains.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/f/f8/M%C3%BCll_im_Eimer.JPG",
            "license": "CC BY-SA 4.0",
            "author": "SammyAusDorfen",
            "resolution": "2576x1932",
            "size_bytes": 654692,
            "waste_context": "Mixed municipal waste and food scraps inside a waste bucket.",
            "qa_inspection_criteria": "Inspect organic waste items inside collection bucket.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        }
    ]

    catalog["streams"]["biological"] = {
        "description": "Authentic discarded fruit peels, apple cores, and food waste from Wikimedia Commons.",
        "target_quota": "10-15 verified images",
        "identified_candidates_count": len(biological_candidates),
        "total_estimated_download_bytes": sum(c["size_bytes"] for c in biological_candidates),
        "status": "SOURCE_IDENTIFIED_AWAITING_DOWNLOAD_APPROVAL",
        "candidates": biological_candidates
    }

    # -------------------------------------------------------------
    # STREAM 3: HARD-NEGATIVE (Tailored to Cabinet FP error)
    # -------------------------------------------------------------
    hard_negative_candidates: List[Dict[str, Any]] = [
        {
            "candidate_id": "wiki_neg_street_cabinet_01",
            "category": "hard_negatives",
            "title": "File:Street_cabinet.jpg",
            "filename": "wiki_neg_street_cabinet.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Street_cabinet.jpg",
            "license": "CC0 1.0 Universal (Public Domain)",
            "author": "Francis Flinch",
            "resolution": "627x790",
            "size_bytes": 519652,
            "domain_relevance": "Direct match for green/gray utility cabinet on urban pavement (addresses rac_online_012 FP error).",
            "qa_inspection_criteria": "Inspect entire frame to guarantee 0 waste items; empty 0-byte label file.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": False
        },
        {
            "candidate_id": "wiki_neg_junction_boxes_02",
            "category": "hard_negatives",
            "title": "File:Junction_boxes_-_geograph.org.uk_-_580156.jpg",
            "filename": "wiki_neg_junction_boxes.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Junction_boxes_-_geograph.org.uk_-_580156.jpg",
            "license": "CC BY-SA 2.0",
            "author": "Thomas Nugent",
            "resolution": "480x640",
            "size_bytes": 110051,
            "domain_relevance": "Street corner utility boxes and brick wall pavement without waste.",
            "qa_inspection_criteria": "Inspect 100% of pixels for zero waste objects.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": False
        },
        {
            "candidate_id": "wiki_neg_stromkasten_bretagne_03",
            "category": "hard_negatives",
            "title": "File:20120925_Stromkasten_Bretagne_DSC07740_PtrQs.jpg",
            "filename": "wiki_neg_stromkasten_outdoor.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/e/e9/20120925_Stromkasten_Bretagne_DSC07740_PtrQs.jpg",
            "license": "CC BY 4.0",
            "author": "PtrQs",
            "resolution": "3580x5369",
            "size_bytes": 19590068,
            "domain_relevance": "Large outdoor electrical distribution cabinet on urban sidewalk.",
            "qa_inspection_criteria": "Inspect frame for zero waste objects.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": False
        },
        {
            "candidate_id": "wiki_neg_utility_box_leeds_001_04",
            "category": "hard_negatives",
            "title": "File:Utility_box,_Wesley_Street,_Beeston,_Leeds_(11th_August_2023)_001.jpg",
            "filename": "wiki_neg_utility_box_leeds_001.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/1/13/Utility_box%2C_Wesley_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29_001.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Mtaylor848",
            "resolution": "5152x3864",
            "size_bytes": 5736159,
            "domain_relevance": "Green utility cabinet on sidewalk against brick building.",
            "qa_inspection_criteria": "Inspect frame for zero waste objects.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": False
        },
        {
            "candidate_id": "wiki_neg_utility_box_corner_05",
            "category": "hard_negatives",
            "title": "File:Utility_box_on_the_corner_of_Wesley_Street_and_Town_Street,_Beeston,_Leeds_(11th_August_2023).jpg",
            "filename": "wiki_neg_utility_box_corner.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/b/b7/Utility_box_on_the_corner_of_Wesley_Street_and_Town_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Mtaylor848",
            "resolution": "5152x3864",
            "size_bytes": 6131636,
            "domain_relevance": "Corner sidewalk utility box against masonry wall.",
            "qa_inspection_criteria": "Inspect frame for zero waste objects.",
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": False
        }
    ]

    catalog["streams"]["hard_negatives"] = {
        "description": "Clean street utility cabinets, junction boxes, and walls with zero waste to abolish FP errors (like rac_online_012).",
        "target_quota": "10 difficult hard-negatives",
        "identified_candidates_count": len(hard_negative_candidates),
        "total_estimated_download_bytes": sum(c["size_bytes"] for c in hard_negative_candidates),
        "status": "SOURCE_IDENTIFIED_AWAITING_DOWNLOAD_APPROVAL",
        "candidates": hard_negative_candidates,
        "local_vietnam_user_photography_guideline": {
            "purpose": "Collect 10-15 clean background scenes from Vietnam urban environments matching project camera.",
            "recommended_subjects": [
                "1. Green/gray electrical distribution cabinets (tủ điện kỹ thuật) on sidewalks in HCM/Hanoi.",
                "2. Clean brick walls (tường gạch) and concrete walls without posters or litter.",
                "3. Clean pedestrian pavements (vỉa hè lát gạch) and asphalt roads without trash.",
                "4. Clean empty public trash bins (thùng rác công cộng rỗng) without overflowing litter."
            ],
            "photography_rules": [
                "Ensure zero litter / waste items within the camera view.",
                "Shoot in varied natural lighting (morning, noon, afternoon).",
                "Resolution: standard smartphone (1920x1080 or 4000x3000 JPEG).",
                "Place files in 'data/acquisition_staging/user_captured_negatives/'.",
                "System will inspect pixels, assign 0-byte text labels, and compute SHA/pHash."
            ]
        }
    }

    # -------------------------------------------------------------
    # STREAM 4: MULTI-WASTE
    # -------------------------------------------------------------
    multi_waste_candidates: List[Dict[str, Any]] = [
        {
            "candidate_id": "wiki_multi_banana_and_cup_01",
            "category": "multi_waste",
            "title": "File:Unbinned_litter_at_Sainsbury's,_Chingford,_London,_England_1.jpg",
            "filename": "wiki_multi_banana_and_paper_cup.jpg",
            "source_url": "https://upload.wikimedia.org/wikipedia/commons/a/a3/Unbinned_litter_at_Sainsbury%27s%2C_Chingford%2C_London%2C_England_1.jpg",
            "license": "CC BY-SA 4.0",
            "author": "Acabashi",
            "resolution": "4923x3692",
            "size_bytes": 14942093,
            "waste_context": "Thrown away banana peel (biological) and paper coffee cup (paper) on street gravel.",
            "objects": [
                {"class_id": 1, "class_name": "biological", "item": "banana skin"},
                {"class_id": 6, "class_name": "paper", "item": "takeaway coffee cup"}
            ],
            "status": "SOURCE_IDENTIFIED",
            "requires_annotation": True
        }
    ]

    catalog["streams"]["multi_waste"] = {
        "description": "Multi-waste scenes containing independent bounding boxes across multiple target classes.",
        "identified_candidates_count": len(multi_waste_candidates),
        "total_estimated_download_bytes": sum(c["size_bytes"] for c in multi_waste_candidates),
        "status": "SOURCE_IDENTIFIED_AWAITING_DOWNLOAD_APPROVAL",
        "candidates": multi_waste_candidates
    }

    # -------------------------------------------------------------
    # SUMMARY OF CURRENT PHYSICAL STAGING INVENTORY
    # -------------------------------------------------------------
    catalog["current_physical_inventory"] = {
        "total_downloaded_images": 3,
        "total_downloaded_bytes": 1099474,
        "pilot_preview_images": [
            {
                "image_id": "078821d86db99fc9",
                "filename": "oi_078821d86db99fc9.jpg",
                "category": "biological (Open Images Apple)",
                "size_bytes": 333549,
                "sha256": "6c83db71cd99f14713b71e7f9ba1877447e52dea559266c7f300430dc30b58ec",
                "phash": "8d4671991575fa89",
                "visual_qa_verdict": "REJECT (Living Fruit On Tree)",
                "training_status": "EXCLUDED_CANNOT_TRAIN"
            },
            {
                "image_id": "74a44994e3d6b962",
                "filename": "oi_74a44994e3d6b962.jpg",
                "category": "biological (Open Images Banana)",
                "size_bytes": 633651,
                "sha256": "f9a1a25e273653539e37f1bff8564779d61de90b314e7cbf0988b9570f05175d",
                "phash": "d24a692da59ad2dc",
                "visual_qa_verdict": "REJECT (Prepared Dining Food)",
                "training_status": "EXCLUDED_CANNOT_TRAIN"
            },
            {
                "image_id": "e29db1c1963628e7",
                "filename": "oi_e29db1c1963628e7.jpg",
                "category": "hard_negatives (Open Images Building)",
                "size_bytes": 132274,
                "sha256": "b769f9e141936b18ec82bab2f51bd55d5fb67d0ed02d600d91220eb0fc232d3f",
                "phash": "d8bd810f6d437e28",
                "visual_qa_verdict": "PENDING_NEGATIVE_REVIEW (Clean scenic monument, but out-of-domain for urban utility cabinets)",
                "training_status": "PENDING_USER_DECISION"
            }
        ],
        "training_expansion_status": {
            "approved_for_training": 0,
            "merged_into_v2": 0,
            "merged_into_v3": 0,
            "active_training_runs": 0
        }
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest successfully created: {MANIFEST_FILE}")
    print(f"Total Streams: {len(catalog['streams'])}")
    print(f"Total Identified Real Candidates: {len(biological_candidates) + len(hard_negative_candidates) + len(multi_waste_candidates)} files")
    print(f"Current Approved for Training: 0 (Strictly Fail-Closed)")
    print("=" * 75)


if __name__ == "__main__":
    main()
