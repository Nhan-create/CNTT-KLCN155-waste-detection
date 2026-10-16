"""Regression test for Candidate Manifest and QA State Lifecycle.

Enforces Sprint AI-UPGRADE-01 rules:
1. Lifecycle state validation (no un-downloaded image can be QA_PASS or EMPTY_LABEL).
2. Proper classification of problematic candidates (Leeds box, Street cabinet, Street food remains).
3. Zero candidates approved for training without explicit user consent.
4. Verified sources have authentic URLs, author, license, and size.
"""

import json
from pathlib import Path
import pytest

MANIFEST_PATH = Path("data/acquisition_staging/manifests/batch_a_candidate_manifest.json")
EVIDENCE_PATH = Path("reports/source_verification/wikimedia_raw_api_responses.json")

VALID_VERIFICATION_STATUSES = {
    "VERIFIED",
    "SOURCE_UNVERIFIED",
    "DOMAIN_REVIEW_REQUIRED",
    "BLOCKED_OR_ERROR",
}

VALID_DOWNLOAD_STATUSES = {
    "DOWNLOADED",
    "DOWNLOADED_PENDING_VISUAL_QA",
    "SOURCE_IDENTIFIED",
    "BLOCKED_SOURCE_UNVERIFIED",
    "AWAITING_APPROVAL",
}

def load_manifest():
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)

def test_manifest_structure_and_counts():
    data = load_manifest()
    assert data.get("manifest_name") == "BATCH_A_MINI_BATCH_01_CANDIDATE_WHITELIST"
    candidates = data.get("candidates", [])
    assert len(candidates) == 25, f"Expected exactly 25 candidates, got {len(candidates)}"
    assert EVIDENCE_PATH.exists(), f"Missing raw evidence file: {EVIDENCE_PATH}"

def test_strict_qa_state_lifecycle():
    data = load_manifest()
    candidates = data.get("candidates", [])

    for c in candidates:
        cid = c["candidate_id"]
        v_status = c.get("verification_status")
        d_status = c.get("download_status")
        qa_status = c.get("qa_status")
        user_app = c.get("user_train_approval")

        # Invariant: No candidate may be approved for training yet
        assert user_app != "APPROVED_FOR_TRAIN", f"{cid} cannot be APPROVED_FOR_TRAIN"

        assert v_status in VALID_VERIFICATION_STATUSES, f"{cid} invalid verification_status: {v_status}"
        assert d_status in VALID_DOWNLOAD_STATUSES, f"{cid} invalid download_status: {d_status}"

        # If not downloaded, cannot be QA_PASS or EMPTY_LABEL
        if d_status not in ["DOWNLOADED", "DOWNLOADED_PENDING_VISUAL_QA"]:
            assert qa_status not in ["QA_PASS", "EMPTY_LABEL"], (
                f"{cid} is not downloaded ({d_status}), cannot have qa_status={qa_status}"
            )

        # Verified candidates must have positive size and non-empty URLs
        if v_status == "VERIFIED":
            assert c.get("file_size_bytes", 0) > 0, f"{cid} verified but size <= 0"
            assert c.get("original_url"), f"{cid} verified but missing original_url"
            assert c.get("license"), f"{cid} verified but missing license"

def test_specific_candidate_corrections():
    data = load_manifest()
    c_map = {c["candidate_id"]: c for c in data.get("candidates", [])}

    # 1. Corroded_Screwdriver_Battery.jpg
    bat = c_map["wiki_bat_corroded_screwdriver"]
    assert bat["author"] == "Eoin"
    assert bat["license"] == "CC BY-SA 4.0"
    assert bat["file_size_bytes"] == 536994

    # 2. Street_cabinet.jpg
    cab = c_map["wiki_neg_street_cabinet_01"]
    assert cab["verification_status"] == "DOMAIN_REVIEW_REQUIRED"
    assert cab["qa_status"] == "DOMAIN_REVIEW_REQUIRED"

    # 3. Apple_core.jpg
    apple = c_map["wiki_bio_apple_core_01"]
    assert apple["author"] == "Adrian Zion"
    assert apple["license"] == "Copyrighted free use"

    # 4. Utility box Leeds
    leeds = c_map["wiki_neg_utility_box_leeds"]
    assert leeds["qa_status"] == "PENDING_OBJECT_IDENTITY_REVIEW"

    # 5. Street food remains
    food = c_map["wiki_bio_street_food_remains"]
    assert "BIOLOGICAL" not in food["qa_status"] or "REJECTED" in food["qa_status"]

def test_proposed_mini_batch_size_within_50mib():
    data = load_manifest()
    c_map = {c["candidate_id"]: c for c in data.get("candidates", [])}
    
    proposed_ids = [
        "wiki_bat_corroded_screwdriver",
        "wiki_bio_apple_core_01",
        "wiki_bio_banana_peel_bucharest_02",
        "wiki_bio_banana_peel_sidewalk_06",
        "wiki_bio_banana_peel_bus_stop_03",
        "wiki_bio_banana_peel_oak_hill_05",
        "wiki_bio_food_scraps_bin_09",
        "wiki_neg_utility_box_corner_05",
        "wiki_neg_brick_wall_clean",
        "wiki_multi_banana_and_cup_01",
    ]
    total_bytes = sum(c_map[cid]["file_size_bytes"] for cid in proposed_ids)
    max_allowed = 50 * 1024 * 1024
    assert total_bytes < max_allowed, f"Proposed mini batch {total_bytes} exceeds {max_allowed} bytes"
