"""
scripts/prepare_acquisition_pilot.py
------------------------------------
Controlled Pilot Data Acquisition Manager:
1. Validates candidate manifest structure, licenses, and filenames.
2. Mathematically verifies zero intersection with authoritative train, val, test, and holdout splits.
3. Strict Safety: Real data downloading is BLOCKED unless explicit user approval flag
   `--approve-pilot-download` is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.export_train_manifest import load_authoritative_splits  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "acquisition_staging" / "manifests" / "candidate_acquisition_manifest.json"
STAGING_DIR = ROOT / "data" / "acquisition_staging"


def load_candidate_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate acquisition manifest at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_candidate_leakage(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Verify zero overlap between candidates and authoritative splits."""
    test_set, val_set, train_set, holdout_set = load_authoritative_splits()

    candidates = manifest.get("candidates", [])
    violations = []
    seen_names: Set[str] = set()

    for item in candidates:
        fn = item.get("proposed_filename") or Path(item.get("staging_target_file", "")).name
        cid = item.get("candidate_id", "unknown")
        sid = item.get("source_image_id", "")
        url = item.get("original_url", "")

        if not fn:
            violations.append({"id": cid, "error": "Empty filename"})
            continue
        if not sid:
            violations.append({"id": cid, "error": "Missing real source_image_id"})
        if not url:
            violations.append({"id": cid, "error": "Missing real original_url"})

        if fn in seen_names:
            violations.append({"id": cid, "error": f"Duplicate filename {fn}"})
        seen_names.add(fn)

        if fn in test_set:
            violations.append({"id": cid, "error": f"{fn} collides with TEST split"})
        if fn in val_set:
            violations.append({"id": cid, "error": f"{fn} collides with VAL split"})
        if fn in train_set:
            violations.append({"id": cid, "error": f"{fn} collides with TRAIN split"})
        if fn in holdout_set:
            violations.append({"id": cid, "error": f"{fn} collides with HOLDOUT manifest"})

    return {
        "total_candidates": len(candidates),
        "unique_filenames": len(seen_names),
        "leakage_violations_count": len(violations),
        "violations": violations,
        "is_leakage_free": len(violations) == 0
    }


def main():
    parser = argparse.ArgumentParser(description="Audit and prepare candidate acquisition pilot staging.")
    parser.add_argument("--check-manifest", action="store_true", help="Validate candidate manifest and verify split isolation")
    parser.add_argument("--download", "--execute-download", action="store_true", dest="execute_download", help="Request download execution")
    parser.add_argument("--approve-pilot-download", action="store_true", help="Explicit user approval confirmation required to download")

    args = parser.parse_args()

    manifest = load_candidate_manifest()
    leakage = audit_candidate_leakage(manifest)

    if args.check_manifest or not args.execute_download:
        print("=" * 65)
        print("CANDIDATE ACQUISITION PILOT MANIFEST AUDIT REPORT")
        print("=" * 65)
        total_target = manifest["manifest_metadata"].get("total_active_candidates", 5)
        proven_oi = manifest["manifest_metadata"].get("total_source_identified_images", 3)
        blocked_rf = manifest["manifest_metadata"].get("total_blocked_access_requests", 2)
        rejected_hist = manifest["manifest_metadata"].get("total_rejected_unverified_history_images", 0)

        print(f"Manifest Version:     {manifest['manifest_metadata']['version']}")
        print(f"Lifecycle Stage:      {manifest['manifest_metadata'].get('current_stage', 'ACQUISITION_TARGET')}")
        print(f"Active Candidates:    {total_target} (Proven OpenImages: {proven_oi}, Blocked Roboflow: {blocked_rf})")
        print(f"Rejected History:     {rejected_hist} unverified items quarantined")
        print(f"Active Breakdown:")
        breakdown = manifest["manifest_metadata"].get("breakdown_active", manifest["manifest_metadata"].get("breakdown", {}))
        for k, v in breakdown.items():
            print(f"  - {k}: {v}")
        print(f"Leakage Audit Status: {'PASSED (Zero Overlap)' if leakage['is_leakage_free'] else 'FAILED'}")
        if not leakage["is_leakage_free"]:
            for viol in leakage["violations"]:
                print(f"  [VIOLATION] {viol['id']}: {viol['error']}")
            sys.exit(1)
        print(f"Staging Environment:  {STAGING_DIR}")
        print("Status:               AWAITING_USER_APPROVAL")
        print("=" * 65)

    if args.execute_download:
        if not args.approve_pilot_download:
            print(
                "\n[ACTION BLOCKED] Real data downloading requires explicit user approval flag "
                "'--approve-pilot-download'. No files downloaded.\n",
                file=sys.stderr
            )
            sys.exit(1)
        else:
            print(
                "\n[STATUS: NOT_IMPLEMENTED / AWAITING_SHORTLIST_CONFIRMATION]\n"
                "User approval confirmation recognized, but network download worker is NOT yet executed.\n"
                "To prevent unverified downloads, individual image whitelist must be confirmed by user.\n"
                "Zero files were downloaded. Staging directory remains 0 bytes.\n"
            )
            sys.exit(0)


if __name__ == "__main__":
    main()
