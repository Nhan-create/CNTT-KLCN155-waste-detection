"""Independent consistency checks for the Phase 3 GT review package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((HERE / "review_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((HERE / "validation_results.json").read_text(encoding="utf-8"))

    assert manifest["decision_state"] == "PROPOSED_ONLY_NOT_FROZEN"
    assert len(manifest["active_disputed_objects"]) == 6
    assert len(manifest["zero_active_images"]) == 10
    assert len(manifest["mixed_image_inactive_objects"]) == 2
    assert manifest["scope"]["zero_active_unique_content"] == 9
    assert len(manifest["exact_duplicate_groups"]) == 1
    assert validation["status"] == "PASS"
    assert validation["checks"]["gt_files_written"] is False

    for relative_path, expected_hash in manifest["source_artifacts"].items():
        path = ROOT / relative_path
        assert path.exists(), path
        assert sha256(path) == expected_hash, path

    for record in manifest["active_disputed_objects"] + manifest["zero_active_images"] + manifest["mixed_image_inactive_objects"]:
        for relative_path in record["visuals"].values():
            assert (HERE / relative_path).exists(), relative_path

    hash_lines = (HERE / "ARTIFACT_SHA256.txt").read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in hash_lines:
        expected_hash, relative_path = line.split("  ", 1)
        path = HERE / relative_path
        assert path.exists(), relative_path
        assert sha256(path) == expected_hash, relative_path
        checked += 1

    result = {
        "status": "PASS",
        "artifact_hashes_checked": checked,
        "source_hashes_checked": len(manifest["source_artifacts"]),
        "active_disputed_objects": 6,
        "zero_active_filenames": 10,
        "mixed_image_inactive_objects": 2,
        "gt_v1_sha256": validation["checks"]["v1_unchanged_sha256"],
        "gt_v2_sha256": validation["checks"]["v2_unchanged_sha256"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
