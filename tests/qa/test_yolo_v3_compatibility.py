import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from verify_yolo_v3_compatibility import verify_yolo_compatibility

def test_yolo_v3_candidate_compatibility():
    """Verify that Dataset v3 Candidate conforms to Ultralytics YOLO format."""
    results = verify_yolo_compatibility()
    
    assert results["data_yaml_exists"] is True, "data.yaml must exist"
    assert results["schema_valid"] is True, f"Schema errors: {results['errors']}"
    assert results["missing_labels"] == 0, f"Missing labels found: {results['missing_labels']}"
    assert results["orphaned_labels"] == 0, f"Orphaned labels found: {results['orphaned_labels']}"
    assert results["total_images"] > 0, "Candidate dataset must contain images"
    assert results["ultralytics_parsable"] is True, "Ultralytics parser must succeed"
    assert len(results["errors"]) == 0, f"Unexpected errors: {results['errors']}"
