from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.download_models import (
    PINNED_RELEASE_TAG,
    compute_sha256,
    download_release_models,
    load_manifest,
)


def _build_mock_zip(zip_path: Path, files: dict[str, bytes]) -> tuple[int, str]:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in files.items():
            zf.writestr(arcname, payload)
    return zip_path.stat().st_size, compute_sha256(zip_path)


def test_manifest_is_pinned_to_specific_release_version() -> None:
    manifest = load_manifest()
    assert manifest["release_tag"] == PINNED_RELEASE_TAG == "v1.0.0-models"
    for pkg in manifest["packages"].values():
        assert f"/releases/download/{PINNED_RELEASE_TAG}/" in pkg["download_url"]
        assert "/releases/latest/" not in pkg["download_url"]
        assert len(pkg["sha256"]) == 64

    paths = [m["relative_path"] for m in manifest["models"]]
    assert paths == [
        "artifacts/official_run/best_model.pt",
        "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt",
        "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
        "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt",
    ]


def test_downloader_verifies_sha256_and_refuses_silent_overwrite(
    tmp_path: Path,
) -> None:
    fake_cls = b"classifier-weights-payload"
    fake_e2 = b"ssdlite-e2-weights-payload"
    fake_e1 = b"yolov8n-e1-weights-payload"
    fake_e3 = b"yolov8n-e3-pilot-payload"

    zip_dir = tmp_path / "remote_zips"
    zip_dir.mkdir()
    app_zip = zip_dir / "app.zip"
    app_size, app_sha = _build_mock_zip(
        app_zip,
        {
            "artifacts/official_run/best_model.pt": fake_cls,
            "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt": fake_e2,
            "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt": fake_e1,
        },
    )
    e3_zip = zip_dir / "e3.zip"
    e3_size, e3_sha = _build_mock_zip(
        e3_zip,
        {
            "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt": fake_e3,
        },
    )

    custom_manifest = {
        "schema_version": "1.0",
        "repository": "https://github.com/Nhan-create/CNTT-KLCN155-waste-detection",
        "release_tag": PINNED_RELEASE_TAG,
        "release_url": f"https://github.com/Nhan-create/CNTT-KLCN155-waste-detection/releases/tag/{PINNED_RELEASE_TAG}",
        "packages": {
            "app-models": {
                "archive_name": "app.zip",
                "download_url": "https://example.invalid/app.zip",
                "size_bytes": app_size,
                "sha256": app_sha,
                "experimental": False,
                "files": [
                    "artifacts/official_run/best_model.pt",
                    "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt",
                    "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
                ],
            },
            "e3-pilot-experimental": {
                "archive_name": "e3.zip",
                "download_url": "https://example.invalid/e3.zip",
                "size_bytes": e3_size,
                "sha256": e3_sha,
                "experimental": True,
                "files": [
                    "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt",
                ],
            },
        },
        "models": [
            {
                "id": "cls",
                "name": "Classifier",
                "role": "cls",
                "experimental": False,
                "default_in_app": True,
                "relative_path": "artifacts/official_run/best_model.pt",
                "package": "app-models",
                "size_bytes": len(fake_cls),
                "sha256": hashlib.sha256(fake_cls).hexdigest(),
            },
            {
                "id": "e2",
                "name": "E2",
                "role": "e2",
                "experimental": False,
                "default_in_app": True,
                "relative_path": "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt",
                "package": "app-models",
                "size_bytes": len(fake_e2),
                "sha256": hashlib.sha256(fake_e2).hexdigest(),
            },
            {
                "id": "e1",
                "name": "E1",
                "role": "e1",
                "experimental": False,
                "default_in_app": True,
                "relative_path": "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
                "package": "app-models",
                "size_bytes": len(fake_e1),
                "sha256": hashlib.sha256(fake_e1).hexdigest(),
            },
            {
                "id": "e3",
                "name": "E3",
                "role": "e3",
                "experimental": True,
                "default_in_app": False,
                "relative_path": "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt",
                "package": "e3-pilot-experimental",
                "size_bytes": len(fake_e3),
                "sha256": hashlib.sha256(fake_e3).hexdigest(),
            },
        ],
    }
    manifest_file = tmp_path / "manifest.json"
    import json

    manifest_file.write_text(json.dumps(custom_manifest), encoding="utf-8")

    def fake_fetch(url: str, dest: Path) -> None:
        if url.endswith("app.zip"):
            dest.write_bytes(app_zip.read_bytes())
        else:
            dest.write_bytes(e3_zip.read_bytes())

    target_root = tmp_path / "workspace"
    summary = download_release_models(
        target_root,
        manifest_path=manifest_file,
        downloader=fake_fetch,
    )
    assert len(summary["models"]) == 3
    assert not (
        target_root / "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt"
    ).exists()

    # Corrupt/modify one existing weight file on disk and confirm overwrite is blocked without --force
    cls_file = target_root / "artifacts/official_run/best_model.pt"
    cls_file.write_bytes(b"different-local-weights")
    with pytest.raises(RuntimeError, match="Refusing to overwrite existing weights"):
        download_release_models(
            target_root,
            manifest_path=manifest_file,
            downloader=fake_fetch,
        )
    assert cls_file.read_bytes() == b"different-local-weights"

    # With --force, it restores the verified release checkpoint
    download_release_models(
        target_root,
        force=True,
        manifest_path=manifest_file,
        downloader=fake_fetch,
    )
    assert cls_file.read_bytes() == fake_cls
