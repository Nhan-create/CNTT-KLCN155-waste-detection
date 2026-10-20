"""Download and verify trained model checkpoints from the pinned GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

PINNED_RELEASE_TAG = "v1.0.0-models"
DEFAULT_MANIFEST_PATH = Path("configs/model_release_manifest.json")

EMBEDDED_MANIFEST: dict[str, Any] = {
    "schema_version": "1.0",
    "repository": "https://github.com/Nhan-create/CNTT-KLCN155-waste-detection",
    "release_tag": PINNED_RELEASE_TAG,
    "release_url": (
        "https://github.com/Nhan-create/CNTT-KLCN155-waste-detection/"
        f"releases/tag/{PINNED_RELEASE_TAG}"
    ),
    "packages": {
        "app-models": {
            "archive_name": "cntt-klcn155-app-models-v1.0.0.zip",
            "download_url": (
                "https://github.com/Nhan-create/CNTT-KLCN155-waste-detection/"
                f"releases/download/{PINNED_RELEASE_TAG}/cntt-klcn155-app-models-v1.0.0.zip"
            ),
            "size_bytes": 77621360,
            "size_mb": 74.03,
            "sha256": "1293dca3b93dadb0b4866d81037c954f7908f3207166c3245f134c8c2335ad2c",
            "description": (
                "Gói trọng số chính thức chạy ứng dụng Streamlit "
                "(MobileNetV3-Large Classifier + E2 SSDLite320 Primary + E1 YOLOv8n Benchmark). "
                "Giữ nguyên cấu trúc thư mục artifacts/..."
            ),
            "experimental": False,
            "files": [
                "artifacts/official_run/best_model.pt",
                "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt",
                "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
            ],
        },
        "e3-pilot-experimental": {
            "archive_name": "cntt-klcn155-e3-pilot-experimental-v1.0.0.zip",
            "download_url": (
                "https://github.com/Nhan-create/CNTT-KLCN155-waste-detection/"
                f"releases/download/{PINNED_RELEASE_TAG}/"
                "cntt-klcn155-e3-pilot-experimental-v1.0.0.zip"
            ),
            "size_bytes": 5656034,
            "size_mb": 5.39,
            "sha256": "613c1cd5c5fb64d7cdbaf7215351a0a2e0e4cfdf9ca72a4ee3da61445c4c5643",
            "description": (
                "Gói trọng số thử nghiệm E3 YOLOv8n v3 Pilot "
                "(đóng gói riêng, KHÔNG thay thế mô hình mặc định của ứng dụng)."
            ),
            "experimental": True,
            "files": [
                "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt",
            ],
        },
    },
    "models": [
        {
            "id": "classifier_mobilenetv3_large",
            "name": "MobileNetV3-Large 10-Class Classifier (Stage-1 & Two-Stage Verifier)",
            "role": "application_default_classifier",
            "experimental": False,
            "default_in_app": True,
            "relative_path": "artifacts/official_run/best_model.pt",
            "package": "app-models",
            "size_bytes": 50929856,
            "size_mb": 48.57,
            "sha256": "c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424",
        },
        {
            "id": "detector_e2_ssdlite320_10cls",
            "name": "SSDLite320-MobileNetV3 10-Class Detector (E2 - Primary Model)",
            "role": "application_default_primary_detector",
            "experimental": False,
            "default_in_app": True,
            "relative_path": "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt",
            "package": "app-models",
            "size_bytes": 28459990,
            "size_mb": 27.14,
            "sha256": "c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7",
        },
        {
            "id": "detector_e1_yolov8n_10cls",
            "name": "YOLOv8n 10-Class Detector (E1 - Benchmark Model)",
            "role": "application_default_benchmark_detector",
            "experimental": False,
            "default_in_app": True,
            "relative_path": "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
            "package": "app-models",
            "size_bytes": 6228451,
            "size_mb": 5.94,
            "sha256": "7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8",
        },
        {
            "id": "detector_e3_yolov8n_pilot",
            "name": "YOLOv8n v3 Pilot 10-Class Detector (E3 - Experimental Pilot Only)",
            "role": "experimental_pilot_detector",
            "experimental": True,
            "default_in_app": False,
            "relative_path": "artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt",
            "package": "e3-pilot-experimental",
            "size_bytes": 6211491,
            "size_mb": 5.92,
            "sha256": "e451cb8d0a0faab9374f5dac90811bba681b2827d51aba07f6e6a120b5364169",
        },
    ],
}


def compute_sha256(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load the release manifest from disk or fall back to the pinned embedded contract."""
    candidate = manifest_path or DEFAULT_MANIFEST_PATH
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if data.get("release_tag") != PINNED_RELEASE_TAG:
            raise RuntimeError(
                f"Manifest release_tag mismatch: expected {PINNED_RELEASE_TAG}, "
                f"got {data.get('release_tag')}"
            )
        return data
    return json.loads(json.dumps(EMBEDDED_MANIFEST))


def _resolve_safe_member_path(target_root: Path, member_name: str) -> Path:
    """Reject Zip-Slip / path traversal and return the resolved destination path."""
    if "\\" in member_name or member_name.startswith("/"):
        raise RuntimeError(f"Unsafe archive entry path rejected: {member_name}")
    posix_path = PurePosixPath(member_name)
    if posix_path.is_absolute() or any(part in ("..", "") for part in posix_path.parts):
        raise RuntimeError(f"Unsafe archive entry path rejected: {member_name}")
    root_resolved = target_root.resolve()
    dest_resolved = (root_resolved / Path(*posix_path.parts)).resolve()
    try:
        dest_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeError(
            f"Archive entry escapes target root: {member_name}"
        ) from error
    return dest_resolved


def _default_url_downloader(url: str, destination: Path) -> None:
    """Download a URL with a standard User-Agent header into destination."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CNTT-KLCN155-ModelDownloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as output_file:
        shutil.copyfileobj(response, output_file, length=1024 * 1024)


def check_existing_models(
    target_root: Path,
    models_meta: list[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Inspect existing model files on disk and block silent overwrite of differing weights."""
    statuses: dict[str, str] = {}
    for model in models_meta:
        rel_path = model["relative_path"]
        dest = _resolve_safe_member_path(target_root, rel_path)
        if not dest.exists():
            statuses[rel_path] = "missing"
            continue
        if not dest.is_file():
            raise RuntimeError(
                f"Target path exists but is not a regular file: {dest}"
            )
        actual_sha = compute_sha256(dest)
        actual_size = dest.stat().st_size
        expected_sha = model["sha256"].lower()
        expected_size = int(model["size_bytes"])
        if actual_sha == expected_sha and actual_size == expected_size:
            statuses[rel_path] = "verified_present"
        else:
            if not force:
                raise RuntimeError(
                    f"Existing checkpoint at '{rel_path}' has different content "
                    f"(expected sha256={expected_sha[:12]}..., found sha256={actual_sha[:12]}...). "
                    "Refusing to overwrite existing weights without --force."
                )
            statuses[rel_path] = "overwrite_forced"
    return statuses


def extract_verified_package(
    archive_path: Path,
    package_meta: dict[str, Any],
    models_by_path: dict[str, dict[str, Any]],
    target_root: Path,
) -> list[dict[str, Any]]:
    """Verify archive SHA-256 and safely extract only expected model checkpoints."""
    expected_archive_sha = package_meta["sha256"].lower()
    expected_archive_size = int(package_meta["size_bytes"])
    actual_archive_size = archive_path.stat().st_size
    actual_archive_sha = compute_sha256(archive_path)
    if actual_archive_sha != expected_archive_sha or actual_archive_size != expected_archive_size:
        raise RuntimeError(
            f"Archive verification failed for {package_meta['archive_name']}: "
            f"expected (size={expected_archive_size}, sha256={expected_archive_sha}), "
            f"got (size={actual_archive_size}, sha256={actual_archive_sha})"
        )

    expected_files = set(package_meta["files"])
    extracted_records: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        archive_members = [info.filename for info in zf.infolist() if not info.is_dir()]
        for member in archive_members:
            _resolve_safe_member_path(target_root, member)
            if member not in expected_files:
                raise RuntimeError(
                    f"Unexpected file '{member}' inside {package_meta['archive_name']}"
                )
        missing_in_zip = expected_files - set(archive_members)
        if missing_in_zip:
            raise RuntimeError(
                f"Archive {package_meta['archive_name']} is missing expected files: "
                f"{sorted(missing_in_zip)}"
            )

        for rel_path in package_meta["files"]:
            model_meta = models_by_path[rel_path]
            dest_path = _resolve_safe_member_path(target_root, rel_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_dest = dest_path.with_suffix(dest_path.suffix + ".part")
            try:
                with zf.open(rel_path, "r") as src_stream, temp_dest.open("wb") as dst_stream:
                    shutil.copyfileobj(src_stream, dst_stream, length=1024 * 1024)
                extracted_size = temp_dest.stat().st_size
                extracted_sha = compute_sha256(temp_dest)
                if (
                    extracted_sha != model_meta["sha256"].lower()
                    or extracted_size != int(model_meta["size_bytes"])
                ):
                    raise RuntimeError(
                        f"Extracted file checksum mismatch for {rel_path}: "
                        f"expected (size={model_meta['size_bytes']}, sha256={model_meta['sha256']}), "
                        f"got (size={extracted_size}, sha256={extracted_sha})"
                    )
                temp_dest.replace(dest_path)
            finally:
                if temp_dest.exists():
                    temp_dest.unlink(missing_ok=True)

            extracted_records.append(
                {
                    "relative_path": rel_path,
                    "name": model_meta["name"],
                    "experimental": bool(model_meta["experimental"]),
                    "size_bytes": extracted_size,
                    "sha256": extracted_sha,
                    "status": "downloaded_and_verified",
                }
            )
    return extracted_records


def download_release_models(
    target_root: Path = Path("."),
    *,
    include_e3_pilot: bool = False,
    only_e3_pilot: bool = False,
    verify_only: bool = False,
    force: bool = False,
    manifest_path: Path | None = None,
    downloader: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Download, extract, and verify pinned release models into target_root."""
    manifest = load_manifest(manifest_path)
    packages_to_process = []
    if only_e3_pilot:
        packages_to_process.append("e3-pilot-experimental")
    else:
        packages_to_process.append("app-models")
        if include_e3_pilot:
            packages_to_process.append("e3-pilot-experimental")

    models_by_path = {m["relative_path"]: m for m in manifest["models"]}
    selected_models = [
        m for m in manifest["models"] if m["package"] in packages_to_process
    ]

    statuses = check_existing_models(target_root, selected_models, force=force)
    if verify_only:
        missing = [path for path, st in statuses.items() if st != "verified_present"]
        if missing:
            raise RuntimeError(
                f"Verification failed; missing or unverified model files: {missing}"
            )
        return {
            "release_tag": manifest["release_tag"],
            "target_root": str(target_root.resolve()),
            "mode": "verify_only",
            "models": [
                {
                    "relative_path": m["relative_path"],
                    "name": m["name"],
                    "experimental": bool(m["experimental"]),
                    "size_bytes": m["size_bytes"],
                    "sha256": m["sha256"],
                    "status": statuses[m["relative_path"]],
                }
                for m in selected_models
            ],
        }

    fetch_fn = downloader or _default_url_downloader
    results: list[dict[str, Any]] = []

    for pkg_key in packages_to_process:
        pkg_meta = manifest["packages"][pkg_key]
        pkg_files = pkg_meta["files"]
        if not force and all(statuses.get(f) == "verified_present" for f in pkg_files):
            for f in pkg_files:
                m = models_by_path[f]
                results.append(
                    {
                        "relative_path": f,
                        "name": m["name"],
                        "experimental": bool(m["experimental"]),
                        "size_bytes": m["size_bytes"],
                        "sha256": m["sha256"],
                        "status": "already_present_verified",
                    }
                )
            continue

        with tempfile.TemporaryDirectory(prefix="cntt_klcn155_models_") as tmp_dir:
            archive_path = Path(tmp_dir) / pkg_meta["archive_name"]
            fetch_fn(pkg_meta["download_url"], archive_path)
            if not archive_path.is_file():
                raise RuntimeError(
                    f"Download did not produce archive file: {pkg_meta['download_url']}"
                )
            extracted = extract_verified_package(
                archive_path, pkg_meta, models_by_path, target_root
            )
            results.extend(extracted)

    return {
        "release_tag": manifest["release_tag"],
        "release_url": manifest["release_url"],
        "target_root": str(target_root.resolve()),
        "included_e3_experimental": "e3-pilot-experimental" in packages_to_process,
        "models": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tải và kiểm tra SHA-256 bộ trọng số mô hình đã huấn luyện từ GitHub Release "
            f"({PINNED_RELEASE_TAG})."
        )
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("."),
        help="Thư mục gốc của dự án để giải nén cấu trúc artifacts/... (mặc định: thư mục hiện tại).",
    )
    parser.add_argument(
        "--include-e3-pilot",
        action="store_true",
        help=(
            "Tải thêm gói mô hình thử nghiệm E3 YOLOv8n v3 Pilot "
            "(đóng gói riêng, KHÔNG thay thế mô hình mặc định của ứng dụng)."
        ),
    )
    parser.add_argument(
        "--only-e3-pilot",
        action="store_true",
        help="Chỉ tải gói mô hình thử nghiệm E3 YOLOv8n v3 Pilot.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Chỉ kiểm tra SHA-256 và dung lượng các file trọng số hiện có trên máy, không tải mới.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phép ghi đè nếu trên máy đang có file trọng số khác SHA-256.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Đường dẫn tới file model_release_manifest.json (tùy chọn).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)
    try:
        summary = download_release_models(
            args.target_root,
            include_e3_pilot=args.include_e3_pilot,
            only_e3_pilot=args.only_e3_pilot,
            verify_only=args.verify_only,
            force=args.force,
            manifest_path=args.manifest,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if summary.get("included_e3_experimental"):
        print(
            "NOTE: E3 (artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt) "
            "là bản thử nghiệm (experimental pilot), KHÔNG thay thế mô hình mặc định của ứng dụng.",
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
