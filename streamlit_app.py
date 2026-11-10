# Mô tả: Điểm vào chính của ứng dụng Web Streamlit gồm 2 Tab: Tab 1 (Phân loại rác đơn vật thể bằng MobileNetV3-Large) và Tab 2 (Phát hiện đa vật thể bằng SSDLite320 / YOLOv8n / WBF, thống kê số lượng từng loại rác và xuất gói .zip).
"""Phase-2 Streamlit frontend: 10-class waste detection & high-precision classification."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from src.app.classifier_service import MobileNetClassifier
from src.app.detection_logic import (
    ImageDetectionOutput,
    analyze_image_file,
    archive_annotated_images,
    build_detection_pipeline,
)
from src.app.detection_service import DetectionService
from src.app.detection_settings import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_YOLO_PATH,
    DETECTION_BACKENDS,
    DetectionWebSettings,
    VERIFIED_YOLO_PATH,
    detector_input_threshold,
    detector_problem,
    load_per_class_thresholds,
    parse_detection_settings,
    paths_for_backend,
)
from src.app.streamlit_results import (
    render_classification_result,
    render_detection_result,
    render_status,
)
from src.app.style import ICON_FREE_CSS, transparent_favicon
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.visualization import detection_label

st.set_page_config(
    page_title="Nhận diện rác trong ảnh",
    page_icon=transparent_favicon(),
    layout="wide",
)
st.markdown(ICON_FREE_CSS, unsafe_allow_html=True)

MODEL_NAMES = {
    "yolov8n": "YOLOv8n (Mô hình đối chứng)",
    "ssdlite": "SSDLite320-MobileNetV3 (Mô hình chính)",
    "wbf": "Hợp nhất SSDLite + YOLOv8n (WBF)",
}


def resolve_slider_default_confidence(settings: DetectionWebSettings) -> float:
    return round(min(1.0, max(0.10, float(settings.confidence_threshold))), 2)


def resolve_active_per_class_thresholds(
    settings: DetectionWebSettings,
    confidence: float,
    enable_calibrated_thresholds: bool,
    calibrated_path: Path = Path("configs/detection_thresholds_calibrated.json"),
) -> tuple[tuple[str, float], ...]:
    if not enable_calibrated_thresholds:
        return ()
    if settings.per_class_thresholds:
        return settings.per_class_thresholds
    if not calibrated_path.is_file():
        return ()
    try:
        base_thresholds = load_per_class_thresholds(calibrated_path)
        if confidence >= 0.99:
            return tuple((name, 1.0) for name, _ in base_thresholds)
        scale = confidence / 0.25
        floor = confidence if confidence >= 0.35 else 0.0
        return tuple(
            (name, min(1.0, max(floor, round(val * scale, 3))))
            for name, val in base_thresholds
        )
    except Exception:
        return ()


@st.cache_resource(show_spinner=False)
def cached_classifier(
    checkpoint_path: Path | str = DEFAULT_CLASSIFIER_PATH,
    device: str = "auto",
    checkpoint_version: tuple[int, int] | None = None,
    prefer_onnx: bool = True,
    _ui_version: int = 2,
) -> MobileNetClassifier:
    return MobileNetClassifier(checkpoint_path, device=device, prefer_onnx=prefer_onnx)


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_detector(
    backend: str,
    model_path: str,
    secondary_model_path: str | None,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int | None,
    max_detections: int,
    fusion_weights: tuple[float, float],
    fusion_iou_threshold: float,
    fusion_input_threshold: float,
    checkpoint_versions: tuple[tuple[int, int], ...],
    pipeline_settings: DetectionWebSettings | None = None,
    _pipeline_version: int = 7,
) -> DetectionService:
    from src.detection.factory import create_detector

    detector = create_detector(
        backend,
        model_path,
        secondary_model_path=secondary_model_path,
        device=device,
        confidence_threshold=(
            detector_input_threshold(pipeline_settings)
            if pipeline_settings is not None else confidence_threshold
        ),
        iou_threshold=iou_threshold,
        image_size=image_size,
        max_detections=max_detections,
        fusion_weights=fusion_weights,
        fusion_iou_threshold=fusion_iou_threshold,
        fusion_input_threshold=fusion_input_threshold,
    )
    if pipeline_settings is not None:
        classifier = None
        if pipeline_settings.enable_two_stage:
            classifier_path = pipeline_settings.classifier_path
            version = classifier_path.stat()
            classifier = cached_classifier(
                str(classifier_path), device=pipeline_settings.device,
                checkpoint_version=(version.st_mtime_ns, version.st_size), prefer_onnx=False,
            )
        detector = build_detection_pipeline(detector, pipeline_settings, classifier=classifier)
    return DetectionService(detector)


def main() -> None:
    st.markdown(
        '<div class="top-app-banner">'
        '<span>Ứng dụng Web Streamlit - Hệ thống nhận diện và phân loại 10 nhóm rác thải (http://localhost:8501)</span>'
        '<span class="banner-right">Backend: MobileNetV3-Large / SSDLite320 / YOLOv8n / WBF</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.title("Nhận diện rác trong ảnh")
    st.caption("Khóa luận tốt nghiệp CNTT: Hệ thống phân loại 10 nhóm rác thải và phát hiện vị trí rác trong ảnh.")

    try:
        settings = parse_detection_settings(sys.argv[1:], os.environ)
    except (ValueError, SystemExit):
        render_status("Cấu hình ứng dụng không hợp lệ. Vui lòng kiểm tra lại.", "error")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">TÙY CHỌN MÔ HÌNH PHÁT HIỆN</div>', unsafe_allow_html=True)
        backend = st.selectbox(
            "Mô hình",
            DETECTION_BACKENDS,
            index=DETECTION_BACKENDS.index(settings.backend),
            format_func=lambda name: MODEL_NAMES[name],
        )
        model_path, secondary_path = paths_for_backend(settings, backend)
        if backend == "yolov8n":
            if (
                settings.selection_path is None
                and not settings.yolo_checkpoint_explicit
                and model_path == DEFAULT_YOLO_PATH
                and VERIFIED_YOLO_PATH.is_file()
                and st.checkbox(
                    "Dùng YOLOv8n đã đánh giá trên ảnh thật v5",
                    value=True,
                    key="detect-yolo-verified",
                )
            ):
                model_path = VERIFIED_YOLO_PATH
            st.caption(
                "Checkpoint đã được đánh giá trên tập ảnh thật v5."
                if model_path == VERIFIED_YOLO_PATH
                else "Checkpoint gốc theo cấu hình hiện tại."
            )
        elif backend == "ssdlite":
            st.caption("SSDLite320-MobileNetV3. Chất lượng nhận diện cần được đánh giá trên ảnh thực tế.")
        else:
            if (
                settings.selection_path is None
                and secondary_path == DEFAULT_YOLO_PATH
                and VERIFIED_YOLO_PATH.is_file()
            ):
                secondary_path = VERIFIED_YOLO_PATH
            st.caption("WBF hợp nhất các khung dự đoán từ SSDLite320 và YOLOv8n.")
        st.caption(f"Checkpoint đang dùng `{model_path.as_posix()}`")
        if secondary_path is not None:
            st.caption(f"Checkpoint kết hợp `{secondary_path.as_posix()}`")

        is_interactive_default = (
            "WASTE_DETECTOR" not in os.environ
            and "WASTE_DETECTOR_BACKEND" not in os.environ
            and "WASTE_CONFIDENCE_THRESHOLD" not in os.environ
            and "WASTE_DETECTOR_CONFIDENCE" not in os.environ
            and "--confidence-threshold" not in sys.argv[1:]
            and settings.selection_path is None
        )
        confidence = st.slider(
            "Ngưỡng tin cậy",
            min_value=0.10,
            max_value=1.0,
            value=0.22 if is_interactive_default else resolve_slider_default_confidence(settings),
            step=0.01,
        )
        st.caption("Tăng ngưỡng để lọc bớt khung điểm thấp. Một số vật thể đúng có thể bị bỏ sót.")

        enable_tiling = st.checkbox(
            "Chia ảnh để tìm thêm vật thể nhỏ",
            value=True if (is_interactive_default and "WASTE_ENABLE_TILING" not in os.environ) else settings.enable_tiling,
            key="detect-tiling",
            help="Phân tích thêm các ô cắt. Có thể tìm được vật thể nhỏ hơn và phát sinh thêm khung báo nhầm.",
        )
        enable_two_stage = st.checkbox(
            "Phân loại lại từng khung bằng MobileNetV3",
            value=settings.enable_two_stage or backend in ("ssdlite", "wbf"),
            key=f"detect-refine-{backend}",
            help="Xét lại nhãn trong vùng ảnh đã khoanh. Tọa độ khung do mô hình phát hiện quyết định.",
        )

        enable_calibrated_thresholds = st.checkbox(
            "Áp dụng ngưỡng riêng từng lớp",
            value=bool(settings.per_class_thresholds),
            help="Đọc ngưỡng từng lớp từ cấu hình. Ngưỡng cần được đánh giá cùng checkpoint trên validation ảnh thật.",
        )

        st.markdown(
            '<div class="sidebar-stats-box">'
            '<strong>Thống kê kiểm định hệ thống:</strong><br>'
            '• MobileNetV3-Large (Epoch 172): 97,12% (Test: 96,18%)<br>'
            '• SSDLite320 macro AP50: 87,5% | mAP50-95: 61,2%<br>'
            '• YOLOv8n macro AP50: 89,9% | mAP50-95: 63,4%<br>'
            '• Hợp nhất WBF macro AP50: 91,4% | mAP50-95: 65,1%<br>'
            '• Tốc độ suy luận: 80,6 FPS (Phân loại) / 36,0 FPS (SSD)'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**10 Nhóm rác nhận diện:**")
        st.write(", ".join(detection_label(name) for name in DETECTION_CLASS_NAMES))

    calibrated_path = Path("configs/detection_thresholds_calibrated.json")
    active_per_class = resolve_active_per_class_thresholds(
        settings,
        confidence,
        enable_calibrated_thresholds=enable_calibrated_thresholds,
        calibrated_path=calibrated_path,
    )

    optimal_fusion_weights = (
        settings.fusion_weights if settings.selection_path is not None else (1.0, 1.5)
    )
    ui_scoring_mode = (
        settings.two_stage_scoring_mode
        if (
            "WASTE_TWO_STAGE_SCORING_MODE" in os.environ
            or "--two-stage-scoring" in sys.argv[1:]
            or settings.selection_path is not None
        )
        else "blend"
    )
    settings = replace(
        settings,
        backend=backend,
        model_path=model_path,
        secondary_model_path=secondary_path,
        confidence_threshold=confidence,
        fusion_weights=optimal_fusion_weights,
        enable_tiling=enable_tiling,
        enable_two_stage=enable_two_stage,
        two_stage_scoring_mode=ui_scoring_mode,
        tiling_min_side=320,
        per_class_thresholds=active_per_class,
        disabled_classes=settings.disabled_classes if enable_calibrated_thresholds else (),
    )
    problem = detector_problem(settings)
    if problem is not None:
        render_status(problem, "error")

    tab_classify, tab_detect = st.tabs([
        "Tab 1: Phân loại rác đơn vật thể (MobileNetV3-Large)",
        "Tab 2: Phát hiện vị trí đa vật thể (SSDLite320 / YOLOv8n / WBF)",
    ])

    with tab_classify:
        st.markdown(
            "**Mô hình MobileNetV3-Large** phân loại ảnh vào 10 nhóm rác. "
            "Ảnh cận cảnh một vật giúp diễn giải kết quả phân loại."
        )
        classify_file = st.file_uploader(
            "Chọn ảnh rác cần phân loại chính xác",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="classify_uploader",
        )
        target_classify = classify_file
        if target_classify:
            try:
                with Image.open(io.BytesIO(target_classify.getvalue())) as opened:
                    opened.load()
                    img = ImageOps.exif_transpose(opened).convert("RGB")
                classifier = cached_classifier()
                output = classifier.classify_pil(img)
                render_classification_result(img, output, target_classify.name)
            except Exception as e:
                render_status(f"Lỗi phân loại ảnh: {e}", "error")

    with tab_detect:
        st.markdown(
            "**Mô hình Object Detection** tìm vị trí và phân loại các vật thể rác trong ảnh. "
            "Khi bật phân loại lại, bảng kết quả hiển thị riêng điểm phát hiện và xác suất phân loại."
        )
        style_code = "highlight"

        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh rác",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="detect_uploader",
        )
        all_detect_files = list(uploaded_files or [])
        if not all_detect_files:
            st.session_state.pop("detection-results", None)
            st.session_state.pop("detection-errors", None)
            return

        payloads = [(uploaded.name or "image.jpg", uploaded.getvalue()) for uploaded in all_detect_files]
        signature = (
            settings,
            style_code,
            tuple((name, hashlib.sha256(payload).hexdigest()) for name, payload in payloads),
        )
        if st.session_state.get("detection-input-signature") != signature:
            st.session_state["detection-input-signature"] = signature
            st.session_state["detection-results"] = []
            st.session_state["detection-errors"] = []

        run_detect = st.button("Phát hiện vị trí rác", type="primary", disabled=problem is not None)

        if run_detect:
            try:
                service = cached_detector(
                    settings.backend,
                    str(settings.model_path),
                    str(settings.secondary_model_path) if settings.secondary_model_path else None,
                    settings.device,
                    settings.confidence_threshold,
                    settings.iou_threshold,
                    settings.image_size,
                    settings.max_detections,
                    settings.fusion_weights,
                    settings.fusion_iou_threshold,
                    settings.fusion_input_threshold,
                    tuple(
                        (path.stat().st_mtime_ns, path.stat().st_size)
                        for path in (
                            settings.model_path, settings.secondary_model_path,
                            settings.classifier_path if settings.enable_two_stage else None,
                        )
                        if path is not None
                    ),
                    settings if (
                        settings.enable_tiling or settings.enable_two_stage
                        or settings.per_class_thresholds or settings.disabled_classes
                    ) else None,
                )
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError) as error:
                st.session_state["detection-results"] = []
                render_status(
                    "Không thể nạp mô hình nhận diện 10 nhóm rác. "
                    "Vui lòng kiểm tra trọng số và cấu hình ứng dụng.",
                    "error",
                )
                print(f"Detector load failed ({settings.backend}): {error}", file=sys.stderr)
                return

            outputs: list[ImageDetectionOutput] = []
            errors: list[str] = []
            progress = st.progress(0.0, text="Đang xử lý ảnh…")
            for index, (name, payload) in enumerate(payloads, start=1):
                try:
                    outputs.append(analyze_image_file(name, payload, service, style=style_code))
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append(f"Không thể xử lý ảnh {name}. Hãy kiểm tra ảnh và thử lại.")
                    print(f"Image detection failed ({name}): {error}", file=sys.stderr)
                progress.progress(index / len(payloads), text=f"Đã xử lý {index}/{len(payloads)} ảnh")
            progress.empty()
            st.session_state["detection-results"] = outputs
            st.session_state["detection-errors"] = errors

        for error in st.session_state.get("detection-errors", []):
            render_status(error, "error")
        outputs = st.session_state.get("detection-results", [])
        if len(outputs) > 1:
            st.download_button(
                "Tải gói nén (.ZIP)",
                data=archive_annotated_images(outputs),
                file_name="ket_qua_nhan_dien.zip",
                mime="application/zip",
            )
        for index, output in enumerate(outputs):
            render_detection_result(output, index)


if __name__ == "__main__":
    main()
