# Mô tả: Điểm vào chính của ứng dụng Web Streamlit gồm 2 Tab: Tab 1 (Phân loại rác đơn vật thể bằng MobileNetV3-Large) và Tab 2 (Phát hiện đa vật thể bằng SSDLite320 / YOLOv8n / WBF, thống kê số lượng từng loại rác và xuất gói .zip).
"""Phase-2 Streamlit frontend: 10-class waste detection & high-precision classification."""

from __future__ import annotations

import hashlib
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
    DETECTION_BACKENDS,
    DetectionWebSettings,
    detector_input_threshold,
    detector_problem,
    load_per_class_thresholds,
    parse_detection_settings,
    paths_for_backend,
)
from src.app.streamlit_results import (
    render_classification_result as render_classification_result,
    render_detection_result as render_detection_result,
    render_status as render_status,
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
    "yolov8n": "YOLOv8n (Mô hình đối chứng - Nhanh & Chuẩn nhất)",
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
def cached_classifier() -> MobileNetClassifier:
    return MobileNetClassifier()


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
    _pipeline_version: int = 4,
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
        classifier = cached_classifier() if pipeline_settings.enable_two_stage else None
        detector = build_detection_pipeline(detector, pipeline_settings, classifier=classifier)
    return DetectionService(detector)


def main() -> None:
    st.title("Nhận diện rác trong ảnh")
    st.caption("Khóa luận tốt nghiệp CNTT: Hệ thống phân loại 10 nhóm rác thải và phát hiện vị trí rác trong ảnh.")

    try:
        settings = parse_detection_settings(sys.argv[1:], os.environ)
    except (ValueError, SystemExit):
        render_status("Cấu hình ứng dụng không hợp lệ. Vui lòng kiểm tra lại.", "error")
        st.stop()

    with st.sidebar:
        backend = st.selectbox(
            "Mô hình",
            DETECTION_BACKENDS,
            index=DETECTION_BACKENDS.index(settings.backend),
            format_func=lambda name: MODEL_NAMES[name],
        )
        if backend == "yolov8n":
            st.caption("Khuyên dùng: YOLOv8n v2 đã hoàn tất huấn luyện, độ chính xác cao nhất.")
        elif backend == "ssdlite":
            st.caption("Mô hình chính SSDLite v2 đã hoàn tất huấn luyện (36 epochs, loss: 3.20, mAP: 12.6%). Tối ưu cho thiết bị nhúng!")

        confidence = st.slider(
            "Ngưỡng tin cậy",
            min_value=0.10,
            max_value=1.0,
            value=resolve_slider_default_confidence(settings),
            step=0.01,
        )
        if confidence > 0.45:
            st.caption(f"⚠️ Ngưỡng {confidence:.0%} rất cao: Chỉ giữ vật thể hoàn hảo. Hãy kéo về **0.25 - 0.35** để bắt trọn rác phức tạp ngoài thực địa.")
        else:
            st.caption("Khuyên dùng 0.25 - 0.35 để bao phủ tối đa các vật thể rác và loại trừ nhiễu.")

        enable_calibrated_thresholds = st.checkbox(
            "Áp dụng ngưỡng hiệu chuẩn riêng từng lớp (Calibrated Per-Class)",
            value=bool(settings.per_class_thresholds),
            help="Khi bật, hệ thống áp dụng ngưỡng nhạy riêng cho từng nhóm rác từ cấu hình đã hiệu chuẩn thay vì dùng chung một ngưỡng đồng nhất.",
        )

        st.markdown("---")
        st.markdown("**10 Nhóm rác nhận diện:**")
        st.write(", ".join(detection_label(name) for name in DETECTION_CLASS_NAMES))

    # Tự động kích hoạt cơ chế tối ưu độ chính xác (Adaptive Tiling + Two-Stage MobileNetV3 + WBF Weighting)
    enable_tiling = True
    enable_two_stage = True
    two_stage_scoring_mode = "product"

    model_path, secondary_path = paths_for_backend(settings, backend)
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
    settings = replace(
        settings,
        backend=backend,
        model_path=model_path,
        secondary_model_path=secondary_path,
        confidence_threshold=confidence,
        fusion_weights=optimal_fusion_weights,
        enable_tiling=enable_tiling,
        enable_two_stage=enable_two_stage,
        two_stage_scoring_mode=two_stage_scoring_mode,
        two_stage_proposal_threshold=0.15,
        two_stage_min_proposal_gate=0.20,
        tiling_min_side=320,
        per_class_thresholds=active_per_class,
    )
    problem = detector_problem(settings)
    if problem is not None:
        render_status(problem, "error")

    tab_classify, tab_detect = st.tabs([
        "🎯 Phân loại rác độ chính xác cao (MobileNetV3 96.2% - Khuyên dùng khi demo)",
        "🔍 Phát hiện vị trí đa vật thể (YOLOv8n / SSDLite / WBF)",
    ])

    with tab_classify:
        st.markdown(
            "**Mô hình MobileNetV3-Large:** Huấn luyện trên **14.829 ảnh rác thực tế**, đạt độ chính xác **96.18%** (Macro-F1: **95.59%**). "
            "Chuyên dùng để phân loại chính xác tuyệt đối loại rác khi chụp cận cảnh vật thể hoặc một đống rác."
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
            "**Mô hình Object Detection:** Huấn luyện phát hiện vị trí (Bounding Box) nhiều vật thể rác đồng thời trong khung hình. "
            "Tự động tích hợp bộ lọc triệt tiêu nhiễu ngoại cảnh."
        )
        visual_style = st.radio(
            "Kiểu hiển thị vị trí:",
            ["Khung viền nổi bật (Highlight Box)", "Hộp chữ nhật viền nét (Bounding Box)"],
            horizontal=True,
            key="detect_visual_style",
        )
        style_code = "highlight" if "nổi bật" in visual_style else "box"

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
            visual_style,
            tuple((name, hashlib.sha256(payload).hexdigest()) for name, payload in payloads),
        )
        if st.session_state.get("detection-input-signature") != signature:
            st.session_state["detection-input-signature"] = signature
            st.session_state["detection-results"] = []
            st.session_state["detection-errors"] = []

        if st.button("Phát hiện vị trí rác", type="primary", disabled=problem is not None):
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
                        for path in (settings.model_path, settings.secondary_model_path)
                        if path is not None
                    ),
                    settings if (
                        settings.enable_tiling or settings.enable_two_stage
                        or settings.per_class_thresholds
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
                "Tải tất cả ảnh kết quả",
                data=archive_annotated_images(outputs),
                file_name="ket_qua_nhan_dien.zip",
                mime="application/zip",
            )
        for index, output in enumerate(outputs):
            render_detection_result(output, index)


if __name__ == "__main__":
    main()
