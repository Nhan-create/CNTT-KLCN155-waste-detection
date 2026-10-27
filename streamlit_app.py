"""Phase-2 Streamlit frontend: six/ten-class waste detection & high-precision classification."""

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

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.visualization import detection_label
from src.app.classifier_service import ClassificationOutput, MobileNetClassifier
from src.app.detection_logic import (
    ImageDetectionOutput,
    analyze_image_file,
    annotated_filename,
    archive_annotated_images,
    build_detection_pipeline,
    count_rows,
    detection_rows,
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
from src.app.style import ICON_FREE_CSS, StatusTone, status_markup, transparent_favicon

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


def render_status(message: str, tone: StatusTone = "neutral", target=None) -> None:
    container = st if target is None else target
    container.markdown(status_markup(message, tone), unsafe_allow_html=True)


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


def render_classification_result(
    image: Image.Image, output: ClassificationOutput, filename: str
) -> None:
    st.subheader(f"Kết quả phân loại: {filename}")
    preview, details = st.columns([3, 2], gap="large")
    with preview:
        st.image(image, width="stretch")
    with details:
        st.metric("Loại rác dự đoán", output.class_name_vi)
        st.metric("Độ tin cậy", f"{output.confidence:.1%}")
        st.metric("Thời gian xử lý", f"{output.processing_ms:.0f} ms")
        if output.confidence < 0.40:
            render_status(
                f"Độ tin cậy thấp ({output.confidence:.1%}): Ảnh chụp toàn cảnh từ xa hoặc chứa nhiều ngoại cảnh phức tạp. "
                "Để nhận diện chính xác nhất, bạn có thể: (1) Cắt (crop) cận cảnh riêng vật thể rác, hoặc (2) Chuyển sang Tab 2 (Phát hiện vị trí) để mô hình tự động tìm khung bao vật thể.",
                "warning",
            )

        st.markdown("**Hướng dẫn phân loại rác tại nguồn:**")
        render_status(
            f"{output.bin_name}: {output.bin_guide}",
            "success" if "Tái chế" in output.bin_name or "hữu cơ" in output.bin_name else "neutral",
        )

        st.markdown("**Xác suất Top 5 lớp rác liên quan:**")
        for cls_id, cls_vi, prob in output.top_probabilities:
            st.write(f"• **{cls_vi}**: `{prob:.1%}`")
            st.progress(float(prob))


def render_detection_result(output: ImageDetectionOutput, index: int) -> None:
    st.subheader(f"Ảnh {index + 1}: {output.original_name}")
    preview, details = st.columns([3, 2], gap="large")
    with preview:
        st.image(output.annotated_png, width="stretch")
        st.download_button(
            "Tải ảnh kết quả",
            data=output.annotated_png,
            file_name=annotated_filename(output.original_name),
            mime="image/png",
            key=f"download-image-{index}",
        )
    with details:
        st.metric("Số vật thể phát hiện", len(output.result.detections))
        st.metric("Thời gian xử lý", f"{output.processing_ms:.0f} ms")
        if output.result.detections:
            st.dataframe(count_rows(output.result), width="stretch", hide_index=True)
            st.dataframe(
                detection_rows(output.result, output.details), width="stretch", hide_index=True
            )
            detected_classes = {d.class_id for d in output.result.detections}
            st.markdown("**Hướng dẫn phân loại rác tại nguồn:**")
            if "biological" in detected_classes:
                render_status("🟢 **Thùng Xanh lá (Rác hữu cơ):** Thức ăn thừa, rau củ quả, lá cây.", "success")
            if detected_classes.intersection({"plastic", "paper", "cardboard", "metal", "glass"}):
                render_status("🟡 **Thùng Vàng / Trắng (Rác tái chế):** Nhựa, giấy, bìa carton, kim loại, thủy tinh.", "success")
            if "battery" in detected_classes:
                render_status("🔴 **Thùng Đỏ / Cam (Rác nguy hại):** Pin, ắc quy, bóng đèn, rác độc hại.", "error")
            if detected_classes.intersection({"clothes", "shoes", "trash"}):
                render_status("⚪ **Thùng Xám / Thu gom riêng:** Quần áo cũ, giày dép (tái sử dụng/quyên góp), rác vô cơ khác.", "neutral")
        else:
            render_status(
                "Không phát hiện vật thể rác đạt ngưỡng tin cậy trong ảnh này. "
                "👉 **Hướng dẫn khắc phục:** Ảnh chụp ngoài thiên nhiên có rác bị dập bẹp, vùi trong đất/lá cây "
                "thường có độ tin cậy từ 30% đến 55%. Bạn hãy kéo thanh trượt **Ngưỡng tin cậy** ở cột bên trái về **0.25 - 0.35** "
                "rồi bấm lại nút 'Phát hiện vị trí rác' để AI hiển thị trọn vẹn các vật thể rác!",
                "warning",
            )


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
            value=max(0.25, settings.confidence_threshold if settings.confidence_threshold <= 0.5 else 0.30),
            step=0.05,
        )
        if confidence > 0.45:
            st.caption(f"⚠️ Ngưỡng {confidence:.0%} rất cao: Chỉ giữ vật thể hoàn hảo. Hãy kéo về **0.25 - 0.35** để bắt trọn rác phức tạp ngoài thực địa.")
        else:
            st.caption("Khuyên dùng 0.25 - 0.35 để bao phủ tối đa các vật thể rác và loại trừ nhiễu.")
        enable_tiling = st.checkbox(
            "Cắt ô ảnh (Tiling) để bắt chi tiết rác nhỏ",
            value=True,
            help="Tự động chia nhỏ ảnh thành các ô chi tiết để nhận diện vật thể nhỏ/vụn và đống rác dày đặc.",
        )
        enable_two_stage = st.checkbox(
            "Phân loại lại từng vật thể bằng MobileNetV3 (Two-Stage)",
            value=True,
            help="Dùng mô hình MobileNetV3 (96.2%) thẩm định lại nhãn từng khung rác, loại trừ hoàn toàn các lỗi nhận diện nhầm logo thành Giày dép/Quần áo và triệt tiêu vật thể ngoại cảnh.",
        )
        two_stage_scoring_mode = "product"
        if enable_two_stage:
            two_stage_scoring_mode = st.radio(
                "Thuật toán kết hợp Two-Stage",
                options=["product", "blend"],
                format_func=lambda x: "🔒 Nhân xác suất Bayesian (Khuyên dùng - Triệt tiêu box ảo ngoại cảnh)" if x == "product" else "✨ Pha trộn điểm (Bao phủ tối đa, dễ dính ngoại cảnh)",
                index=0,
            )

        enable_calibrated_thresholds = st.checkbox(
            "Áp dụng ngưỡng hiệu chuẩn riêng từng lớp",
            value=False,
            help="Hiệu chuẩn ngưỡng riêng theo từng loại rác (ví dụ: hạ trần nhựa 22%, kim loại/pin 20% và nâng trần quần áo/giày dép 35% để tránh nhận nhầm). Khi TẮT, hệ thống áp dụng đúng 100% ngưỡng sàn đã chọn trên thanh trượt cho mọi lớp.",
        )
        if enable_calibrated_thresholds:
            st.caption("ℹ️ Đang kích hoạt ngưỡng riêng: Nhựa (22%), Pin/Kim loại (20%), Quần áo/Giày (35%), các loại khác (25%) – tự động co giãn theo thanh trượt.")

        st.markdown("---")
        st.markdown("**10 Nhóm rác nhận diện:**")
        st.write(", ".join(detection_label(name) for name in DETECTION_CLASS_NAMES))

    model_path, secondary_path = paths_for_backend(settings, backend)
    calibrated_path = Path("configs/detection_thresholds_calibrated.json")
    active_per_class = settings.per_class_thresholds
    if enable_calibrated_thresholds and not active_per_class and calibrated_path.is_file():
        try:
            base_thresholds = load_per_class_thresholds(calibrated_path)
            if confidence >= 0.99:
                active_per_class = tuple((name, 1.0) for name, _ in base_thresholds)
            else:
                scale = confidence / 0.25
                active_per_class = tuple(
                    (
                        name,
                        min(
                            1.0,
                            max(
                                confidence if confidence >= 0.35 else round(val * scale, 3),
                                round(val * scale, 3),
                            ),
                        ),
                    )
                    for name, val in base_thresholds
                )
        except Exception:
            active_per_class = ()
    elif not enable_calibrated_thresholds:
        active_per_class = ()

    settings = replace(
        settings,
        backend=backend,
        model_path=model_path,
        secondary_model_path=secondary_path,
        confidence_threshold=confidence,
        enable_tiling=enable_tiling,
        enable_two_stage=enable_two_stage,
        two_stage_scoring_mode=two_stage_scoring_mode,
        two_stage_proposal_threshold=0.12 if two_stage_scoring_mode == "blend" else 0.15,
        two_stage_min_proposal_gate=0.28,
        tiling_min_side=800 if enable_tiling else settings.tiling_min_side,
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
