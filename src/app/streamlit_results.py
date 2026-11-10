"""Streamlit status, classification, and detection result panels."""

from __future__ import annotations

import csv
import io

import streamlit as st
from PIL import Image

from src.app.classifier_service import ClassificationOutput, summarize_bin_guidance
from src.app.detection_logic import (
    ImageDetectionOutput,
    annotated_filename,
    count_rows,
    detection_rows,
)
from src.app.style import StatusTone, status_markup


def render_status(message: str, tone: StatusTone = "neutral", target=None) -> None:
    container = st if target is None else target
    container.markdown(status_markup(message, tone), unsafe_allow_html=True)


def _rows_to_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def render_classification_result(
    image: Image.Image, output: ClassificationOutput, filename: str
) -> None:
    st.subheader(f"Kết quả phân loại: {filename}")
    preview, details = st.columns([3, 2], gap="large")
    with preview:
        st.caption("Ảnh gốc tải lên (đã đồng bộ hướng xoay EXIF RGB):")
        st.image(image, width="stretch")
        st.caption(f"Kích thước gốc: {image.width} x {image.height} px | Định dạng: RGB")
    with details:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Loại rác dự đoán", output.class_name_vi)
        with m2:
            st.metric("Độ tin cậy", f"{output.confidence:.1%}")
        with m3:
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

        st.markdown("**Phân phối xác suất Top 5 lớp rác liên quan (Softmax):**")
        for _, cls_vi, prob in output.top_probabilities:
            st.write(f"• **{cls_vi}**: `{prob:.2%}`")
            st.progress(float(prob))


def render_detection_result(output: ImageDetectionOutput, index: int) -> None:
    st.subheader(f"Ảnh {index + 1}: {output.original_name}")
    preview, details = st.columns([3, 2], gap="large")
    with preview:
        st.image(output.annotated_png, width="stretch")
        st.download_button(
            "Tải ảnh kết quả (.PNG)",
            data=output.annotated_png,
            file_name=annotated_filename(output.original_name),
            mime="image/png",
            key=f"download-image-{index}",
        )
    with details:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Số vật thể phát hiện", len(output.result.detections))
        with m2:
            st.metric("Thời gian xử lý", f"{output.processing_ms:.0f} ms")
        if output.result.detections:
            det_rows = detection_rows(output.result, output.details)
            st.markdown("**Bảng chi tiết tọa độ khung bao và thống kê số lượng:**")
            st.dataframe(det_rows, width="stretch", hide_index=True)
            st.dataframe(count_rows(output.result), width="stretch", hide_index=True)
            st.download_button(
                "Tải bảng phát hiện (CSV)",
                data=_rows_to_csv_bytes(det_rows),
                file_name=f"{Path_stem(output.original_name)}_detections.csv",
                mime="text/csv",
                key=f"download-csv-{index}",
            )
            st.markdown("**Thống kê số lượng và hướng dẫn phân loại rác tại nguồn:**")
            for msg, tone in summarize_bin_guidance(d.class_id for d in output.result.detections):
                render_status(msg, tone)  # type: ignore[arg-type]
        else:
            render_status(
                "Không phát hiện vật thể rác đạt ngưỡng tin cậy trong ảnh này. "
                "Bạn có thể chụp gần hơn, giữ ảnh rõ nét và kiểm tra lại cấu hình mô hình. "
                "Giảm ngưỡng có thể tìm thêm vật thể nhưng cũng làm tăng khung báo nhầm. "
                "Hãy đối chiếu kết quả với từng vật trong ảnh.",
                "warning",
            )


def Path_stem(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].strip()
    return stem or "waste"
