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
        for _, cls_vi, prob in output.top_probabilities:
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
            det_rows = detection_rows(output.result, output.details)
            st.dataframe(count_rows(output.result), width="stretch", hide_index=True)
            st.dataframe(det_rows, width="stretch", hide_index=True)
            st.download_button(
                "Tải bảng phát hiện (CSV)",
                data=_rows_to_csv_bytes(det_rows),
                file_name=f"{Path_stem(output.original_name)}_detections.csv",
                mime="text/csv",
                key=f"download-csv-{index}",
            )
            st.markdown("**Hướng dẫn phân loại rác tại nguồn:**")
            for msg, tone in summarize_bin_guidance(d.class_id for d in output.result.detections):
                render_status(msg, tone)  # type: ignore[arg-type]
        else:
            render_status(
                "Không phát hiện vật thể rác đạt ngưỡng tin cậy trong ảnh này. "
                "👉 **Hướng dẫn khắc phục:** Ảnh chụp ngoài thiên nhiên có rác bị dập bẹp, vùi trong đất/lá cây "
                "thường có độ tin cậy từ 30% đến 55%. Bạn hãy kéo thanh trượt **Ngưỡng tin cậy** ở cột bên trái về **0.25 - 0.35** "
                "rồi bấm lại nút 'Phát hiện vị trí rác' để AI hiển thị trọn vẹn các vật thể rác!",
                "warning",
            )


def Path_stem(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].strip()
    return stem or "waste"
