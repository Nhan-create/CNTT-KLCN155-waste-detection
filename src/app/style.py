# Mô tả: Cung cấp các hàm sinh mã HTML/CSS và favicon trong suốt giúp giao diện Streamlit hiển thị đồng nhất, rõ ràng.
"""Text-only presentation primitives for the Streamlit frontend."""

from __future__ import annotations

from html import escape
import re
from typing import Literal

from PIL import Image

StatusTone = Literal["neutral", "success", "warning", "error"]

_STATUS_CLASSES: dict[StatusTone, str] = {
    "neutral": "text-status--neutral",
    "success": "text-status--success",
    "warning": "text-status--warning",
    "error": "text-status--error",
}

ICON_FREE_CSS = """
<style>
span[data-testid="stIconMaterial"],
[data-testid="stHeaderActionElements"],
[data-testid="stElementToolbar"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stTooltipIcon"],
[data-testid="stDownloadButton"] svg,
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileChipDeleteBtn"] svg,
[data-testid="stCameraInput"] svg {
  display: none !important;
}

/* Ẩn ô vuông trắng thừa bên trái file chip và xóa icon tròn đỏ, chỉ hiển thị dấu X văn bản thuần */
[data-testid="stFileChip"] > div:first-child {
  display: none !important;
}
[data-testid="stFileChip"] {
  background: #f8fafc !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 6px !important;
  padding: 4px 12px !important;
  color: #0f172a !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  max-width: none !important;
}
[data-testid="stFileChipName"],
[data-testid="stFileChip"] div {
  color: #0f172a !important;
  font-weight: 600 !important;
  max-width: none !important;
  overflow: visible !important;
  text-overflow: unset !important;
}
[data-testid="stFileChipDeleteBtn"] {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  margin-left: 6px !important;
  width: auto !important;
  height: auto !important;
}
[data-testid="stFileChipDeleteBtn"] button {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  color: #334155 !important;
  cursor: pointer !important;
  padding: 0 4px !important;
  margin: 0 !important;
  width: auto !important;
  height: auto !important;
  min-height: 0 !important;
}
[data-testid="stFileChipDeleteBtn"] button::after {
  content: "✕" !important;
  font-family: Arial, sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  color: #334155 !important;
  line-height: 1 !important;
}
[data-testid="stFileChipDeleteBtn"] button:hover::after {
  color: #dc2626 !important;
}

/* Giao diện sáng học thuật chuẩn hình báo cáo KLCN */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
  background-color: #f8fafc !important;
  color: #0f172a !important;
}
[data-testid="stHeader"] {
  display: none !important;
}
.block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 1.5rem !important;
  padding-left: 1.8rem !important;
  padding-right: 1.8rem !important;
  max-width: 100% !important;
}
.top-app-banner {
  background: #17375e;
  color: #ffffff !important;
  padding: 10px 20px;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: "Times New Roman", Georgia, serif;
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 14px;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.12);
}
.top-app-banner span {
  color: #ffffff !important;
}
.top-app-banner .banner-right {
  color: #93c5fd !important;
  font-size: 0.95rem;
}

/* Sidebar sáng với các khung thông tin đóng hộp */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
  background-color: #eef2f6 !important;
  border-right: 1.5px solid #cbd5e1 !important;
  color: #0f172a !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
  color: #0f172a !important;
}
.sidebar-section-title {
  font-family: "Times New Roman", Georgia, serif;
  font-size: 1.02rem;
  font-weight: 700;
  color: #17375e !important;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  margin: 4px 0 8px 0;
}
.sidebar-stats-box {
  background: #f0f7ff;
  border: 1.5px solid #3b82f6;
  border-radius: 6px;
  padding: 10px 12px;
  margin-top: 10px;
  font-size: 0.85rem;
  line-height: 1.55;
  color: #0f172a !important;
}
.sidebar-stats-box strong {
  color: #17375e !important;
}

/* Tiêu đề chính và thanh Tabs dạng nút chữ nhật đồng bộ hình báo cáo */
h1 {
  font-family: "Times New Roman", Georgia, serif !important;
  color: #0f172a !important;
  font-size: 1.85rem !important;
  font-weight: 700 !important;
  margin-bottom: 0.15rem !important;
}
h2, h3 {
  font-family: "Times New Roman", Georgia, serif !important;
  color: #0f172a !important;
}
[data-baseweb="tab-list"],
[role="tablist"] {
  gap: 12px !important;
  background: transparent !important;
  border-bottom: none !important;
  margin-bottom: 10px !important;
  display: flex !important;
}
[data-baseweb="tab"],
[data-testid="stTab"],
[role="tab"] {
  flex: 1 1 0% !important;
  justify-content: center !important;
  background-color: #e2e8f0 !important;
  border: 1.5px solid #cbd5e1 !important;
  border-radius: 6px !important;
  padding: 8px 16px !important;
  height: auto !important;
  text-align: center !important;
}
[data-baseweb="tab"] p,
[data-testid="stTab"] p,
[role="tab"] p {
  font-family: "Times New Roman", Georgia, serif !important;
  font-size: 1.02rem !important;
  font-weight: 700 !important;
  color: #334155 !important;
  margin: 0 !important;
}
[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"],
[role="tab"][aria-selected="true"] {
  background-color: #1b4d6b !important;
  border-color: #17375e !important;
}
[data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] p,
[role="tab"][aria-selected="true"] p {
  color: #ffffff !important;
}
[data-baseweb="tab"]::after,
[data-testid="stTab"]::after,
[role="tab"]::after,
[data-baseweb="tab-highlight"] {
  display: none !important;
}
[data-testid="stImage"] img {
  max-height: 360px !important;
  object-fit: contain !important;
  border-radius: 6px !important;
  border: 1px solid #cbd5e1 !important;
  background: #ffffff !important;
}

/* Hộp tải ảnh sáng rõ */
[data-testid="stFileUploaderDropzone"] {
  background-color: #ffffff !important;
  border: 1.5px dashed #64748b !important;
  border-radius: 8px !important;
  color: #0f172a !important;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
  background-color: #1b4d6b !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] p {
  color: #ffffff !important;
}

/* Các thẻ Metric đóng khung màu học thuật */
[data-testid="stMetric"] {
  background: #f0f9ff !important;
  border: 1.5px solid #1b4d6b !important;
  border-radius: 6px !important;
  padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] p {
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
  color: #15803d !important;
  font-family: "Times New Roman", Georgia, serif !important;
  font-weight: 700 !important;
  font-size: 1.22rem !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
  line-height: 1.25 !important;
}

/* Nút bấm chính và nút tải xuống */
button[data-testid="stBaseButton-primary"] {
  background-color: #17375e !important;
  border-color: #17375e !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}
[data-testid="stDownloadButton"] button {
  background-color: #17375e !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  width: 100% !important;
}
[data-testid="stDownloadButton"] button p {
  color: #ffffff !important;
}

.text-status {
  border-left: 4px solid #64748b;
  border-radius: 4px;
  margin: 0.5rem 0 1rem;
  padding: 0.75rem 1rem;
  line-height: 1.5;
  font-weight: 500;
}
.text-status--neutral { background: #f8fafc; border-color: #64748b; color: #0f172a !important; }
.text-status--success { background: #f0fdf4; border-color: #15803d; color: #14532d !important; }
.text-status--warning { background: #fffbeb; border-color: #b45309; color: #78350f !important; }
.text-status--error { background: #fef2f2; border-color: #b91c1c; color: #7f1d1d !important; }
</style>
"""


def transparent_favicon() -> Image.Image:
    """Return a fully transparent favicon accepted by Streamlit."""

    return Image.new("RGBA", (1, 1), (255, 255, 255, 0))


def status_markup(message: str, tone: StatusTone = "neutral") -> str:
    """Render escaped, text-only status markup for one supported tone."""

    try:
        css_class = _STATUS_CLASSES[tone]
    except KeyError as error:
        raise ValueError(f"Unsupported status tone: {tone}") from error
    role = "alert" if tone == "error" else "status"
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(message))
    return (
        f'<div role="{role}" class="text-status {css_class}">'
        f"{formatted}</div>"
    )
