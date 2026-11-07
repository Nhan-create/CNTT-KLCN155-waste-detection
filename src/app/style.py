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
[data-testid="stDownloadButton"] svg,
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stCameraInput"] svg {
  display: none !important;
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
