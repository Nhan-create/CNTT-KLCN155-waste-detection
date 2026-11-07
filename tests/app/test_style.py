# Mô tả: Kiểm thử các hàm sinh mã HTML/CSS trạng thái và biểu tượng favicon trong suốt cho giao diện Streamlit.
import pytest

from src.app.style import ICON_FREE_CSS, status_markup, transparent_favicon


def test_transparent_favicon_has_no_visible_pixel() -> None:
    favicon = transparent_favicon()

    assert favicon.mode == "RGBA"
    assert favicon.size == (1, 1)
    assert favicon.getpixel((0, 0))[3] == 0


def test_status_markup_escapes_text_and_contains_no_icon_markup() -> None:
    markup = status_markup('<script>alert("x")</script>', "error")

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert 'role="alert"' in markup
    assert "<svg" not in markup


def test_neutral_status_markup_uses_status_role() -> None:
    markup = status_markup("Sẵn sàng")

    assert 'role="status"' in markup
    assert "Sẵn sàng" in markup


def test_style_hides_decorative_framework_icons() -> None:
    assert 'span[data-testid="stIconMaterial"]' in ICON_FREE_CSS
    assert '[data-testid="stHeaderActionElements"]' in ICON_FREE_CSS
    assert '[data-testid="stElementToolbar"]' in ICON_FREE_CSS
    assert '[data-testid="stFileUploaderDropzone"] svg' in ICON_FREE_CSS
    assert '[data-testid="stCameraInput"] svg' in ICON_FREE_CSS


def test_status_markup_rejects_unknown_tone() -> None:
    with pytest.raises(ValueError, match="Unsupported status tone"):
        status_markup("message", "blue")  # type: ignore[arg-type]


def test_status_markup_renders_bold_and_high_contrast_text_colors() -> None:
    markup = status_markup("🟡 **Thùng Vàng:** Nhựa, kim loại.", "success")
    assert "<strong>Thùng Vàng:</strong>" in markup
    assert "**" not in markup
    assert "color: #14532d !important;" in ICON_FREE_CSS
    assert "color: #0f172a !important;" in ICON_FREE_CSS

