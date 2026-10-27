# Mô tả: Kiểm thử hợp đồng cấu trúc giao diện Streamlit (streamlit_app.py) và các điểm kết nối dịch vụ suy luận.
from pathlib import Path


def test_phase2_demo_is_static_image_detection_without_icons() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.title("Nhận diện rác trong ảnh")' in source
    assert "create_detector" in source
    assert "accept_multiple_files=True" in source
    assert "LiveDetectionCallback" not in source
    assert "webrtc_streamer" not in source
    assert "Mô hình phân loại toàn bộ khung hình" not in source
    assert "♻" not in source
    assert "view.icon" not in source
    assert "row.icon" not in source
    assert "label.icon" not in source
    assert "latest.icon" not in source
    assert "page_icon=transparent_favicon()" in source
    for alert_call in (
        "st.info(",
        "st.warning(",
        "st.error(",
        "st.success(",
        "st.spinner(",
        "status.info(",
    ):
        assert alert_call not in source
    for label in ("Tải video", "Camera trực tiếp"):
        assert f'"{label}"' not in source
    assert "camera_input" not in source


def test_phase2_demo_opens_with_missing_weights_and_can_change_models(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WASTE_DETECTOR", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("WASTE_DETECTOR_BACKEND", "ssdlite")
    root_app = Path(__file__).resolve().parent.parent.parent / "streamlit_app.py"
    app = AppTest.from_file(str(root_app)).run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "ssdlite"
    assert "Chưa có mô hình đã huấn luyện" in " ".join(item.value for item in app.markdown)
    assert len(app.slider) == 1
    uploaders = app.get("file_uploader")
    assert len(uploaders) >= 1
    assert any("detect" in u.key or "detect" in u.label.lower() or u.accept_multiple_files for u in uploaders)
    app.selectbox[0].select("wbf").run()
    assert not app.exception
    assert app.selectbox[0].value == "wbf"
