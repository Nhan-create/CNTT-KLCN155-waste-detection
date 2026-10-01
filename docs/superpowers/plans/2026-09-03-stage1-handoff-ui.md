# Stage 1 Handoff and Icon-Free UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện bàn giao giai đoạn 1 bằng tài liệu kỹ thuật trung thực và giao diện MobileNetV3 10 lớp không có icon, trong khi giữ nguyên bốn chế độ ảnh/camera/video.

**Architecture:** Giữ nguyên toàn bộ model, checkpoint và pipeline suy luận MobileNetV3. Thu gọn presentation model thành `display_name + color`, thêm module web thuần để tạo favicon trong suốt, CSS ẩn biểu tượng trang trí và HTML trạng thái đã escape, rồi dùng các primitive này trong Streamlit. Tài liệu bàn giao Markdown là nguồn tham chiếu chính trong GitHub và phân biệt rõ mã nguồn đã hoàn thành với kết quả thực nghiệm chưa có.

**Tech Stack:** Python 3.11, PyTorch/Torchvision, Streamlit, streamlit-webrtc, Pillow, PyQt5, pytest, Ruff, Markdown.

**Spec:** `docs/superpowers/specs/2026-09-03-stage1-handoff-ui-design.md`

## Global Constraints

- Model mặc định vẫn là `mobilenet_v3_large` với đúng 10 class ID hiện tại.
- Không thay đổi checkpoint contract, class order, transform hoặc thuật toán smoothing.
- Giữ đủ bốn tab: `Tải ảnh`, `Chụp ảnh`, `Tải video`, `Camera trực tiếp`.
- Không hiển thị emoji, icon do dự án thêm, favicon nhìn thấy hoặc banner xanh đã chỉ định.
- Trạng thái lỗi, cảnh báo và tiến trình phải còn đọc được bằng chữ.
- Không công bố accuracy, mAP hoặc FPS nếu repository chưa chứa artifact đo từ dữ liệu của nhóm.
- Camera/video thuộc phạm vi mở rộng của giai đoạn 1; bounding box, sáu lớp detection, SSDLite, YOLOv8n và WBF thuộc giai đoạn 2.

---

### Task 1: Thu gọn presentation contract về chữ và màu

**Files:**
- Modify: `tests/ui/test_presentation.py`
- Modify: `tests/ui/test_main_window.py`
- Create: `tests/web/test_app_logic.py`
- Modify: `src/ui/presentation.py`
- Modify: `src/ui/main_window.py`
- Modify: `src/web/app_logic.py`

**Interfaces:**
- Consumes: `Prediction`, `ScoredClass` và 10 class ID trong `src.data.schema.CLASS_NAMES`.
- Produces: `LabelPresentation(display_name: str, color: str)`, `ScoredClassView(class_id, display_name, color, probability)` và `PredictionView(class_id, display_name, color, probability, topk, low_confidence)`.

- [ ] **Step 1: Viết kiểm thử presentation không còn trường icon**

Thay import và assertion trong `tests/ui/test_presentation.py` bằng hợp đồng sau:

```python
from src.data.schema import CLASS_NAMES
from src.ui.presentation import COLOR_BY_CLASS, DISPLAY_NAMES_VI, present_label


def test_every_class_has_the_approved_vietnamese_name_and_color() -> None:
    assert set(DISPLAY_NAMES_VI) == set(CLASS_NAMES)
    assert set(COLOR_BY_CLASS) == set(CLASS_NAMES)


def test_known_class_has_text_only_display_metadata() -> None:
    result = present_label("glass")
    assert result.display_name == "Thủy tinh"
    assert result.color.startswith("#")
    assert not hasattr(result, "icon")


def test_unknown_class_uses_neutral_text_fallback() -> None:
    result = present_label("future_class")
    assert result.display_name == "Không xác định"
    assert result.color == "#64748B"
    assert not hasattr(result, "icon")
```

- [ ] **Step 2: Siết kiểm thử PyQt về chuỗi hiển thị thuần chữ**

Trong `tests/ui/test_main_window.py`, sau `window.handle_result(result)` thêm các assertion chính xác:

```python
assert window.result_label.text() == "Thủy tinh"
assert window.top_three.item(0).text() == "Thủy tinh: 62.0%"
assert window.top_three.item(1).text() == "Nhựa: 23.0%"
assert window.top_three.item(2).text() == "Kim loại: 10.0%"
```

- [ ] **Step 3: Tạo kiểm thử view model của web**

Tạo `tests/web/test_app_logic.py`:

```python
from src.inference.predict import Prediction, ScoredClass
from src.web.app_logic import prediction_view


def test_prediction_view_contains_text_probability_and_no_icon() -> None:
    topk = (
        ScoredClass(4, "glass", 0.62),
        ScoredClass(7, "plastic", 0.23),
        ScoredClass(5, "metal", 0.10),
    )
    view = prediction_view(Prediction(topk[0], topk, False))

    assert view.display_name == "Thủy tinh"
    assert view.probability == 0.62
    assert [row.display_name for row in view.topk] == [
        "Thủy tinh",
        "Nhựa",
        "Kim loại",
    ]
    assert not hasattr(view, "icon")
    assert all(not hasattr(row, "icon") for row in view.topk)
```

- [ ] **Step 4: Chạy kiểm thử để xác nhận hợp đồng cũ thất bại**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/ui/test_presentation.py tests/ui/test_main_window.py tests/web/test_app_logic.py -q
```

Expected: FAIL vì `COLOR_BY_CLASS` chưa tồn tại và view model vẫn có `icon`.

- [ ] **Step 5: Sửa presentation model**

Trong `src/ui/presentation.py`, đổi cấu trúc style thành:

```python
@dataclass(frozen=True)
class LabelPresentation:
    display_name: str
    color: str


COLOR_BY_CLASS: dict[str, str] = {
    "battery": "#B45309",
    "biological": "#15803D",
    "cardboard": "#92400E",
    "clothes": "#7E22CE",
    "glass": "#0369A1",
    "metal": "#475569",
    "paper": "#2563EB",
    "plastic": "#0F766E",
    "shoes": "#9D174D",
    "trash": "#52525B",
}

UNKNOWN_PRESENTATION = LabelPresentation("Không xác định", "#64748B")


def present_label(class_id: str) -> LabelPresentation:
    display_name = DISPLAY_NAMES_VI.get(class_id)
    color = COLOR_BY_CLASS.get(class_id)
    if display_name is None or color is None:
        return UNKNOWN_PRESENTATION
    return LabelPresentation(display_name, color)
```

- [ ] **Step 6: Loại icon khỏi consumer PyQt và web**

Trong `src/ui/main_window.py`, dùng:

```python
self.result_label.setText(presentation.display_name)
self.top_three.addItem(
    f"{label.display_name}: {scored.probability:.1%}"
)
```

Trong `src/web/app_logic.py`, xóa thuộc tính `icon` khỏi hai dataclass và xóa hai đối số `icon=...` khi dựng view.

- [ ] **Step 7: Chạy kiểm thử Task 1**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\python.exe -m pytest tests/ui/test_presentation.py tests/ui/test_main_window.py tests/web/test_app_logic.py -q
```

Expected: tất cả kiểm thử được chọn PASS.

- [ ] **Step 8: Commit presentation contract**

```powershell
git add src/ui/presentation.py src/ui/main_window.py src/web/app_logic.py tests/ui/test_presentation.py tests/ui/test_main_window.py tests/web/test_app_logic.py
git commit -m "refactor(ui): remove label icons"
```

---

### Task 2: Tạo Streamlit shell không icon và giữ bốn chế độ

**Files:**
- Create: `src/web/style.py`
- Create: `tests/web/test_style.py`
- Create: `tests/web/test_streamlit_contract.py`
- Modify: `streamlit_app.py`

**Interfaces:**
- Consumes: text trạng thái và `tone` thuộc `neutral|success|warning|error`.
- Produces: `transparent_favicon() -> PIL.Image.Image`, `status_markup(message: str, tone: StatusTone) -> str`, hằng `ICON_FREE_CSS`, và helper Streamlit `render_status(message, tone, target=None) -> None`.

- [ ] **Step 1: Viết kiểm thử style thuần**

Tạo `tests/web/test_style.py`:

```python
import pytest

from src.web.style import ICON_FREE_CSS, status_markup, transparent_favicon


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


def test_style_hides_decorative_framework_icons() -> None:
    assert 'span[data-testid="stIconMaterial"]' in ICON_FREE_CSS
    assert '[data-testid="stHeaderActionElements"]' in ICON_FREE_CSS
    assert '[data-testid="stFileUploaderDropzone"] svg' in ICON_FREE_CSS
    assert '[data-testid="stCameraInput"] svg' in ICON_FREE_CSS


def test_status_markup_rejects_unknown_tone() -> None:
    with pytest.raises(ValueError, match="Unsupported status tone"):
        status_markup("message", "blue")  # type: ignore[arg-type]
```

- [ ] **Step 2: Viết kiểm thử hợp đồng tĩnh của Streamlit**

Tạo `tests/web/test_streamlit_contract.py`:

```python
from pathlib import Path


def test_streamlit_is_text_only_and_keeps_all_four_modes() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.title("Phân loại rác bằng MobileNetV3")' in source
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
    for label in ("Tải ảnh", "Chụp ảnh", "Tải video", "Camera trực tiếp"):
        assert f'"{label}"' in source
    assert "render_uploaded_video(service, settings)" in source
    assert "render_live_camera(service, settings)" in source
```

- [ ] **Step 3: Chạy kiểm thử để xác nhận UI hiện tại thất bại**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/web/test_style.py tests/web/test_streamlit_contract.py -q
```

Expected: FAIL vì `src.web.style` chưa tồn tại và Streamlit còn emoji/banner.

- [ ] **Step 4: Tạo primitive style an toàn**

Tạo `src/web/style.py` với cấu trúc sau:

```python
from __future__ import annotations

from html import escape
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
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stCameraInput"] svg {
  display: none !important;
}
.text-status {
  border-left: 4px solid #64748b;
  border-radius: 4px;
  margin: 0.5rem 0 1rem;
  padding: 0.75rem 1rem;
}
.text-status--neutral { background: #f8fafc; border-color: #64748b; }
.text-status--success { background: #f0fdf4; border-color: #15803d; }
.text-status--warning { background: #fffbeb; border-color: #b45309; }
.text-status--error { background: #fef2f2; border-color: #b91c1c; }
</style>
"""


def transparent_favicon() -> Image.Image:
    return Image.new("RGBA", (1, 1), (255, 255, 255, 0))


def status_markup(message: str, tone: StatusTone = "neutral") -> str:
    try:
        css_class = _STATUS_CLASSES[tone]
    except KeyError as error:
        raise ValueError(f"Unsupported status tone: {tone}") from error
    role = "alert" if tone == "error" else "status"
    return (
        f'<div role="{role}" class="text-status {css_class}">'
        f"{escape(message)}</div>"
    )
```

- [ ] **Step 5: Dùng favicon trong suốt và CSS ngay sau page config**

Trong `streamlit_app.py`, import các primitive mới, đặt:

```python
st.set_page_config(
    page_title="Phân loại rác 10 lớp",
    page_icon=transparent_favicon(),
    layout="wide",
)
st.markdown(ICON_FREE_CSS, unsafe_allow_html=True)
```

Thêm helper:

```python
def render_status(message: str, tone: StatusTone = "neutral", target=None) -> None:
    container = st if target is None else target
    container.markdown(status_markup(message, tone), unsafe_allow_html=True)
```

- [ ] **Step 6: Thay toàn bộ output icon/alert trong Streamlit**

Áp dụng các chuỗi thuần chữ:

```python
st.markdown(f"### {view.display_name}")
left.write(f"{rank}. {row.display_name}")
"Lớp": label.display_name
render_status(
    f"Gần nhất: {latest.display_name} ({latest.probability:.1%})",
    target=status,
)
```

Thay `st.info`, `st.warning`, `st.error`, `st.success` bằng `render_status` với
tone tương ứng. Xóa context `st.spinner` vì spinner là biểu tượng động; progress
bar và text hiện tại tiếp tục thể hiện tiến trình. Đổi tiêu đề thành:

```python
st.title("Phân loại rác bằng MobileNetV3")
```

Xóa hoàn toàn lệnh `st.info` ngay sau tiêu đề có câu bắt đầu bằng
`Mô hình phân loại toàn bộ khung hình`.

- [ ] **Step 7: Chạy kiểm thử Task 2**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/web/test_style.py tests/web/test_streamlit_contract.py tests/web/test_app_logic.py -q
```

Expected: tất cả kiểm thử được chọn PASS.

- [ ] **Step 8: Kiểm tra cú pháp và chuỗi emoji còn sót**

Run:

```powershell
.venv\Scripts\python.exe -m compileall src streamlit_app.py
rg -n "♻|🔋|🌿|view\.icon|row\.icon|label\.icon|latest\.icon|Mô hình phân loại toàn bộ khung hình" streamlit_app.py src
```

Expected: compile thành công; `rg` không trả dòng nào.

- [ ] **Step 9: Commit Streamlit presentation**

```powershell
git add src/web/style.py streamlit_app.py tests/web/test_style.py tests/web/test_streamlit_contract.py
git commit -m "feat(web): make Streamlit presentation icon-free"
```

---

### Task 3: Viết tài liệu bàn giao giai đoạn 1

**Files:**
- Create: `docs/giai-doan-1-ban-giao.md`
- Modify: `README.md`
- Modify: `tests/test_repository_contract.py`
- Modify: `docs/superpowers/specs/2026-09-03-stage1-handoff-ui-design.md`

**Interfaces:**
- Consumes: lệnh chạy, class order, model provenance và giới hạn đã có trong repository.
- Produces: tài liệu bàn giao GitHub với các heading ổn định và liên kết từ README.

- [ ] **Step 1: Viết kiểm thử hợp đồng tài liệu**

Mở rộng `test_documentation_matches_the_pinned_ten_class_system` trong
`tests/test_repository_contract.py`:

```python
handoff_path = Path("docs/giai-doan-1-ban-giao.md")
assert handoff_path.is_file()
handoff = handoff_path.read_text(encoding="utf-8")
assert "[Bàn giao giai đoạn 1](docs/giai-doan-1-ban-giao.md)" in readme
for heading in (
    "# Bàn giao giai đoạn 1",
    "## Phạm vi và kết quả",
    "## Cách chạy sản phẩm",
    "## Trạng thái thực nghiệm",
    "## Đầu vào cho giai đoạn 2",
):
    assert heading in handoff
assert "checkpoint demo" in handoff.lower()
assert "không phải kết quả huấn luyện hai dataset của nhóm" in handoff
assert "SSDLite320-MobileNetV3" in handoff
assert "YOLOv8n" in handoff
assert "Weighted Boxes Fusion" in handoff
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận tài liệu chưa tồn tại**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_repository_contract.py::test_documentation_matches_the_pinned_ten_class_system -q
```

Expected: FAIL tại `handoff_path.is_file()`.

- [ ] **Step 3: Tạo tài liệu bàn giao với cấu trúc cố định**

Tạo `docs/giai-doan-1-ban-giao.md` với các phần và nội dung bắt buộc:

```markdown
# Bàn giao giai đoạn 1

## Phạm vi và kết quả

Giai đoạn 1 là bộ phân loại toàn khung hình MobileNetV3-Large, 10 lớp và một
nhãn chính cho mỗi ảnh. Bốn đầu vào minh họa gồm tải ảnh, chụp ảnh, tải video và
camera trực tiếp. Đây không phải detector và không tạo bounding box.

## Kiến trúc và dữ liệu

Mô tả pipeline `merge -> mapping -> SHA-256/pHash -> group split 70/15/15 ->
train -> checkpoint -> evaluate -> inference`, hai dataset đã khóa phiên bản và
thứ tự 10 class ID.

## Cách chạy sản phẩm

Ghi nguyên lệnh tạo `artifacts/ecovision/best.pt`, chạy Streamlit, train,
evaluate và CLI inference đang có trong README.

## Trạng thái thực nghiệm

Phân biệt ba nhóm: mã nguồn đã hoàn thành; artifact cần chạy bằng dữ liệu thật;
checkpoint demo. Ghi rõ checkpoint demo không phải kết quả huấn luyện hai
dataset của nhóm và không ghi số accuracy chưa đo.

## Đầu vào cho giai đoạn 2

Liệt kê phần tái sử dụng được và backlog `taxonomy 6 lớp`, `scene_id`, CVAT,
bounding box, Albumentations bbox-aware, Copy-Paste có mask,
SSDLite320-MobileNetV3, YOLOv8n, Weighted Boxes Fusion và COCO mAP.

## Checklist bàn giao

Danh sách file cấu hình, manifest, checkpoint, metrics, lệnh chạy, provenance,
giới hạn và người nhận bàn giao cần kiểm tra.
```

Phần taxonomy chuyển tiếp phải nói rõ: tài liệu giai đoạn 2 mô tả một schema
VN-trash khác schema `Alu/Carton/Foam_box/...` của dataset hiện đã chọn; vì vậy
không tự ánh xạ `Other` hay nhãn mơ hồ thành lớp detection.

- [ ] **Step 4: Liên kết README và cập nhật trạng thái spec**

Thêm ngay sau đoạn mở đầu README:

```markdown
Tài liệu tổng hợp phạm vi, cách chạy, trạng thái thực nghiệm và đầu vào chuyển
tiếp nằm tại [Bàn giao giai đoạn 1](docs/giai-doan-1-ban-giao.md).
```

Đổi `Trạng thái: Chờ duyệt trước khi triển khai` trong spec thành
`Trạng thái: Đã duyệt ngày 2026-09-03`.

- [ ] **Step 5: Chạy kiểm thử tài liệu**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_repository_contract.py::test_documentation_matches_the_pinned_ten_class_system -q
```

Expected: PASS.

- [ ] **Step 6: Kiểm tra liên kết/lệnh quan trọng bằng tìm kiếm tĩnh**

Run:

```powershell
rg -n "Bàn giao giai đoạn 1|import_ecovision_checkpoint|streamlit run|MobileNetV3-Large|checkpoint demo|SSDLite320-MobileNetV3|YOLOv8n|Weighted Boxes Fusion" README.md docs/giai-doan-1-ban-giao.md
```

Expected: mọi thuật ngữ xuất hiện trong tài liệu bàn giao; README có liên kết.

- [ ] **Step 7: Commit tài liệu bàn giao**

```powershell
git add README.md docs/giai-doan-1-ban-giao.md docs/superpowers/specs/2026-09-03-stage1-handoff-ui-design.md tests/test_repository_contract.py
git commit -m "docs: add stage 1 handoff"
```

---

### Task 4: Xác minh sản phẩm và bàn giao nhánh

**Files:**
- Verify only: `streamlit_app.py`, `src/ui/presentation.py`, `src/web/app_logic.py`, `src/web/style.py`, `docs/giai-doan-1-ban-giao.md`, `README.md`

**Interfaces:**
- Consumes: checkpoint `artifacts/ecovision/best.pt` và server Streamlit localhost.
- Produces: bằng chứng cú pháp/kiểm thử sạch, ảnh chụp trực quan không icon/banner và nhánh đã push.

- [ ] **Step 1: Chạy quality gate có phạm vi**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check src/ui src/web streamlit_app.py tests/ui tests/web tests/test_repository_contract.py
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\python.exe -m pytest tests/ui/test_presentation.py tests/ui/test_main_window.py tests/web tests/test_repository_contract.py -q
.venv\Scripts\python.exe -m compileall src streamlit_app.py
```

Expected: ba lệnh thành công với exit code 0.

- [ ] **Step 2: Xác nhận MobileNetV3 checkpoint vẫn suy luận**

Run:

```powershell
.venv\Scripts\python.exe -c "from PIL import Image; from src.inference.predict import WastePredictor; result = WastePredictor('artifacts/ecovision/best.pt', device='cpu').predict_pil(Image.new('RGB', (224, 224), (128, 128, 128)), top_k=3); print(result)"
```

Expected: đối tượng `Prediction` có `top1`, `topk`, `low_confidence`; không có
lỗi compatibility và không cần ghi ảnh thử ra đĩa.

- [ ] **Step 3: Làm mới ứng dụng Streamlit đang chạy**

Nếu server cũ không tự tải lại, chạy:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false -- --checkpoint artifacts/ecovision/best.pt --device cpu
```

Expected: `http://localhost:8501` phản hồi và log không có traceback.

- [ ] **Step 4: Kiểm tra trực quan bằng in-app browser**

Tại `http://localhost:8501`, xác nhận:

1. tab trình duyệt không có favicon nhìn thấy;
2. tiêu đề là `Phân loại rác bằng MobileNetV3` và không có icon/link anchor;
3. banner xanh cũ không tồn tại;
4. bốn tab vẫn hiện đúng tên;
5. màn hình tải ảnh/chụp ảnh không có biểu tượng trang trí;
6. tải ảnh mẫu cho kết quả top-1/top-3 thuần chữ;
7. console không có lỗi JavaScript.

- [ ] **Step 5: Kiểm tra trạng thái Git và push nhánh**

Run:

```powershell
git status --short
git log -5 --oneline
git push origin feat/mobilenetv3-10-class
```

Expected: worktree sạch trước push; remote nhận đủ các commit mới.
