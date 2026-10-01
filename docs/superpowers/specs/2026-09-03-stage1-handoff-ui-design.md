# Thiết kế hoàn thiện và bàn giao giai đoạn 1

Ngày: 2026-09-03  
Trạng thái: Chờ duyệt trước khi triển khai

## 1. Quyết định phạm vi

Giai đoạn 1 tiếp tục là hệ thống phân loại ảnh toàn khung hình bằng
MobileNetV3-Large với 10 lớp:

`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`,
`plastic`, `shoes`, `trash`.

Không thay MobileNetV3 bằng ViT hoặc ensemble trong lần hoàn thiện này. Camera
và video vẫn là chức năng mở rộng của bộ phân loại giai đoạn 1. Bounding box,
taxonomy sáu lớp, SSDLite320-MobileNetV3, YOLOv8n và Weighted Boxes Fusion thuộc
giai đoạn 2, không được mô tả như chức năng đã có của giai đoạn 1.

## 2. Tài liệu bàn giao

Tạo `docs/giai-doan-1-ban-giao.md` làm tài liệu bàn giao chính trong GitHub.
Tài liệu phải đủ để một thành viên khác có thể hiểu, chạy và tiếp tục dự án mà
không cần đọc toàn bộ lịch sử phát triển. Nội dung gồm:

1. mục tiêu, phạm vi và ranh giới giữa hai giai đoạn;
2. kiến trúc và luồng dữ liệu của bộ phân loại 10 lớp;
3. nguồn dữ liệu, taxonomy, quy tắc ánh xạ và chống rò rỉ;
4. mô hình, huấn luyện, checkpoint, đánh giá và suy luận;
5. cách chuẩn bị checkpoint demo, chạy Streamlit và dùng bốn chế độ;
6. trạng thái sản phẩm theo ba mức `đã hoàn thành`, `cần kết quả chạy thật` và
   `chuyển sang giai đoạn 2`;
7. giới hạn trung thực: checkpoint demo không phải kết quả huấn luyện hai bộ dữ
   liệu của nhóm, chưa có số liệu thực nghiệm chính thức trong repository;
8. danh sách đầu vào bàn giao cho giai đoạn 2 và các việc không được tự động suy
   diễn, đặc biệt là taxonomy VN Trash đang khác mô tả trong đề cương.

README liên kết đến tài liệu bàn giao và giữ hướng dẫn chạy ngắn gọn. Tài liệu
không công bố accuracy, mAP hoặc FPS khi chưa có artifact đo thực tế.

## 3. Giao diện Streamlit không biểu tượng

Giao diện giữ bốn tab hiện có: tải ảnh, chụp ảnh, tải video và camera trực tiếp.
Không thay đổi pipeline suy luận, checkpoint hoặc cách làm mượt video.

Các thay đổi trình bày:

- tiêu đề chỉ còn chữ `Phân loại rác bằng MobileNetV3`;
- dùng favicon trong suốt để tab trình duyệt không hiển thị biểu tượng dự án hoặc
  biểu tượng mặc định của Streamlit;
- xóa banner xanh giải thích bộ phân loại toàn khung hình;
- bỏ emoji/icon khỏi kết quả top-1, top-3, thống kê video và camera;
- bỏ icon khỏi lớp presentation dùng chung để giao diện PyQt không tái sử dụng
  emoji;
- thay các hộp trạng thái có icon bằng khối trạng thái chỉ có chữ;
- ẩn các biểu tượng trang trí trong widget tải file/camera bằng selector ổn định,
  nhưng không ẩn nút chức năng hoặc nội dung hỗ trợ khả năng sử dụng.

Giao diện vẫn phải hiển thị rõ lỗi thiếu checkpoint, confidence thấp, video vượt
giới hạn và lỗi media. Việc bỏ icon không được làm mất thông tin trạng thái.

## 4. Luồng chạy sau thay đổi

1. Người dùng chạy `python -m scripts.import_ecovision_checkpoint` nếu chưa có
   checkpoint demo.
2. Người dùng chạy `streamlit run streamlit_app.py`.
3. Streamlit tải cùng `WastePredictor` và checkpoint MobileNetV3 như trước.
4. Ảnh/chụp ảnh được phân loại trực tiếp; video/camera được lấy mẫu và làm mượt
   như hiện tại.
5. Mọi kết quả chỉ dùng chữ, xác suất và thanh tiến trình, không có emoji hoặc
   favicon nhận diện.

## 5. Xử lý lỗi và tính trung thực

- Thiếu checkpoint: hiển thị thông báo chữ và lệnh chuẩn bị model.
- Ảnh/video lỗi: giữ thông báo nguyên nhân, không dùng alert có icon.
- Confidence thấp: giữ cảnh báo bằng màu/chữ, không thay đổi top-1 ngầm.
- Tài liệu bàn giao phải phân biệt rõ mã nguồn đã hoàn thành với thí nghiệm chưa
  được chạy trên hai dataset của nhóm.
- Không trình bày việc xóa biểu tượng như một biện pháp che giấu nguồn gốc công
  việc; đây chỉ là quyết định thiết kế giao diện tối giản.

## 6. Các tệp dự kiến thay đổi

- `streamlit_app.py`: favicon, CSS, tiêu đề, banner và cách hiển thị trạng thái.
- `src/ui/presentation.py`: loại bỏ dữ liệu emoji khỏi presentation model.
- `src/ui/main_window.py`: hiển thị nhãn thuần chữ.
- `src/web/app_logic.py`: loại bỏ trường icon khỏi view model.
- `README.md`: cập nhật mô tả giao diện và liên kết tài liệu bàn giao.
- `docs/giai-doan-1-ban-giao.md`: tài liệu bàn giao mới.
- Các kiểm thử giao diện/view model bị ảnh hưởng: cập nhật theo hợp đồng không
  icon, không thay đổi kiểm thử mô hình hoặc pipeline dữ liệu.

## 7. Tiêu chí nghiệm thu

1. MobileNetV3-Large vẫn tạo đúng 10 logits và checkpoint hiện tại vẫn tải được.
2. Bốn chế độ Streamlit vẫn hiện diện và dùng cùng luồng suy luận trước đó.
3. Không còn emoji/icon do dự án thêm vào, favicon nhìn thấy hoặc banner xanh đã
   chỉ định trên trang chính.
4. Các trạng thái lỗi/cảnh báo vẫn đọc được chỉ bằng văn bản.
5. Tài liệu bàn giao nêu đúng phần đã hoàn thành, phần thiếu bằng chứng thực
   nghiệm và backlog giai đoạn 2.
6. README dẫn đến tài liệu bàn giao và lệnh chạy hiện tại vẫn đúng.
7. Kiểm tra cú pháp, kiểm thử mục tiêu và kiểm tra trực quan trang localhost
   không phát hiện lỗi hồi quy.
