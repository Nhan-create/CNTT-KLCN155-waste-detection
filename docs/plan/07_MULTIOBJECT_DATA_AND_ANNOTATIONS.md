# 07 — MULTIOBJECT DATA AND ANNOTATIONS: DỮ LIỆU VÀ GÁN NHÃN ĐA RÁC (GIAI ĐOẠN B) (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã loại bỏ hoàn toàn CVAT, kiểm toán nguồn tập v1)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Xây dựng một bộ dữ liệu phát hiện đối tượng đa rác (Multi-Object Waste Detection Dataset) hoàn chỉnh, chuẩn hóa theo danh mục 10 lớp thống nhất (với cơ chế ánh xạ đối chiếu sang 6 nhóm vật liệu của đề cương cũ khi cần); kiểm soát nghiêm ngặt nguồn gốc dữ liệu; bóc tách và cô lập dữ liệu tổng hợp (synthetic); triển khai quy trình thẩm định và gán nhãn Bounding Box gọn nhẹ qua công cụ Streamlit nội bộ [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) và script Python, loại bỏ hoàn toàn việc sử dụng CVAT.

### 1.2. Vấn đề cốt lõi cần giải quyết
1. **Loại bỏ CVAT theo chỉ đạo dứt khoát của PM:** Không sử dụng CVAT hay các giải pháp Docker cồng kềnh. Sử dụng công cụ xem và gán nhãn nhẹ viết bằng Streamlit và Python có sẵn trong repository để nhóm sinh viên thẩm định trực tiếp trên máy local.
2. **Kiểm toán và cô lập ảnh tổng hợp trong tập v1:** Kết quả kiểm toán [data/audit/audit_summary.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/audit_summary.json) xác nhận trong 1.419 ảnh của tập `detection/v1` có tới **1.305 ảnh (91,97%) là ảnh tổng hợp (Mendeley Synthetic)** và chỉ 114 ảnh (8,03%) là ảnh thực tế từ OpenImages. Ảnh tổng hợp chỉ dùng cho bước pretraining sơ khởi, tuyệt đối không đưa vào tập Test đánh giá thực tế.
3. **Bẫy gán nhãn một phần (Partial Annotation Trap):** Trong một bức ảnh có nhiều món rác hỗn hợp, bắt buộc phải vẽ bounding box cho toàn bộ các vật thể nhìn thấy được để tránh việc các vật thể chưa được gán nhãn bị coi là vùng nền âm tính (negative background), triệt tiêu gradient của mô hình.

---

## 2. Hiện trạng tập dữ liệu Detection v1 đã xác minh (`VERIFIED`)

Thư mục lưu trữ: [data/detection/v1](file:///D:/CNTT-KLCN155-waste-detection/data/detection/v1)
- **Tổng số ảnh:** 1.419 ảnh.
- **Tổng số Bounding Boxes:** 4.602 hộp.
- **Cơ cấu nguồn gốc:**
  - **Mendeley Synthetic:** 1.305 ảnh ($91,97\%$) — Rác nhân tạo ghép đồ họa trên mặt phẳng.
  - **OpenImages:** 114 ảnh ($8,03\%$) — Ảnh chụp thực tế ngoài trời.
- **Quyết định xử lý dữ liệu:**
  - Tách riêng nhãn nguồn gốc `is_synthetic` trong manifest.
  - Chỉ dùng dữ liệu tổng hợp để hỗ trợ tập Train; toàn bộ tập Validation và Test của Giai đoạn B bắt buộc dùng ảnh thực tế chụp tại TP.HCM và OpenImages đã qua thẩm định bằng công cụ nội bộ.

---

## 3. Quy trình Gán nhãn và Thẩm định bằng Công cụ Nội bộ (In-Repo Review Tool)

### 3.1. Kiến trúc Công cụ Thẩm định Nội bộ
Thay vì cài đặt phần mềm bên thứ ba phức tạp, dự án sử dụng module:
- **Công cụ trực quan:** [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py) chạy trên Streamlit (`streamlit run src/ui/review_tool.py`).
- **Chức năng:**
  - Hiển thị ảnh kèm các bounding box hiện có (vẽ overlay bằng PIL/OpenCV).
  - Cho phép người duyệt bấm duyệt (`APPROVE`), sửa nhãn (`RE-LABEL`), hoặc từ chối ảnh rác (`REJECT`).
  - Hỗ trợ nhập tọa độ chỉnh sửa bounding box trực tiếp trên web UI hoặc thông qua tệp JSON cấu hình.
  - Tự động xuất tệp nhãn YOLO `.txt` chuẩn và cập nhật manifest kiểm toán.

### 3.2. Tiêu chuẩn vẽ Bounding Box
1. **Ôm sát mép (Tight Box):** Box phải bao trọn vật thể, tính cả nắp chai, quai túi hoặc phần nhô ra. Không để thừa khoảng trống nền quá lớn.
2. **Che khuất (Occluded):** Nếu vật thể bị che khuất một phần (ví dụ hộp xốp đè lên chai nhựa), vẽ box bao trọn phần nhìn thấy được của vật thể bị đè.
3. **Vật thể trong suốt:** Vẽ box dựa trên viền phản quang và nắp chai.
4. **Ảnh nền không có rác (Background / Negative Images):** Đưa vào tập Train khoảng $5\% - 8\%$ ảnh mặt đường, vỉa hè hoặc mặt bàn sạch không có rác (tệp `.txt` để trống 0 bytes) để dạy mô hình triệt tiêu báo động giả.

---

## 4. Kế hoạch Thu thập và Gán nhãn Ảnh thực tế TP.HCM

```
Nguồn 1: OpenImages v1 đã duyệt                          ==> 114 ảnh thực tế
Nguồn 2: Ảnh chụp thực địa tại TP.HCM (HUIT, đường phố)   ==> 300–400 ảnh thực tế
Nguồn 3: Mendeley Synthetic (Chỉ dùng cho Train)         ==> 1,305 ảnh tổng hợp
-----------------------------------------------------------------------------------------
TẬP DỮ LIỆU ĐA RÁC GIAI ĐOẠN B:                          ==> ~1,800 ảnh (đủ cho 10 lớp)
```

- **Quy trình kiểm tra chéo (Cross-Review) giữa 3 sinh viên:**
  - Sinh viên 1 (Ngô Thanh Nhân): Chụp và tiền xử lý lô ảnh $\rightarrow$ Sinh viên 2 (Võ Gia Ninh) thẩm định qua Review Tool.
  - Sinh viên 2 (Võ Gia Ninh): Chụp lô ảnh tiếp theo $\rightarrow$ Sinh viên 3 (Vũ Trường Vinh) thẩm định.
  - Sinh viên 3 (Vũ Trường Vinh): Chụp lô ảnh tiếp theo $\rightarrow$ Sinh viên 1 (Ngô Thanh Nhân) thẩm định.
  - Các ca bất đồng ý kiến về nhãn lớp sẽ được đưa vào biên bản trao đổi và chốt bởi nhóm trưởng.

---

## 5. Tiêu chí nghiệm thu bộ dữ liệu đa rác

1. **100% ảnh tập Test là ảnh thực tế:** Không chứa bất kỳ ảnh tổng hợp Mendeley Synthetic nào trong tập Test.
2. **Định dạng nhãn chuẩn YOLO:** Mỗi tệp `.txt` chứa các dòng `<class_id> <x_center> <y_center> <w> <h>` chuẩn hóa trong đoạn $[0, 1]$.
3. **Không rò rỉ bối cảnh:** Các ảnh chụp cùng một địa điểm / thời điểm được gom cụm vào cùng một split để tránh hiện tượng học thuộc nền.
