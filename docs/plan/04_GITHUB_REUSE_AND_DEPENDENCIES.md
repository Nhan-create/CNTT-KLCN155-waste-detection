# 04 — GITHUB REUSE AND DEPENDENCIES: TÁI SỬ DỤNG MÃ NGUỒN MỞ VÀ QUẢN LÝ PHỤ THUỘC (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã xác minh nguồn gốc và bóc tách metadata)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Xác định rõ ràng, minh bạch các kho lưu trữ mã nguồn mở (GitHub / Hugging Face) và trọng số pre-trained được kế thừa trong dự án; phân định ranh giới pháp lý về giấy phép mã nguồn mở (Licenses); kiểm toán nguy cơ rò rỉ tập pre-training của các checkpoint cộng đồng; chỉ rõ phần mã nguồn được kế thừa và các đóng góp khoa học/kỹ thuật cốt lõi của nhóm sinh viên; thiết lập danh mục phụ thuộc thư viện đồng bộ.

### 1.2. Vấn đề cần giải quyết
1. **Minh bạch nguy cơ rò rỉ từ checkpoint cộng đồng (`Ecovision`):** Checkpoint `AmadFR/ecovision_mobilenetv3` được train trên Garbage V2 nhưng tác giả không công bố split train/test cụ thể. Trạng thái: `PRETRAINING_OVERLAP_UNKNOWN`. Nếu đánh giá checkpoint này trên tập test Garbage V2, kết quả có thể bị thiên vị do Data Contamination. Giải pháp: Huấn luyện chính thức sẽ bắt đầu từ **ImageNet-1K chuẩn** của Torchvision; Ecovision chỉ đóng vai trò baseline đối chứng ngoại bộ và chỉ đánh giá trên VN Trash và ảnh thực tế TP.HCM.
2. **Làm rõ bản chất tệp `yolo26n.pt`:** Đã bóc tách metadata xác nhận đây là mô hình COCO 80 lớp (`tune-yolo26n-objv1-coco`), không phải mô hình rác. Huấn luyện đa rác Giai đoạn B sẽ sử dụng bộ khung chuẩn của Ultralytics `yolov8n.pt`.
3. **Tính liêm chính học thuật (Academic Integrity):** Tuyệt đối không xóa tên tác giả gốc, không che giấu nguồn gốc mã nguồn tham khảo, không biến công trình mã nguồn mở của cộng đồng thành "tự phát triển 100%".

---

## 2. Bảng phân tích các kho lưu trữ tham khảo và giấy phép (`VERIFIED`)

| Tên nguồn / Dự án | Liên kết / Revision | Giấy phép | Thành phần kế thừa | Trạng thái kiểm toán & Điểm tích hợp |
|:---|:---|:---:|:---|:---|
| **Torchvision MobileNetV3** | PyTorch / Torchvision (`models.mobilenet_v3_large`) | BSD-3-Clause | Trọng số ImageNet-1K (`weights='DEFAULT'`) | `VERIFIED`: Trọng số khởi tạo chính thức cho huấn luyện Giai đoạn A |
| **Ecovision MobileNetV3** | HuggingFace: `AmadFR/ecovision_mobilenetv3` (Rev: `7c2daee`) | MIT License | Checkpoint weights `ecovision_mobilenetv3_ST.pth` (17.1 MB) | `PRETRAINING_OVERLAP_UNKNOWN`: [artifacts/ecovision/best.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/ecovision/best.pt); chỉ dùng làm baseline tham chiếu ngoại bộ |
| **Ultralytics YOLOv8** | GitHub: `ultralytics/ultralytics` (v8.4.x) | AGPL-3.0 | Kiến trúc mạng YOLOv8n, giải thuật Decoupled Head, DFL/CIoU loss | `VERIFIED`: [src/detection/yolo.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py); mô hình phát hiện chính Giai đoạn B |
| **Torchvision SSDLite320** | PyTorch / TorchVision (`models.detection`) | BSD-3-Clause | Kiến trúc `ssdlite320_mobilenet_v3_large` | `VERIFIED`: [src/detection/ssdlite.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/ssdlite.py); mô hình đối chứng (chuyển giao 258 keys backbone) |
| **Weighted Boxes Fusion** | GitHub: `ZFTurbo/Weighted-Boxes-Fusion` (`ensemble-boxes` v1.0.9) | MIT License | Giải thuật WBF (Solovyev et al.) hợp nhất bounding boxes | `VERIFIED`: [src/detection/fusion.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/fusion.py); module thử nghiệm có điều kiện |
| **Albumentations** | GitHub: `albumentations-team/albumentations` (v2.0.8) | MIT License | Pipeline biến đổi hình học, ánh sáng, màu sắc | `VERIFIED`: Phục vụ các nhánh thực nghiệm tăng cường dữ liệu |

---

## 3. Phân định ranh giới đóng góp của dự án (Contribution Matrix)

| Hạng mục kỹ thuật | Kế thừa từ cộng đồng mã nguồn mở | Đóng góp nghiên cứu và kỹ thuật của Nhóm dự án |
|:---|:---|:---|
| **Kiến trúc mô hình** | MobileNetV3 (Torchvision), YOLOv8 (Ultralytics), SSDLite320 (PyTorch), WBF (`ensemble-boxes`). | Tích hợp thành một hệ thống đồng nhất có thể chuyển đổi linh hoạt các backend trên cùng một giao diện Web. |
| **Dữ liệu & Nhãn** | Bộ ảnh thô Kaggle Garbage Classification V2 và Kaggle VN Trash Classification. | (1) Thuật toán khử trùng bằng SHA-256 + pHash BK-Tree loại bỏ 923 ảnh trùng; (2) Cơ chế Cluster-based Split chống rò rỉ; (3) Bộ ảnh thực tế tự thu thập tại TP.HCM kèm nhãn bounding box chuẩn 10 lớp; (4) Công cụ thẩm định/gán nhãn nội bộ [src/ui/review_tool.py](file:///D:/CNTT-KLCN155-waste-detection/src/ui/review_tool.py). |
| **Thực nghiệm học sâu** | Trọng số pre-trained ImageNet-1K và COCO. | (1) Huấn luyện fine-tune MobileNetV3 10 lớp từ ImageNet-1K; (2) Nghiên cứu đối chứng SSDLite320 vs YOLOv8n trên rác sinh hoạt với cơ chế chuyển giao 258 backbone keys; (3) Tối ưu hóa trọng số WBF qua Grid Search trên tập Validation; (4) Khảo sát thực nghiệm Ablation tăng cường dữ liệu. |
| **Ứng dụng phần mềm** | Nền tảng Streamlit cơ bản. | Xây dựng ứng dụng hoàn chỉnh: hỗ trợ cả phân loại đơn rác và phát hiện đa rác, đo độ trễ suy luận chính xác, xuất tệp kết quả ZIP và dữ liệu kiểm toán CSV/JSON. |

---

## 4. Hiện trạng môi trường và thư viện trong workspace độc lập (`VERIFIED`)

- **Hệ điều hành:** Windows 11 Home Single Language (Build 26100).
- **Phần cứng:** AMD Ryzen 5 6600H, RAM 16GB, GPU NVIDIA GeForce RTX 2050 (4GB VRAM), Driver 591.86.
- **Python Workspace:** Python 3.12.10 tại `D:\CNTT-KLCN155-waste-detection\.venv`.
- **Thư viện đã cài đặt trong venv mới:** `pillow`, `pandas`, `pyyaml`, `numpy`.
- **Lệnh cài đặt PyTorch CUDA cho Task 01 tiếp theo:**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  .\.venv\Scripts\python.exe -m pip install ultralytics streamlit albumentations ensemble-boxes pycocotools opencv-python pytest
  ```
