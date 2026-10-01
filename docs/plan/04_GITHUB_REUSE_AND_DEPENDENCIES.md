# 04 — GITHUB REUSE AND DEPENDENCIES: TÁI SỬ DỤNG MÃ NGUỒN MỞ VÀ QUẢN LÝ PHỤ THUỘC (R2.1)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2.1 — Data Gate Audit, Leakage Verification & Smoke Test  
**Trạng thái:** `VERIFIED` (Đã cài đặt PyTorch CUDA cu121, đối soát 258/308 keys backbone và giải quyết mâu thuẫn VN Trash)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Xác định rõ ràng, minh bạch các kho lưu trữ mã nguồn mở và trọng số pre-trained được kế thừa trong dự án; phân định ranh giới pháp lý về giấy phép mã nguồn mở (Licenses); kiểm toán nguy cơ rò rỉ tập pre-training của các checkpoint cộng đồng; chỉ rõ phần mã nguồn được kế thừa và các đóng góp khoa học/kỹ thuật cốt lõi của nhóm sinh viên; thiết lập danh mục phụ thuộc thư viện đồng bộ trên môi trường GPU thực tế.

### 1.2. Vấn đề khoa học đã làm rõ trong R2.1
1. **Bác bỏ hoàn toàn nhận định "VN Trash là tập test ngoại bộ độc lập" (`CONFLICT -> RESOLVED`):**
   - Trong bản thảo cũ và R2 sơ bộ từng ghi: "VN Trash dùng làm out-of-distribution test set".
   - Đối soát thực tế trên [data/audit/split_manifest.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/split_manifest.csv) và [data/audit/duplicate_groups.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/duplicate_groups.csv) chứng minh:
     * **1.801 ảnh VN Trash đang nằm trực tiếp trong tập Train** (chiếm 17,3% tập Train).
     * **879 ảnh VN Trash trùng khớp băm MD5/SHA-256 tuyệt đối với Garbage Classification V2**.
     * Do đó, việc gọi VN Trash là "tập test độc lập" là một sai lầm khoa học nghiêm trọng. Trong R2.1, VN Trash được định vị chính xác là **nguồn dữ liệu gộp chung để huấn luyện/kiểm thử nội bộ**, không được coi là tập test ngoại bộ.
2. **Minh bạch nguy cơ rò rỉ từ checkpoint cộng đồng (`Ecovision`):**
   - Checkpoint `AmadFR/ecovision_mobilenetv3` được train trên Garbage V2 nhưng tác giả không công bố split train/test cụ thể. Trạng thái: `PRETRAINING_OVERLAP_UNKNOWN`.
   - Vì VN Trash chứa 879 ảnh trùng tuyệt đối với Garbage V2, không thể dùng VN Trash để kiểm chứng xem Ecovision có bị "nhớ ảnh" hay không.
   - **Quyết định kiến trúc:** Mô hình phân loại của dự án bắt buộc phải fine-tune từ **ImageNet-1K chuẩn của PyTorch Torchvision** (`MobileNet_V3_Large_Weights.DEFAULT`), tuyệt đối không dựa vào checkpoint bên thứ ba không rõ nguồn gốc.
3. **Chuyển giao trọng số Backbone sang SSDLite320 (`VERIFIED`):**
   - Đã chạy script kiểm toán thực nghiệm [scripts/audit_backbone_transfer.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py), xuất bảng ánh xạ [data/audit/backbone_transfer_keys.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_keys.csv) và JSON [data/audit/backbone_transfer_audit.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json).
   - Tỷ lệ khớp: **258 / 308 feature keys (83,8%)** khớp hoàn hảo tên và kích thước tensor giữa Classifier và Detector Backbone.
   - 50 keys thuộc các block cuối (`features.14, 15, 16`) khác biệt về kênh chiếu; 168 keys của detector (Feature Pyramid Extra Layers + Bounding Box / Class Detection Heads) bắt buộc phải huấn luyện từ đầu trên nhãn bounding box.
   - Đã thực hiện nạp thử nghiệm (`load_state_dict(strict=False)`) và chạy forward pass thành công trên tensor ngẫu nhiên `(1, 3, 320, 320)`, trích xuất đúng 300 boxes dự đoán.

---

## 2. Bảng phân tích các kho lưu trữ tham khảo và giấy phép (`VERIFIED`)

| Tên nguồn / Dự án | Liên kết / Revision | Giấy phép | Thành phần kế thừa | Trạng thái kiểm toán & Điểm tích hợp |
|:---|:---|:---:|:---|:---|
| **Torchvision MobileNetV3** | PyTorch / Torchvision (`models.mobilenet_v3_large`) | BSD-3-Clause | Trọng số ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`) | `VERIFIED`: Trọng số khởi tạo chính thức cho huấn luyện Giai đoạn A |
| **Ecovision MobileNetV3** | HuggingFace: `AmadFR/ecovision_mobilenetv3` (Rev: `7c2daee`) | MIT License | Checkpoint weights `ecovision_mobilenetv3_ST.pth` (17.1 MB) | `PRETRAINING_OVERLAP_UNKNOWN`: [artifacts/ecovision/best.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/ecovision/best.pt); chỉ dùng làm đối chứng baseline tham khảo |
| **Ultralytics YOLOv8** | GitHub: `ultralytics/ultralytics` (v8.4.x) | AGPL-3.0 | Kiến trúc mạng YOLOv8n, giải thuật Decoupled Head, DFL/CIoU loss | `VERIFIED`: [src/detection/yolo.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/yolo.py); mô hình phát hiện chính Giai đoạn B |
| **Torchvision SSDLite320** | PyTorch / TorchVision (`models.detection`) | BSD-3-Clause | Kiến trúc `ssdlite320_mobilenet_v3_large` | `VERIFIED`: [src/detection/ssdlite.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/ssdlite.py); mô hình đối chứng (chuyển giao 258 keys backbone) |
| **Weighted Boxes Fusion** | GitHub: `ZFTurbo/Weighted-Boxes-Fusion` (`ensemble-boxes` v1.0.9) | MIT License | Giải thuật WBF (Solovyev et al.) hợp nhất bounding boxes | `VERIFIED`: [src/detection/fusion.py](file:///D:/CNTT-KLCN155-waste-detection/src/detection/fusion.py); module thử nghiệm có điều kiện |
| **Albumentations** | GitHub: `albumentations-team/albumentations` (v2.0.8) | MIT License | Pipeline biến đổi hình học, ánh sáng, màu sắc | `VERIFIED`: Phục vụ các nhánh thực nghiệm tăng cường dữ liệu |

---

## 3. Hiện trạng môi trường và thư viện trong workspace độc lập (`VERIFIED`)

- **Thư mục dự án độc lập:** `D:\CNTT-KLCN155-waste-detection` (Thư mục vật lý 100%, Git repo độc lập trên branch `main`).
- **Python Virtualenv:** `D:\CNTT-KLCN155-waste-detection\.venv` (Python 3.12.10).
- **Phần cứng xác thực:** NVIDIA GeForce RTX 2050 Laptop GPU (4.095,5 MB VRAM), CUDA Driver 591.86.
- **Thư viện Deep Learning cài đặt và xác thực:**
  - `torch==2.5.1+cu121` (`torch.cuda.is_available() == True`).
  - `torchvision==0.20.1+cu121`.
  - `scikit-learn==1.9.1`.
  - `matplotlib==3.11.2`, `tqdm==4.70.1`, `pandas==2.2.3`, `pillow==12.3.0`, `pyyaml==6.0.3`.
