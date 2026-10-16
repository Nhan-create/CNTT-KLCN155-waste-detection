# BÁO CÁO REVIEW VÀ ĐỀ XUẤT NGHIỆM THU GROUND TRUTH V2 (HOLDOUT)
## (GROUND TRUTH V2 REVIEW FOR STAKEHOLDER APPROVAL)

**Dự án:** Nhận diện Rác thải Đa lớp Thông minh (CNTT-KLCN155)  
**Tập dữ liệu:** Holdout Độc lập `D:\Hinh_Anh_Rac` (27 ảnh kiểm toán)  
**Tài liệu tham chiếu:** [`reports/qa_acceptance/ground_truth_hinh_anh_rac.json`](file:///D:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/ground_truth_hinh_anh_rac.json) (v1 Frozen)  
**Tệp đề xuất:** [`reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json`](file:///D:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json)  
**Trạng thái:** DỪNG VÀ XIN Ý KIẾN PHÊ DUYỆT (CHƯA ĐÓNG BĂNG V2 KHI CHƯA ĐƯỢC CHẤP THUẬN)

---

## 1. TỔNG QUAN HIỆN TRẠNG VÀ CƠ CẤU V2 PROPOSED

Tệp Ground Truth gốc (v1) gồm **27 ảnh** và **57 vật thể**, được giữ nguyên vẹn 100% (SHA-256: `f390f0f9440a977d0a0afe9e55b862562f1aafd3011ab18313e1772b18d470c0`).

Trong phiên bản đề xuất chuẩn hóa v2:
- **Tổng số vật thể:** 57 objects (bảo toàn toàn bộ danh mục v1, bổ sung trường trạng thái `v2_status`).
- **ACCEPTED (Hợp lệ, rời rạc rõ ràng):** **39 objects** (68.42%).
- **QUARANTINED (Cách ly hoàn toàn):** **4 objects** (7.02%).
- **FLAGGED_UNSEGREGATED_PILE (Đống rác/bãi rác cận cảnh):** **8 objects** (14.04%).
- **Mục còn tranh chấp / Cần thống nhất quy ước (6 objects):**
  - FLAGGED_CONTAINER_NESTING: 1 object.
  - FLAGGED_OVERLAPPING_OBJECT: 2 objects.
  - AMBIGUOUS_COMPOSITE_TAXONOMY: 1 object.
  - AMBIGUOUS_BAGGED_WASTE: 2 objects.
- **Active objects hiện tại (khi tính cả 6 mục tranh chấp):** $39 + 6 = 45$ objects.
- **Active objects nghiêm ngặt (chỉ tính ACCEPTED):** 39 objects.

---

## 2. CHI TIẾT 6 ĐỐI TƯỢNG ACTIVE CHƯA THỐNG NHẤT CẦN DUYỆT

### Đối tượng 1: Thùng carton chứa hồ sơ giấy (`images (12).jpg` – `img06_obj01`)
- **Tệp:** `images (12).jpg` ($554 \times 554$)
- **Object ID:** `img06_obj01`
- **Class hiện tại:** `cardboard` (ID 2)
- **Class đề xuất:** Giữ nguyên `cardboard` (nhưng tinh chỉnh ranh giới về viền thùng ngoài)
- **Bounding box:** `[32, 36, 535, 519]` (Diện tích: **79.16%**)
- **Vật thể bên trong:** `img06_obj02` (`paper`, ID 6, box `[110, 90, 475, 445]`, diện tích 42.22%)
- **Vấn đề kỹ thuật:** Quan hệ lồng nhau giữa vật chứa (container) và vật được chứa (contained). Thùng carton bao kín trọn tập hồ sơ giấy. Nếu detector phát hiện cả 2 box thì có nguy cơ bị NMS loại trừ hoặc phạt IoU lồng ghép.
- **Mức độ chắc chắn:** **Cao (85%)**
- **Tác động đến Evaluation:** Nếu giữ cả 2 box, mô hình phải phát hiện đồng thời `cardboard` và `paper` tại cùng một vùng không gian. Đề xuất: Chấp nhận cả 2 box với quy ước hierarchy (Cha: `cardboard`, Con: `paper`).

### Đối tượng 2: Đống quần áo lồng nhau (`images (10).jpg` – `img04_obj01`)
- **Tệp:** `images (10).jpg` ($640 \times 398$)
- **Object ID:** `img04_obj01`
- **Class hiện tại:** `clothes` (ID 3)
- **Class đề xuất:** `clothes` (Hợp nhất hoặc xóa box nhỏ)
- **Bounding box:** `[0, 160, 640, 398]` (Diện tích: **59.80%**)
- **Vật thể bị lồng:** `img04_obj02` (`clothes`, box `[0, 120, 325, 295]`, diện tích 22.33%)
- **Vấn đề kỹ thuật:** Cả 2 box đều cùng class `clothes` và nằm chồng lên nhau trong cùng một cụm vải dệt.
- **Mức độ chắc chắn:** **Cao (90%)**
- **Tác động đến Evaluation:** Khi 2 box cùng class có độ trùng lắp lớn trong GT, detector dự đoán 1 box duy nhất sẽ chỉ khớp được 1 GT (1 TP) và bị phạt 1 FN đối với box còn lại. Đề xuất: Hợp nhất thành 1 box `clothes` duy nhất bao quát toàn bộ cụm rác vải.

### Đối tượng 3: Đống quần áo sa mạc chồng lấn (`OIP (1).webp` – `img20_obj01`)
- **Tệp:** `OIP (1).webp` ($313 \times 237$)
- **Object ID:** `img20_obj01`
- **Class hiện tại:** `clothes` (ID 3)
- **Class đề xuất:** `clothes` (Điều chỉnh ranh giới)
- **Bounding box:** `[0, 95, 313, 237]` (Diện tích: **59.92%**)
- **Vật thể kề cận:** `img20_obj02` (`clothes`, box `[0, 70, 160, 175]`, diện tích 22.65%)
- **Vấn đề kỹ thuật:** Hai gò đống quần áo phế thải trên sườn sa mạc Atacama có phần giao nhau ở nửa trái ảnh.
- **Mức độ chắc chắn:** **Trung bình (80%)**
- **Tác động đến Evaluation:** Cần phân định ranh giới rõ giữa tiền cảnh bên phải và gò đồi bên trái để tránh việc detector dự đoán 1 box lớn bị tính là False Positive với box nhỏ.

### Đối tượng 4: Hộp sữa tiệt trùng đa lớp Tetra Pak (`Phan-Loai-Rac-Tai-Ng-01.jpg` – `img22_obj04`)
- **Tệp:** `Phan-Loai-Rac-Tai-Ng-01.jpg` ($960 \times 720$)
- **Object ID:** `img22_obj04`
- **Class hiện tại:** `cardboard` (ID 2)
- **Class đề xuất:** Thống nhất là `cardboard` (hoặc `paper`) theo quy ước GLOSSARY.md
- **Bounding box:** `[680, 570, 765, 650]` (Diện tích: **0.98%**)
- **Vấn đề kỹ thuật:** Vỏ hộp sữa TH true MILK có cấu tạo 75% giấy bìa cứng, 20% màng nhựa PE và 5% lá nhôm. Một số detector nhận là `paper`, một số nhận là `cardboard`.
- **Mức độ chắc chắn:** **Cao (90%)**
- **Tác động đến Evaluation:** Nếu mô hình dự đoán `paper` mà GT quy định `cardboard`, mô hình bị phạt kép: 1 FP cho `paper` và 1 FN cho `cardboard`. Cần thống nhất quy ước taxonomy toàn dự án.

### Đối tượng 5 & 6: Bao tải và túi rác chứa rác hỗn hợp (`Phan-Loai-Rac-Tai-Ng-01.jpg` – `img22_obj01`, `img22_obj02`)
- **Tệp:** `Phan-Loai-Rac-Tai-Ng-01.jpg` ($960 \times 720$)
- **Object ID:** `img22_obj01` (Bao xanh) và `img22_obj02` (Túi đỏ)
- **Class hiện tại:** `plastic` (ID 7)
- **Class đề xuất:** `trash` (ID 9) nếu xét theo bản chất rác bên trong; hoặc giữ `plastic` nếu chỉ xét vỏ bao bì
- **Bounding box:**
  - `img22_obj01`: `[230, 220, 460, 550]` (Diện tích: **10.98%**)
  - `img22_obj02`: `[340, 445, 675, 705]` (Diện tích: **12.60%**)
- **Vấn đề kỹ thuật:** Vỏ ngoài là túi nilon (plastic), nhưng bên trong là rác sinh hoạt tổng hợp (thức ăn thừa, hộp xốp, giấy vụn). Nếu gán `plastic`, mô hình học rằng mọi bao rác hỗn hợp đều là nhựa tái chế.
- **Mức độ chắc chắn:** **Trung bình (75%)**
- **Tác động đến Evaluation:** Đây là điểm nghẽn phân loại lớn giữa `plastic` và `trash`.

---

## 3. PHÂN ĐỊNH 10 ẢNH CÓ ZERO ACTIVE OBJECTS TRONG GT V2

Trong tệp v2 đề xuất, có đúng **10 ảnh** không còn bất kỳ active object nào cho bài toán Object Detection vật thể đơn lẻ:
1. `images (18).jpg` – Thùng rác công cộng 2 bánh xe đô thị (`QUARANTINED_UTILITY_RECEPTACLE`)
2. `images (8).jpg` – Sọt đựng rác văn phòng đục lỗ (`QUARANTINED_UTILITY_RECEPTACLE`)
3. `images (13).jpg` – Đống chai nhựa 100% canvas (File trùng lặp byte với `images (7)`)
4. `images (7).jpg` – Đống chai nhựa cận cảnh 100% canvas (`FLAGGED_UNSEGREGATED_PILE`)
5. `images (6).jpg` – Rác hữu cơ sinh hoạt 100% canvas (`FLAGGED_UNSEGREGATED_PILE`)
6. `images (9).jpg` – Mảnh bìa carton sóng 100% canvas (`FLAGGED_UNSEGREGATED_PILE`)
7. `images (19).jpg` – Đống rác hữu cơ ủ phân compost 84.92% canvas (`FLAGGED_UNSEGREGATED_PILE`)
8. `OIP.webp` – Đống quần áo cũ 100% canvas (`FLAGGED_UNSEGREGATED_PILE`)
9. `photo-5-1642556...webp` – Ngọn núi rác quần áo sa mạc 96.01% canvas (`FLAGGED_UNSEGREGATED_PILE`)
10. `rac-thai-huu-co-1.jpg` – Rác hữu cơ nhà bếp phân hủy 100% canvas (`FLAGGED_UNSEGREGATED_PILE`)

### Nguyên tắc phân loại nghiêm ngặt:
- **Tuyệt đối KHÔNG coi 8 ảnh đống rác là ảnh cảnh nền sạch (Negative):** Vì thực tế các ảnh này chứa đầy rác thải. Nếu coi là Negative, bất kỳ dự đoán đúng nào của detector đều bị tính oan là False Positive!
- **Quy tắc nghiệm thu đề xuất:**
  - Đối với bài toán **Object Detection vật thể đơn lẻ**: 10 ảnh này được đánh dấu là `EXCLUDED_FROM_INSTANCE_DETECTION` (loại khỏi mẫu tính mAP của detector rời rạc).
  - Đối với bài toán **Waste Pile / Dump Estimation**: 8 ảnh đống rác được chuyển sang tập đánh giá riêng biệt cho bài toán Phân đoạn ngữ nghĩa (Semantic Segmentation) hoặc Phân loại mức độ ô nhiễm (Image-level Pollution Severity).

---

## 4. BẢNG TỔNG HỢP KIỂM DUYỆT XIN Ý KIẾN CHỦ DỰ ÁN

| Danh mục | Số lượng | Trạng thái hiện tại | Đề xuất nghiệm thu | Cần duyệt |
|---|---|---|---|---|
| **Vật thể rời rạc chuẩn** | 39 objects | ACCEPTED | Đưa vào GT v2 chính thức | **Phê duyệt** |
| **Hạ tầng thu gom** | 2 objects (`images (8)`, `(18)`) | QUARANTINED | Chuyển sang Negative / Receptacle | **Phê duyệt** |
| **Box trùng đè toàn ảnh** | 2 objects (`download (2)`, `base64...`) | QUARANTINED | Xóa bỏ box nền; giữ box con | **Phê duyệt** |
| **Đống rác hỗn hợp cận cảnh** | 8 objects | FLAGGED_PILE | Tách riêng bài toán Pile Detection | **Phê duyệt** |
| **Box lồng nhau / Chồng lấn** | 3 objects (`images (10)`, `(12)`, `OIP (1)`) | FLAGGED_NESTED | Điều chỉnh ranh giới / Hợp nhất | **Xin ý kiến** |
| **Tranh chấp vật liệu vs. rác** | 3 objects (`img22_01`, `02`, `04`) | AMBIGUOUS | Áp dụng chuẩn hóa GLOSSARY.md | **Xin ý kiến** |

---
*Báo cáo được lập để xin ý kiến chỉ đạo trực tiếp của Chủ dự án trước khi thực hiện freeze Ground Truth v2.*
