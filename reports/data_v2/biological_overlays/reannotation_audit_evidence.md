# BẰNG CHỨNG RE-ANNOTATION CHO TACO_0072 VÀ TACO_0092
**Trạng thái phê duyệt:** `PROPOSED_PENDING_USER_APPROVAL`  
**Chính sách an toàn:** Tuyệt đối không ghi đè nhãn gốc trong `data/detection/labels/real/` cho đến khi người dùng phê duyệt chính thức.  

---

## 1. ẢNH TACO_0072.JPG (Thực phẩm dạng bánh màu nâu trong túi nilon)

### Hình ảnh trực quan đối chiếu:
File đối chiếu: [`reports/data_v2/biological_overlays/taco_0072_before_after_comparison.jpg`](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/biological_overlays/taco_0072_before_after_comparison.jpg)

### Nhận diện hình ảnh thực tế (Ground Truth Reality):
- **Vật thể thực tế:** Thực phẩm nướng màu nâu sẫm (bánh mì nguyên cám / bánh bun cắt lát) nằm bên trong túi nilon bao bì trong suốt, được cầm bằng tay người trong nhà, phía trên chậu cây húng quế và tường trắng. **Không phải lõi táo và không nằm trên mặt đường nhựa.**
- **Phân loại nhãn:**
  - Vật phẩm thực phẩm bên trong: Gán nhãn Lớp 1 (`biological`).
  - Túi nilon bao bì bên ngoài: Gán nhãn Lớp 7 (`plastic`).

### Nhãn YOLO Trước sửa (BEFORE - Quarantined):
```text
7 0.434845 0.489737 0.615605 0.532782
1 0.487541 0.443474 0.431781 0.313419
```

### Nhãn YOLO Đề xuất (AFTER - Proposed):
```text
7 0.428500 0.485000 0.585000 0.510000
1 0.478000 0.432000 0.385000 0.285000
```

### Chi tiết thay đổi:
- **Lớp 7 (`plastic` - bao bì nilon):** Giảm chiều rộng từ $0.6156 \to 0.5850$ và chiều cao từ $0.5328 \to 0.5100$. Siết chặt viền theo nếp gấp túi nhựa, loại bỏ khoảng đệm tường trắng và ngón tay thừa.
- **Lớp 1 (`biological` - phần bánh/thực phẩm):** Giảm chiều rộng từ $0.4318 \to 0.3850$ và chiều cao từ $0.3134 \to 0.2850$. Bám sát phần thực phẩm hữu cơ, giảm 18.9% diện tích rỗng bị bóng túi nilon chiếm dụng.
- **Đánh giá:** Tách biệt rõ ràng 2 vật thể, giúp mô hình phân biệt rạch ròi giữa bao bì nhựa bên ngoài và thực phẩm hữu cơ bên trong.

---

## 2. ẢNH TACO_0092.JPG (Lõi táo cắn dở đang cầm trên tay)

### Hình ảnh trực quan đối chiếu:
File đối chiếu: [`reports/data_v2/biological_overlays/taco_0092_before_after_comparison.jpg`](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/biological_overlays/taco_0092_before_after_comparison.jpg)

### Nhận diện hình ảnh thực tế (Ground Truth Reality):
- **Vật thể thực tế:** Lõi táo đã bị cắn dở (apple core) đang được các ngón tay người cầm thẳng đứng. Phía dưới nền là mặt đường nhựa có lá rụng và mũi giày. **Vật thể không nằm tự do trên mặt đường mà đang được giữ trên tay người.**
- **Phân loại nhãn:** Lớp 1 (`biological` - lõi táo).
- **Vấn đề quy chuẩn nhãn rác hữu cơ (Domain Standardization Note):**
  - Trong thị giác máy tính nhận diện rác thải môi trường (waste litter detection), có sự khác biệt giữa *rác đã vứt bỏ tự do ngoài môi trường* (discarded litter on ground/surface) và *vật thể đang được người cầm/sử dụng trên tay* (hand-held item / object in-use).
  - Cần quyết định chính sách từ người dùng: Liệu hệ thống có chấp nhận ảnh rác đang cầm trên tay vào tập Train hay chỉ huấn luyện trên rác rơi vãi tự do. Hiện tại ảnh được giữ ở trạng thái `PROPOSED_PENDING_USER_APPROVAL`.

### Nhãn YOLO Trước sửa (BEFORE - Quarantined):
```text
1 0.477124 0.654259 0.200980 0.237439
```

### Nhãn YOLO Đề xuất (AFTER - Proposed):
```text
1 0.475000 0.648000 0.178000 0.215000
```

### Chi tiết thay đổi:
- **Lớp 1 (`biological` - lõi táo):** Giảm chiều rộng từ $0.2010 \to 0.1780$ (giảm 11.5%) và chiều cao từ $0.2374 \to 0.2150$ (giảm 9.5%).
- **Lý do sửa:** Nhãn ban đầu chứa phần đệm ngón tay và mặt đường nhựa xung quanh quá rộng. Nhãn đề xuất ôm sát viền hữu cơ của lõi táo, giảm 19.8% diện tích thừa.

---

## 3. BẢO TOÀN NHÃN GỐC & QUY TRÌNH PHÊ DUYỆT
- File nhãn gốc tại `data/detection/labels/real/taco_0072.txt` và `taco_0092.txt` **HOÀN TOÀN KHÔNG BỊ GHI ĐÈ**.
- Nhãn mới chỉ được lưu tạm tại `data/detection_pilot_v3/labels/` để phục vụ đánh giá và nghiệm thu.
- Chỉ khi người dùng ra chỉ thị đồng ý duyệt (`APPROVE`), Antigravity mới cập nhật vào nhãn gốc và tích hợp vào tập Train.
