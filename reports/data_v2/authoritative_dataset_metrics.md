# BẢNG ĐỐI SOÁT SỐ LIỆU CHUẨN XÁC TỪ DỮ LIỆU ĐĨA THẬT
**Thời điểm đo (UTC):** 2026-10-11T18:38:26Z  
**Trạng thái kiểm toán:** HOÀN TOÀN NHẤT QUÁN 100% GIỮA FILE ĐĨA, CSV MANIFEST VÀ BÁO CÁO  

---

## 1. GIẢI TRÌNH NGUYÊN NHÂN SAI LỆCH SỐ LIỆU (ROOT CAUSE)
Trong báo cáo Phase 4B.1, bảng thống kê đã ghi nhận số liệu từ một bản nháp ước lượng thủ công cũ:
- Ghi v2 = 5.590 (thừa 6 box so với nhãn đĩa thật 5.584).
- Ghi v3 = 4.337 (thừa 6 box so với nhãn đĩa thật 4.331).
- Ghi plastic v3 = 1.902 (thừa 12 box so với nhãn đĩa thật 1.890).
- Ghi biological v3 = 5 (thiếu 1 box so với nhãn đĩa thật 6).

**Bằng chứng xác thực từ đĩa:**
Các file `real_v2_split_manifest.csv`, `candidate_annotation_manifest.csv`, các file nhãn `.txt` và `data.yaml` trên đĩa **luôn luôn có số liệu chuẩn xác là 5.584 (v2) và 4.331 (v3)**. Tuyệt đối không có sự mâu thuẫn trong dữ liệu gốc. Phép toán trừ giữa v2 và v3 khớp chính xác 100% trên từng class:

**Công thức đối soát:** 5.584 (v2) - 1.253 (loại bỏ) = 4.331 (v3 candidate)

---

## 2. BẢNG TỔNG HỢP THEO PHÂN TÁCH (SPLITS)

| Dataset | Tổng ảnh | Train (ảnh / box) | Val (ảnh / box) | Test (ảnh / box) | Tổng Box | Manifest SHA-256 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Real v2** | **1.996** | 1.398 / 3.963 | 299 / 901 | 299 / 720 | **5.584** | `14f54b7f71250f62...` |
| **v3 Candidate** | **1.820** | 1.277 / 3.052 | 269 / 684 | 274 / 595 | **4.331** | `3dfb3cbb933d09c7...` |
| **Chênh lệch (Loại bỏ)** | **-176** | -121 / -911 | -30 / -217 | -25 / -125 | **-1.253** | *176 ảnh dính cờ Quality Gate* |

---

## 3. BẢNG PHÂN BỐ CHI TIẾT 10 LỚP (AUTHORITATIVE CLASS DISTRIBUTION)

| Lớp (Class) | v2 Tổng | v2 Train | v2 Val | v2 Test | Loại bỏ (176 ảnh) | v3 Tổng | v3 Train | v3 Val | v3 Test | Đánh giá bảo toàn |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `0: battery` | **2** | 1 | 0 | 1 | **0** | **2** | 1 | 0 | 1 | **Bảo toàn 100% (0 box mất)** |
| `1: biological` | **6** | 3 | 2 | 1 | **0** | **6** | 3 | 2 | 1 | **Bảo toàn 100% (0 box mất)** |
| `2: cardboard` | **244** | 169 | 39 | 36 | **36** | **208** | 143 | 30 | 35 | Loại bỏ box sub-resolution |
| `3: clothes` | **527** | 368 | 68 | 91 | **90** | **437** | 299 | 56 | 82 | Làm sạch ảnh thời trang |
| `4: glass` | **327** | 213 | 82 | 32 | **122** | **205** | 155 | 26 | 24 | Làm sạch box trùng lặp kính vỡ |
| `5: metal` | **567** | 420 | 74 | 73 | **97** | **470** | 331 | 71 | 68 | Loại bỏ box sub-resolution |
| `6: paper` | **237** | 160 | 43 | 34 | **42** | **195** | 137 | 28 | 30 | Loại bỏ box nhiễu |
| `7: plastic` | **2.216** | 1.560 | 382 | 274 | **326** | **1.890** | 1.313 | 335 | 242 | Giảm bias class quá áp đảo |
| `8: shoes` | **295** | 215 | 41 | 39 | **8** | **287** | 207 | 41 | 39 | Bảo toàn 97.3% |
| `9: trash` | **1.163** | 854 | 170 | 139 | **532** | **631** | 463 | 95 | 73 | **Cắt giảm mạnh box catch-all nhiễu** |
| **TỔNG CỘNG** | **5.584** | **3.963** | **901** | **720** | **1.253** | **4.331** | **3.052** | **684** | **595** | **Khớp 100% toán học** |

