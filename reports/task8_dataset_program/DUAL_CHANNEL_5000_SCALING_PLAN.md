# TASK 8I — KẾ HOẠCH MỞ RỘNG KÉP (DUAL-CHANNEL SCALING PLAN) ĐẾN 500 & 5.000 ẢNH/LỚP

## 1. Nguyên tắc điều phối Kênh A & Kênh B
- **Kênh A (Bộ dữ liệu công khai chuyên biệt):** Khai thác có kiểm soát giấy phép kép (`photo` + `observation/record`), loại trừ vĩnh viễn các nguồn `EXHAUSTED` và không thu thập đại trà `plastic` (`987` ảnh train, đã vượt xa `M1 = 500`).
- **Kênh B (First-Party Field Capture):** Chụp ảnh thực địa theo `FIRST_PARTY_FIELD_CAPTURE_GUIDE.md` và kiểm định tự động qua `scripts/field_capture_import_and_qa.py`, bắt đầu bằng lô Pilot `50–100` ảnh thật.

## 2. Lộ trình từng lớp đến mốc M1 (`500` ảnh/lớp) và M4 (`5.000` ảnh/lớp) sau Task 8I

| Class ID | Lớp | Verified RW Train (Post-8I) | Unique Specimens | Unique Sessions | Verified RW Val | Thiếu tới `500` (`M1`) | Chỉ tiêu Kênh A (`M1`) | Chỉ tiêu Kênh B Thực địa (`M1`) | Số buổi chụp ước tính (`M1`) | Thiếu tới `5.000` (`M4`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `battery` | **86** | 82 | 77 | 31 | 414 | 166 | **248** | 6 | 4914 |
| 1 | `biological` | **122** | 115 | 92 | 29 | 378 | 151 | **227** | 6 | 4878 |
| 2 | `cardboard` | **185** | 185 | 58 | 44 | 315 | 126 | **189** | 5 | 4815 |
| 3 | `clothes` | **55** | 55 | 28 | 4 | 445 | 178 | **267** | 6 | 4945 |
| 4 | `glass` | **111** | 111 | 49 | 32 | 389 | 156 | **233** | 6 | 4889 |
| 5 | `metal` | **305** | 305 | 93 | 57 | 195 | 78 | **117** | 3 | 4695 |
| 6 | `paper` | **201** | 200 | 71 | 47 | 299 | 120 | **179** | 4 | 4799 |
| 7 | `plastic` | **987** | 986 | 161 | 177 | 0 | 0 | **0** | 0 | 4013 |
| 8 | `shoes` | **78** | 78 | 57 | 15 | 422 | 169 | **253** | 6 | 4922 |
| 9 | `trash` | **361** | 360 | 96 | 72 | 139 | 56 | **83** | 2 | 4639 |
