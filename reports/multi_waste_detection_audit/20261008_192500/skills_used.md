# Báo Cáo Phân Loại & Đánh Giá Sử Dụng 16 AI Coding Skills (Phiên Bản V3)
**Dự án:** Kiểm toán độc lập mô hình nhận diện đa loại rác (`CNTT-KLCN155-waste-detection`)  
**Tài liệu tham chiếu:** `huong-dan-chi-tiet-16-skills(7).md`  
**Run ID:** `20261008_192500_v3` | **Thời điểm cập nhật:** 2026-10-08T22:45:00+07:00  

---

## 1. Bảng Tổng Hợp 16 AI Coding Skills Gốc

Tuân thủ nghiêm ngặt yêu cầu của PM Review (V2-10 & F10): Toàn bộ 16 kỹ năng gốc từ cẩm nang được liệt kê đầy đủ kèm đường dẫn cấu hình thực tế (`SKILL.md`), trạng thái thực thi và bằng chứng kết xuất cụ thể. Trợ lý AI chỉ thực thi các kỹ năng phục vụ trực tiếp nhiệm vụ điều tra nguyên nhân gốc rễ, kiên quyết bỏ qua các kỹ năng gây lãng phí tài nguyên hoặc vượt phạm vi kiểm toán.

| STT | Kỹ năng (Skill) | Đường dẫn SKILL.md cục bộ | Trạng thái trong Audit | Mục đích & Phương thức áp dụng | Bằng chứng / Đầu ra cụ thể |
|:---:|:---|:---|:---:|:---|:---|
| 1 | `/systematic-debugging` | `~/.gemini\config\skills\systematic-debugging\SKILL.md` | **SỬ DỤNG CHÍNH** | Phương pháp luận cốt lõi: Tái hiện lỗi $\rightarrow$ Thu thập dữ liệu khách quan $\rightarrow$ Kiểm chứng giả thuyết cạnh tranh $\rightarrow$ Kết luận gốc rễ có dẫn chứng. | `traces/case_001_trace.json` $\rightarrow$ `case_004_trace.json`, `cases.csv`, `REPORT.md` (Mục 2–7). |
| 2 | `/understand` | `~/.gemini\config\skills\understand\SKILL.md` | **ĐÃ SỬ DỤNG** | Rà soát toàn bộ luồng luân chuyển dữ liệu từ upload ảnh, decode, preprocessing, model inference, postprocessing (NMS/WBF) đến UI rendering. | Phân tích chi tiết tại `src/web/detection_logic.py`, `src/detection/ssdlite.py`, `src/detection/yolo.py`, `src/detection/fusion.py`. |
| 3 | `/writing-plans` | `~/.gemini\config\skills\writing-plans\SKILL.md` | **ĐÃ SỬ DỤNG** | Lập kế hoạch kiểm toán 5 giai đoạn độc lập trước khi thực thi, xác định rõ ranh giới kiểm toán và sản phẩm bàn giao (DoD). | `docs/superpowers/plans/2026-10-08-multi-waste-detection-audit-plan.md`. |
| 4 | `/ponytail` | `~/.gemini\config\skills\ponytail\SKILL.md` | **ĐÃ SỬ DỤNG** | Tái sử dụng tối đa mã nguồn, script và mô hình đã có; kiên quyết từ chối cài thêm framework nặng, dashboard phụ hoặc viết lại kiến trúc. | Script kiểm toán đặt gọn gàng trong `reports/.../scripts/`, không can thiệp thay đổi mã nguồn sản phẩm `src/`. |
| 5 | `/interview-me` | `~/.gemini\config\skills\interview-me\SKILL.md` | **ĐÃ SỬ DỤNG** | Xác nhận các nghi vấn thiết yếu về đường dẫn ảnh gốc và các checkpoint lịch sử mà không suy đoán tùy tiện. | Truy vết thành công ảnh gốc tại `data/holdout/rac-1562828175-1562828212-3852-1562828313.webp`. |
| 6 | `/brainstorming` | `~/.gemini\config\skills\brainstorming\SKILL.md` | **ĐÃ SỬ DỤNG** | Xây dựng cây giả thuyết cạnh tranh để phân biệt: Lỗi do dữ liệu (data taxonomy), lỗi mô hình (model capacity), lỗi tích hợp (pipeline/WBF), hay lỗi giao diện (UI threshold). | Phân tích 4 giả thuyết tại Mục 1 & 2 của `REPORT.md`. |
| 7 | `/speckit-specify` | `~/.gemini\config\skills\speckit-specify\SKILL.md` | **ĐÃ SỬ DỤNG** | Định vị và đối chiếu các tiêu chí nghiệm thu (acceptance criteria) trong tài liệu đề cương và cấu hình gốc của dự án. | Đối chiếu với `configs/detection_dataset.yaml` và `configs/detection_source_mapping.yaml`. |
| 8 | `/speckit-tasks` | `~/.gemini\config\skills\speckit-tasks\SKILL.md` | **ĐÃ SỬ DỤNG** | Phân rã 10 hạng mục phản biện V2-01 đến V2-10 của PM Review thành checklist công việc có thứ tự phụ thuộc chặt chẽ. | Bảng đối chiếu V2-01 đến V2-10 tại Mục 2 của `REPORT.md`. |
| 9 | `/graphify` | `~/.gemini\config\skills\graphify\SKILL.md` | *Bỏ qua* | Cấu trúc module của dự án đơn giản, luồng xử lý thẳng (linear pipeline). Đọc trực tiếp mã nguồn bằng `view_file` đã đủ rõ ràng. | Không cần sinh knowledge graph đồ sộ để tránh lãng phí context. |
| 10 | `/archify` | `~/.gemini\config\skills\archify\SKILL.md` | *Bỏ qua* | Không cần vẽ diagram kiến trúc động phức tạp trong phạm vi điều tra lỗi; sơ đồ Mermaid trong `REPORT.md` là đủ trực quan. | Sơ đồ luồng xử lý Mermaid tại Mục 1 của `REPORT.md`. |
| 11 | `/test-driven-development` | `~/.gemini\config\skills\test-driven-development\SKILL.md` | **ĐÃ SỬ DỤNG (Characterization)** | Viết bộ regression & characterization test suite để chứng minh lỗi hoán đổi nhãn, phạt điểm WBF và biến đổi tọa độ bbox. Không tuyên bố là fix sản phẩm. | `scripts/test_pipeline_regressions.py`, kết quả `metrics/junit_report.xml` (4 passed) và `raw_logs/pytest_stdout.log`. |
| 12 | `/performance-optimization` | `~/.gemini\config\skills\performance-optimization\SKILL.md` | **ĐÃ SỬ DỤNG (Benchmark)** | Đo đạc phân phối thời gian thực thi (min, max, mean, std, p50, p95, FPS), raw samples và bộ nhớ tiêu thụ thực tế (`cpu_rss_end_mb`) trên ảnh thực địa. | `metrics/benchmark_latency.json`, `scripts/benchmark_latency.py`. |
| 13 | `/impeccable` | `~/.gemini\config\skills\impeccable\SKILL.md` | **ĐÃ SỬ DỤNG (Audit UI)** | Đánh giá tính chính xác của overlay box, màu sắc nhãn, nhãn tiếng Việt, bảng thống kê số lượng và trạng thái session state trên Streamlit UI. Không redesign UI. | Ảnh chụp màn hình kiểm toán UI thực tế: ma trận 6 ảnh `images/browser_*.png`. |
| 14 | `/ui-ux-pro-max` | `~/.gemini\config\skills\ui-ux-pro-max\SKILL.md` | *Bỏ qua* | Giao diện Streamlit là cố định theo đề tài; kiểm toán chỉ đánh giá độ tin cậy hiển thị, không thay đổi thiết kế màu sắc/typography. | Tuân thủ phạm vi kiểm toán. |
| 15 | `/design-system` | `~/.gemini\config\skills\design-system\SKILL.md` | *Bỏ qua* | Dự án không xây dựng design system mở rộng hoặc CSS framework mới. | Tuân thủ phạm vi kiểm toán. |
| 16 | `/caveman` | `~/.gemini\config\skills\caveman\SKILL.md` | **ĐÃ SỬ DỤNG** | Cập nhật tiến độ ngắn gọn, súc tích trong giao tiếp; toàn bộ chứng cứ khoa học, bảng số liệu và file:line được lưu trữ đầy đủ trong báo cáo. | Lịch sử trao đổi và tóm tắt tiến độ. |

---

## 2. Kỹ Năng Bổ Sung Phụ Trợ (Ngoài 16 Kỹ Năng Gốc)

* **`browser-testing-with-devtools` (Chrome DevTools MCP):**
  - **Mục đích:** Thao tác trực tiếp trên trình duyệt thật đang mở ứng dụng Streamlit (`http://localhost:8501/`).
  - **Hành động thực tế:**
    1. Kết nối vào phiên làm việc trình duyệt Chrome.
    2. Upload ảnh thực tế người dùng báo lỗi (`case_001_raw.webp`).
    3. Thao tác chọn radio button: YOLOv8n, SSDLite320, WBF Ensemble.
    4. Thao tác điều chỉnh slider ngưỡng confidence giữa `0.45` và `0.20`.
    5. Chụp ảnh màn hình toàn bộ ma trận 3 mô hình $\times$ 2 mức ngưỡng để làm chứng cứ kiểm toán độc lập.
  - **Bằng chứng đầu ra (6 ảnh chụp màn hình trong `images/`):**
    1. `images/browser_yolo_conf045.png` (YOLO @ 0.45: 2 boxes, 1.489 ms).
    2. `images/browser_yolo_conf020.png` (YOLO @ 0.20: 4 boxes, 1.463 ms).
    3. `images/browser_ssdlite_conf045.png` (SSDLite @ 0.45: 0 box, 1.738 ms).
    4. `images/browser_ssdlite_conf020.png` (SSDLite @ 0.20: 7 boxes, 2.018 ms).
    5. `images/browser_wbf_conf045.png` (WBF @ 0.45: 0 box, 2.587 ms).
    6. `images/browser_wbf_conf020.png` (WBF @ 0.20: 2 boxes, 2.071 ms).
