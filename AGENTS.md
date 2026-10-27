# QUY CHUẨN HOẠT ĐỘNG VÀ PHÂN CÔNG SUB-AGENTS (CNTT-KLCN155)
> Hệ thống Multi-Agent: Antigravity Integrator + 4 Chuyên trách

## 1. Vai trò của Integrator (Main Agent)
- Giải quyết dependencies, phân rã và điều phối sub-agents.
- Tích hợp thay đổi tuần tự, giải quyết xung đột mã nguồn.
- Kiểm soát tài nguyên phần cứng: Chỉ duy nhất 1 tác vụ huấn luyện/đánh giá GPU nặng tại một thời điểm.
- Thu thập bằng chứng thực thi thật, ghi nhận Agent IDs, nhật ký run_id và đối chiếu DoD nghiêm ngặt.

---

## 2. Danh sách Sub-Agents chuyên trách

| Agent | Định danh vai trò | Phạm vi & Quyền sửa mã nguồn | Đầu ra bắt buộc |
|---|---|---|---|
| **A** | **Data / Mapping** | - `src/detection/schema.py`<br>- `configs/detection_source_mapping.yaml`<br>- `reports/reproduce_scripts/audit_mendeley_mapping.py`<br>- `tests/detection/test_real_mapping_integration.py`<br>- `data/detection/labels/` (đánh dấu/quarantine) | Source samples/metadata thật, expected độc lập, output converter, patch và regression tests. |
| **B** | **Model / Inference** | - `src/detection/ssdlite.py`<br>- `src/detection/ssdlite_train.py`<br>- `src/detection/yolo.py`<br>- `src/detection/factory.py`<br>- `configs/` liên quan training/inference | Config, split fingerprint, checkpoint hash, training logs, best/completed epochs và predictions thật. |
| **C** | **Coordinates / Rendering** | - `src/detection/visualization.py`<br>- `src/detection/fusion.py`<br>- `src/app/detection_logic.py`<br>- `src/app/detection_settings.py`<br>- `streamlit_app.py`<br>- Chuyển đổi tọa độ, EXIF, letterbox, clipping | Patch, kiểm thử tọa độ xuôi/ngược, overlay qua luồng ứng dụng thực tế. |
| **D** | **QA Độc lập** | - `reports/qa_acceptance/`<br>- Manifest và Ground Truth độc lập `data/holdout`<br>- Đánh giá baseline và nghiệm thu DoD<br>*(Tuyệt đối không sửa code mô hình/inference, không sửa GT để khớp prediction)* | Manifest 27 ảnh `data/holdout`, SHA-256, Ground Truth từng ảnh/vật thể, bảng đánh giá một-một, lỗi và bằng chứng. |

---

## 3. Ràng buộc bất biến
1. **SSDLite320-MobileNetV3** là mô hình chính. **YOLOv8n** là mô hình đối chứng. Đánh giá riêng biệt.
2. Thứ tự 10 lớp cố định: `[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`.
3. SSDLite: Background = 0, Foreground = 1..10. Target load cộng 1, decode trừ 1. YOLO dùng 0..9.
4. Mọi kết quả phải từ thực thi thật; không hardcode nhãn/tọa độ theo filename hay checksum.
5. Bảo toàn dirty changes hiện hữu của người dùng, không `git reset --hard` hay xóa dữ liệu thô.

---

## AI Skills Routing (Điều phối Kho Skills Toàn Máy)

Khi thực hiện nhiệm vụ trong dự án này, Agent tuân thủ quy trình chọn và đọc Skills tại bộ tài liệu trung tâm trên máy:
- **Hướng dẫn chính & Quy trình 7 bước:** `~/.agents\docs\AI_SKILLS_GUIDE.md`
- **Ma trận chọn Skill theo 11 nhóm nhiệm vụ & 4 cấp quy mô:** `~/.agents\docs\TASK_SKILL_ROUTING_MATRIX.md`
- **Hướng dẫn theo từng công cụ (Antigravity, Codex, Claude Code, Orca):** `~/.agents\docs\TOOL_SKILL_GUIDES.md`
- **Danh mục tổng (1.636 nội dung duy nhất):** `~/.agents\docs\SKILLS_CATALOG.md`
- **Nguyên tắc bắt buộc:** Mỗi bước chọn đúng **01 Skill chính** (chỉ thêm Skill bổ trợ khi có lý do rõ ràng), nêu ngắn gọn chọn skill nào và dùng công cụ đọc file (`view_file` / `Read`) đọc đầy đủ `SKILL.md` trước khi làm. Luôn giữ nguyên các ràng buộc riêng của dự án ở phía trên làm ưu tiên cao nhất.
