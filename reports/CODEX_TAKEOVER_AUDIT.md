# Codex takeover audit — CNTT-KLCN155

Ngày: 2026-10-11 (Asia/Saigon) · Run ID: `codex-20261011-eval-p0-01` · Nhánh: `main`

Phạm vi thực hiện: tiếp nhận Phase 1, xác minh và sửa evaluator P0 (Phase 2). Các phần thu thập dữ liệu, sửa GT, đổi mô hình và huấn luyện chưa được thực hiện. Kiểm toán viên: Integrator `/root` (evaluator, test, tổng hợp); Agent A `/root/data_mapping`; Agent B `/root/model_inventory`; Agent D `/root/qa_gt`. Các agent A/B/D chỉ đọc. Không có tác vụ GPU nặng.

## 1. Tình trạng nhận bàn giao và những gì đã kiểm chứng

`git status --short` lúc bắt đầu có 19 tệp tracked đã sửa và nhiều tệp untracked, gồm chính script evaluator, tài liệu và dữ liệu báo cáo. Codex không reset, xóa hoặc ghi đè chúng. `AGENTS.md` trong repo cũng là untracked và khớp quy tắc người dùng gửi. Tài liệu handoff trong Downloads là chỉ dẫn điều tra, không phải chứng cứ máy này đã chạy thí nghiệm.

Các artifact mà đợt Anti đã để lại và Codex xác minh được: 3 tệp raw predictions của SSDLite before/after quick fix và YOLOv8n E1; GT v1, GT v2 proposed, manifest 27 ảnh, 27 overlay review; split real_v2; checkpoint E2 SSDLite và E1 YOLO. Cả hai checkpoint thực sự nạp được trên CPU và metadata giữ đúng thứ tự 10 lớp. Không thể suy từ artifact rằng mọi lệnh hoặc kết luận trong các Markdown cũ đã được tái chạy; các số dưới đây được tính lại trực tiếp.

## 2. Pipeline và môi trường thực tế

- Ứng dụng mặc định dùng SSDLite E2 `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt`, confidence `0.45`. Luồng mặc định bật tiling và two-stage: ảnh/camera → preprocessing → SSDLite → `TiledDetector` → `TwoStageDetector` với classifier `artifacts/official_run/best_model.pt` → per-class threshold → Streamlit overlay. Benchmark trong bảng dưới là raw predictions của từng pipeline đã lưu, không phải phép suy luận mới từ giao diện đang chạy.
- YOLOv8n E1 `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` là mô hình đối chứng riêng. SSDLite dùng background 0 và foreground 1..10 rồi decode về 0..9; YOLO dùng 0..9. Không đổi thứ tự `[battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash]`.
- SHA-256 checkpoint: SSDLite `c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7`; YOLO `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8`; classifier `c824fec4f3d1dffff946cd8ad7e2d4f121be33e684dda4e6af92b5bda343e424`.
- `.venv`: Python 3.12.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, Ultralytics 8.4.171, CUDA khả dụng trên RTX 2050 4 GiB. Các phiên bản này khác `requirements.txt`; Python hệ thống thiếu torch. Metadata E2 ghi 36 epoch hoàn thành, best epoch 30; trường cấu hình epochs=50 trong metadata/summary khác epochs=60 trong checkpoint. Chưa xác định nguyên nhân.

## 3. P0 evaluator: tái hiện, sửa và giao thức

`scripts/compare_eval_v1_vs_v2.py` cũ chọn GT có IoU lớn nhất trong mọi class, sau đó mới xét class. Tại `images (12).jpg`, prediction `paper` confidence `0.4424` có IoU `0.7961` với GT `cardboard` và `0.6700` với GT `paper`: script cũ ghi FP dù đủ điều kiện TP. Test tổng hợp và test trên raw file đều **thất bại trước sửa** (`0/1/2` thay vì `1/0/1`; `12/9/45` thay vì `13/8/44`). Sau sửa, thuật toán chọn GT chưa ghép **cùng class** trước khi so IoU, theo thứ tự confidence giảm dần, ghép một-một. Khi `class_aware=False`, matching vẫn xét mọi class.

Giao thức là precision/recall/F1 tại **một** ngưỡng confidence `>=0.25` và IoU `>=0.50`. Đây không phải AP/mAP; AP cần tích lũy đường cong precision–recall qua nhiều score threshold, và COCO AP còn đánh giá theo từng category, nhiều IoU threshold/area/maxDets. [COCOeval chính thức](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) tách GT/detection theo `(image_id, category_id)` khi `useCats=1`. Script hiện không áp dụng đầy đủ quy tắc crowd/ignore, area ranges và maxDets của COCO; không gắn nhãn các số sau là COCO AP.

| Pipeline | GT | GT box | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SSDLite trước quick fix | v1 frozen | 57 | 21 | 13 | 8 | 44 | 61.90% | 22.81% | 33.33% |
| SSDLite trước quick fix | v2 proposed active | 45 | 21 | 10 | 11 | 35 | 47.62% | 22.22% | 30.30% |
| SSDLite trước quick fix | v2 proposed ACCEPTED | 39 | 21 | 9 | 12 | 30 | 42.86% | 23.08% | 30.00% |
| SSDLite sau quick fix | v1 frozen | 57 | 11 | 8 | 3 | 49 | 72.73% | 14.04% | 23.53% |
| SSDLite sau quick fix | v2 proposed active | 45 | 11 | 7 | 4 | 38 | 63.64% | 15.56% | 25.00% |
| SSDLite sau quick fix | v2 proposed ACCEPTED | 39 | 11 | 6 | 5 | 33 | 54.55% | 15.38% | 24.00% |
| YOLOv8n E1 | v1 frozen | 57 | 32 | 5 | 27 | 52 | 15.62% | 8.77% | 11.24% |
| YOLOv8n E1 | v2 proposed active | 45 | 32 | 5 | 27 | 40 | 15.62% | 11.11% | 12.99% |
| YOLOv8n E1 | v2 proposed ACCEPTED | 39 | 32 | 5 | 27 | 34 | 15.62% | 12.82% | 14.08% |

`reports/DATASET_AUDIT_AND_TRAINING_READINESS.md:159-161` hiện còn bảng **cũ** cho SSDLite trước quick fix; không sửa số Markdown thủ công. Nguồn máy đọc được: `reports/CODEX_EVAL_V1_V2_2026-10-11.json`, chứa toàn bộ 9 hàng, giao thức và SHA-256 của 5 tệp đầu vào. Log console: `reports/CODEX_TAKEOVER_EVAL_2026-10-11.log`.
Một phép tính độc lập bằng `reports/qa_acceptance/practical_metrics.py::precision_recall_f1` trên raw GT v1 cho cùng ba bộ `(TP, FP, FN)`: `(13, 8, 44)`, `(8, 3, 49)`, `(5, 27, 52)`.

## 4. Dataset, GT và hash

- real_v2: train `1,398` ảnh/`3,963` box, val `299`/`902`, test `299`/`720`. Train theo thứ tự 10 lớp: `[1, 3, 169, 368, 213, 420, 160, 1560, 215, 854]`. Không thiếu file ảnh/label trong splits, không có label rỗng hoặc zero-byte. Do đó hiện thiếu hard negatives; `battery` và `biological` hầu như không có dữ liệu train.
- Đầu ra kiểm đếm trực tiếp lưu ở `reports/CODEX_TAKEOVER_DATASET_2026-10-11.log`; kết quả gọi parser trên 2 nhãn lỗi lưu ở `reports/CODEX_TAKEOVER_LABEL_BOUNDS_2026-10-11.log`.
- Manifest holdout 27 ảnh còn đủ, kích thước và SHA từng ảnh khớp; thư mục `D:\Hinh_Anh_Rac` hiện có 147 ảnh (thêm 20 `kho_` và 100 `rac_online_`). Một cặp holdout trùng byte: `images (13).jpg` và `images (7).jpg`. GT v1 có 57 box hợp lệ về tọa độ và class ID. GT v2 proposed có 57 object giữ **nguyên class/bbox v1**, chỉ thêm trạng thái; 39 ACCEPTED, 8 pile, 4 quarantine, 3 nested/overlap, 3 ambiguous. `45 active` vẫn gồm 6 mục nested/ambiguous; chưa nghiệm thu. Có 10 ảnh zero-active theo v2, cần quy tắc đánh giá riêng.
- SHA-256 thực tế: manifest `6c57e733d78f0c1631fa2deb3fdbbde3ce853dad159b5af8657aba04b2494027`; GT v1 `f390f0f9440a977d0a0afe9e55b862562f1aafd3011ab18313e1772b18d470c0`; v2 proposed `6acea648293d545a23bc0a94b8410abc656134bdd7506ee5789b79f577857ac0`. Bảng cũ báo GT v1 `5c9f...`, v2 `f3c3...`, không khớp bytes repo. Nguồn gốc số cũ chưa xác định; không đụng GT.
- Hai nhãn TACO đang trong split vượt biên chuẩn hóa: train `taco_0112.txt:12` y_min `-0.000548`, test `taco_0457.txt:5` y_min `-0.000240`. `read_yolo_boxes` từ chối; split builder chấp nhận sai số ±0.01. Chuyển lại từ annotation TACO gốc cho 2 dòng này cho box khác file hiện hành; cần đối chiếu thủ công và sửa theo quy trình dữ liệu.
- Bốn ảnh Open Images train thiếu bản ghi trong `real_v2_source_manifest.csv`; năm ảnh khác có số box khác CSV nguồn hiện có. Mapping OI của clothing/footwear còn `REQUIRES_REVIEW` nhưng collector nhận mọi class có `class_id`; OI đóng góp toàn bộ 528 box clothes và 288/295 box shoes trong các split. Chưa chứng minh nhãn sai, nhưng provenance/mapping chưa đạt DoD. 16 annotation TACO mơ hồ đã tách vào `real_v2_ambiguous_boxes.json`.

## 5. Kiểm thử và những lỗi khác

- Trước sửa: 2 regression test evaluator mới đều fail. Sau sửa: `2 passed` (log `reports/CODEX_TAKEOVER_P0_TESTS_2026-10-11.log`). Nhóm `tests/detection tests/qa tests/web`: **173 passed, 3 warnings** (log `reports/CODEX_TAKEOVER_DETECTION_TESTS_2026-10-11.log`). Điều này xác nhận phần từng công bố 171 test có thể chạy cộng thêm 2 test mới, không chứng minh toàn bộ suite đều qua.
- Script evaluator chạy được từ `C:\Users\ad\Downloads` bằng đường dẫn tuyệt đối, xác nhận nó không còn phụ thuộc working directory của repo.
- Chạy `python -m pytest -q` trên **toàn repo** dừng ở collection với 10 lỗi do `.venv` thiếu `imagehash`, `seaborn`, `nbformat`, `kagglehub`, `PyQt5`. Log nguyên bản: `reports/CODEX_TAKEOVER_PYTEST_2026-10-11.log`. Chưa cài thêm dependencies hay tuyên bố full suite pass.
- Trong `streamlit_app.py:99`, `cached_classifier()` được gọi không truyền `settings.classifier_path`, nên tùy chọn classifier path bị bỏ qua ở nhánh two-stage này. Cần test luồng app trước khi sửa.
- Các chỉ số v2 trong tài liệu hiện tại chưa thể dùng làm kết quả nghiệm thu: GT chưa duyệt, 6 active vẫn cần review, và raw predictions có thể đã chịu ảnh hưởng của quick fix chọn trên holdout cũ. Không dùng holdout này để chọn hyperparameter tiếp.

## 6. File Codex sửa, diff và rollback

- Sửa `scripts/compare_eval_v1_vs_v2.py`: matching cùng class trước khi tìm IoU, nạp dữ liệu theo `__file__`, không còn side effect khi import, tùy chọn `--output` ghi kết quả/hashes.
- Thêm `tests/qa/test_compare_eval_v1_vs_v2.py`: test tổng hợp và test raw GT v1.
- Thêm báo cáo này, JSON kết quả và log kiểm thử. Diff evaluator lưu ở `reports/CODEX_EVALUATOR_PATCH_2026-10-11.diff`; bản script trước sửa lấy từ ZIP bàn giao và lưu `reports/reproduce_scripts/compare_eval_v1_vs_v2_before_codex.py`. Bản script evaluator ban đầu là **untracked**, nên Git không tự khôi phục được nó; rollback bằng cách chép lại bản lưu, sau khi so diff để giữ thay đổi người dùng.

Tái lập từ root repository:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\qa\test_compare_eval_v1_vs_v2.py -q
.\.venv\Scripts\python.exe scripts\compare_eval_v1_vs_v2.py --output reports\CODEX_EVAL_V1_V2_2026-10-11.json
.\.venv\Scripts\python.exe -m pytest tests\detection tests\qa tests\web -q
```

## 7. Ưu tiên tiếp theo và quyết định cần duyệt

1. **P0 kỹ thuật dữ liệu:** truy nguyên/sửa 2 nhãn TACO vượt biên, 4 ảnh thiếu provenance và 5 ảnh lệch box; thêm kiểm thử đi qua chính parser và nguồn thật. Không tự sửa nhãn theo dự đoán model.
2. **P0 QA:** người dùng/QA duyệt 18 object trong `reports/qa_acceptance/suspicious_gt_audit_table.json`, đặc biệt 6 active còn tranh chấp; quyết định xử lý cặp ảnh trùng và 10 ảnh zero-active. Chỉ sau đó mới freeze GT v2 và hash mới.
3. **P0 dữ liệu:** DATA-01 hard negatives, DATA-02 battery/biological, DATA-03 ảnh nhiều vật thể, DATA-05 split chống leakage theo nguồn và pHash. Các mục tiêu số lượng trong ticket hiện là kế hoạch, chưa có dữ liệu được thu thập/kiểm chứng quyền sử dụng. Xác minh license từng nguồn trước khi thêm.
4. **P1 đánh giá và sản phẩm:** sửa đường dẫn classifier tùy chỉnh sau test app, hoàn tất dependency/test suite, đánh giá riêng SSDLite và YOLO trên holdout độc lập với TP/FP/FN, AP50, AP50-95, latency/FPS và bộ nhớ đo thật. Không có số latency/FPS mới trong audit này.
5. **P1 mô hình:** sau khi dữ liệu và protocol được duyệt, so phương án fine-tune SSDLite chính, YOLOv8n đối chứng và cải thiện preprocessing/postprocessing. Chưa đủ bằng chứng để khẳng định phương án nào tăng mAP hay tốc độ. Cần người dùng duyệt GT v2, kế hoạch thu thập dữ liệu và ngân sách huấn luyện GPU dài; không chạy huấn luyện trong run này.
