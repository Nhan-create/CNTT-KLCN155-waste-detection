# Huấn luyện nhận diện nhiều vật thể rác

## Vì sao phải tạo dataset mới

Dataset classification hiện tại chỉ nói "ảnh này thuộc lớp plastic". Object detector
cần thêm vị trí của **từng** vật thể: `class_id x_center y_center width height`.
Không thể suy ra bounding box chính xác từ tên thư mục. Có thể dùng model/segmentation
để pre-label, nhưng mọi box phải được người gán nhãn kiểm tra lại.

## Chuẩn nhãn bắt buộc

Thứ tự class không được đổi:

```text
0 battery     1 biological  2 cardboard  3 clothes  4 glass
5 metal       6 paper       7 plastic    8 shoes    9 trash
```

Quy tắc annotation:

- Khoanh sát phần nhìn thấy của từng vật thể; không khoanh cả đống rác thành một box.
- Mọi vật thể thuộc 10 lớp trong ảnh đều phải có box. Bỏ sót biến vật thể đó thành
  false negative trong lúc train.
- Vật thể bị che vẫn được gán nhãn nếu người gán nhãn còn xác định chắc chắn; thống
  nhất khoanh phần nhìn thấy hoặc toàn vật thể, không trộn hai chính sách.
- `trash` chỉ dùng cho vật thể rác khác xác định được, không dùng làm nhãn unknown cho
  mọi background.
- Milk carton, cốc giấy phủ nhựa và vật liệu composite phải có quy tắc riêng được
  review trước khi annotate hàng loạt.
- Ảnh không có rác được giữ làm hard negative với file label rỗng.

## Layout YOLO

```text
data/detection/v1/
  images/{train,val,test}/...
  labels/{train,val,test}/...
```

Mỗi `images/train/a.jpg` có `labels/train/a.txt`. Một dòng label:

```text
7 0.512 0.483 0.231 0.620
```

Các tọa độ được chuẩn hóa về `[0,1]`. Cấu hình class/path nằm tại
`configs/detection_dataset.yaml`.

## Cách chia dữ liệu để metric không bị ảo

- Chia theo **vật thể vật lý / video session**, không chia ngẫu nhiên từng frame.
- Không để các frame liền nhau của cùng video xuất hiện ở cả train và test.
- Test nên chứa thiết bị, địa điểm và thời gian chụp khác train.
- Giữ một test thực tế hoàn toàn khóa; chỉ dùng val để chọn model/threshold.
- Mỗi lớp nên có nhiều thiết bị, background, ánh sáng và kích thước object. Mục tiêu
  thực dụng ban đầu là 1.500–3.000 object instance/lớp, sau đó bổ sung theo error
  analysis thay vì chỉ tăng ảnh sạch.

## Validation trước khi dùng GPU

```powershell
python -m src.detection.dataset --data configs/detection_dataset.yaml
```

Validator kiểm tra class order, đủ train/val/test, cặp image-label, class index và box
normalized hợp lệ. Sau đó train:

```powershell
python -m src.detection.train `
  --data configs/detection_dataset.yaml `
  --config configs/detection_training.yaml `
  --device 0
```

Pipeline mặc định fine-tune `yolo26s.pt` trong 200 epoch, chọn best bằng validation,
rồi đánh giá `best.pt` đúng một lần trên test. Artifact chính:

```text
artifacts/detection/waste-yolo26s/weights/best.pt
artifacts/detection/waste-yolo26s/test_summary.json
```

## Chọn kích thước model

Không có một model vừa chính xác nhất vừa nhanh nhất trên mọi điện thoại. Chạy cùng
một split với:

```powershell
# Trần accuracy, dùng để biết dữ liệu còn giới hạn model hay không
python -m src.detection.train --model yolo26m.pt --device 0

# Candidate triển khai mobile
python -m src.detection.train --model yolo26n.pt --device 0
python -m src.detection.train --model yolo26s.pt --device 0
```

Chọn theo Pareto frontier của `mAP50-95`, recall từng lớp, model size và latency p95
trên **thiết bị thật**. Với vật thể nhỏ, thử `imgsz=960` khi train/evaluate, nhưng
benchmark lại vì chi phí tăng mạnh.

## Augmentation và hard cases

Cấu hình đã bật mosaic, mixup/cutmix, affine, perspective và HSV. Để bám camera thực,
nên thêm Albumentations có xác suất vừa phải cho motion blur, defocus, JPEG artifact,
ISO noise, shadow, glare và low-light. Không áp dụng enhancement chỉ ở inference.

Sau mỗi lần train:

1. Xem confusion matrix và PR curve theo lớp.
2. Lọc false positive confidence cao và false negative.
3. Bổ sung đúng hard cases đó vào train.
4. Không chuyển mẫu từ test sang train; tạo test version mới khi cần.

Metric cần báo cáo: mAP50-95, mAP50, precision, recall từng lớp, confusion matrix,
false positives/image, false negatives/image và latency p50/p95. Threshold production
được chọn trên validation theo precision/recall nghiệp vụ, không dùng mặc định 0.35
như một con số cuối cùng.

## Export Mobile

LiteRT dùng cùng định dạng `.tflite`. Export nên chạy trên Linux x86_64 hoặc macOS:

```powershell
python -m src.detection.export `
  --model artifacts/detection/waste-yolo26s/weights/best.pt `
  --format litert `
  --imgsz 640
```

Chỉ bật INT8 sau khi có calibration data đại diện camera thật:

```powershell
python -m src.detection.export `
  --model artifacts/detection/waste-yolo26s/weights/best.pt `
  --format litert `
  --imgsz 640 `
  --int8 `
  --data configs/detection_dataset.yaml
```

Sau export phải chạy validation lại model `.tflite` và đo chênh lệch mAP với `.pt`.
Ứng dụng Mobile phải lấy class mapping từ metadata model, scale box về đúng kích
thước frame sau letterbox và benchmark camera end-to-end.

Tài liệu Ultralytics chính thức:

- https://docs.ultralytics.com/tasks/detect/
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/integrations/litert/

Ultralytics dùng AGPL-3.0 hoặc giấy phép Enterprise; cần review giấy phép nếu ứng dụng
được phân phối thương mại.

