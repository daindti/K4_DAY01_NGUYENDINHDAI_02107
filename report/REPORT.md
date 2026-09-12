# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`: 468, `class_name`: "cab, `rank`:1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K"):
- Record này mô tả toàn ảnh như thế nào? record này mô tả toàn ảnh là traffic
- Ai định nghĩa class list mà checkpoint có thể dự đoán?người tạo dataset
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? giữ ID để model có thể nhận diện class, tên lớp để con người có thể hiểu được, tên taxonomy cần giữ lại để xác định bộ nhãn được sử dụng
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? phải phân rõ ranh giới giữa các object
- Vì sao model score không phải ground truth? vì ground truth là guideline do chính con người tạo ra để quy ước. các guideline này chỉ mang tính tương đối chứ không phải giá trị tuyệt đối chính xác.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`: "bowl", `score`: 0.719063, `bbox_xyxy`:[32.65, 342.12, 100.16, 384.93], `bbox_width`:67.51, `bbox_height`:42.81):
- Diễn giải vị trí box bằng lời: có 1 box ở rìa bên trái bị cắt mép(truncated)
- So sánh số prediction ở hai threshold: số prediction ở threshold nhỏ hơn sẽ lớn hơn số prediction ở threshold lớn hơn (threshold tỉ lệ nghịch với số prediction)
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? khi threshold lớn dần lên thì số lượng vật thể không được đánh box cũng tăng dần lên dẫn tới độ bao phủ và khối lượng mà reviewer cần xem sẽ giảm dần
- Đề xuất một quy tắc box chặt: cần phải bám sát vào vật thể, không thừa nền, mọi vật thể trong classlist đều có box, đối với những trường hợp truncated thì vẫn cần vẽ box cho vật thể dừng lại ở rìa frame ảnh, đối với trường hợp occluded vẫn phải vẽ box, đánh dấu lại.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? guideline cần quy định có đánh dấu hay không, đánh dấu như thế nào, khoảng bao nhiêu % bị che khuất thì không đánh dấu. nếu vượt qua ngưỡng này thì cần có guideline riêng để giải quyết.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box? polygon sẽ đánh chính xác biên của vật thể
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id là định danh duy nhất cho mỗi object, dùng để phân biệt các object với nhau, nó khác với class_id dùng để định danh lớp, cũng khác sample_id dùng để nhận biết case
- Đề xuất một quy tắc biên mask: cần bám sát vào vật thể, không thừa nền, mọi vật thể trong classlist đều được mark riêng
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? guideline cần quy định có đánh dấu hay không, đánh dấu như thế nào, khoảng bao nhiêu % bị che khuất thì không đánh dấu. nếu vượt qua ngưỡng này thì cần có guideline riêng để giải quyết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class | Sai class, thiếu label | Chọn đúng class, flag nếu mơ hồ | Class đúng guideline không  |
| Phát hiện vật thể | bounding box | Sai class, thiếu/thừa box, box lệch, quá rộng/hẹp, xử lý occlusion/truncation sai | Vẽ box sát object + gán class | Class, số lượng object, vị trí/kích thước box, guideline  |
| Instance segmentation | polygon | Sai class, thiếu/thừa mask, polygon lệch biên, ăn background, thiếu vùng object, lấn object khác | Vẽ polygon/mask sát biên + gán class/instance | Class, boundary, mask coverage, instance, occlusion/truncation  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không sao chép, tải xuống, chia sẻ hoặc sử dụng dữ liệu ngoài mục đích và phạm vi được cấp quyền
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:  Team Lead / Data Manager hoặc người phụ trách dự án theo quy trình escalation.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
