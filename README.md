# PneumoVision — Phân loại viêm phổi từ ảnh X-quang ngực

PneumoVision là một project Computer Vision xây dựng bằng PyTorch với mục tiêu phân loại ảnh X-quang ngực thành hai nhóm `NORMAL` và `PNEUMONIA`. Với mỗi ảnh đầu vào, mô hình trả về xác suất `P(PNEUMONIA)` và một nhãn dự đoán dựa trên một ngưỡng đã được xác định trước.

Project sử dụng dataset **Chest X-Ray Images (Pneumonia)** và tập trung vào một vấn đề thực tế thường gặp khi xây dựng mô hình phân loại ảnh: kết quả tốt trên một tập validation không đồng nghĩa với việc mô hình sẽ hoạt động giống như vậy trên dữ liệu chưa từng được nhìn thấy. Vì vậy, thay vì chỉ huấn luyện một mô hình rồi báo cáo accuracy, project đi qua toàn bộ quá trình từ kiểm tra dữ liệu, tạo validation set, so sánh các cách huấn luyện, phân tích lỗi, lựa chọn threshold, kiểm tra vùng mô hình tập trung bằng Grad-CAM, khóa cấu hình cuối cùng và cuối cùng mới đánh giá một lần trên test set.

> **Lưu ý:** Đây là project Machine Learning / Computer Vision trên một public dataset, không phải hệ thống chẩn đoán y khoa. Kết quả trong project không được dùng để đưa ra quyết định lâm sàng và chưa có clinical validation hay đánh giá trên một cohort độc lập.

---

## 1. Bài toán

Bài toán được đặt dưới dạng **binary image classification**. Mỗi ảnh X-quang thuộc một trong hai lớp:

* `NORMAL`: không thuộc nhóm viêm phổi trong dataset.
* `PNEUMONIA`: thuộc nhóm viêm phổi trong dataset.

Luồng xử lý của mô hình khá đơn giản về mặt đầu ra:

```text
Chest X-ray
    ↓
Preprocessing
    ↓
ResNet18
    ↓
P(PNEUMONIA)
    ↓
Threshold = 0.30
    ↓
NORMAL / PNEUMONIA
```

Điểm quan trọng nằm ở phần giữa của quá trình. Dataset ban đầu có sự mất cân bằng giữa hai lớp, validation gốc chỉ có 16 ảnh nên không đủ để dùng cho việc lựa chọn mô hình, và kết quả validation sau khi tạo lại cũng khác đáng kể so với test ở vị trí threshold cuối cùng. Những vấn đề này khiến việc xây dựng một quy trình đánh giá rõ ràng quan trọng hơn việc chỉ tìm một con số accuracy cao.

---

## 2. Dữ liệu

Dataset được đặt trong:

```text
data/chest_xray/
```

Dataset gốc có ba phần:

| Split   | NORMAL | PNEUMONIA |  Tổng |
| ------- | -----: | --------: | ----: |
| `train` |  1,341 |     3,875 | 5,216 |
| `val`   |      8 |         8 |    16 |
| `test`  |    234 |       390 |   624 |

Phần `train` có số lượng ảnh PNEUMONIA gần gấp ba lần NORMAL. Trong khi đó, validation gốc chỉ có 16 ảnh. Với số lượng nhỏ như vậy, việc dùng trực tiếp `val` để chọn model hoặc threshold sẽ khiến kết quả rất dễ bị ảnh hưởng bởi chỉ một vài ảnh. Vì vậy, project không sử dụng validation gốc để lựa chọn mô hình mà tạo một validation set mới từ `train`.

Tập `test` được giữ nguyên và không được sử dụng trong quá trình lựa chọn model hay threshold.

### Tạo lại validation set

Từ `train` ban đầu, project thực hiện một stratified split với `val_ratio=0.15` và `seed=42`, đồng thời kiểm tra SHA-256 để tránh việc các file ảnh giống hệt nhau bị chia sang những split khác nhau.

Sau khi chia, dữ liệu được sử dụng như sau:

| Split      | NORMAL | PNEUMONIA |  Tổng |
| ---------- | -----: | --------: | ----: |
| Train      |  1,140 |     3,292 | 4,432 |
| Validation |    201 |       583 |   784 |
| Test       |    234 |       390 |   624 |

Thông tin của split được lưu tại:

```text
experiments/data_split.json
```

Kiểm tra duplicate bằng SHA-256 cho thấy **không có hash ảnh trùng nhau giữa train, validation và test** (`n_cross_split_duplicate_hashes = 0`). Điều này giúp loại bỏ trường hợp đơn giản nhất trong đó cùng một file ảnh xuất hiện ở nhiều split.

Một điểm khác cần lưu ý là tên file trong dataset có các pattern như `personNNNN`, `IM-*` và `NORMAL2-IM-*`. Project có kiểm tra sự trùng lặp của phần `personNNNN` giữa các split và nhận thấy có overlap. Tuy nhiên, đây chỉ là thông tin được lấy từ tên file; project không có metadata để xác nhận những chuỗi này thực sự tương ứng với patient ID duy nhất. Vì vậy, không thể từ đó kết luận rằng dataset chắc chắn có patient leakage.

---

## 3. Chuẩn bị ảnh

Ảnh được đọc bằng PIL và chuyển sang RGB nếu cần. Tất cả ảnh sau đó được resize về `224 × 224`.

Trong quá trình huấn luyện, project sử dụng một số phép biến đổi nhẹ như horizontal flip, rotation nhỏ, affine transformation và thay đổi nhẹ brightness/contrast. Mục đích là tạo ra một số biến thể của ảnh huấn luyện mà không làm thay đổi bản chất của bài toán.

Validation, test và inference sử dụng preprocessing cố định, không có augmentation:

```text
Resize → ToTensor → ImageNet normalization
```

Các giá trị normalization là:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Cách preprocessing này được dùng thống nhất cho validation và test để hai tập được đánh giá trong cùng một điều kiện.

Phần xử lý dữ liệu nằm trong:

```text
src/data/preprocessing.py
```

---

## 4. Các mô hình được thử nghiệm

Project bắt đầu bằng một Custom CNN đơn giản. Mục đích của mô hình này chủ yếu là kiểm tra pipeline đọc dữ liệu, DataLoader và training engine có hoạt động đúng hay không. Vì đây không phải mô hình được dùng cho quá trình lựa chọn cuối cùng nên kết quả của Custom CNN không được đưa vào bảng so sánh chính.

Sau đó, project chuyển sang các mô hình pretrained trên ImageNet để xem mức độ khác biệt giữa việc chỉ huấn luyện classifier và việc cho phép một phần backbone học lại từ dữ liệu X-quang.

### ResNet18 frozen

Ở phiên bản đầu tiên, ResNet18 pretrained được giữ nguyên phần lớn backbone và chỉ huấn luyện `fc`. Cách này cho phép kiểm tra xem các feature có sẵn từ ImageNet có thể sử dụng trực tiếp đến mức nào cho bài toán này.

### ResNet18 fine-tune

Tiếp theo, `layer4` và `fc` được mở để huấn luyện, trong khi các block trước đó được giữ nguyên. Backbone và classifier sử dụng learning rate khác nhau:

```text
head:     5e-4
backbone: 1e-4
```

Cách này cho phép phần cuối của ResNet18 thích nghi với dữ liệu X-quang nhưng vẫn giữ lại phần lớn feature đã học từ pretrained model.

### EfficientNet-B0

EfficientNet-B0 cũng được thử nghiệm để kiểm tra liệu một kiến trúc khác có tạo ra kết quả tốt hơn trên validation hay không. Phần được huấn luyện tập trung vào các block cuối (`features.7`, `features.8`) và classifier.

### ResNet18 + weighted loss

Ngoài thay đổi model, project cũng thử thay đổi cách tính loss để xử lý class imbalance. Với phiên bản này, CrossEntropyLoss sử dụng class weight được tính **chỉ từ tập train**:

```text
NORMAL     ≈ 1.944
PNEUMONIA  ≈ 0.673
```

Như vậy, lỗi trên lớp NORMAL được đặt trọng số cao hơn do NORMAL có ít mẫu hơn trong tập train.

---

## 5. Huấn luyện

Các mô hình được huấn luyện bằng AdamW, cosine learning-rate scheduler và early stopping dựa trên `val_loss`.

Thiết lập chính:

```text
Seed:             42
Batch size:       16
Maximum epochs:   10
Early stopping:   patience = 3
Optimizer:        AdamW
```

Với mô hình ResNet18 fine-tune, learning rate được chia thành hai mức cho head và backbone như mô tả ở trên. Với phiên bản chỉ huấn luyện classifier, learning rate của head là `1e-3`.

Project được phát triển theo hướng **CPU-first**, vì môi trường hiện tại không có NVIDIA GPU. Việc này không thay đổi cách đánh giá mô hình, nhưng thời gian training có thể dài hơn so với môi trường có GPU.

---

## 6. Kết quả trên validation

Sau khi huấn luyện, các model được đánh giá trên validation ở threshold mặc định `0.5` trước khi đi sâu vào việc chọn threshold.

| Model                            | Accuracy | Macro F1 | Pneumonia Recall | FN | FP | ROC-AUC |
| -------------------------------- | -------: | -------: | ---------------: | -: | -: | ------: |
| ResNet18 frozen                  |    0.949 |    0.933 |            0.966 | 20 | 20 |   0.987 |
| ResNet18 fine-tune               |    0.977 |    0.971 |            0.974 | 15 |  3 |   0.998 |
| ResNet18 fine-tune + weighted CE |    0.981 |    0.975 |            0.983 | 10 |  5 |   0.998 |
| EfficientNet-B0                  |    0.968 |    0.959 |            0.964 | 21 |  4 |   0.997 |

Các kết quả này cho thấy việc mở `layer4` của ResNet18 để fine-tune thay vì chỉ huấn luyện classifier đã thay đổi đáng kể hành vi của mô hình trên validation. Phiên bản weighted loss tiếp tục giảm số ca FN ở threshold `0.5`, từ 15 xuống 10.

Tuy nhiên, đây vẫn chưa phải bước cuối cùng. Threshold `0.5` chỉ là ngưỡng mặc định của binary classification; với bài toán này, thay đổi threshold sẽ trực tiếp thay đổi sự cân bằng giữa FN và FP. Vì vậy, trước khi chọn model cuối cùng, project tiếp tục phân tích lỗi và threshold trên validation.

---

## 7. Phân tích lỗi

Với `resnet18_finetune` ở threshold mặc định khoảng `0.5`, validation có:

* 3 false positive.
* 15 false negative.

Điều đáng chú ý là các false positive không chỉ nằm sát threshold. Ba ảnh NORMAL bị dự đoán thành PNEUMONIA có xác suất PNEUMONIA trung bình khoảng `0.90`, với giá trị nằm trong khoảng `0.84–0.94`. Ngược lại, các false negative có xác suất PNEUMONIA trung bình khoảng `0.33`, với giá trị từ khoảng `0.057` đến `0.49`.

Điều này cho thấy một số ảnh NORMAL bị mô hình đánh giá khá chắc chắn, trong khi một số ảnh PNEUMONIA lại có score khá thấp. Do đó, chỉ nhìn vào accuracy hoặc thay đổi threshold một cách tùy ý sẽ không giải thích được hết hành vi của mô hình.

Một false positive xuất hiện trong quá trình kiểm tra là:

```text
NORMAL2-IM-0554-0001.jpeg
```

Các hình ảnh lỗi được lưu tại:

```text
reports/figures/error_analysis/resnet18_finetune/fp_grid.png
reports/figures/error_analysis/resnet18_finetune/fn_grid.png
```

Phần phân tích này chỉ nhằm hiểu mô hình đang mắc lỗi như thế nào. Không nên dùng prediction của từng ảnh riêng lẻ để đưa ra kết luận y khoa.

---

## 8. Chọn threshold

Sau khi xem xét lỗi ở threshold `0.5`, project thực hiện threshold analysis trên validation với các giá trị từ `0.10` đến `0.90`.

Quy tắc lựa chọn được xác định trước như sau:

1. Ưu tiên pneumonia F1 cao hơn.
2. Nếu F1 bằng nhau, ưu tiên pneumonia recall cao hơn.
3. Nếu vẫn bằng nhau, chọn threshold thấp hơn.
4. Khi các model vẫn tương đương theo rule trên, dùng số FP để phân biệt.

Quy tắc này được lưu cùng kết quả trong:

```text
experiments/threshold_analysis/comparison_summary.json
```

Đối với `resnet18_finetune`, thay đổi threshold từ `0.5` xuống `0.3` làm:

| Threshold | Pneumonia Recall | Pneumonia F1 | FN | FP |
| --------- | ---------------: | -----------: | -: | -: |
| 0.50      |            0.974 |        0.984 | 15 |  3 |
| 0.30      |            0.991 |        0.991 |  5 |  5 |

Như vậy, threshold `0.30` giúp giảm FN từ 15 xuống 5 trên validation, đồng thời số FP tăng từ 3 lên 5.

Khi áp dụng rule trên cho hai phiên bản ResNet18:

| Model                            | Threshold được chọn | Pneumonia F1 | Recall | FN | FP |
| -------------------------------- | ------------------: | -----------: | -----: | -: | -: |
| ResNet18 fine-tune               |                0.30 |        0.991 |  0.991 |  5 |  5 |
| ResNet18 fine-tune + weighted CE |                0.20 |        0.988 |  0.991 |  5 |  9 |

Theo rule đã đặt ra trước đó, `resnet18_finetune` với threshold `0.30` được chọn làm cấu hình cuối cùng.

Đường cong threshold được lưu tại:

```text
reports/figures/threshold_analysis/resnet18_finetune_threshold_curves.png
```

Điểm quan trọng ở đây là threshold `0.30` được **chọn từ validation**, không phải từ test.

---

## 9. Grad-CAM

Sau khi chọn `resnet18_finetune`, project sử dụng Grad-CAM trên `layer4` để quan sát những vùng nào trong ảnh có ảnh hưởng đến prediction của mô hình.

Bốn trường hợp được kiểm tra gồm:

| Trường hợp     | Ảnh                             |
| -------------- | ------------------------------- |
| True Positive  | `person1003_bacteria_2934.jpeg` |
| True Negative  | `IM-0411-0001.jpeg`             |
| False Positive | `NORMAL2-IM-0554-0001.jpeg`     |
| False Negative | `person730_virus_1351.jpeg`     |

Các hình kết quả:

```text
reports/figures/gradcam/resnet18_finetune_TP_gradcam.png
reports/figures/gradcam/resnet18_finetune_TN_gradcam.png
reports/figures/gradcam/resnet18_finetune_FP_gradcam.png
reports/figures/gradcam/resnet18_finetune_FN_gradcam.png
```

Tóm tắt được lưu tại:

```text
experiments/gradcam/resnet18_finetune_gradcam_summary.json
```

Grad-CAM ở đây được dùng như một công cụ quan sát để hiểu thêm về prediction của mô hình. Vùng được highlight không chứng minh rằng mô hình đã xác định đúng một tổn thương viêm phổi, cũng không thể thay thế đánh giá chuyên môn hoặc clinical validation.

---

## 10. Khóa model và cấu hình cuối cùng

Sau khi hoàn thành quá trình lựa chọn trên validation, project khóa model và threshold vào một manifest:

```text
experiments/final_selection/final_selection_manifest.json
```

Đồng thời tạo marker:

```text
experiments/final_selection/LOCKED
```

Cấu hình cuối cùng là:

```text
Model:
    ResNet18

Trainable:
    layer4 + fc

Loss:
    CrossEntropyLoss
    class_weighted_loss = false

Checkpoint:
    checkpoints/resnet18_finetune/resnet18_finetune_best.pt

Decision threshold:
    0.30
```

Quy tắc dự đoán là:

```text
P(PNEUMONIA) >= 0.30  →  PNEUMONIA
P(PNEUMONIA) <  0.30  →  NORMAL
```

Từ thời điểm này, test set mới được sử dụng.

---

## 11. Kết quả trên test set

Test set gồm 624 ảnh và được giữ nguyên từ đầu project:

```text
NORMAL:     234
PNEUMONIA:  390
```

Sau khi model và threshold đã được khóa, `scripts/final_test_evaluation.py` được chạy để đánh giá mô hình trên test.

Kết quả:

| Metric              |   Test |
| ------------------- | -----: |
| Accuracy            | 0.8510 |
| ROC-AUC             | 0.9670 |
| PR-AUC              | 0.9754 |
| Pneumonia Precision | 0.8113 |
| Pneumonia Recall    | 0.9923 |
| Pneumonia F1        | 0.8927 |
| FN                  |      3 |
| FP                  |     90 |

Confusion matrix:

```text
                Pred NORMAL    Pred PNEUMONIA
Actual NORMAL        144              90
Actual PNEUMONIA      3              387
```

Ở threshold `0.30`, mô hình chỉ bỏ sót `3/390` ảnh PNEUMONIA, nhưng đồng thời có `90/234` ảnh NORMAL bị dự đoán thành PNEUMONIA, tương đương khoảng **38.5% số ảnh NORMAL của test set**.

Đây là một kết quả quan trọng của project. Các metric như ROC-AUC và PR-AUC vẫn ở mức cao, cho thấy model vẫn phân biệt và xếp hạng hai nhóm khá tốt theo score. Tuy nhiên, khi chuyển score thành nhãn bằng threshold `0.30`, số false positive trên test lại tăng mạnh.

Để tham khảo, nếu dùng threshold `0.5` trực tiếp trên test thì số FP giảm từ 90 xuống 77 trong khi FN vẫn là 3. Con số này chỉ được báo cáo để quan sát ảnh hưởng của threshold trên test; **threshold 0.5 không được dùng để thay đổi cấu hình cuối cùng sau khi đã nhìn thấy test**.

Kết quả đầy đủ được lưu tại:

```text
experiments/final_evaluation/final_test_report.json
```

---

## 12. Vì sao validation và test khác nhau?

Sau khi hoàn thành final test, project thực hiện một bước riêng để tìm hiểu sự khác biệt giữa validation và test. Bước này không thay đổi model hoặc threshold đã khóa.

Một trong những khác biệt rõ nhất nằm ở score của các ảnh NORMAL.

Với model cuối cùng và threshold `0.30`:

|                                        | Validation |    Test |
| -------------------------------------- | ---------: | ------: |
| Mean `P(PNEUMONIA)` trên ảnh NORMAL    |    ≈ 0.035 | ≈ 0.338 |
| NORMAL có score ≥ 0.30                 |     ≈ 2.5% | ≈ 38.5% |
| False Positive                         |          5 |      90 |
| Mean `P(PNEUMONIA)` trên ảnh PNEUMONIA |    ≈ 0.970 | ≈ 0.983 |
| False Negative                         |          5 |       3 |

Trong khi score của các ảnh PNEUMONIA khá tương đồng giữa hai tập, score của các ảnh NORMAL khác nhau rất rõ. Điều này giải thích trực tiếp vì sao threshold `0.30` hoạt động tốt trên validation nhưng tạo ra nhiều FP trên test.

Project cũng kiểm tra các file `NORMAL2-IM-*` và `IM-*` trên test. Với threshold `0.30`, nhóm `NORMAL2-IM-*` có `74/165` ảnh bị FP, khoảng `44.8%`, trong khi nhóm `IM-*` có `16/69`, khoảng `23.2%`.

Đây là những tín hiệu đáng chú ý, nhưng chưa đủ để kết luận nguyên nhân là domain shift, khác biệt bệnh viện, cách chụp hay patient leakage. Project không có metadata lâm sàng hoặc thông tin nguồn gốc đủ để xác nhận những giả thuyết đó.

Ngoài ra, kiểm tra SHA-256 cho thấy không có ảnh trùng hoàn toàn giữa các split. Tuy nhiên, việc một phần `personNNNN` xuất hiện ở nhiều split vẫn là một tín hiệu cần lưu ý. Vì đây chỉ là identifier lấy từ filename chứ không phải patient ID đã được xác minh, project không dùng nó để kết luận có hay không có patient-level leakage.

Các kết quả của bước điều tra này được lưu tại:

```text
experiments/final_evaluation/val_test_gap_investigation.json
```

và các hình:

```text
reports/figures/final_evaluation/val_test_score_distributions.png
reports/figures/final_evaluation/test_false_positive_scores.png
```

Điểm quan trọng là bước investigation này được thực hiện **sau khi test đã hoàn thành và không quay lại thay đổi model hoặc threshold**.

---

## 13. Inference

Project có một CLI riêng để chạy prediction trên một ảnh bất kỳ:

```bash
python scripts/inference.py --image path/to/xray.jpeg
```

Inference không tự đặt model hoặc threshold trong code. Thay vào đó, nó đọc cấu hình từ final selection manifest:

```text
experiments/final_selection/final_selection_manifest.json
```

Có thể chỉ định manifest khác nếu cần:

```bash
python scripts/inference.py \
    --image path/to/xray.jpeg \
    --manifest experiments/final_selection/final_selection_manifest.json
```

Output gồm prediction, xác suất PNEUMONIA và threshold đang được sử dụng.

Pipeline inference sử dụng cùng preprocessing deterministic với validation/test, vì vậy cách chuẩn bị ảnh không thay đổi giữa lúc đánh giá và lúc chạy prediction.

Một số smoke test đã được kiểm tra:

* Ảnh PNEUMONIA mẫu → `PNEUMONIA`, `P(PNEUMONIA)=1.0000`.
* Ảnh NORMAL mẫu → `NORMAL`, `P(PNEUMONIA)=0.0000`.
* Đường dẫn không tồn tại → báo lỗi rõ ràng và exit code `1`.
* File không phải ảnh → báo lỗi rõ ràng và exit code `1`.

Inference không truy cập test set và không thay đổi các experiment artifact.

---

## 14. Cấu trúc project

```text
pneumonia-detection/
├── configs/
├── checkpoints/
├── data/
│   └── chest_xray/
├── experiments/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
└── requirements.txt
```

Một số artifact quan trọng nếu muốn kiểm tra lại quá trình thực nghiệm:

```text
experiments/data_split.json
experiments/threshold_analysis/comparison_summary.json
experiments/final_selection/final_selection_manifest.json
experiments/final_evaluation/final_test_report.json
experiments/final_evaluation/val_test_gap_investigation.json
```

---

## 15. Chạy project

Cài dependency:

```bash
pip install -r requirements.txt
```

Tạo lại data split:

```bash
python scripts/create_data_split.py
```

Huấn luyện một configuration:

```bash
python scripts/train.py \
    --config configs/resnet18_finetune.yaml
```

Đánh giá checkpoint trên validation:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/resnet18_finetune/resnet18_finetune_best.pt \
    --split val \
    --model resnet18_finetune
```

Chạy inference:

```bash
python scripts/inference.py \
    --image path/to/xray.jpeg
```

Các bước phân tích chính được thực hiện bởi:

```text
scripts/error_analysis.py
scripts/threshold_analysis.py
scripts/run_gradcam.py
scripts/lock_final_selection.py
scripts/final_test_evaluation.py
scripts/investigate_val_test_gap.py
```

Toàn bộ quá trình sử dụng `seed=42` cho các bước cần tính tái lập.

Training có thể chạy trên CPU nhưng có thể mất nhiều thời gian hơn so với môi trường có GPU. Inference và evaluation nhẹ hơn và có thể chạy trực tiếp trên môi trường CPU-first hiện tại.

---

## 16. Những điểm cần lưu ý

Project có một số giới hạn rõ ràng từ chính dataset và cách thiết kế thực nghiệm.

Thứ nhất, dữ liệu bị mất cân bằng giữa NORMAL và PNEUMONIA, đồng thời validation gốc chỉ có 16 ảnh nên phải tạo validation mới từ train. Điều này giúp quá trình lựa chọn có cơ sở hơn nhưng không làm cho dataset trở thành một clinical dataset đầy đủ.

Thứ hai, threshold được lựa chọn dựa trên validation có thể không phù hợp với phân bố score của một tập dữ liệu khác. Trong project này, điều đó thể hiện khá rõ khi threshold `0.30` cho validation chỉ có 5 FP nhưng tạo ra 90 FP trên test. Vì vậy, kết quả test cho thấy việc chọn threshold cũng là một phần của bài toán, chứ không chỉ đơn giản là chọn model có ROC-AUC cao.

Thứ ba, project chưa có external validation. Tất cả kết quả đều đến từ cùng một public dataset, nên chưa thể biết mô hình sẽ hoạt động như thế nào trên một nguồn dữ liệu hoàn toàn khác.

Cuối cùng, Grad-CAM chỉ cung cấp một cách trực quan để quan sát activation của model. Nó không phải bằng chứng rằng model đang nhìn đúng vùng tổn thương hay rằng prediction có ý nghĩa lâm sàng.

---

## 17. Những gì project cho thấy

Một kết quả đáng chú ý của project không phải chỉ là con số accuracy cuối cùng, mà là sự khác biệt giữa **model performance** và **operating point**.

ResNet18 fine-tune cho kết quả validation tốt hơn rõ ràng so với phiên bản chỉ train classifier, và việc dùng weighted loss giúp giảm FN ở threshold `0.5`. Tuy nhiên, khi threshold được điều chỉnh để tăng recall trên validation, hiệu quả đó không được giữ nguyên trên test. Threshold `0.30` giảm FN trên validation nhưng đồng thời tạo ra rất nhiều false positive trên test.

Điều này cho thấy một mô hình có ROC-AUC hoặc validation F1 cao vẫn có thể gặp vấn đề khi được sử dụng ở một threshold cụ thể trên dữ liệu mới. Trong một bài toán thực tế, việc xác định model và cách chuyển probability thành decision cần được xem xét cùng nhau.

Project cũng cố gắng giữ quá trình đánh giá tách biệt: validation được dùng để lựa chọn, model và threshold được khóa trước khi test, test chỉ được đánh giá một lần, và việc tìm hiểu val/test gap được thực hiện sau đó mà không quay lại tối ưu theo test.

---

## 18. Hướng phát triển

Nếu tiếp tục phát triển project, bước đầu tiên hợp lý sẽ là tìm một cách xác định patient identity đáng tin cậy hơn để có thể thực hiện **patient-level split** thay vì chỉ dựa vào filename.

Bên cạnh đó, có thể đánh giá model trên một external dataset để xem sự khác biệt giữa validation và test hiện tại có tiếp tục xuất hiện trên một nguồn dữ liệu khác hay không. Calibration và việc phân tích nhiều operating point cũng là những hướng tiếp theo nếu muốn sử dụng probability của model một cách cẩn thận hơn.

Với chính dataset hiện tại, có thể tiếp tục xem xét các nhóm ảnh NORMAL khó, đặc biệt những ảnh thường xuyên bị dự đoán thành PNEUMONIA. Những ảnh này có thể được dùng cho quá trình data curation hoặc hard-negative mining trong các vòng huấn luyện tiếp theo.

Các hướng trên cần được thực hiện với một protocol đánh giá được xác định trước để tránh biến test set thành một phần của quá trình tuning.

---

## 19. Kết luận

PneumoVision bắt đầu từ một bài toán phân loại ảnh X-quang tương đối đơn giản, nhưng quá trình thực hiện tập trung vào toàn bộ vòng đời của một model thay vì chỉ huấn luyện một mạng CNN và báo cáo accuracy.

Từ việc kiểm tra dataset và xây dựng lại validation set, project thử nghiệm nhiều cách sử dụng pretrained model, fine-tuning, class weighting và threshold. Sau đó, các lỗi trên validation được phân tích, Grad-CAM được sử dụng để quan sát prediction, model và threshold được khóa trước khi đánh giá test, và cuối cùng sự khác biệt giữa validation và test được điều tra mà không thay đổi lại kết quả cuối cùng.

Kết quả test cho thấy model vẫn đạt `ROC-AUC = 0.9670` và pneumonia recall `0.9923`, nhưng cũng có `90` false positive trên `234` ảnh NORMAL ở threshold `0.30`. Đây là một giới hạn quan trọng của kết quả và cũng là phần có giá trị nhất để tiếp tục nghiên cứu, bởi nó cho thấy một pipeline Machine Learning không kết thúc ở việc tìm được một model có metric validation cao; cách dữ liệu được chia, cách threshold được chọn và cách kết quả được kiểm tra trên dữ liệu chưa từng thấy đều ảnh hưởng trực tiếp đến hành vi cuối cùng của hệ thống.
