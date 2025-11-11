# Tài Liệu Chi Tiết: Bước 1 & 2 - Phân Tích và Tiền Xử Lý Dữ Liệu

## Tổng Quan

Bước 1 & 2 là giai đoạn quan trọng nhất trong quy trình xây dựng hệ thống chẩn đoán bệnh tật. Ở đây, chúng ta không chỉ hiểu dữ liệu mà còn chuẩn bị nó cho các bước huấn luyện mô hình tiếp theo.

## Bản Chất

### Tại Sao Cần Phân Tích Dữ Liệu (EDA - Exploratory Data Analysis)?

EDA là quá trình khám phá dữ liệu để:
- **Hiểu cấu trúc dữ liệu**: Kích thước, kiểu dữ liệu, phân bố
- **Phát hiện vấn đề**: Dữ liệu thiếu, outlier, phân bố không cân bằng
- **Khám phá mối quan hệ**: Giữa các biến, đặc biệt là giữa triệu chứng và bệnh
- **Hướng dẫn quyết định**: Lựa chọn thuật toán, kỹ thuật preprocessing phù hợp

### Tại Sao Cần Tiền Xử Lý Dữ Liệu?

Dữ liệu thô thường không sẵn sàng cho machine learning:
- **Định dạng không phù hợp**: Text labels cần chuyển thành số
- **Thiếu tính nhất quán**: Thứ tự features phải giống nhau trong train/test
- **Cần chia tập dữ liệu**: Để đánh giá mô hình công bằng

## Cách Thực Hiện

### 1. Load Dữ Liệu

```python
import pandas as pd
df = pd.read_csv('../data/Disease and symptoms dataset.csv')
```

**Giải thích**: Sử dụng pandas để đọc file CSV. Dữ liệu bao gồm 246,945 mẫu với 378 cột (1 cột bệnh + 377 cột triệu chứng).

### 2. Phân Tích Cơ Bản

```python
print(f"Shape: {df.shape}")
print(df.head())
print(df.info())
```

**Kết quả**:
- **Shape**: (246945, 378) - 246,945 mẫu, 378 cột
- **Dtypes**: 377 cột int64 (triệu chứng), 1 cột object (bệnh)
- **Memory usage**: 712.2 MB - Dữ liệu khá lớn

### 3. Phân Tích Bệnh (Target Analysis)

```python
unique_diseases = df['diseases'].nunique()  # 773 bệnh duy nhất
disease_counts = df['diseases'].value_counts()
```

**Nhận xét quan trọng**:
- **773 loại bệnh** khác nhau
- **Phân bố không cân bằng**: Một số bệnh có >1000 mẫu, một số chỉ có vài mẫu
- **Top bệnh**: cystitis (1219), vulvodynia (1218), nose disorder (1218)

**Ý nghĩa**: Sự mất cân bằng này ảnh hưởng đến hiệu suất mô hình. Các bệnh ít mẫu sẽ khó dự đoán chính xác.

### 4. Phân Tích Triệu Chứng (Feature Analysis)

```python
symptom_sums = df.drop('diseases', axis=1).sum().sort_values(ascending=False)
```

**Top triệu chứng phổ biến**:
1. sharp abdominal pain: 32,307
2. vomiting: 27,874
3. headache: 24,719
4. cough: 24,296
5. sharp chest pain: 24,016

**Nhận xét**: Các triệu chứng phổ biến chủ yếu liên quan đến đau bụng, nôn mửa, đau đầu - có thể là triệu chứng của nhiều bệnh khác nhau.

### 5. Tiền Xử Lý Dữ Liệu

#### Tách Features và Target
```python
X = df.drop('diseases', axis=1)  # Features: 377 triệu chứng
y = df['diseases']               # Target: tên bệnh
```

#### Mã Hóa Nhãn (Label Encoding)
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```

**Tại sao cần?**
- Machine learning algorithms yêu cầu target là số
- `label_encoder` lưu mapping giữa tên bệnh và số
- Quan trọng cho việc inverse_transform sau này

#### Lưu Artifacts Quan Trọng
```python
import joblib
joblib.dump(label_encoder, '../models/label_encoder.joblib')
joblib.dump(X.columns.tolist(), '../models/feature_list.joblib')
```

**Ý nghĩa của feature_list**:
- Đảm bảo thứ tự triệu chứng giống nhau trong tất cả các bước
- Khi người dùng nhập triệu chứng, tạo vector theo đúng thứ tự này
- Nếu thứ tự sai, dự đoán sẽ hoàn toàn sai

#### Chia Dữ Liệu
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

**Lưu ý**: Ban đầu dùng `stratify=y_encoded` nhưng gặp lỗi vì một số class chỉ có 1 mẫu. Đã bỏ stratify để tránh lỗi.

**Kích thước sau chia**:
- Train: 197,556 mẫu (80%)
- Test: 49,389 mẫu (20%)

#### Lưu Dữ Liệu Đã Xử Lý
```python
import numpy as np
np.save('../data/processed/X_train.npy', X_train.values)
np.save('../data/processed/X_test.npy', X_test.values)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/y_test.npy', y_test)
```

**Lý do lưu .npy**:
- Nhanh hơn CSV cho dữ liệu số lớn
- Giữ nguyên dtype (int64)
- Dễ load với numpy

## Kết Quả và Nhận Xét

### Thống Kê Chính
- **Tổng mẫu**: 246,945
- **Số triệu chứng**: 377
- **Số bệnh**: 773
- **Tập huấn luyện**: 197,556 mẫu
- **Tập kiểm tra**: 49,389 mẫu

### Vấn Đề Phát Hiện
1. **Mất cân bằng dữ liệu**: Một số bệnh có rất nhiều mẫu, số khác ít
2. **Dữ liệu nhị phân**: Tất cả triệu chứng là 0/1 (có/không có)
3. **Không có missing values**: Dữ liệu khá sạch

### Ý Nghĩa Cho Các Bước Tiếp Theo
- **Bước 3 (Huấn luyện)**: Cần chọn thuật toán xử lý tốt với dữ liệu mất cân bằng
- **Bước 4 (Tuning)**: Tập trung vào các hyperparameters phù hợp
- **Bước 5 (Ứng dụng)**: Đảm bảo feature_list được sử dụng đúng

## Files Được Tạo

### Dữ Liệu
- `data/processed/X_train.npy`: Features tập huấn luyện
- `data/processed/X_test.npy`: Features tập kiểm tra
- `data/processed/y_train.npy`: Labels tập huấn luyện
- `data/processed/y_test.npy`: Labels tập kiểm tra

### Models/Artifacts
- `models/label_encoder.joblib`: Encoder cho bệnh
- `models/feature_list.joblib`: Danh sách thứ tự triệu chứng

### Báo Cáo
- `reports/data_analysis_report.html`: Báo cáo HTML với biểu đồ
- `reports/disease_distribution.png`: Biểu đồ phân bố bệnh
- `reports/symptom_frequency.png`: Biểu đồ tần suất triệu chứng

## Câu Hỏi Thường Gặp

### Tại sao không dùng stratify trong train_test_split?
Vì một số class chỉ có 1 mẫu, stratify yêu cầu ít nhất 2 mẫu mỗi class để chia đều.

### Tại sao lưu feature_list?
Để đảm bảo khi tạo vector input từ triệu chứng người dùng, thứ tự các feature giống hệt thứ tự đã học.

### Dữ liệu có missing values không?
Không, dữ liệu khá sạch, tất cả là số nguyên 0/1.

## Kết Luận

Bước 1 & 2 đã hoàn thành thành công:
- ✅ Hiểu rõ cấu trúc dữ liệu
- ✅ Phát hiện đặc điểm quan trọng
- ✅ Chuẩn bị dữ liệu cho huấn luyện
- ✅ Lưu trữ artifacts cần thiết
- ✅ Tạo báo cáo trực quan

Dữ liệu đã sẵn sàng cho Bước 3: Huấn luyện mô hình.