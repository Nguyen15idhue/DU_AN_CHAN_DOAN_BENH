# Quy Trình Huấn Luyện Mô Hình - Bước 3

## Tổng Quan

Bước 3 là giai đoạn "cuộc thi" giữa các thuật toán Machine Learning để tìm ra mô hình hoạt động hiệu quả nhất trên dữ liệu đã tiền xử lý. Đây là bước quan trọng để đảm bảo chất lượng dự đoán của hệ thống chẩn đoán bệnh.

## Bản Chất Thực Hiện

### Tại Sao Cần So Sánh Nhiều Mô Hình?

- **Không có mô hình nào tốt nhất cho mọi bài toán**: Hiệu suất phụ thuộc vào đặc điểm dữ liệu
- **Đánh giá toàn diện**: So sánh không chỉ accuracy mà còn precision, recall, F1-score
- **Cân bằng trade-offs**: Một số mô hình nhanh nhưng kém chính xác, ngược lại
- **Hiểu rõ điểm mạnh/yếu**: Chuẩn bị cho việc tinh chỉnh siêu tham số

### Nguyên Tắc Lựa Chọn Thuật Toán

Dựa trên đặc điểm dữ liệu:
- **377 features nhị phân**: Phù hợp với tree-based models và linear models
- **773 classes**: Multi-class classification
- **Dữ liệu mất cân bằng**: Cần mô hình robust với imbalance
- **Kích thước lớn**: Cần cân nhắc thời gian huấn luyện

## Các Thuật Toán Được Chọn

### 1. Random Forest Classifier
**Lý do chọn:**
- Xuất sắc với tabular data có nhiều features
- Xử lý tốt missing values và outliers
- Robust với overfitting nhờ ensemble
- Tự động feature selection

**Ưu điểm:**
- Accuracy cao
- Không cần scale features
- Interpretability tốt

**Nhược điểm:**
- Chậm với dữ liệu lớn
- Khó interpret chi tiết

### 2. Logistic Regression
**Lý do chọn:**
- Baseline model đơn giản và nhanh
- Tốt cho binary/multi-class classification
- Probabilistic output
- Dễ hiểu và implement

**Ưu điểm:**
- Nhanh training/inference
- Không overfit nếu regularized
- Feature importance rõ ràng

**Nhược điểm:**
- Giả định linear relationship
- Cần scale features
- Kém với non-linear data

### 3. Multinomial Naive Bayes
**Lý do chọn:**
- Tốt cho high-dimensional data
- Nhanh và memory-efficient
- Robust với irrelevant features
- Tốt cho text classification (mặc dù đây là symptoms)

**Ưu điểm:**
- Rất nhanh
- Ít parameters
- Không overfit

**Nhược điểm:**
- Giả định independence (thường không đúng)
- Có thể underfit
- Không tốt với correlated features

## Quy Trình Thực Hiện Chi Tiết

### 1. Chuẩn Bị Dữ Liệu
```python
# Load processed data từ Bước 2
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')
```

### 2. Định Nghĩa Mô Hình
```python
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    'MultinomialNB': MultinomialNB()
}
```

### 3. Huấn Luyện và Đánh Giá
**Cho mỗi mô hình:**
1. **Training**: Fit trên tập train
2. **Timing**: Ghi lại thời gian huấn luyện
3. **Prediction**: Dự đoán trên tập test
4. **Metrics**: Tính toán các chỉ số

### 4. Các Metrics Đánh Giá

#### Accuracy
```
Accuracy = (Số dự đoán đúng) / (Tổng số mẫu)
```
- Dễ hiểu nhưng misleading với dữ liệu imbalance

#### Macro Average Metrics
- **Precision**: TP / (TP + FP) cho mỗi class, sau đó average
- **Recall**: TP / (TP + FN) cho mỗi class, sau đó average
- **F1-Score**: Harmonic mean của precision và recall

**Tại sao macro average?**
- Trả trọng số bằng nhau cho tất cả classes
- Quan trọng với dữ liệu imbalance
- Phản ánh hiệu suất tổng thể

### 5. So Sánh và Lựa Chọn

#### Tiêu Chí Lựa Chọn
1. **Accuracy cao nhất** (ưu tiên chính)
2. **Macro F1-Score cao** (cân bằng precision/recall)
3. **Thời gian huấn luyện hợp lý**
4. **Stability** (ít biến động)

#### Kết Quả Mong Đợi
- Random Forest thường thắng về accuracy
- Logistic Regression nhanh nhất
- Naive Bayes có thể surprise với high-dimensional data

## Điểm Lưu Ý Quan Trọng

### 1. Hyperparameters
- Sử dụng default parameters cho comparison công bằng
- Tinh chỉnh sẽ thực hiện ở Bước 4

### 2. Cross-Validation
- Không dùng CV trong bước này để tiết kiệm thời gian
- Sẽ dùng ở bước tuning

### 3. Memory Management
- Dữ liệu lớn (197k samples × 377 features)
- Sử dụng n_jobs=-1 để tận dụng CPU
- Monitor memory usage

### 4. Reproducibility
- random_state=42 cho consistency
- Same train/test split

### 5. Error Handling
- zero_division=0 trong classification_report
- Handle potential convergence issues

## Kết Quả Thực Tế

Sau khi chạy, kết quả sẽ bao gồm:
- Bảng so sánh chi tiết
- Mô hình tốt nhất được chọn
- Model được lưu với joblib

## Chuẩn Bị Cho Bước Tiếp Theo

### Bước 4: Hyperparameter Tuning
- Lấy mô hình tốt nhất từ bước này
- Tìm bộ parameters tối ưu
- Sử dụng GridSearchCV hoặc RandomizedSearchCV

### Bước 5: Deployment
- Model đã tune sẽ được dùng trong ứng dụng
- Cần load label_encoder và feature_list

## Troubleshooting

### Nếu Memory Error
- Giảm n_estimators của RandomForest
- Sử dụng mini-batch training
- Tăng virtual memory

### Nếu Convergence Warning
- Tăng max_iter của LogisticRegression
- Scale features nếu cần

### Nếu Poor Performance
- Kiểm tra data preprocessing
- Thử thêm features engineering
- Consider deep learning approaches

## Kết Luận

Bước 3 cung cấp foundation cho hệ thống:
- **Model Selection**: Chọn thuật toán phù hợp
- **Performance Baseline**: Thiết lập expectations
- **Understanding**: Hiểu strengths/weaknesses của mỗi approach

Kết quả sẽ hướng dẫn việc tinh chỉnh và deployment tiếp theo.