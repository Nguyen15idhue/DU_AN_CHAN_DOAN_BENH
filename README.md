# Hệ Thống Chẩn Đoán Bệnh Tật

Dự án này xây dựng một hệ thống chẩn đoán bệnh tật dựa trên triệu chứng sử dụng Machine Learning.

## Cấu Trúc Dự Án

```
DU_AN_CHAN_DOAN_BENH/
├── data/
│   └── Disease and symptoms dataset.csv  # Dữ liệu triệu chứng và bệnh
├── notebooks/
│   └── 1_EDA_and_Preprocessing.ipynb     # Phân tích và tiền xử lý dữ liệu
├── src/
│   ├── train.py                          # Huấn luyện mô hình
│   └── app.py                            # Ứng dụng chẩn đoán
├── models/                               # Lưu trữ mô hình và encoder
├── requirements.txt                      # Thư viện cần thiết
└── README.md                             # Tài liệu dự án
```

## Cài Đặt

1. Clone repository:
   ```
   git clone <repository-url>
   cd DU_AN_CHAN_DOAN_BENH
   ```

2. Tạo môi trường ảo:
   ```
   python -m venv venv
   .\venv\Scripts\activate  # Trên Windows
   ```

3. Cài đặt thư viện:
   ```
   pip install -r requirements.txt
   ```

## Sử Dụng

1. Chạy notebook để phân tích dữ liệu: Mở `notebooks/1_EDA_and_Preprocessing.ipynb`

2. Huấn luyện mô hình:
   ```
   python src/train.py
   ```

3. Chạy ứng dụng chẩn đoán:
   ```
   python src/app.py
   ```

## Các Bước Thực Hiện

- Bước 1 & 2: Phân tích và tiền xử lý dữ liệu
- Bước 3: Huấn luyện và lựa chọn mô hình
- Bước 4: Tinh chỉnh siêu tham số
- Bước 5: Xây dựng ứng dụng chẩn đoán

## Thư Viện Sử Dụng

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Tác Giả

[Your Name]

## Giấy Phép

[License]