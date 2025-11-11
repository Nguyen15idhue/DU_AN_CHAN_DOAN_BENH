import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def load_data():
    """Load the disease symptoms dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Disease and symptoms dataset.csv')
    df = pd.read_csv(data_path)
    return df

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    print("=== EXPLORATORY DATA ANALYSIS ===")
    print("\n1. Basic Information:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())

    print("\n2. Disease Analysis:")
    unique_diseases = df['diseases'].nunique()
    print(f"Number of unique diseases: {unique_diseases}")

    disease_counts = df['diseases'].value_counts()
    print(f"\nTop 10 diseases by sample count:")
    print(disease_counts.head(10))

    # Plot top 20 diseases
    plt.figure(figsize=(12, 6))
    top_20_diseases = disease_counts.head(20)
    top_20_diseases.plot(kind='bar')
    plt.title('Top 20 Diseases by Sample Count')
    plt.xlabel('Disease')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n3. Symptom Analysis:")
    symptom_sums = df.drop('diseases', axis=1).sum().sort_values(ascending=False)
    print(f"\nTop 10 most common symptoms:")
    print(symptom_sums.head(10))

    # Plot top 20 symptoms
    plt.figure(figsize=(12, 6))
    top_20_symptoms = symptom_sums.head(20)
    top_20_symptoms.plot(kind='bar')
    plt.title('Top 20 Most Common Symptoms')
    plt.xlabel('Symptom')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/symptom_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    return disease_counts, symptom_sums

def preprocess_data(df):
    """Perform data preprocessing"""
    print("\n=== DATA PREPROCESSING ===")

    # Separate features and target
    X = df.drop('diseases', axis=1)
    y = df['diseases']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Encoded target classes: {len(label_encoder.classes_)}")
    print(f"Sample classes: {label_encoder.classes_[:5]}...")

    # Save important artifacts
    os.makedirs('../models', exist_ok=True)
    joblib.dump(label_encoder, '../models/label_encoder.joblib')
    joblib.dump(X.columns.tolist(), '../models/feature_list.joblib')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("\nData split:")
    print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

    # Save processed data
    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/X_train.npy', X_train.values)
    np.save('../data/processed/X_test.npy', X_test.values)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/y_test.npy', y_test)

    return X_train, X_test, y_train, y_test, label_encoder

def generate_report(disease_counts, symptom_sums, df_shape, train_shape, test_shape):
    """Generate HTML report"""
    html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích Dữ Liệu - Hệ Thống Chẩn Đoán Bệnh</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .conclusion {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Báo Cáo Phân Tích Dữ Liệu</h1>
        <h2>Hệ Thống Chẩn Đoán Bệnh Tật</h2>
    </div>

    <div class="section">
        <h2>📈 Tổng Quan Dữ Liệu</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{df_shape[0]}</div>
                <div>Tổng số mẫu</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{df_shape[1]-1}</div>
                <div>Số triệu chứng</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(disease_counts)}</div>
                <div>Số bệnh duy nhất</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{train_shape[0]}</div>
                <div>Mẫu huấn luyện</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{test_shape[0]}</div>
                <div>Mẫu kiểm tra</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🏥 Phân Tích Bệnh</h2>
        <p>Dữ liệu chứa {len(disease_counts)} loại bệnh khác nhau. Phân bố số mẫu cho mỗi bệnh không đồng đều.</p>
        <img src="disease_distribution.png" alt="Phân bố bệnh">
        <h3>Top 10 Bệnh Theo Số Mẫu</h3>
        <table>
            <tr>
                <th>Bệnh</th>
                <th>Số Mẫu</th>
            </tr>
"""

    for disease, count in disease_counts.head(10).items():
        html_content += f"""
            <tr>
                <td>{disease}</td>
                <td>{count}</td>
            </tr>"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>🩺 Phân Tích Triệu Chứng</h2>
        <p>Các triệu chứng có tần suất xuất hiện khác nhau trong dữ liệu.</p>
        <img src="symptom_frequency.png" alt="Tần suất triệu chứng">
        <h3>Top 10 Triệu Chứng Phổ Biến Nhất</h3>
        <table>
            <tr>
                <th>Triệu Chứng</th>
                <th>Tần Suất</th>
            </tr>
"""

    for symptom, freq in symptom_sums.head(10).items():
        html_content += f"""
            <tr>
                <td>{symptom}</td>
                <td>{freq}</td>
            </tr>"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>🔄 Tiền Xử Lý Dữ Liệu</h2>
        <h3>Các Bước Đã Thực Hiện:</h3>
        <ul>
            <li><strong>Tách Features và Target:</strong> Tách các cột triệu chứng (X) và cột bệnh (y)</li>
            <li><strong>Mã Hóa Nhãn:</strong> Chuyển đổi tên bệnh thành số sử dụng LabelEncoder</li>
            <li><strong>Lưu Artifacts:</strong> Lưu label_encoder và feature_list để sử dụng sau</li>
            <li><strong>Chia Dữ Liệu:</strong> Chia thành tập huấn luyện (80%) và kiểm tra (20%) với stratify</li>
        </ul>
        <div class="conclusion">
            <strong>Kết Luận:</strong> Dữ liệu đã được chuẩn bị sẵn sàng cho việc huấn luyện mô hình. Các artifacts quan trọng đã được lưu trữ để đảm bảo tính nhất quán trong quá trình dự đoán.
        </div>
    </div>
</body>
</html>
"""

    # Create reports directory
    os.makedirs('../reports', exist_ok=True)
    with open('../reports/data_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("\nHTML report generated: ../reports/data_analysis_report.html")

def main():
    # Create reports directory
    os.makedirs('../reports', exist_ok=True)

    # Load data
    df = load_data()

    # Perform EDA
    disease_counts, symptom_sums = perform_eda(df)

    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    # Generate HTML report
    generate_report(disease_counts, symptom_sums, df.shape, X_train.shape, X_test.shape)

    print("\n✅ Bước 1 & 2 hoàn thành!")
    print("📊 Báo cáo HTML: reports/data_analysis_report.html")
    print("💾 Dữ liệu đã xử lý lưu trong: data/processed/")
    print("🗂️  Artifacts lưu trong: models/")

if __name__ == "__main__":
    main()