import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# Setup logging
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to terminal
        logging.FileHandler(log_filename)  # Log to file
    ]
)

logger = logging.getLogger(__name__)

def load_data():
    """Load processed training and test data"""
    logger.info("Loading processed data...")
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')

    logger.info(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    logger.info(f"y_train {y_train.shape}, y_test {y_test.shape}")
    return X_train, X_test, y_train, y_test

def define_models():
    """Define the 3 best models for comparison"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'MultinomialNB': MultinomialNB()
    }
    logger.info("Models defined: RandomForest, LogisticRegression, MultinomialNB")
    return models

def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model and evaluate its performance"""
    logger.info(f"Training {model_name}...")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info(".2f")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    logger.info(".4f")
    logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Macro Precision: {report['macro avg']['precision']:.4f}")
    logger.info(f"Macro Recall: {report['macro avg']['recall']:.4f}")

    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'training_time': training_time,
        'y_pred': y_pred,
        'report': report
    }

def save_best_model(results):
    """Save the best performing model"""
    best_result = max(results, key=lambda x: x['accuracy'])
    model_path = f"models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_result['model'], model_path)
    logger.info(f"Best model ({best_result['model_name']}) saved to {model_path}")
    return best_result, model_path

def create_visualization_html(results, y_test):
    """Create HTML file with visualizations"""
    logger.info("Creating visualization HTML...")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Training Results - Step 3</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .best {{ background-color: #e8f5e8; border: 2px solid #4CAF50; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Model Training Results - Step 3</h1>
        <p>Training completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro F1-Score</th>
                <th>Macro Precision</th>
                <th>Macro Recall</th>
                <th>Training Time (s)</th>
            </tr>
    """

    best_accuracy = max(r['accuracy'] for r in results)

    for result in results:
        is_best = result['accuracy'] == best_accuracy
        css_class = "best" if is_best else ""
        html_content += f"""
            <tr class="{css_class}">
                <td>{result['model_name']}</td>
                <td>{result['accuracy']:.4f}</td>
                <td>{result['macro_f1']:.4f}</td>
                <td>{result['macro_precision']:.4f}</td>
                <td>{result['macro_recall']:.4f}</td>
                <td>{result['training_time']:.2f}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Detailed Classification Reports</h2>
    """

    for result in results:
        html_content += f"""
        <div class="metric">
            <h3>{result['model_name']}</h3>
            <pre>{classification_report(y_test, result['y_pred'], zero_division=0)}</pre>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    html_path = f"reports/model_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    os.makedirs('reports', exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Visualization HTML saved to {html_path}")
    return html_path

def create_md_report(results, best_result, model_path, html_path, log_filename):
    """Create MD file with detailed process description"""
    logger.info("Creating MD report...")

    md_content = f"""# Model Training Results - Step 3

## Tổng Quan

Bước 3 đã hoàn thành thành công vào {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Đây là giai đoạn so sánh và lựa chọn mô hình Machine Learning tốt nhất cho hệ thống chẩn đoán bệnh.

## Quy Trình Thực Hiện

### 1. Chuẩn Bị Dữ Liệu
- **Dữ liệu huấn luyện**: {results[0]['model'].n_features_in_} features, {len(results[0]['y_pred'])} samples trong tập test
- **Loại bài toán**: Multi-class classification với 773 classes
- **Đặc điểm dữ liệu**: 377 features nhị phân biểu diễn triệu chứng bệnh

### 2. Thuật Toán Được Chọn

#### Tại Sao Chọn 3 Thuật Toán Này?
1. **Random Forest Classifier**: Xuất sắc với tabular data, robust với overfitting
2. **Logistic Regression**: Baseline nhanh, dễ interpret
3. **Multinomial Naive Bayes**: Tốt cho high-dimensional data, memory-efficient

### 3. Tham Số Mô Hình

#### Random Forest Classifier
- `n_estimators`: 100
- `random_state`: 42
- `n_jobs`: -1 (sử dụng tất cả CPU cores)

#### Logistic Regression
- `random_state`: 42
- `max_iter`: 1000
- `n_jobs`: -1

#### Multinomial Naive Bayes
- Không có hyperparameters đặc biệt (default)

## Kết Quả So Sánh

| Model | Accuracy | Macro F1-Score | Macro Precision | Macro Recall | Training Time (s) |
|-------|----------|----------------|-----------------|--------------|-------------------|
"""

    for result in results:
        md_content += f"| {result['model_name']} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} | {result['macro_precision']:.4f} | {result['macro_recall']:.4f} | {result['training_time']:.2f} |\n"

    md_content += f"""

### Mô Hình Tốt Nhất
- **Algorithm**: {best_result['model_name']}
- **Accuracy**: {best_result['accuracy']:.4f}
- **Macro F1-Score**: {best_result['macro_f1']:.4f}
- **Được lưu tại**: `{model_path}`

## Phân Tích Chi Tiết

### Điểm Mạnh Của Mô Hình Tốt Nhất
"""

    if best_result['model_name'] == 'RandomForest':
        md_content += """
- Robust với missing values và outliers
- Tự động feature selection
- Interpretability tốt qua feature importance
- Ensemble method giảm overfitting
"""
    elif best_result['model_name'] == 'LogisticRegression':
        md_content += """
- Training và inference nhanh
- Probabilistic output
- Dễ hiểu và implement
- Feature importance rõ ràng
"""
    else:  # MultinomialNB
        md_content += """
- Rất nhanh và memory-efficient
- Ít parameters cần tune
- Robust với irrelevant features
- Tốt cho high-dimensional data
"""

    md_content += f"""
## Files Được Tạo

### Logs
- **Training log**: `{log_filename}`
- Chứa toàn bộ thông tin training process và metrics

### Reports
- **HTML Visualization**: `{html_path}`
- **MD Report**: File này

### Models
- **Best Model**: `{model_path}`
- Sử dụng joblib để serialize model

## Khuyến Nghị Tiếp Theo

### Bước 4: Hyperparameter Tuning
- Tune mô hình `{best_result['model_name']}` để cải thiện performance
- Sử dụng GridSearchCV hoặc RandomizedSearchCV
- Focus vào parameters quan trọng nhất

### Bước 5: Deployment
- Integrate model vào ứng dụng web
- Load label encoder và feature list
- Implement prediction API

## Troubleshooting (Nếu Cần)

### Memory Issues
- Giảm `n_estimators` của RandomForest
- Sử dụng mini-batch training
- Tăng virtual memory

### Convergence Issues
- Tăng `max_iter` của LogisticRegression
- Scale features nếu cần

### Poor Performance
- Kiểm tra data preprocessing
- Thử feature engineering
- Consider deep learning approaches

## Kết Luận

Bước 3 đã thành công:
- ✅ So sánh 3 thuật toán Machine Learning
- ✅ Chọn mô hình tốt nhất: **{best_result['model_name']}**
- ✅ Tạo visualizations và reports
- ✅ Lưu model và logs

Hệ thống sẵn sàng cho bước tinh chỉnh hyperparameters tiếp theo.
"""

    md_path = f"reports/model_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    logger.info(f"MD report saved to {md_path}")
    return md_path

def main():
    logger.info("Starting Model Training - Step 3")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Define models
    models = define_models()

    # Train and evaluate each model
    results = []
    for model_name, model in models.items():
        result = train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    # Save best model
    best_result, model_path = save_best_model(results)

    # Create visualizations
    html_path = create_visualization_html(results, y_test)

    # Create MD report
    md_path = create_md_report(results, best_result, model_path, html_path, log_filename)

    logger.info("Model Training - Step 3 completed successfully!")
    logger.info(f"Best model: {best_result['model_name']} with accuracy {best_result['accuracy']:.4f}")
    logger.info(f"Files created: {html_path}, {md_path}, {model_path}")

if __name__ == "__main__":
    main()
