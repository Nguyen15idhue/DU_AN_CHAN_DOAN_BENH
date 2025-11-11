import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os

def load_data():
    """Load processed data from step 2"""
    print("Loading processed data...")
    X_train = np.load('../data/processed/X_train.npy')
    X_test = np.load('../data/processed/X_test.npy')
    y_train = np.load('../data/processed/y_train.npy')
    y_test = np.load('../data/processed/y_test.npy')

    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_models():
    """Define the models to train"""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1  # Use all CPU cores
        ),
        'MultinomialNB': MultinomialNB()
    }
    return models

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
    print(f"\n--- Training {model_name} ---")

    # Start timing
    start_time = time.time()

    # Train model
    model.fit(X_train, y_train)

    # End timing
    train_time = time.time() - start_time

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Get classification report (detailed metrics)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract macro averages
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    print(".4f")
    print(".4f")
    print(".4f")
    print(".2f")

    return {
        'model': model,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'train_time': train_time
    }

def compare_models(results):
    """Compare and display model results"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)

    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': ".4f",
            'Macro Precision': ".4f",
            'Macro Recall': ".4f",
            'Macro F1-Score': ".4f",
            'Training Time (s)': ".2f"
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Accuracy', ascending=False)

    print(df_comparison.to_string(index=False))

    # Select best model
    best_model_name = df_comparison.iloc[0]['Model']
    best_accuracy = df_comparison.iloc[0]['Accuracy']

    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy})")

    return best_model_name, results[best_model_name]['model']

def save_best_model(best_model, model_name):
    """Save the best model"""
    os.makedirs('../models', exist_ok=True)
    filename = f'../models/{model_name.lower()}_model.joblib'
    joblib.dump(best_model, filename)
    print(f"Best model saved as: {filename}")

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Get models
    models = get_models()

    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        result = train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test)
        results[model_name] = result

    # Compare models and select best
    best_model_name, best_model = compare_models(results)

    # Save best model
    save_best_model(best_model, best_model_name)

    print("\n✅ Step 3 completed!")
    print(f"📊 Best model: {best_model_name}")
    print("💾 Model saved in: models/")

if __name__ == "__main__":
    main()