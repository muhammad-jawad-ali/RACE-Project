import os
import json
import joblib
import pandas as pd
from src.utils import load_data, log_message, ensure_directory
from src.model_a_supervised import train_and_evaluate_supervised
from src.model_a_unsupervised import train_and_evaluate_clustering
from src.model_b_distractors import evaluate_distractors
from src.model_b_hints import evaluate_hints

def main():
    # 1. Setup paths
    DATA_DIR = "data/processed"
    MODELS_DIR = "models"
    ensure_directory(MODELS_DIR)

    # 2. Load data
    log_message("Loading processed datasets...")
    train_df, val_df, test_df = load_data(DATA_DIR)

    # 3. Load preprocessing artifacts (assumed to exist)
    log_message("Loading preprocessing artifacts...")
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
    # Load pre-vectorized features
    X_train = joblib.load(os.path.join(MODELS_DIR, "X_train.pkl"))
    X_val = joblib.load(os.path.join(MODELS_DIR, "X_val.pkl"))
    X_test = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(MODELS_DIR, "y_train.pkl"))
    y_val = joblib.load(os.path.join(MODELS_DIR, "y_val.pkl"))
    y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))

    # 4. Train and Evaluate Supervised Models
    log_message("--- Training Supervised Models ---")
    supervised_metrics_df = train_and_evaluate_supervised(X_train, y_train, X_val, y_val, MODELS_DIR)
    supervised_metrics = supervised_metrics_df.to_dict(orient='records')

    # 5. Train and Evaluate Unsupervised Models (K-Means)
    log_message("--- Training Unsupervised Models ---")
    clustering_metrics = train_and_evaluate_clustering(X_train, X_val, y_val, tfidf, MODELS_DIR)

    # 6. Evaluate Model B (Distractors and Hints)
    log_message("--- Evaluating Distractors ---")
    distractor_metrics = evaluate_distractors(test_df, tfidf)

    log_message("--- Evaluating Hints ---")
    hint_metrics = evaluate_hints(test_df, tfidf)

    # 7. Save combined metrics for the dashboard
    all_metrics = {
        "supervised": supervised_metrics,
        "clustering": clustering_metrics,
        "distractors": distractor_metrics,
        "hints": hint_metrics
    }
    
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    log_message(f"All models trained and metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
