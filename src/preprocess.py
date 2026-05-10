"""
Preprocessing Pipeline:
  1. Data Cleaning — lowercasing, punctuation removal
  2. Feature Engineering — TF-IDF vectorization with feature text
  3. Label Encoding — One-Hot / ordinal encoding of answer labels
  4. Feature Scaling — TF-IDF inherently normalizes; explicit note
  5. Data Transformation — sparse matrix output
  6. Imbalanced Data Check — class distribution analysis
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Tuple
from src.utils import log_message, create_feature_text, save_model, ensure_directory


class Preprocessor:
    """
    Handles full preprocessing pipeline for the QA dataset:
    cleaning → feature engineering → vectorization → encoding.
    """
    
    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        """
        Initialize the preprocessor.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer
            stop_words: Stop words to use ('english')
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True,          # explicit lowercasing
            token_pattern=r'(?u)\b\w\w+\b'  # word tokenization
        )
        self.label_encoder = LabelEncoder()
    
    # ──────────────────────────────────────────
    # Step 1: Data Cleaning
    # ──────────────────────────────────────────
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a text string:
          - Convert to lowercase
          - Remove punctuation (keep alphanumeric + spaces)
          - Collapse multiple whitespace
        """
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)   # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # collapse whitespace
        return text
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning to all text columns."""
        df = df.copy()
        text_cols = ['article', 'question', 'A', 'B', 'C', 'D']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self.clean_text)
        log_message(f"  Cleaned {len(df)} rows (lowercase + punctuation removal)")
        return df
    
    # ──────────────────────────────────────────
    # Step 2: Imbalanced Data Check
    # ──────────────────────────────────────────
    
    @staticmethod
    def check_class_balance(y: np.ndarray, name: str = ""):
        """Report class distribution and whether data is imbalanced."""
        counts = pd.Series(y).value_counts().sort_index()
        ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        
        print(f"\n  Class distribution ({name}):")
        for cls, cnt in counts.items():
            print(f"    Class {cls}: {cnt} ({100*cnt/len(y):.1f}%)")
        
        if ratio < 1.5:
            print(f"  → Max/Min ratio = {ratio:.2f} — Balanced. No resampling needed.")
        else:
            print(f"  → Max/Min ratio = {ratio:.2f} — Imbalanced! Consider class_weight='balanced'.")
        
        return ratio
        
    def fit_transform(self, 
                     train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame) -> Tuple:
        """
        Full preprocessing pipeline:
          1. Clean text (lowercase, punctuation removal)
          2. Create feature text
          3. TF-IDF vectorization (fit on train, transform all)
          4. Label encoding
          5. Class balance check
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Step 1: Clean text
        log_message("Step 1: Cleaning text (lowercase + punctuation removal)...")
        train_df = self.clean_dataframe(train_df)
        val_df = self.clean_dataframe(val_df)
        test_df = self.clean_dataframe(test_df)
        
        # Step 2: Feature engineering — create combined feature text
        log_message("Step 2: Creating feature texts (article + question + options)...")
        train_df['feature_text'] = train_df.apply(create_feature_text, axis=1)
        val_df['feature_text'] = val_df.apply(create_feature_text, axis=1)
        test_df['feature_text'] = test_df.apply(create_feature_text, axis=1)
        
        # Step 3: TF-IDF vectorization (inherently applies feature scaling via IDF)
        log_message("Step 3: Fitting TF-IDF vectorizer on training data (feature scaling via IDF norms)...")
        X_train = self.tfidf.fit_transform(train_df['feature_text'])
        
        log_message("  Transforming validation and test data...")
        X_val = self.tfidf.transform(val_df['feature_text'])
        X_test = self.tfidf.transform(test_df['feature_text'])
        
        # Step 4: Encode labels (A=0, B=1, C=2, D=3)
        log_message("Step 4: Encoding labels...")
        y_train = self.label_encoder.fit_transform(train_df['answer'])
        y_val = self.label_encoder.transform(val_df['answer'])
        y_test = self.label_encoder.transform(test_df['answer'])
        
        # Step 5: Class balance check
        log_message("Step 5: Checking class balance...")
        self.check_class_balance(y_train, "train")
        self.check_class_balance(y_val, "val")
        
        # Print statistics
        self._print_stats(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _print_stats(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("PREPROCESSING STATISTICS")
        print("="*60)
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"\nTrain class distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
        print(f"\nVal class distribution:\n{pd.Series(y_val).value_counts().sort_index()}")
        print(f"\nTest class distribution:\n{pd.Series(y_test).value_counts().sort_index()}")
        print(f"TF-IDF vocabulary size: {len(self.tfidf.get_feature_names_out())}")
        print(f"Feature scaling: L2-normalized TF-IDF (inherent)")
        print("="*60 + "\n")

def preprocess_and_save(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, 
                       output_dir: str) -> Tuple:
    """
    Preprocess data and save all artifacts.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        output_dir: Directory to save preprocessed data
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    ensure_directory(output_dir)
    
    preprocessor = Preprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(
        train_df, val_df, test_df
    )
    
    # Save artifacts
    log_message("Saving preprocessed data...")
    save_model(preprocessor.tfidf, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
    save_model(preprocessor.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    
    joblib.dump(X_train, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_val, os.path.join(output_dir, 'X_val.pkl'))
    joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_val, os.path.join(output_dir, 'y_val.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    
    log_message(f"All artifacts saved to {output_dir}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    from src.utils import load_data
    train_df, val_df, test_df = load_data("data/processed")
    preprocess_and_save(train_df, val_df, test_df, "models")
