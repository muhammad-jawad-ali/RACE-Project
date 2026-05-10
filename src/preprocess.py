"""
Preprocessing module: TF-IDF vectorization and label encoding.
"""

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
    Handles TF-IDF vectorization and label encoding for the QA dataset.
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
            stop_words=stop_words
        )
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, 
                     train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame) -> Tuple:
        """
        Fit TF-IDF and label encoder on training data, transform all datasets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Create feature text for all datasets
        log_message("Creating feature texts...")
        train_df['feature_text'] = train_df.apply(create_feature_text, axis=1)
        val_df['feature_text'] = val_df.apply(create_feature_text, axis=1)
        test_df['feature_text'] = test_df.apply(create_feature_text, axis=1)
        
        # Fit and transform training data
        log_message("Fitting TF-IDF vectorizer on training data...")
        X_train = self.tfidf.fit_transform(train_df['feature_text'])
        
        # Transform validation and test data
        log_message("Transforming validation and test data...")
        X_val = self.tfidf.transform(val_df['feature_text'])
        X_test = self.tfidf.transform(test_df['feature_text'])
        
        # Encode labels (A=0, B=1, C=2, D=3)
        log_message("Encoding labels...")
        y_train = self.label_encoder.fit_transform(train_df['answer'])
        y_val = self.label_encoder.transform(val_df['answer'])
        y_test = self.label_encoder.transform(test_df['answer'])
        
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
