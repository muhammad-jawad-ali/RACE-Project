"""
Utility functions for data loading, feature engineering, and model persistence.
"""

import os
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Any

logger_enabled = True

def log_message(msg: str):
    """Simple logging utility."""
    if logger_enabled:
        print(f"[LOG] {msg}")

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets from CSV files.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Raises:
        FileNotFoundError: If any required CSV is missing
    """
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val file not found at {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    log_message(f"Loaded train: {len(train_df)} rows")
    log_message(f"Loaded val: {len(val_df)} rows")
    log_message(f"Loaded test: {len(test_df)} rows")
    
    return train_df, val_df, test_df

def create_feature_text(row: pd.Series) -> str:
    """
    Combine article, question, and all answer options into a single feature text.
    
    Expected columns in row: article, question, A, B, C, D
    
    Args:
        row: A pandas Series (row from DataFrame)
        
    Returns:
        Combined feature text string
    """
    article = str(row.get('article', ''))
    question = str(row.get('question', ''))
    option_a = str(row.get('A', ''))
    option_b = str(row.get('B', ''))
    option_c = str(row.get('C', ''))
    option_d = str(row.get('D', ''))
    
    feature_text = f"{article} {question} {option_a} {option_b} {option_c} {option_d}"
    return feature_text.strip()

def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk using joblib.
    
    Args:
        model: The model object to save
        path: File path where model should be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log_message(f"Model saved to {path}")

def load_model(path: str) -> Any:
    """
    Load a model from disk using joblib.
    
    Args:
        path: File path of saved model
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    model = joblib.load(path)
    log_message(f"Model loaded from {path}")
    return model

def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

# ────────────────────────────────────────────────────────────
# GENERATION METRICS (BLEU, ROUGE-L, METEOR)
# ────────────────────────────────────────────────────────────

from collections import Counter
import numpy as np

def compute_bleu(reference_toks: list, hypothesis_toks: list, n: int = 1) -> float:
    """
    Simple n-gram BLEU between two token lists (unigram by default).
    """
    if len(hypothesis_toks) == 0:
        return 0.0
    ref_ngrams = Counter(
        tuple(reference_toks[i:i + n]) for i in range(len(reference_toks) - n + 1))
    hyp_ngrams = Counter(
        tuple(hypothesis_toks[i:i + n]) for i in range(len(hypothesis_toks) - n + 1))
    overlap = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
                  for ng in hyp_ngrams)
    return overlap / max(sum(hyp_ngrams.values()), 1)

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L (longest common subsequence based F-measure).
    """
    ref_toks = reference.lower().split()
    hyp_toks = hypothesis.lower().split()
    m, n = len(ref_toks), len(hyp_toks)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_toks[i - 1] == hyp_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_meteor(reference: str, hypothesis: str) -> float:
    """
    Simplified METEOR: unigram F-measure with 0.9 weight on recall.
    """
    ref_toks = set(reference.lower().split())
    hyp_toks = set(hypothesis.lower().split())
    if not ref_toks or not hyp_toks:
        return 0.0
    matches = len(ref_toks & hyp_toks)
    prec = matches / len(hyp_toks)
    rec = matches / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    alpha = 0.9
    return (prec * rec) / (alpha * prec + (1 - alpha) * rec)
