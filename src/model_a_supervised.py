"""
Supervised learning models: Logistic Regression, SVM, Naive Bayes, and ensemble voting.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import joblib
import os
from typing import Dict, Tuple
from src.utils import log_message, save_model, ensure_directory


class SupervisedModels:
    """
    Trains and evaluates supervised models (LR, SVM, NB, and ensemble).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize supervised models.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.lr = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            multi_class='multinomial'
        )
        # Wrap LinearSVC with CalibratedClassifierCV so it supports predict_proba
        self.svm = CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                random_state=random_state,
                dual=False
            ),
            cv=3
        )
        self.nb = MultinomialNB(alpha=1.0)
        
        # Soft-voting ensemble across all three classifiers
        self.ensemble = VotingClassifier(
            estimators=[
                ('lr', self.lr),
                ('svm', self.svm),
                ('nb', self.nb)
            ],
            voting='soft'
        )
        self.metrics = {}
        self.confusion_matrices = {}
        
    def train(self, X_train, y_train):
        """
        Train all models on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        log_message("Training Logistic Regression...")
        self.lr.fit(X_train, y_train)
        
        log_message("Training Calibrated Linear SVM...")
        self.svm.fit(X_train, y_train)
        
        log_message("Training Multinomial Naive Bayes...")
        self.nb.fit(X_train, y_train)
        
        log_message("Training Soft Voting Ensemble (LR + SVM + NB)...")
        self.ensemble.fit(X_train, y_train)
    
    def evaluate(self, X_val, y_val) -> pd.DataFrame:
        """
        Evaluate all models on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            DataFrame with metrics for each model
        """
        metrics_dict = {
            'Model': [],
            'Accuracy': [],
            'Precision (Macro)': [],
            'Recall (Macro)': [],
            'Macro F1': [],
            'Weighted F1': []
        }
        
        models = {
            'Logistic Regression': self.lr,
            'Linear SVM': self.svm,
            'Naive Bayes': self.nb,
            'Soft Voting Ensemble': self.ensemble
        }
        
        for model_name, model in models.items():
            log_message(f"Evaluating {model_name}...")
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            weighted_f1 = f1_score(y_val, y_pred, average='weighted')
            
            metrics_dict['Model'].append(model_name)
            metrics_dict['Accuracy'].append(round(accuracy, 4))
            metrics_dict['Precision (Macro)'].append(round(prec, 4))
            metrics_dict['Recall (Macro)'].append(round(rec, 4))
            metrics_dict['Macro F1'].append(round(macro_f1, 4))
            metrics_dict['Weighted F1'].append(round(weighted_f1, 4))
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            self.confusion_matrices[model_name] = cm.tolist()
            
            print(f"\n{model_name} Classification Report:")
            print(classification_report(y_val, y_pred))
            print(f"Confusion Matrix:\n{cm}\n")
        
        metrics_df = pd.DataFrame(metrics_dict)
        self.metrics = metrics_dict
        
        print("\n" + "="*60)
        print("SUPERVISED MODELS COMPARISON")
        print("="*60)
        print(metrics_df.to_string(index=False))
        print("="*60 + "\n")
        
        return metrics_df
    
    def save_models(self, output_dir: str):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        ensure_directory(output_dir)
        save_model(self.lr, os.path.join(output_dir, 'logistic_regression.pkl'))
        save_model(self.svm, os.path.join(output_dir, 'linear_svm.pkl'))
        save_model(self.nb, os.path.join(output_dir, 'naive_bayes.pkl'))
        save_model(self.ensemble, os.path.join(output_dir, 'voting_ensemble.pkl'))
        
        # Save confusion matrices
        import json
        cm_path = os.path.join(output_dir, 'confusion_matrices.json')
        with open(cm_path, 'w') as f:
            json.dump(self.confusion_matrices, f, indent=2)
        log_message(f"Confusion matrices saved to {cm_path}")


def train_and_evaluate_supervised(X_train, y_train, X_val, y_val, 
                                  output_dir: str) -> pd.DataFrame:
    """
    Convenience function to train and evaluate supervised models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save models
        
    Returns:
        DataFrame with metrics
    """
    supervisor = SupervisedModels()
    supervisor.train(X_train, y_train)
    metrics_df = supervisor.evaluate(X_val, y_val)
    supervisor.save_models(output_dir)
    
    return metrics_df
