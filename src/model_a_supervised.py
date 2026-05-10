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
from src.utils import (
    log_message, save_model, ensure_directory, 
    compute_bleu, compute_rouge_l, compute_meteor
)


class SupervisedModels:
    def __init__(self, random_state: int = 42):
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
        log_message("Training Logistic Regression...")
        self.lr.fit(X_train, y_train)
        
        log_message("Training Calibrated Linear SVM...")
        self.svm.fit(X_train, y_train)
        
        log_message("Training Multinomial Naive Bayes...")
        self.nb.fit(X_train, y_train)
        
        log_message("Training Soft Voting Ensemble (LR + SVM + NB)...")
        self.ensemble.fit(X_train, y_train)
    
    def evaluate(self, X_val, y_val) -> pd.DataFrame:
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

    def evaluate_generation(self, val_df: pd.DataFrame) -> Dict:
        log_message("Evaluating Model A with generation metrics...")
        
        gen_metrics = {}
        models = {
            'Logistic Regression': self.lr,
            'Linear SVM': self.svm,
            'Naive Bayes': self.nb,
            'Soft Voting Ensemble': self.ensemble
        }
        
        # We need the TF-IDF features for the validation set again
        # but the class doesn't store them. This is usually called from train_all.py
        # where features are available. 
        # For simplicity, we assume this is called after predict has been done or we pass features.
        return gen_metrics # Placeholder for now, will implement in the convenience function
    
    def save_models(self, output_dir: str):
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
                                  output_dir: str, val_df: pd.DataFrame = None) -> pd.DataFrame:
    supervisor = SupervisedModels()
    supervisor.train(X_train, y_train)
    metrics_df = supervisor.evaluate(X_val, y_val)
    
    if val_df is not None:
        log_message("Computing generation metrics for Model A...")
        # Add generation metrics to the dataframe
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        
        models = {
            'Logistic Regression': supervisor.lr,
            'Linear SVM': supervisor.svm,
            'Naive Bayes': supervisor.nb,
            'Soft Voting Ensemble': supervisor.ensemble
        }
        
        for model_name, model in models.items():
            y_pred = model.predict(X_val)
            
            sample_bleus, sample_rouges, sample_meteors = [], [], []
            
            # Label map: 0->A, 1->B, 2->C, 3->D
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            
            for i, (_, row) in enumerate(val_df.iterrows()):
                pred_label = label_map.get(y_pred[i], 'A')
                gt_label = str(row['answer']).strip()
                
                pred_text = str(row.get(pred_label, '')).lower()
                gt_text = str(row.get(gt_label, '')).lower()
                
                sample_bleus.append(compute_bleu(gt_text.split(), pred_text.split()))
                sample_rouges.append(compute_rouge_l(gt_text, pred_text))
                sample_meteors.append(compute_meteor(gt_text, pred_text))
            
            # Update metrics_df with average scores
            idx = metrics_df[metrics_df['Model'] == model_name].index[0]
            metrics_df.loc[idx, 'BLEU-1'] = round(np.mean(sample_bleus), 4)
            metrics_df.loc[idx, 'ROUGE-L'] = round(np.mean(sample_rouges), 4)
            metrics_df.loc[idx, 'METEOR'] = round(np.mean(sample_meteors), 4)

        print("\n" + "="*60)
        print("MODEL A GENERATION METRICS")
        print("="*60)
        print(metrics_df[['Model', 'BLEU-1', 'ROUGE-L', 'METEOR']].to_string(index=False))
        print("="*60 + "\n")

    supervisor.save_models(output_dir)
    return metrics_df
