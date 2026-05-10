"""
Unsupervised learning: K-Means clustering with analysis, evaluation,
and comparison against supervised models.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import Counter
import joblib
import os
from typing import Dict, List, Tuple
from src.utils import log_message, save_model, ensure_directory


def clustering_purity(y_true, cluster_labels) -> float:
    """
    Compute clustering purity.
    
    For each cluster, assign it the label of the majority class inside that
    cluster.  Purity = (total correctly assigned) / N.
    
    Args:
        y_true: Ground-truth labels (ints)
        cluster_labels: Predicted cluster assignments
        
    Returns:
        Purity score in [0, 1]
    """
    contingency = {}
    for cl, gt in zip(cluster_labels, y_true):
        contingency.setdefault(cl, Counter())
        contingency[cl][gt] += 1
    total_correct = sum(counter.most_common(1)[0][1] for counter in contingency.values())
    return total_correct / len(y_true)


class KMeansClustering:
    """
    K-Means clustering on TF-IDF features with automatic cluster selection.
    """
    
    def __init__(self, min_clusters: int = 4, max_clusters: int = 10, random_state: int = 42):
        """
        Initialize K-Means clustering.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            random_state: Random state for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.kmeans = None
        self.best_k = None
        self.silhouette_scores = {}
        
    def find_optimal_clusters(self, X_train, metric: str = 'cosine'):
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            X_train: Training features (sparse matrix)
            metric: Distance metric ('cosine' or 'euclidean')
        """
        log_message(f"Finding optimal clusters (k={self.min_clusters} to {self.max_clusters})...")
        
        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans_temp = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = kmeans_temp.fit_predict(X_train)
            
            # Compute silhouette score
            sil_score = silhouette_score(X_train, cluster_labels, sample_size=500)
            self.silhouette_scores[k] = sil_score
            log_message(f"k={k}: silhouette_score={sil_score:.4f}")
        
        # Select best k
        self.best_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        log_message(f"Optimal number of clusters: {self.best_k}")
    
    def train(self, X_train):
        """
        Train K-Means with optimal number of clusters.
        
        Args:
            X_train: Training features
        """
        log_message(f"Training K-Means with k={self.best_k}...")
        self.kmeans = KMeans(
            n_clusters=self.best_k,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X_train)
        
    def evaluate(self, X_val, y_val) -> Dict:
        """
        Evaluate clustering using silhouette score, adjusted rand index,
        and clustering purity.
        
        Args:
            X_val: Validation features
            y_val: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        cluster_labels = self.kmeans.predict(X_val)
        
        sil_score = silhouette_score(X_val, cluster_labels, sample_size=min(500, len(y_val)))
        ari_score = adjusted_rand_score(y_val, cluster_labels)
        purity = clustering_purity(y_val, cluster_labels)
        
        metrics = {
            'Silhouette Score': round(sil_score, 4),
            'Adjusted Rand Index': round(ari_score, 4),
            'Clustering Purity': round(purity, 4),
            'Num Clusters': self.best_k
        }
        
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING EVALUATION")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
        
        return metrics
    
    def compare_with_supervised(self, supervised_metrics: list, clustering_metrics: dict):
        """
        Print a comparison table between supervised and unsupervised results.
        
        Args:
            supervised_metrics: List of dicts from supervised evaluation
            clustering_metrics: Dict from clustering evaluation
        """
        print("\n" + "="*70)
        print("SUPERVISED vs. UNSUPERVISED COMPARISON")
        print("="*70)
        print(f"{'Approach':<25} {'Metric':<20} {'Value':<10}")
        print("-"*55)
        for m in supervised_metrics:
            print(f"{m['Model']:<25} {'Accuracy':<20} {m['Accuracy']:<10}")
        print(f"{'K-Means Clustering':<25} {'Purity':<20} {clustering_metrics['Clustering Purity']:<10}")
        print(f"{'K-Means Clustering':<25} {'Silhouette':<20} {clustering_metrics['Silhouette Score']:<10}")
        print(f"{'K-Means Clustering':<25} {'ARI':<20} {clustering_metrics['Adjusted Rand Index']:<10}")
        print("="*70)
        print("Note: Supervised models significantly outperform unsupervised K-Means,")
        print("which is expected since K-Means lacks label information during training.")
        print("="*70 + "\n")
    
    def analyze_centroids(self, tfidf_vectorizer, top_n: int = 10):
        """
        Analyze cluster centroids and print top terms.
        
        Args:
            tfidf_vectorizer: Fitted TF-IDF vectorizer
            top_n: Number of top terms to display per cluster
        """
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        print("\n" + "="*60)
        print("CLUSTER CENTROIDS ANALYSIS")
        print("="*60)
        
        for cluster_id in range(self.best_k):
            centroid = self.kmeans.cluster_centers_[cluster_id]
            
            # Get indices of top terms
            top_indices = centroid.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            print(f"\nCluster {cluster_id} (top {top_n} terms):")
            print(", ".join(top_terms))
        
        print("\n" + "="*60 + "\n")
    
    def save_model(self, output_dir: str):
        """
        Save trained K-Means model.
        
        Args:
            output_dir: Directory to save model
        """
        ensure_directory(output_dir)
        save_model(self.kmeans, os.path.join(output_dir, 'kmeans_model.pkl'))
        
        # Also save silhouette scores
        joblib.dump(self.silhouette_scores, os.path.join(output_dir, 'silhouette_scores.pkl'))


def train_and_evaluate_clustering(X_train, X_val, y_val, tfidf_vectorizer, 
                                  output_dir: str) -> Dict:
    """
    Convenience function to train and evaluate K-Means clustering.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_val: Validation labels
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        output_dir: Directory to save models
        
    Returns:
        Dictionary with evaluation metrics
    """
    clusterer = KMeansClustering()
    clusterer.find_optimal_clusters(X_train)
    clusterer.train(X_train)
    metrics = clusterer.evaluate(X_val, y_val)
    clusterer.analyze_centroids(tfidf_vectorizer)
    clusterer.save_model(output_dir)
    
    return metrics
