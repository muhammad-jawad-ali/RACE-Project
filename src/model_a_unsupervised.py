"""
Unsupervised learning: K-Means clustering with analysis and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import joblib
import os
from typing import Dict, List, Tuple
from src.utils import log_message, save_model, ensure_directory

class KMeansClustering:
    """
    K-Means clustering on TF-IDF features with automatic cluster selection.
    """
    
    def __init__(self, min_clusters: int = 5, max_clusters: int = 15, random_state: int = 42):
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
        Evaluate clustering using silhouette score and adjusted rand index.
        
        Args:
            X_val: Validation features
            y_val: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        cluster_labels = self.kmeans.predict(X_val)
        
        sil_score = silhouette_score(X_val, cluster_labels, sample_size=min(500, len(y_val)))
        ari_score = adjusted_rand_score(y_val, cluster_labels)
        
        metrics = {
            'Silhouette Score': round(sil_score, 4),
            'Adjusted Rand Index': round(ari_score, 4),
            'Num Clusters': self.best_k
        }
        
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING EVALUATION")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
        
        return metrics
    
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
