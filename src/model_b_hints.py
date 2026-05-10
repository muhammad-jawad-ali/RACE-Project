"""
Model B: Extract and evaluate hints from articles.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from src.utils import log_message

def extract_hints(article: str, 
                 question: str, 
                 tfidf_vectorizer, 
                 num_hints: int = 3) -> List[str]:
    """
    Extract hints (relevant sentences) from article using TF-IDF similarity to question.
    
    Args:
        article: Article text
        question: Question text
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        num_hints: Number of hints to extract
        
    Returns:
        List of hint sentences (sorted by relevance descending)
    """
    # Split article into sentences (simple split on ". ")
    sentences = [s.strip() for s in article.split('. ') if s.strip()]
    
    if len(sentences) == 0:
        return []
    
    # Vectorize question and sentences
    try:
        texts_to_vectorize = [question] + sentences
        vectors = tfidf_vectorizer.transform(texts_to_vectorize)
    except:
        log_message("Warning: Could not vectorize hints")
        return sentences[:num_hints]
    
    question_vector = vectors[0]
    sentence_vectors = vectors[1:]
    
    # Compute similarities
    similarities = cosine_similarity(question_vector, sentence_vectors)[0]
    
    # Sort sentences by similarity (descending)
    sorted_indices = np.argsort(-similarities)
    
    # Extract top hints
    hints = [sentences[i] for i in sorted_indices[:num_hints]]
    
    return hints

def evaluate_hints(test_df: pd.DataFrame, tfidf_vectorizer) -> Dict:
    """
    Evaluate hint quality (simplified: just report average hint similarity).
    
    Args:
        test_df: Test dataframe
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Dictionary with evaluation metrics
    """
    log_message("Evaluating hints on test set...")
    
    total_similarity = 0
    total_samples = 0
    
    for idx, row in test_df.iterrows():
        article = row['article']
        question = row['question']
        
        hints = extract_hints(article, question, tfidf_vectorizer, num_hints=3)
        
        if len(hints) > 0:
            # Vectorize
            try:
                vectors = tfidf_vectorizer.transform([question] + hints)
                question_vector = vectors[0]
                hint_vectors = vectors[1:]
                similarities = cosine_similarity(question_vector, hint_vectors)[0]
                total_similarity += np.mean(similarities)
                total_samples += 1
            except:
                pass
    
    avg_similarity = total_similarity / total_samples if total_samples > 0 else 0
    
    metrics = {
        'Average Hint Similarity': round(avg_similarity, 4)
    }
    
    print("\n" + "="*60)
    print("HINTS EVALUATION")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    return metrics
