"""
Model B: Generate and evaluate distractors for multiple-choice questions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from src.utils import log_message

def extract_candidate_phrases(article: str, question: str, correct_answer: str) -> List[str]:
    """
    Extract candidate noun phrases (simple: all unique words length > 2, exclude correct answer).
    
    Args:
        article: Article text
        question: Question text
        correct_answer: Correct answer string
        
    Returns:
        List of candidate phrases
    """
    # Combine article and question
    text = f"{article} {question}".lower()
    
    # Split into words and filter
    words = text.split()
    candidates = []
    
    for word in words:
        # Remove punctuation
        clean_word = ''.join(c for c in word if c.isalnum() or c == '-')
        
        # Keep if: length > 2, not in correct answer, and not trivial
        if (len(clean_word) > 2 and 
            clean_word != correct_answer.lower() and
            clean_word not in ['the', 'and', 'for', 'are', 'was', 'with', 'from', 'that']):
            candidates.append(clean_word)
    
    # Return unique candidates
    return list(set(candidates))

def generate_distractors(article: str, 
                        question: str, 
                        correct_answer: str, 
                        tfidf_vectorizer, 
                        top_n: int = 3, 
                        diversity_penalty: float = 0.3) -> List[str]:
    """
    Generate distractors using TF-IDF similarity with diversity penalty.
    
    Args:
        article: Article text
        question: Question text
        correct_answer: Correct answer string
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        top_n: Number of distractors to return
        diversity_penalty: Penalty for selecting similar distractors
        
    Returns:
        List of generated distractor strings
    """
    # Extract candidates
    candidates = extract_candidate_phrases(article, question, correct_answer)
    
    if len(candidates) < top_n:
        log_message(f"Warning: Only {len(candidates)} candidates for {top_n} distractors")
        return candidates[:top_n]
    
    # Vectorize correct answer and candidates
    correct_answer_lower = correct_answer.lower()
    all_texts = [correct_answer_lower] + candidates
    
    try:
        vectors = tfidf_vectorizer.transform(all_texts)
    except:
        log_message("Warning: Could not vectorize distractors")
        return candidates[:top_n]
    
    correct_vector = vectors[0]
    candidate_vectors = vectors[1:]
    
    # Compute similarities
    similarities = cosine_similarity(correct_vector, candidate_vectors)[0]
    
    # Iteratively select distractors with diversity penalty
    selected_distractors = []
    used_indices = set()
    
    for _ in range(top_n):
        if len(used_indices) >= len(candidates):
            break
        
        # Compute adjusted scores
        adjusted_scores = similarities.copy()
        
        for used_idx in used_indices:
            # Compute similarity between candidate and already selected
            sim_to_selected = cosine_similarity(
                candidate_vectors[used_idx:used_idx+1], 
                candidate_vectors
            )[0]
            # Penalize similar candidates
            adjusted_scores -= diversity_penalty * sim_to_selected
        
        # Select best candidate not yet used
        valid_indices = [i for i in range(len(candidates)) if i not in used_indices]
        best_idx = valid_indices[np.argmax(adjusted_scores[valid_indices])]
        
        selected_distractors.append(candidates[best_idx])
        used_indices.add(best_idx)
    
    return selected_distractors

def evaluate_distractors(test_df: pd.DataFrame, tfidf_vectorizer) -> Dict:
    """
    Evaluate generated distractors using ground-truth wrong options.
    
    Args:
        test_df: Test dataframe with columns: article, question, answer, A, B, C, D
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Dictionary with Precision@3 and Recall@3
    """
    log_message("Evaluating distractors on test set...")
    
    correct_matches = 0
    total_samples = 0
    
    for idx, row in test_df.iterrows():
        article = row['article']
        question = row['question']
        correct_answer = row['answer']
        
        # Get ground-truth wrong options
        ground_truth_wrong = {
            row['A'], row['B'], row['C'], row['D']
        }
        ground_truth_wrong.discard(correct_answer)
        
        if len(ground_truth_wrong) == 0:
            continue
        
        # Generate distractors
        generated = generate_distractors(
            article, question, correct_answer, tfidf_vectorizer, top_n=3
        )
        
        # Compute matches
        matches = len(set(generated) & ground_truth_wrong)
        correct_matches += min(matches, 3)  # Cap at 3
        total_samples += 1
    
    precision_at_3 = correct_matches / (3 * total_samples) if total_samples > 0 else 0
    recall_at_3 = correct_matches / (3 * total_samples) if total_samples > 0 else 0
    
    metrics = {
        'Precision@3': round(precision_at_3, 4),
        'Recall@3': round(recall_at_3, 4)
    }
    
    print("\n" + "="*60)
    print("DISTRACTORS EVALUATION")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    return metrics
