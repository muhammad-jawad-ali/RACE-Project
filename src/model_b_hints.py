"""
Model B: Extract and evaluate graduated hints from articles.

Graduated Hint Levels:
  Hint 1 (General)       — least relevant sentence (broad context)
  Hint 2 (Specific)      — moderately relevant sentence
  Hint 3 (Near-Explicit) — the sentence most closely related to the answer
"""

import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
from src.utils import log_message


# ────────────────────────────────────────────────────────────
# Sentence-level feature helpers
# ────────────────────────────────────────────────────────────

def _keyword_overlap(sentence: str, question: str) -> float:
    """Fraction of question content-words that appear in the sentence."""
    stop = {'the','a','an','is','are','was','were','of','in','to','and','or',
            'for','on','at','by','it','he','she','they','this','that','with'}
    q_words = set(question.lower().split()) - stop
    s_words = set(sentence.lower().split()) - stop
    if not q_words:
        return 0.0
    return len(q_words & s_words) / len(q_words)


def _sentence_position(idx: int, total: int) -> float:
    """Normalised sentence position (0 = first, 1 = last)."""
    return idx / max(total - 1, 1)


def _sentence_length(sentence: str) -> int:
    """Number of tokens in the sentence."""
    return len(sentence.split())


# ────────────────────────────────────────────────────────────
# Core hint extraction
# ────────────────────────────────────────────────────────────

def extract_hints(article: str,
                  question: str,
                  tfidf_vectorizer,
                  correct_answer: str = "",
                  num_hints: int = 3) -> List[str]:
    """
    Extract **graduated** hints from the article.

    Scoring uses TF-IDF cosine similarity to the question, combined with
    keyword overlap, sentence position, and sentence length as features.

    Returns:
        hints[0] = general (low relevance)
        hints[1] = specific (medium relevance)
        hints[2] = near-explicit (highest relevance / closest to answer)
    """
    # Split article into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', article) if s.strip()]

    if len(sentences) == 0:
        return []

    # ── TF-IDF cosine similarity ──
    try:
        texts = [question] + sentences
        vectors = tfidf_vectorizer.transform(texts)
        question_vec = vectors[0]
        sentence_vecs = vectors[1:]
        cos_sims = cosine_similarity(question_vec, sentence_vecs)[0]
    except Exception:
        cos_sims = np.zeros(len(sentences))

    # ── Keyword-overlap & positional features ──
    kw_scores = np.array([_keyword_overlap(s, question) for s in sentences])
    pos_scores = np.array([_sentence_position(i, len(sentences))
                           for i in range(len(sentences))])
    len_scores = np.array([_sentence_length(s) for s in sentences])
    # Normalise length
    max_len = max(len_scores.max(), 1)
    len_norm = len_scores / max_len

    # ── Combined score (weighted) ──
    combined = 0.5 * cos_sims + 0.25 * kw_scores + 0.15 * len_norm + 0.10 * pos_scores

    # If we have the correct answer text, boost sentences that contain it
    if correct_answer:
        ans_lower = correct_answer.lower()
        for i, s in enumerate(sentences):
            if ans_lower in s.lower():
                combined[i] += 0.3  # heavy boost

    # Rank sentences by combined score
    ranked_indices = np.argsort(combined)  # ascending

    # ── Build graduated hints ──
    if len(sentences) < 3:
        # Not enough sentences — return what we have, least-relevant first
        return [sentences[i] for i in ranked_indices]

    # Hint 1 (General): pick from the lower third of relevance
    low_third = ranked_indices[:max(len(ranked_indices) // 3, 1)]
    hint_1_idx = low_third[-1]  # best of the low third

    # Hint 3 (Near-Explicit): highest relevance
    hint_3_idx = ranked_indices[-1]

    # Hint 2 (Specific): median relevance (avoid repeating hint 1 / 3)
    mid = len(ranked_indices) // 2
    hint_2_idx = ranked_indices[mid]
    if hint_2_idx == hint_1_idx or hint_2_idx == hint_3_idx:
        # Try neighbour
        for offset in (1, -1, 2, -2):
            alt = mid + offset
            if 0 <= alt < len(ranked_indices) and ranked_indices[alt] not in (hint_1_idx, hint_3_idx):
                hint_2_idx = ranked_indices[alt]
                break

    hints = [
        sentences[hint_1_idx],  # general
        sentences[hint_2_idx],  # specific
        sentences[hint_3_idx],  # near-explicit
    ]

    return hints


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────

def evaluate_hints(test_df: pd.DataFrame, tfidf_vectorizer) -> Dict:
    """
    Evaluate hint quality:
      - Average cosine similarity of each hint level to the question
      - Graduation check: Hint 3 should be more relevant than Hint 1
    """
    log_message("Evaluating hints on test set...")

    sims_by_level = {0: [], 1: [], 2: []}
    graduation_correct = 0
    total_samples = 0

    for _, row in test_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        ans_letter = str(row['answer']).strip()
        correct_text = str(row.get(ans_letter, ''))

        hints = extract_hints(article, question, tfidf_vectorizer,
                              correct_answer=correct_text, num_hints=3)

        if len(hints) < 3:
            continue

        try:
            vecs = tfidf_vectorizer.transform([question] + hints)
            q_vec = vecs[0]
            h_vecs = vecs[1:]
            sims = cosine_similarity(q_vec, h_vecs)[0]
        except Exception:
            continue

        for lvl in range(3):
            sims_by_level[lvl].append(sims[lvl])

        # Check graduation (hint 3 > hint 1)
        if sims[2] >= sims[0]:
            graduation_correct += 1
        total_samples += 1

    avg_sims = {lvl: round(float(np.mean(vals)), 4) if vals else 0.0
                for lvl, vals in sims_by_level.items()}

    grad_rate = round(graduation_correct / max(total_samples, 1), 4)

    metrics = {
        'Hint 1 (General) Avg Sim': avg_sims[0],
        'Hint 2 (Specific) Avg Sim': avg_sims[1],
        'Hint 3 (Near-Explicit) Avg Sim': avg_sims[2],
        'Graduation Correctness Rate': grad_rate,
        'Average Hint Similarity': round(
            float(np.mean([v for vals in sims_by_level.values() for v in vals])), 4
        ) if any(sims_by_level.values()) else 0.0
    }

    print("\n" + "=" * 60)
    print("HINTS EVALUATION (Graduated)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=" * 60 + "\n")

    return metrics
