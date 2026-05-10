"""
Model B: Generate and evaluate distractors for multiple-choice questions.

Pipeline:
  1. Candidate Extraction — noun-phrase & word candidates from passage
  2. Feature Engineering — cosine similarity, char-level match, passage frequency
  3. ML Ranker (Logistic Regression) — trained to score candidates
  4. Diversity Penalty — ensures distractors are lexically distinct
  5. Evaluation — BLEU, ROUGE-L, METEOR, Precision@3
"""

import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Tuple, Dict
import joblib
import os
from src.utils import log_message, ensure_directory

# ────────────────────────────────────────────────────────────
# 1. CANDIDATE EXTRACTION
# ────────────────────────────────────────────────────────────

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'can', 'could', 'and', 'but', 'or',
    'nor', 'not', 'so', 'yet', 'for', 'at', 'by', 'to', 'from', 'in',
    'on', 'of', 'with', 'as', 'that', 'this', 'it', 'its', 'he', 'she',
    'they', 'them', 'we', 'you', 'i', 'me', 'my', 'his', 'her', 'our',
    'your', 'their', 'what', 'which', 'who', 'whom', 'than', 'very',
    'just', 'also', 'about', 'into', 'over', 'after', 'before'
}


def extract_candidate_phrases(article: str, question: str,
                               correct_answer: str) -> List[str]:
    """
    Extract candidate distractor phrases from the passage.

    Strategy:
      - Split article into sentences, then into n-grams (1–3 words).
      - Filter out stop-only phrases and the correct answer itself.
      - Return unique candidates.
    """
    text = f"{article} {question}".lower()
    # Tokenise
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)*", text)

    correct_lower = correct_answer.lower().strip()
    candidates = set()

    # Unigrams
    for tok in tokens:
        if tok not in STOP_WORDS and len(tok) > 2 and tok != correct_lower:
            candidates.add(tok)

    # Bigrams / trigrams (phrase extraction)
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            # At least one content word
            if any(t not in STOP_WORDS for t in tokens[i:i + n]):
                if gram != correct_lower:
                    candidates.add(gram)

    return list(candidates)

# ────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING FOR RANKING
# ────────────────────────────────────────────────────────────

def _char_match_score(candidate: str, correct: str) -> float:
    """Character-level overlap (Jaccard of character bigrams)."""
    def char_bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else {s}
    bg_c = char_bigrams(candidate.lower())
    bg_a = char_bigrams(correct.lower())
    inter = bg_c & bg_a
    union = bg_c | bg_a
    return len(inter) / len(union) if union else 0.0


def _passage_frequency(candidate: str, article: str) -> int:
    """Count how many times the candidate appears in the article."""
    return article.lower().count(candidate.lower())


def compute_candidate_features(candidates: List[str],
                                correct_answer: str,
                                article: str,
                                tfidf_vectorizer) -> np.ndarray:
    """
    Build a feature matrix for each candidate:
      f0: TF-IDF cosine similarity to correct answer
      f1: character-level match score
      f2: passage frequency (log-scaled)
    """
    correct_lower = correct_answer.lower()
    all_texts = [correct_lower] + [c.lower() for c in candidates]

    try:
        vectors = tfidf_vectorizer.transform(all_texts)
        cos_sims = cosine_similarity(vectors[0:1], vectors[1:])[0]
    except Exception:
        cos_sims = np.zeros(len(candidates))

    features = []
    for i, cand in enumerate(candidates):
        f0 = cos_sims[i]
        f1 = _char_match_score(cand, correct_lower)
        f2 = np.log1p(_passage_frequency(cand, article))
        features.append([f0, f1, f2])

    return np.array(features) if features else np.empty((0, 3))

# ────────────────────────────────────────────────────────────
# 3. ML RANKER
# ────────────────────────────────────────────────────────────

def train_distractor_ranker(train_df: pd.DataFrame,
                            tfidf_vectorizer,
                            output_dir: str) -> LogisticRegression:
    """
    Train a Logistic Regression ranker that predicts whether a candidate
    is a ground-truth distractor (positive) or not (negative).

    Persisted via joblib to ``output_dir/distractor_ranker.pkl``.
    """
    log_message("Building training data for distractor ranker...")
    X_all, y_all = [], []

    for _, row in train_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        ans_letter = str(row['answer']).strip()
        correct_text = str(row.get(ans_letter, ''))

        # Ground-truth distractors
        gt_distractors = set()
        for opt in ['A', 'B', 'C', 'D']:
            if opt != ans_letter:
                gt_distractors.add(str(row[opt]).lower().strip())

        candidates = extract_candidate_phrases(article, question, correct_text)
        if len(candidates) == 0:
            continue

        feats = compute_candidate_features(candidates, correct_text,
                                           article, tfidf_vectorizer)
        if feats.shape[0] == 0:
            continue

        for i, cand in enumerate(candidates):
            label = 1 if cand.lower().strip() in gt_distractors else 0
            X_all.append(feats[i])
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    log_message(f"Ranker training set: {len(X_all)} samples, "
                f"{y_all.sum()} positives")

    ranker = LogisticRegression(max_iter=500, class_weight='balanced',
                                random_state=42)
    ranker.fit(X_all, y_all)

    ensure_directory(output_dir)
    path = os.path.join(output_dir, 'distractor_ranker.pkl')
    joblib.dump(ranker, path)
    log_message(f"Distractor ranker saved to {path}")

    return ranker

# ────────────────────────────────────────────────────────────
# 4. GENERATE DISTRACTORS (with diversity)
# ────────────────────────────────────────────────────────────

def generate_distractors(article: str,
                         question: str,
                         correct_answer: str,
                         tfidf_vectorizer,
                         ranker=None,
                         top_n: int = 3,
                         diversity_penalty: float = 0.3) -> List[str]:
    """
    Generate distractors using the ML ranker (if available) with diversity
    penalty.

    Falls back to TF-IDF cosine similarity when no ranker is provided.
    """
    candidates = extract_candidate_phrases(article, question, correct_answer)

    if len(candidates) < top_n:
        return candidates[:top_n]

    feats = compute_candidate_features(candidates, correct_answer,
                                       article, tfidf_vectorizer)
    if feats.shape[0] == 0:
        return candidates[:top_n]

    # Score each candidate
    if ranker is not None:
        scores = ranker.predict_proba(feats)[:, 1]  # P(distractor)
    else:
        # Fallback: use cosine similarity as score
        scores = feats[:, 0]  # f0 = cosine similarity

    # Greedy selection with diversity penalty
    selected: List[str] = []
    used = set()

    for _ in range(top_n):
        if len(used) >= len(candidates):
            break

        adj = scores.copy()
        for idx in used:
            sim = _char_match_score(candidates[idx], "")
            for j in range(len(candidates)):
                if j not in used:
                    adj[j] -= diversity_penalty * _char_match_score(
                        candidates[idx], candidates[j])

        valid = [i for i in range(len(candidates)) if i not in used]
        best = valid[np.argmax(adj[valid])]
        selected.append(candidates[best])
        used.add(best)

    return selected

# ────────────────────────────────────────────────────────────
# 5. EVALUATION (BLEU / ROUGE-L / METEOR + Precision@3)
# ────────────────────────────────────────────────────────────

def _bleu_score(reference: List[str], hypothesis: List[str], n: int = 1) -> float:
    """
    Simple n-gram BLEU between two token lists (unigram by default).
    """
    if len(hypothesis) == 0:
        return 0.0
    ref_ngrams = Counter(
        tuple(reference[i:i + n]) for i in range(len(reference) - n + 1))
    hyp_ngrams = Counter(
        tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))
    overlap = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
                  for ng in hyp_ngrams)
    return overlap / max(sum(hyp_ngrams.values()), 1)


def _rouge_l(reference: str, hypothesis: str) -> float:
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


def _meteor_simple(reference: str, hypothesis: str) -> float:
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


def evaluate_distractors(test_df: pd.DataFrame,
                         tfidf_vectorizer,
                         ranker=None) -> Dict:
    """
    Evaluate generated distractors against ground-truth wrong options using
    BLEU, ROUGE-L, METEOR, and Precision@3.
    """
    log_message("Evaluating distractors on test set...")

    bleu_scores, rouge_scores, meteor_scores = [], [], []
    correct_matches = 0
    total_samples = 0

    for _, row in test_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        ans_letter = str(row['answer']).strip()
        correct_text = str(row.get(ans_letter, ''))

        # Ground-truth distractors
        gt_wrong = []
        for opt in ['A', 'B', 'C', 'D']:
            if opt != ans_letter:
                gt_wrong.append(str(row[opt]).strip())

        if not gt_wrong:
            continue

        generated = generate_distractors(
            article, question, correct_text, tfidf_vectorizer,
            ranker=ranker, top_n=3
        )

        # Token-level metrics
        ref_text = " ".join(gt_wrong).lower()
        hyp_text = " ".join(generated).lower()
        ref_toks = ref_text.split()
        hyp_toks = hyp_text.split()

        bleu_scores.append(_bleu_score(ref_toks, hyp_toks, n=1))
        rouge_scores.append(_rouge_l(ref_text, hyp_text))
        meteor_scores.append(_meteor_simple(ref_text, hyp_text))

        # Precision@3 (exact match)
        gen_set = set(g.lower() for g in generated)
        gt_set = set(g.lower() for g in gt_wrong)
        matches = len(gen_set & gt_set)
        correct_matches += min(matches, 3)
        total_samples += 1

    prec3 = correct_matches / (3 * total_samples) if total_samples else 0

    metrics = {
        'Precision@3': round(prec3, 4),
        'BLEU-1': round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0,
        'ROUGE-L': round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0,
        'METEOR': round(float(np.mean(meteor_scores)), 4) if meteor_scores else 0,
    }

    print("\n" + "=" * 60)
    print("DISTRACTORS EVALUATION")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=" * 60 + "\n")

    return metrics
