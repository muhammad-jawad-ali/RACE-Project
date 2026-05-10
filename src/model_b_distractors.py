

import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Tuple, Dict
import joblib
import os
from src.utils import (
    log_message, ensure_directory, compute_bleu, compute_rouge_l, compute_meteor
)

# Candidate Extraction

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

# Feature Engineering

def _char_match_score(candidate: str, correct: str) -> float:
    def char_bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else {s}
    bg_c = char_bigrams(candidate.lower())
    bg_a = char_bigrams(correct.lower())
    inter = bg_c & bg_a
    union = bg_c | bg_a
    return len(inter) / len(union) if union else 0.0


def _passage_frequency(candidate: str, article: str) -> int:
    return article.lower().count(candidate.lower())


def compute_candidate_features(candidates: List[str],
                               correct_answer: str,
                               article: str,
                               tfidf_vectorizer) -> np.ndarray:
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

# ML Ranker

def train_distractor_ranker(train_df: pd.DataFrame,
                            tfidf_vectorizer,
                            output_dir: str) -> LogisticRegression:
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

# Generate Distractors

def generate_distractors(article: str,
                         question: str,
                         correct_answer: str,
                         tfidf_vectorizer,
                         ranker=None,
                         top_n: int = 3,
                         diversity_penalty: float = 0.3) -> List[str]:
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

# Evaluation

# Metrics are now imported from src.utils


def evaluate_distractors(test_df: pd.DataFrame,
                         tfidf_vectorizer,
                         ranker=None) -> Dict:
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

        bleu_scores.append(compute_bleu(ref_toks, hyp_toks, n=1))
        rouge_scores.append(compute_rouge_l(ref_text, hyp_text))
        meteor_scores.append(compute_meteor(ref_text, hyp_text))

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
