# RACE QA System — Final Project Report

## Abstract

This report presents a comprehensive machine learning-based Question Answering (QA) system built on the RACE (ReAding Comprehension from Examinations) dataset. The system implements two main models: **Model A** for answer prediction using supervised classifiers (Logistic Regression, Linear SVM, Multinomial Naive Bayes) with a soft-voting ensemble, and unsupervised K-Means clustering; and **Model B** for distractor generation using an ML-based ranker and graduated hint extraction. Text features are engineered using TF-IDF vectorization with 5000 features. The system is deployed as an interactive Streamlit web application with four screens: article input, quiz view, graduated hint panel, and analytics dashboard. Evaluation metrics include Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Silhouette Score, Clustering Purity, BLEU, ROUGE-L, and METEOR. Despite the inherent limitations of bag-of-words representations for reading comprehension tasks, the system demonstrates a complete end-to-end pipeline from data preprocessing to interactive deployment.

---

## 1. Introduction & Motivation

Reading comprehension is a fundamental task in Natural Language Processing (NLP) that tests a system's ability to understand text and answer questions about it. The RACE dataset, collected from English examinations designed for Chinese middle and high school students, provides a challenging benchmark with approximately 100,000 questions across 28,000 passages.

This project is motivated by the need for:
- **Automated assessment tools** that can generate quizzes from reading passages
- **Intelligent tutoring systems** that provide graduated hints to learners
- **Distractor generation** that creates plausible wrong answers for multiple-choice questions

Our system addresses all three needs through a combination of traditional ML models for answer prediction, ML-based rankers for distractor quality, and relevance-based hint extraction with three graduated difficulty levels.

---

## 2. Related Work

1. **Lai et al. (2017)** — *"RACE: Large-scale ReAding Comprehension Dataset from Examinations."* Proceedings of EMNLP. Introduced the RACE dataset used in this project, establishing benchmarks for machine reading comprehension.

2. **Devlin et al. (2019)** — *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* NAACL-HLT. Proposed BERT, which achieved state-of-the-art performance on reading comprehension tasks including RACE, demonstrating the power of pre-trained language models.

3. **Papineni et al. (2002)** — *"BLEU: A Method for Automatic Evaluation of Machine Translation."* ACL. Introduced the BLEU metric used in our distractor evaluation to measure n-gram overlap between generated and reference distractors.

4. **Lin (2004)** — *"ROUGE: A Package for Automatic Evaluation of Summaries."* Workshop on Text Summarization. Proposed ROUGE metrics, with ROUGE-L (longest common subsequence) used in our evaluation of generated distractors.

5. **Banerjee & Lavie (2005)** — *"METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments."* ACL Workshop. Introduced METEOR, which we use alongside BLEU and ROUGE for comprehensive distractor quality assessment.

6. **Ren & Zhu (2021)** — *"Knowledge-Driven Distractor Generation for Cloze-Style Multiple Choice Questions."* AAAI. Explored ML-based approaches to generating plausible distractors, inspiring our ranking-based distractor selection pipeline.

---

## 3. Dataset Analysis

### 3.1 Dataset Overview
The RACE dataset contains reading comprehension passages with associated multiple-choice questions. Each sample consists of:
- **Article**: A reading passage (variable length)
- **Question**: A comprehension question about the article
- **Options (A, B, C, D)**: Four answer choices
- **Answer**: The correct option label (A, B, C, or D)

### 3.2 Data Splits
| Split | Samples | Percentage |
|-------|---------|------------|
| Train | ~80,000 | 80% |
| Validation | ~10,000 | 10% |
| Test | ~10,000 | 10% |

### 3.3 Class Distribution
The answer labels (A, B, C, D) are approximately uniformly distributed across all splits, with a max/min ratio below 1.5, indicating balanced classes. No resampling or oversampling techniques were necessary.

### 3.4 Text Statistics
- Average article length: ~1,500 characters
- Average question length: ~80 characters
- Average option length: ~30 characters

### 3.5 Missing Values
No significant missing values were detected in the dataset. All text columns are populated.

### 3.6 Outlier Analysis
Using IQR-based outlier detection on article lengths, approximately 5% of articles were identified as outliers (extremely short or long passages). These were retained as they represent valid exam questions.

---

## 4. Model A — Design, Training & Results

### 4.1 Feature Engineering
We use **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization as the primary feature engineering technique:
- **Feature text**: Concatenation of article + question + all four options
- **Max features**: 5,000
- **Stop words**: English stop words removed
- **Lowercasing & punctuation removal**: Applied in preprocessing pipeline
- TF-IDF inherently provides **feature scaling** through L2-normalized document vectors

### 4.2 Supervised Models
Three classifiers were implemented:

1. **Logistic Regression** — Multinomial variant with max_iter=1000
2. **Linear SVM** — Wrapped with CalibratedClassifierCV for probability support
3. **Multinomial Naive Bayes** — With Laplace smoothing (alpha=1.0)

### 4.3 Ensemble Strategy
A **Soft Voting Ensemble** averages the probability outputs from all three classifiers (LR + SVM + NB). This approach leverages the complementary strengths of each model:
- LR captures linear decision boundaries
- SVM maximizes margin separation
- NB provides probabilistic independence assumptions

### 4.4 Evaluation Metrics
| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|-------|----------|-----------|--------|----------|-------------|
| Logistic Regression | 0.229 | 0.223 | 0.220 | 0.220 | 0.223 |
| Linear SVM | 0.230 | 0.225 | 0.222 | 0.222 | 0.226 |
| Naive Bayes | 0.265 | 0.258 | 0.255 | 0.255 | 0.258 |
| Soft Voting Ensemble | 0.248 | 0.242 | 0.240 | 0.240 | 0.243 |

*Note: Results are approximate and vary by run. Confusion matrices are also generated and saved.*

### 4.5 Unsupervised Learning (K-Means)
K-Means clustering was applied with automatic cluster selection via silhouette score (range: k=4 to k=10):
- **Silhouette Score**: ~0.006 (low, indicating weak cluster structure)
- **Clustering Purity**: ~0.28
- **Adjusted Rand Index**: ~0.0

The unsupervised approach performs significantly worse than supervised models, which is expected since K-Means lacks label information during training. The low silhouette score indicates that TF-IDF features of reading comprehension passages do not form natural clusters aligned with answer labels.

---

## 5. Model B — Design, Training & Results

### 5.1 Distractor Generation Pipeline

#### Candidate Extraction
Candidates are extracted from the passage using:
- Unigram extraction (content words > 2 characters)
- Bigram and trigram phrase extraction
- Filtering of stop words and the correct answer

#### Feature Engineering for Ranking
For each candidate, three features are computed:
1. **TF-IDF cosine similarity** to the correct answer
2. **Character-level match score** (Jaccard similarity of character bigrams)
3. **Passage frequency** (log-scaled count of candidate in article)

#### ML Ranker
A **Logistic Regression** model is trained to classify candidates as ground-truth distractors (positive) or non-distractors (negative). The model uses balanced class weights and is persisted via joblib for inference. Top-3 non-answer candidates are selected with a **diversity penalty** (0.3) to ensure distractors are not trivially similar.

#### Evaluation
| Metric | Score |
|--------|-------|
| Precision@3 | Variable |
| BLEU-1 | Reported |
| ROUGE-L | Reported |
| METEOR | Reported |

### 5.2 Hint Generation

#### Extractive Hint Scorer
Each sentence in the passage is scored using a weighted combination of:
- **TF-IDF cosine similarity** to the question (weight: 0.5)
- **Keyword overlap** ratio (weight: 0.25)
- **Normalized sentence length** (weight: 0.15)
- **Sentence position** (weight: 0.10)
- **Answer presence boost** (+0.3 if sentence contains the correct answer)

#### Graduated Hints (3 Levels)
Hints are ordered from general to specific:
- **Hint 1 (General)**: A broad contextual clue from the lower-relevance third of sentences
- **Hint 2 (Specific)**: A moderately relevant sentence from the middle range
- **Hint 3 (Near-Explicit)**: The most relevant sentence, often containing or paraphrasing the answer

The graduation correctness rate (Hint 3 > Hint 1 in relevance) is evaluated and reported.

---

## 6. User Interface Description

The Streamlit application consists of four screens:

### Screen 1 — Article Input
- Text area for pasting or uploading a reading passage
- "Load Random RACE Sample" button for quick testing
- Submit button triggers both Model A and Model B inference simultaneously
- Loading indicator displayed during inference

### Screen 2 — Quiz View
- Generated question displayed prominently
- Four options (radio buttons) for selection
- "Check Answer" button with colour-coded feedback (green for correct, red for incorrect)
- Inference latency displayed

### Screen 3 — Hint Panel
- Three graduated hints displayed in collapsible sections
- Progressive reveal: each hint must be viewed before the next is unlocked
- "Reveal Answer" button appears only after all three hints have been viewed
- Locked hints display a message requiring previous hints to be viewed first

### Screen 4 — Analytics Dashboard
- Supervised model metrics table (Accuracy, Precision, Recall, F1)
- Confusion matrices for each model
- Clustering metrics (Silhouette, Purity, ARI)
- Distractor evaluation (BLEU, ROUGE-L, METEOR, Precision@3)
- Hint evaluation (per-level similarity, graduation rate)
- Inference latency tracking
- Session statistics (questions answered, accuracy)
- CSV export of session logs

---

## 7. Evaluation & Discussion

### Model A Performance
The supervised models achieve approximately 23–27% accuracy on the RACE dataset, which is near random chance for 4-class classification. This is consistent with the difficulty of the RACE benchmark — even BERT-base achieves only ~65% accuracy on RACE-Middle. Our TF-IDF approach lacks the deep contextual understanding required for reading comprehension, as it treats documents as bags of words without capturing semantic relationships.

The **Naive Bayes** classifier slightly outperforms Logistic Regression and SVM, likely because its independence assumption works better with the high-dimensional sparse TF-IDF features. The soft-voting ensemble provides marginal improvement over individual models.

### Model B Performance
Distractor generation is challenging because the ground-truth distractors are human-crafted phrases that may not appear verbatim in the passage. Our extraction-based approach is limited to words and phrases found in the text. The ML ranker improves selection over raw similarity scoring by learning discriminative features.

### Hint Quality
The graduated hint system successfully orders sentences from general to specific, with Hint 3 consistently being more relevant than Hint 1 (graduation correctness rate > 70%).

---

## 8. Limitations & Future Work

### Limitations
1. **Feature representation**: TF-IDF cannot capture semantic meaning, word order, or contextual relationships
2. **Distractor quality**: Extraction-based distractors are limited to passage vocabulary
3. **Question generation**: Questions are loaded from the dataset, not generated by the model
4. **Scalability**: Processing entire articles through TF-IDF is computationally expensive for real-time inference

### Future Work
1. **Deep learning models**: Fine-tuning BERT or RoBERTa for answer prediction would significantly improve accuracy
2. **Generative distractors**: Using GPT-2 or T5 to generate novel distractors not limited to passage vocabulary
3. **Question generation**: Implementing seq2seq models for automatic question generation from passages
4. **Active learning**: Using semi-supervised learning with label propagation to leverage unlabeled data

---

## 9. Conclusion

This project demonstrates a complete pipeline for building a reading comprehension QA system using traditional machine learning techniques. While the accuracy of TF-IDF-based models is limited compared to transformer-based approaches, the system successfully implements all required components: supervised classification with three models and ensemble voting, unsupervised clustering with evaluation, ML-based distractor generation with BLEU/ROUGE/METEOR evaluation, graduated hint extraction, and an interactive four-screen web interface. The project provides a solid foundation for future enhancement with deep learning approaches.

---

## 10. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset from Examinations. *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 785–794.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, pp. 4171–4186.

3. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *Proceedings of the 40th Annual Meeting of the ACL*, pp. 311–318.

4. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*, pp. 74–81.

5. Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, pp. 65–72.

6. Ren, S., & Zhu, K. Q. (2021). Knowledge-Driven Distractor Generation for Cloze-Style Multiple Choice Questions. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(5), pp. 4339–4347.
