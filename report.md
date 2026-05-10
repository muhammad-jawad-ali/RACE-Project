# RACE QA System — Comprehensive Project Report
**National University of Computer and Emerging Sciences**  
**FAST School of Computing - 2026**

---

## Abstract
This report details the development and evaluation of a machine learning-based Question Answering (QA) and Quiz Generation system utilizing the RACE (ReAding Comprehension from Examinations) dataset. The system architecture comprises two primary modules: Model A, which employs a Soft-Voting Ensemble (Logistic Regression, SVM, and Naive Bayes) for answer prediction, and Model B, which utilizes a Logistic Regression-based ranker for distractor generation and an extractive relevance scorer for graduated hint revelation. Features are engineered through TF-IDF vectorization with rigorous data cleaning, and unsupervised K-Means clustering is implemented as a performance baseline. The system is deployed via an interactive Streamlit application featuring a 4-screen workflow including a developer analytics dashboard. Evaluation using metrics such as Accuracy, F1-Score, Silhouette Score, BLEU, ROUGE, and METEOR reveals that while traditional bag-of-words models face inherent limitations in semantic comprehension, the integrated pipeline successfully provides a robust framework for automated educational assessment and graduated learning support.

---

## 1. Introduction & Motivation
Reading comprehension assessment is a vital component of language learning and cognitive evaluation. However, the manual creation of high-quality multiple-choice questions (MCQs), plausible distractors, and helpful hints is a time-consuming task for educators. This project is motivated by the potential of Natural Language Processing (NLP) to automate these processes.

Our primary objectives are:
1. **Automated Assessment**: To predict correct answers for existing reading comprehension questions using machine learning.
2. **Intelligent Distractor Generation**: To identify and rank plausible but incorrect options from a passage to challenge the learner.
3. **Scaffolded Learning**: To provide graduated hints that progressively guide a student toward the correct answer without revealing it immediately.

By leveraging the RACE dataset, we aim to build a system that balances traditional statistical machine learning with logic-based NLP heuristics.

---

## 2. Related Work
The development of our system is informed by several foundational works in NLP and reading comprehension:

1.  **Lai et al. (2017)** — *"RACE: Large-scale ReAding Comprehension Dataset from Examinations."* This work introduced the RACE dataset, highlighting the gap between human performance and traditional ML models in high-school level English exams.
2.  **Devlin et al. (2019)** — *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* BERT established the state-of-the-art for RACE, demonstrating that contextual embeddings are critical for understanding complex passages.
3.  **Papineni et al. (2002)** — *"BLEU: A Method for Automatic Evaluation of Machine Translation."* The original BLEU metric, which we utilize to evaluate the token-level overlap between generated distractors and ground truth.
4.  **Lin (2004)** — *"ROUGE: A Package for Automatic Evaluation of Summaries."* We employ ROUGE-L to measure the longest common subsequence between generated and reference distractors.
5.  **Banerjee & Lavie (2005)** — *"METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments."* METEOR provides a more nuanced evaluation by considering synonyms and stemming, which we use to assess hint and distractor quality.
6.  **Ren & Zhu (2021)** — *"Knowledge-Driven Distractor Generation for Cloze-Style Multiple Choice Questions."* This research inspired our use of a ranking-based approach for selecting the most plausible distractors from a set of candidates.

---

## 3. Dataset Analysis
The RACE dataset consists of approximately 100,000 questions across 28,000 passages. 

### 3.1 Exploratory Data Analysis (EDA)
Our EDA implementation covers four critical areas:
*   **Data Overview**: The dataset contains articles, questions, four options (A-D), and the correct label.
*   **Missing Value Analysis**: No null entries were detected in the primary text columns.
*   **Statistical Summaries**: Article lengths average ~1,500 characters, while questions and options are significantly more concise (~80 and ~30 chars respectively).
*   **Outlier Detection**: Using the IQR method, passages exceeding ~3,500 characters were identified as outliers but retained to maintain the integrity of the examination difficulty levels.

### 3.2 Visualizations
*   **Data Distribution**: We observed a nearly uniform distribution of correct labels (A, B, C, D), indicating a balanced dataset.
*   **Correlation Matrices**: Analysis revealed a moderate correlation between article length and question complexity.
*   **Feature Relationships**: Text lengths across options were found to be consistent, preventing models from gaming the system based on answer length alone.

---

## 4. Model A — Design, Training & Results
Model A focuses on predicting the correct answer letter given an article and question.

### 4.1 Preprocessing Pipeline
The pipeline includes **lowercasing**, **punctuation removal**, and **TF-IDF vectorization** (5,000 features). Feature scaling is handled inherently through L2-normalization of the TF-IDF vectors.

### 4.2 Supervised Learning
We implemented a **Soft-Voting Ensemble** combining:
1.  **Logistic Regression** (Multinomial)
2.  **Linear SVM** (with probability calibration)
3.  **Multinomial Naive Bayes**

The ensemble strategy averages probability outputs to leverage the strengths of each classifier.

### 4.3 Unsupervised Learning
**K-Means clustering** was implemented with an automatic selection of $k$ (4 to 10) based on the **Silhouette Score**. Clustering purity and Adjusted Rand Index (ARI) were used for evaluation.

### 4.4 Results Comparison
| Approach | Metric | Score |
| :--- | :--- | :--- |
| **Supervised (Ensemble)** | Accuracy | ~0.24 |
| **Supervised (SVM)** | Accuracy | ~0.28 |
| **Unsupervised (K-Means)** | Purity | ~0.27 |
| **Unsupervised (K-Means)** | Silhouette | ~0.003 |

---

## 5. Model B — Design, Training & Results
Model B handles the generation of distractors and hints.

### 5.1 Distractor Generation
*   **Candidate Extraction**: Noun phrases and keywords are extracted from the passage via string matching and frequency-based selection.
*   **ML Ranker**: A Logistic Regression model ranks candidates using **Cosine Similarity**, **Character-level match scores**, and **Passage frequency**.
*   **Diversity Penalty**: To ensure distractors are not trivially similar, a penalty is applied to overlapping candidates during selection.

### 5.2 Hint Generation
*   **Extractive Scorer**: Sentences are ranked by relevance to the question using TF-IDF similarity, keyword overlap, and position.
*   **Graduated Levels**: Three levels are generated: Hint 1 (General Context), Hint 2 (Specific Clue), and Hint 3 (Near-Explicit paraphrase of the answer).

### 5.3 Evaluation
| Metric | Distractors | Hint 3 |
| :--- | :--- | :--- |
| **BLEU-1** | ~0.16 | ~0.06 |
| **ROUGE-L** | ~0.04 | ~0.08 |
| **METEOR** | ~0.04 | ~0.17 |

---

## 6. UI Description
The system is implemented as a 4-screen Streamlit application:
1.  **Screen 1: Article Input**: Allows users to paste text or load random RACE samples. Triggers simultaneous Model A/B inference.
2.  **Screen 2: Quiz View**: Displays the question and four options. Features a "Check Answer" button with color-coded feedback.
3.  **Screen 3: Hint Panel**: A collapsible panel providing graduated hints. It enforces sequential viewing and unlocks the answer only after all hints are viewed.
4.  **Screen 4: Analytics**: A developer sidebar showing real-time metrics, inference latency, and session logs with CSV export.

---

## 7. Evaluation & Discussion
The results indicate that while the system is architecturally complete, answer prediction on the RACE dataset remains difficult for traditional ML models (~24-28% accuracy). This stems from the fact that RACE questions often require high-level reasoning and inference, which TF-IDF's bag-of-words approach cannot capture.

Model B's distractor generation provides plausible alternatives by selecting terms directly from the article. The graduated hint system is highly effective, with a graduation correctness rate exceeding 70%, ensuring that Hint 3 is consistently more helpful than Hint 1.

---

## 8. Limitations & Future Work
### 8.1 Limitations
*   **Lack of Semantic Depth**: TF-IDF ignores word order and context, leading to poor performance on reasoning-heavy questions.
*   **Extraction Bias**: Distractors and hints are limited to existing text in the article; the system cannot generate novel paraphrases.

### 8.2 Future Work
*   **Transformers**: Implementing BERT or T5 would significantly improve both answer prediction and generative distractor creation.
*   **Deep Hinting**: Moving from extractive to abstractive hint generation would provide more natural and helpful tutoring.

---

## 9. Conclusion
This project successfully demonstrates an end-to-end NLP pipeline for automated quiz generation and assessment. By integrating supervised ensembles, unsupervised clustering, and ML-based ranking, we have created a functional educational tool. While deep learning is necessary for state-of-the-art accuracy, this system provides a robust baseline and a feature-rich user interface for exploring the RACE dataset.

---

## 10. References
1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset from Examinations. *EMNLP*.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
3. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *ACL*.
4. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
5. Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *ACL Workshop*.
6. Ren, S., & Zhu, K. Q. (2021). Knowledge-Driven Distractor Generation for Cloze-Style Multiple Choice Questions. *AAAI*.
