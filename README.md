# QA System - Quiz Master

A comprehensive machine learning-based question answering system with multiple-choice quiz generation, hint extraction, and distractor generation.

## Overview

This project implements:
- **Model A (Supervised)**: Logistic Regression, SVM, and Voting Ensemble for answer prediction
- **Model A (Unsupervised)**: K-Means clustering with automatic cluster selection
- **Model B (Distractors)**: Intelligent distractor generation using TF-IDF similarity
- **Model B (Hints)**: Automatic hint extraction from articles
- **Streamlit UI**: Interactive web application with 4 screens and developer dashboard

## Features

### Models
- **TF-IDF Vectorization**: 5000 features with English stop words
- **Supervised Learning**: Logistic Regression, Linear SVM, and Soft Voting Ensemble
- **Clustering**: K-Means with silhouette-based optimal cluster selection
- **Text Analysis**: TF-IDF based similarity for hints and distractors

### Streamlit App (4 Screens)
1. **Article Input**: Load articles manually or randomly from RACE dataset
2. **Quiz View**: Multiple-choice questions with answer checking and hints
3. **Hint Panel**: Progressive hints with reveal functionality
4. **Developer Dashboard**: Model metrics, session statistics, and log export

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Directory Structure
project/
├── requirements.txt
├── README.md
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   ├── voting_ensemble.pkl
│   ├── kmeans_model.pkl
│   ├── label_encoder.pkl
│   └── metrics.json
├── src/
│   ├── init.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── model_a_supervised.py
│   ├── model_a_unsupervised.py
│   ├── model_b_distractors.py
│   └── model_b_hints.py
├── notebooks/
│   └── colab_training.ipynb
└── app/
└── streamlit_app.py

## Training

### Google Colab (Recommended)

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Mount Google Drive
3. Upload your dataset to `/My Drive/data/` (train.csv, val.csv, test.csv)
4. Run all cells sequentially
5. Models will be saved to `/My Drive/models/`

### Local Training

```bash
python -m src.preprocess  # Preprocess data
python -m src.model_a_supervised  # Train supervised models
python -m src.model_a_unsupervised  # Train clustering
python -m src.model_b_distractors  # Evaluate distractors
python -m src.model_b_hints  # Evaluate hints
```

## Running the App

### Local
```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`

### Colab (Optional)
```python
from pyngrok import ngrok
ngrok.connect(8501)
# Run streamlit in background
```

## Dataset Format

CSV files should have the following columns:
- `article`: Full article text
- `question`: Question text
- `answer`: Correct answer (single letter: A, B, C, or D)
- `A`, `B`, `C`, `D`: Answer options

Example:
article,question,answer,A,B,C,D
"Once upon a time...",Who is the main character?,A,Alice,Bob,Charlie,Diana

## Models Performance

After training, check `models/metrics.json` for:
- **Supervised**: Accuracy and F1 scores for LR, SVM, and Ensemble
- **Clustering**: Silhouette score and Adjusted Rand Index
- **Distractors**: Precision@3 and Recall@3
- **Hints**: Average relevance similarity

## Configuration

### TF-IDF
- Max features: 5000
- Stop words: English
- Location: `src/preprocess.py`

### K-Means
- Cluster range: 5-15
- Selection: Silhouette score
- Location: `src/model_a_unsupervised.py`

### Distractors
- Diversity penalty: 0.3
- Top candidates: 3
- Location: `src/model_b_distractors.py`

### Hints
- Default extraction: 3 hints
- Ranking: TF-IDF cosine similarity
- Location: `src/model_b_hints.py`

## Troubleshooting

### Models Not Loading
- Ensure models are trained and saved in `models/` directory
- Check that model filenames match: `tfidf_vectorizer.pkl`, `logistic_regression.pkl`, etc.

### Data Not Found
- Place CSV files in `data/` directory
- Verify column names: article, question, answer, A, B, C, D

### Streamlit Errors
- Clear cache: `streamlit cache clear`
- Check Python version: 3.8+

## Production Notes

- Error handling implemented for missing models and data
- Session state management for persistent UI state
- Logging utilities for debugging
- Caching for model and data loading
- CSV export for session logs

## License

MIT License - See LICENSE file for details

## Contact

For questions or improvements, please open an issue or submit a pull request.
Now I'll create the __init__.py file for the src package:
