"""
QA System Package - Machine Learning based question answering system.

Modules:
- utils: Data loading and utility functions
- preprocess: TF-IDF vectorization and preprocessing
- model_a_supervised: Supervised learning models (LR, SVM, Ensemble)
- model_a_unsupervised: Unsupervised learning (K-Means clustering)
- model_b_distractors: Distractor generation and evaluation
- model_b_hints: Hint extraction and evaluation
"""

__version__ = "1.0.0"
__author__ = "QA System Team"

from . import utils
from . import preprocess
from . import model_a_supervised
from . import model_a_unsupervised
from . import model_b_distractors
from . import model_b_hints

__all__ = [
    'utils',
    'preprocess',
    'model_a_supervised',
    'model_a_unsupervised',
    'model_b_distractors',
    'model_b_hints'
]

