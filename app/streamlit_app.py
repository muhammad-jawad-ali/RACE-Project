import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_b_distractors import generate_distractors
from src.model_b_hints import extract_hints

# Configuration

st.set_page_config(
    page_title="QA System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define model paths
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

@st.cache_resource
def load_models():
    try:
        models = {
            'tfidf': joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl'),
            'lr': joblib.load(MODELS_DIR / 'logistic_regression.pkl'),
            'svm': joblib.load(MODELS_DIR / 'linear_svm.pkl'),
            'ensemble': joblib.load(MODELS_DIR / 'voting_ensemble.pkl'),
            'kmeans': joblib.load(MODELS_DIR / 'kmeans_model.pkl'),
            'label_encoder': joblib.load(MODELS_DIR / 'label_encoder.pkl')
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Model loading error: {e}")
        st.info("Please train models first using the Colab notebook.")
        return None

@st.cache_resource
def load_test_data():
    try:
        return pd.read_csv(DATA_DIR / 'test.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_metrics():
    try:
        with open(MODELS_DIR / 'metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Session State

if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {
        'article': '',
        'question': '',
        'correct_answer': '',
        'options': [],
        'selected_option': None,
        'is_answered': False,
        'answer_correct': None,
        'hints_used': 0,
        'available_hints': [],
        'current_hint_idx': 0
    }

if 'session_log' not in st.session_state:
    st.session_state.session_log = []

# Helpers

def extract_question(article: str) -> str:
    sentences = article.split('. ')
    wh_words = ['who', 'what', 'where', 'when', 'why', 'how']
    
    for sentence in sentences:
        if any(wh in sentence.lower() for wh in wh_words):
            return sentence.strip() + '?'
    
    # Fallback: first sentence
    return (sentences[0].strip() + '?') if sentences else 'What is this about?'

def get_inference_time(start_time) -> float:
    from time import time
    return round((time() - start_time) * 1000, 2)

def log_session(article: str, question: str, answer: str, is_correct: bool, model_used: str):
    st.session_state.session_log.append({
        'timestamp': datetime.now().isoformat(),
        'article': article[:100],
        'question': question[:100],
        'answer_selected': answer,
        'correct': is_correct,
        'model': model_used
    })

def export_session_logs() -> str:
    if not st.session_state.session_log:
        return ""
    
    df = pd.DataFrame(st.session_state.session_log)
    return df.to_csv(index=False)

# Screens

def screen_1_article_input():
    st.header("Article Input")
    
    test_df = load_test_data()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        article_input = st.text_area(
            "Paste an article:",
            value=st.session_state.quiz_state['article'],
            height=200,
            key='article_input'
        )
    
    with col2:
        if test_df is not None and st.button("Load Random from RACE"):
            random_row = test_df.sample(1).iloc[0]
            st.session_state.quiz_state['article'] = random_row['article']
            st.session_state.quiz_state['question'] = random_row['question']
            
            # Get the actual text of the correct answer (A, B, C, or D)
            ans_letter = random_row['answer'].strip()
            st.session_state.quiz_state['correct_answer'] = random_row[ans_letter]
            
            # Reset options and state
            st.session_state.quiz_state['options'] = []
            st.session_state.quiz_state['is_answered'] = False
            st.session_state.quiz_state['hints_used'] = 0
            st.session_state.quiz_state['available_hints'] = []
            st.rerun()
    
    st.session_state.quiz_state['article'] = article_input
    
    if st.button("Submit Article", use_container_width=True, type='primary'):
        if article_input.strip():
            st.session_state.quiz_state['question'] = extract_question(article_input)
            # Reset options so they are regenerated for the new question
            st.session_state.quiz_state['options'] = []
            st.session_state.quiz_state['is_answered'] = False
            st.session_state.quiz_state['hints_used'] = 0
            st.session_state.quiz_state['available_hints'] = []
            st.rerun()
        else:
            st.error("Please enter an article!")

def screen_2_quiz_view():
    st.header("Quiz")
    
    models = load_models()
    if models is None:
        st.error("Models not loaded!")
        return
    
    quiz = st.session_state.quiz_state
    
    # Display question
    st.subheader("Question:")
    st.write(quiz['question'])
    
    # Generate options if not already done
    if not quiz['options']:
        from time import time
        start_time = time()
        
        with st.spinner("Generating options..."):
            # Generate distractors
            distractors = generate_distractors(
                quiz['article'],
                quiz['question'],
                quiz['correct_answer'] or "answer",
                models['tfidf'],
                top_n=3
            )
            
            # Create options (correct + distractors)
            correct = quiz['correct_answer'] or distractors[0] if distractors else "Answer"
            options = [correct] + (distractors[:3] if len(distractors) >= 3 else distractors)
            np.random.shuffle(options)
            
            quiz['options'] = options
            quiz['inference_time'] = get_inference_time(start_time)
    
    # Display options
    st.subheader("Choose your answer:")
    quiz['selected_option'] = st.radio(
        "Options:",
        options=quiz['options'],
        key='option_radio'
    )
    
    # Check if all hints are used for Reveal Answer restriction
    hints_ready = (quiz['hints_used'] >= 3)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check Answer is always enabled if an option is selected
        if st.button("Check Answer", use_container_width=True, type='primary'):
            if quiz['selected_option']:
                # Predict using ensemble
                from time import time
                start_time = time()
                
                feature_text = f"{quiz['article']} {quiz['question']} {quiz['selected_option']}"
                feature_vector = models['tfidf'].transform([feature_text])
                
                prediction = models['ensemble'].predict(feature_vector)[0]
                pred_label = models['label_encoder'].inverse_transform([prediction])[0]
                
                is_correct = (quiz['selected_option'] == quiz['correct_answer'])
                quiz['is_answered'] = True
                quiz['answer_correct'] = is_correct
                quiz['inference_time'] = get_inference_time(start_time)
                
                log_session(
                    quiz['article'],
                    quiz['question'],
                    quiz['selected_option'],
                    is_correct,
                    'Ensemble'
                )
            else:
                st.warning("Please select an option first!")
    
    def handle_hint_request():
        if not quiz['available_hints']:
            quiz['available_hints'] = extract_hints(
                quiz['article'],
                quiz['question'],
                models['tfidf'],
                correct_answer=quiz['correct_answer'],
                num_hints=3
            )
        if quiz['hints_used'] < 3:
            quiz['hints_used'] += 1

    with col2:
        st.button(
            "Get Next Hint", 
            use_container_width=True, 
            disabled=quiz['hints_used'] >= 3,
            on_click=handle_hint_request
        )
    
    with col3:
        if st.button("Reveal Answer", use_container_width=True, disabled=not hints_ready):
            quiz['is_answered'] = True
            st.info(f"Correct Answer: {quiz['correct_answer']}")
        if not hints_ready:
            st.caption("View all 3 hints to reveal.")

    # Graduated Hint Panel (Screen 3 requirements)
    if quiz['hints_used'] > 0 and quiz['available_hints']:
        st.divider()
        st.subheader("Graduated Hints")
        
        for i in range(quiz['hints_used']):
            if i < len(quiz['available_hints']):
                level = ["General", "Specific", "Near-Explicit"][i]
                # Keep all revealed hints expanded as they are opened one by one
                with st.expander(f"Hint {i+1}: {level}", expanded=True):
                    st.write(quiz['available_hints'][i])

    # Show result
    if quiz['is_answered']:
        st.divider()
        if quiz['answer_correct']:
            st.success("Correct!")
        else:
            st.error(f"Wrong! Correct answer: {quiz['correct_answer']}")

def screen_3_hint_panel():
    pass

def screen_4_dashboard():
    st.sidebar.header("Developer Dashboard")
    
    metrics = load_metrics()
    
    if metrics:
        st.sidebar.subheader("Model Metrics")
        
        # Supervised models
        if 'supervised' in metrics:
            st.sidebar.write("**Supervised Models:**")
            for model_info in metrics['supervised']:
                st.sidebar.metric(
                    label=model_info['Model'],
                    value=f"{model_info['Accuracy']:.4f}",
                    delta="Accuracy"
                )
        
        # Clustering
        if 'clustering' in metrics:
            st.sidebar.write("**Clustering:**")
            c_met = metrics['clustering']
            st.sidebar.metric("Silhouette Score", f"{c_met.get('Silhouette Score', 0):.4f}")
            st.sidebar.metric("Clustering Purity", f"{c_met.get('Clustering Purity', 0):.4f}")
            st.sidebar.metric("Adjusted Rand Index", f"{c_met.get('Adjusted Rand Index', 0):.4f}")
        
        # Distractors
        if 'distractors' in metrics:
            st.sidebar.write("**Distractors:**")
            st.sidebar.metric(
                label="Precision@3",
                value=f"{metrics['distractors'].get('Precision@3', 0):.4f}"
            )
        
        # Hints
        if 'hints' in metrics:
            st.sidebar.write("**Hints:**")
            st.sidebar.metric(
                label="Graduation Rate",
                value=f"{metrics['hints'].get('Graduation Correctness Rate', 0):.4f}"
            )
    
    st.sidebar.divider()
    
    # Session stats
    st.sidebar.subheader("Session Stats")
    total_answered = len(st.session_state.session_log)
    correct_answers = sum(1 for log in st.session_state.session_log if log['correct'])
    
    st.sidebar.metric("Questions Answered", total_answered)
    st.sidebar.metric(
        "Correct Answers",
        correct_answers,
        f"{100*correct_answers/total_answered if total_answered > 0 else 0:.1f}%"
    )
    st.sidebar.metric("Hints Used", st.session_state.quiz_state['hints_used'])
    
    # Export logs
    st.sidebar.divider()
    if st.sidebar.button("Export Session Logs"):
        csv_data = export_session_logs()
        if csv_data:
            st.sidebar.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Main

def main():
    # Dashboard in sidebar
    screen_4_dashboard()
    
    # Main content
    st.title("QA System - Quiz Master")
    st.write("Learn from articles with AI-powered multiple choice questions!")
    
    st.divider()
    
    # Check if article is loaded
    if not st.session_state.quiz_state['article']:
        screen_1_article_input()
    else:
        # Show article and quiz
        tab1, tab2 = st.tabs(["Article", "Quiz"])
        
        with tab1:
            st.subheader("Current Article:")
            st.text_area(
                "Article:",
                value=st.session_state.quiz_state['article'],
                height=200,
                disabled=True,
                key='article_display'
            )
            if st.button("Load Different Article"):
                st.session_state.quiz_state = {
                    'article': '',
                    'question': '',
                    'correct_answer': '',
                    'options': [],
                    'selected_option': None,
                    'is_answered': False,
                    'answer_correct': None,
                    'hints_used': 0,
                    'available_hints': [],
                    'current_hint_idx': 0
                }
                st.rerun()
        
        with tab2:
            screen_2_quiz_view()

if __name__ == '__main__':
    main()