from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
import streamlit as st

from clean_text import clean_text
from features import extract_numeric_features


# Path to trained pipeline
PIPELINE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pipeline.joblib"


# ---------------- Page Configuration ---------------- #

st.set_page_config(
    page_title="ReviewGuard – Review Authenticity Check",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 ReviewGuard")
st.caption("NLP-based review credibility analysis using TF-IDF and behavioral signals")


# ---------------- Load Trained Pipeline ---------------- #

@st.cache_resource
def load_model_pipeline():
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)
    return None


pipeline = load_model_pipeline()


# ---------------- User Input ---------------- #

review_text = st.text_area(
    "Enter a product review for analysis",
    height=200,
    placeholder="Absolutely amazing product!!! Best deal ever, got it for free and love it..."
)

threshold = st.slider(
    "Fake review sensitivity threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01
)

analyze_btn = st.button("Run Analysis", type="primary")


# ---------------- Prediction Logic ---------------- #

if analyze_btn:
    if pipeline is None:
        st.error("Trained model not found. Please train the model first using `python src/train.py`.")

    elif not review_text.strip():
        st.warning("Please enter some review text before running the analysis.")

    else:
        # Clean and prepare text
        cleaned_text = clean_text(review_text)

        text_df = pd.DataFrame([{
            "text": review_text,
            "text_clean": cleaned_text
        }])

        # Extract numeric and behavioral features
        numeric_features = extract_numeric_features([review_text])

        # Combine all features
        input_data = pd.concat([text_df, numeric_features], axis=1)

        # Generate prediction probability
        fake_probability = float(pipeline.predict_proba(input_data)[0, 1])

        prediction_label = "FAKE" if fake_probability >= threshold else "REAL"

        # Display result
        st.metric("Classification Result", prediction_label)

        st.progress(
            fake_probability if prediction_label == "FAKE" else 1 - fake_probability,
            text=f"Fake likelihood: {fake_probability:.1%} | Threshold: {threshold:.2f}"
        )


# ---------------- Helpful Guidance ---------------- #

st.info(
    "Indicators such as excessive punctuation, ALL-CAPS usage, repetitive praise, "
    "and references to free samples or discounts often correlate with deceptive reviews.",
    icon="💡"
)
