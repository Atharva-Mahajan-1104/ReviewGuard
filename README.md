

# 🧠 ReviewGuard – AI Review Authenticity Detection

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Analytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📖 Overview

Online product reviews strongly influence consumer decisions, yet many platforms suffer from **manipulated or deceptive reviews**.
**ReviewGuard** is designed to automatically assess the **credibility of written product reviews** using a combination of **linguistic analysis, sentiment cues, and behavioral patterns**.

The system goes beyond plain text classification by incorporating **how a review is written**, not just **what is written**. An interactive Streamlit interface allows users to experiment with review inputs and observe how classification confidence changes.

---

## ✨ Key Capabilities

* Robust text preprocessing and normalization
* Multi-source feature extraction:

  * Lexical features using TF-IDF
  * Emotional intensity indicators
  * Writing behavior analysis (capitalization, punctuation)
  * Detection of overused promotional language
* Transparent and interpretable classification model
* Adjustable decision threshold for sensitivity tuning
* Visual evaluation using standard classification metrics
* Web-based UI for instant predictions

---

## 🧩 System Architecture

```text
User Review Text
        ↓
Text Cleaning & Normalization
        ↓
Feature Engineering
  ├─ Linguistic (TF-IDF)
  ├─ Sentiment Signals
  └─ Behavioral Patterns
        ↓
Logistic Regression Classifier
        ↓
Credibility Score + Prediction
```

---

## 📁 Project Layout

```bash
reviewguard/
├── app/
│   └── web_interface.py
├── core/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── inference.py
├── data/
│   └── sample_reviews.csv
├── artifacts/
│   ├── trained_pipeline.joblib
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. **Input**
   Review text is provided via CSV or the web interface.

2. **Processing**
   The text undergoes cleaning, normalization, and pattern extraction.

3. **Feature Construction**
   Both linguistic representation and behavioral signals are combined.

4. **Classification**
   A logistic regression model estimates the likelihood of deception.

5. **Decision Logic**
   Users can adjust the classification threshold to control sensitivity.

---

## 🖥️ Web Application

The Streamlit-based interface allows users to:

* Paste or type review content
* View real-time predictions
* Inspect probability scores
* Adjust detection strictness
* Understand why a review was flagged

---

## 📊 Model Performance

The model is evaluated using:

* Confusion Matrix
* ROC Curve
* Precision–Recall Curve

On balanced sample data, the system achieves:

* **AUC ≈ 1.00**
* **Average Precision ≈ 1.00**

*(Results may vary on real-world datasets.)*

---

## 🚀 Setup & Usage

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model
python core/train_model.py --csv data/sample_reviews.csv --outdir artifacts

# Run inference from CLI
python core/inference.py --pipeline artifacts/trained_pipeline.joblib \
                         --text "Absolutely amazing product, best ever!!!"

# Launch the web app
streamlit run app/web_interface.py
```

---

## 🔍 Insights Observed

* Excessive emotional language often correlates with deceptive intent
* ALL-CAPS and repetitive praise are strong indicators of manipulation
* Genuine reviews typically contain balanced sentiment and concrete details
* Combining behavioral cues with NLP improves detection reliability

---

## 🔮 Future Enhancements

* Expand training using real-world labeled datasets
* Replace TF-IDF with transformer-based embeddings
* Add explainability tools (SHAP / LIME)
* Deploy publicly via Streamlit Cloud or Hugging Face Spaces

---

## ⭐ Why This Project Matters

ReviewGuard demonstrates how **practical NLP systems** can be designed to:

* Improve platform trust
* Assist moderation workflows
* Provide explainable AI decisions
* Balance accuracy with interpretability

---

## 🤝 Contributions

Suggestions, issues, and improvements are welcome.
Feel free to fork the repository or submit a pull request.

---


