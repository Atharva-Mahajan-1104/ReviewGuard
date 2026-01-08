#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split

from clean_text import batch_clean
from features import extract_numeric_features


# ---------------- Data Loading ---------------- #

def load_review_data(csv_file: Path) -> pd.DataFrame:
    """
    Load review dataset and normalize labels.
    Expected columns: text, label
    """
    df = pd.read_csv(csv_file)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'text' and 'label' columns.")

    label_mapping = {
        "FAKE": 1, "fake": 1, 1: 1, "1": 1, True: 1,
        "REAL": 0, "real": 0, 0: 0, "0": 0, False: 0
    }

    df["y"] = df["label"].map(label_mapping)

    if df["y"].isna().any():
        raise ValueError("Labels must be REAL or FAKE (or 0/1 equivalents).")

    # Clean text in batch
    df["text_clean"] = batch_clean(df["text"])

    return df[["text", "text_clean", "y"]]


# ---------------- Pipeline Construction ---------------- #

def create_model_pipeline(max_features: int = 20_000) -> Pipeline:
    """
    Build the full preprocessing + classification pipeline.
    """
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf_vectorizer, "text_clean"),
            (
                "numeric",
                StandardScaler(with_mean=False),
                [
                    "sentiment",
                    "exclamation_count",
                    "all_caps_tokens",
                    "repeated_phrases",
                    "char_length",
                    "unique_word_ratio"
                ]
            )
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    classifier = LogisticRegression(max_iter=300)

    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier)
    ])


# ---------------- Feature Augmentation ---------------- #

def append_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment and behavioral numeric features to the dataset.
    """
    numeric_features = extract_numeric_features(df["text"])
    return pd.concat(
        [df.reset_index(drop=True), numeric_features.reset_index(drop=True)],
        axis=1
    )


# ---------------- Training Routine ---------------- #

def train_model(
    csv_path: Path,
    output_dir: Path,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> dict:
    """
    Train the fake review detector and save artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    data = append_numeric_features(load_review_data(csv_path))

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["y", "text"]),
        data["y"].values,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=data["y"].values
    )

    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Metrics
    results = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "avg_precision": float(average_precision_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["REAL", "FAKE"],
            output_dict=True
        )
    }

    # Save metrics and model
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    joblib.dump(pipeline, output_dir / "pipeline.joblib")

    # ---------------- Visualizations ---------------- #

    import matplotlib
    matplotlib.use("Agg")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"]
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(4, 3))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(4, 3))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    return results


# ---------------- CLI Entry Point ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train an NLP-based fake review detection model."
    )

    parser.add_argument("--csv", default="data/reviews_sample.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    metrics = train_model(
        Path(args.csv),
        Path(args.outdir),
        args.test_size,
        args.seed
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
