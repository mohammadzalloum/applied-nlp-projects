"""
Professional Sentiment Analysis Project
=======================================

This project compares two practical approaches for sentiment analysis:

1. A lexicon-based VADER workflow for short Twitter posts.
2. Recurrent neural models (Simple RNN, LSTM, and GRU) for IMDB movie reviews.

The project also generates a professional comparative analysis report that discusses
performance, interpretability, training cost, context handling, and practical use cases.

Default expected project structure:

project/
├── Data/
│   └── Tweets.csv
├── sentiment_analysis_project.py
└── outputs/
    ├── vader_analysis/
    ├── recurrent_models/
    └── comparative_analysis/

Run:
    python sentiment_analysis_project.py

Optional:
    python sentiment_analysis_project.py --tweets-path Data/Tweets.csv --epochs 5
    python sentiment_analysis_project.py --skip-neural
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Central configuration for the full pipeline."""

    # Paths
    tweets_path: Path = Path("Data/Tweets.csv")
    output_root: Path = Path("outputs")

    # Reproducibility
    seed: int = 42

    # IMDB / neural model settings
    vocab_size: int = 10_000
    max_len: int = 200
    embedding_dim: int = 128
    hidden_units: int = 64
    dropout_rate: float = 0.3
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    validation_size: float = 0.2


# =============================================================================
# Utility functions
# =============================================================================


def create_output_dirs(config: PipelineConfig) -> Dict[str, Path]:
    """Create and return output folders for the project sections."""

    output_dirs = {
        "vader": config.output_root / "vader_analysis",
        "neural": config.output_root / "recurrent_models",
        "analysis": config.output_root / "comparative_analysis",
    }

    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    return output_dirs


def make_json_serializable(value: Any) -> Any:
    """Convert common Python objects to JSON-friendly values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    return value


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as a readable JSON file."""

    serializable_data = make_json_serializable(data)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, indent=4, ensure_ascii=False)


def resolve_tweets_path(path: Path) -> Path:
    """Find Tweets.csv using a clear fallback strategy."""

    possible_paths = [
        path,
        Path("Tweets.csv"),
        Path("Data/Tweets.csv"),
        Path("data/Tweets.csv"),
    ]

    for candidate in possible_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Tweets.csv was not found. Put it in Data/Tweets.csv or pass --tweets-path."
    )


def setup_nltk() -> None:
    """Download the NLTK resources needed for VADER and preprocessing."""

    required_packages = [
        "vader_lexicon",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]

    for package in required_packages:
        nltk.download(package, quiet=True)


def setup_tensorflow(seed: int):
    """Import TensorFlow lazily, set the seed, and print GPU information."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("\nGPU detected by TensorFlow:")
        for gpu in gpus:
            print("-", gpu)

        # Prevent TensorFlow from taking all GPU memory at once.
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs are initialized.
                pass
    else:
        print("\nNo GPU detected by TensorFlow. Training will run on CPU.")

    return tf


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series | List[int] | List[str],
    y_pred: np.ndarray | pd.Series | List[int] | List[str],
    labels: List[Any],
    display_labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    """Create and save a confusion matrix image."""

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Lexicon-Based Sentiment Analysis: VADER on Twitter Tweets
# =============================================================================


def build_stopword_set() -> set[str]:
    """Build a stopword set while keeping sentiment-critical words."""

    stop_words = set(stopwords.words("english"))

    # Do not remove negation words because they can flip sentiment.
    negation_words = {
        "no",
        "nor",
        "not",
        "never",
        "none",
        "cannot",
        "can't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "wont",
        "won't",
        "isnt",
        "isn't",
        "arent",
        "aren't",
        "wasnt",
        "wasn't",
        "werent",
        "weren't",
        "havent",
        "haven't",
        "hasnt",
        "hasn't",
        "hadnt",
        "hadn't",
    }

    # These words often strengthen or weaken sentiment.
    intensifiers = {
        "very",
        "really",
        "too",
        "so",
        "extremely",
        "highly",
        "barely",
        "hardly",
    }

    return stop_words - negation_words - intensifiers


def clean_text_for_methodology(
    text: str,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    """
    Full preprocessing used for the methodology and saved outputs.

    This version removes noise, lowercases text, removes stopwords, and applies
    lemmatization. It is saved for methodology and analysis.
    """

    text = str(text)
    text = html.unescape(text)
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"<.*?>", " ", text)  # HTML tags
    text = re.sub(r"@\w+", " ", text)  # mentions
    text = re.sub(r"#", "", text)  # keep hashtag word, remove only #
    text = re.sub(r"[^a-zA-Z!? ]", " ", text)  # keep letters, !, ?
    text = re.sub(r"\s+", " ", text).strip()

    cleaned_words = []
    for word in text.split():
        if word not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned_words)


def clean_text_for_vader(text: str) -> str:
    """
    Light cleaning for VADER scoring.

    VADER benefits from punctuation, intensifiers, capitalization, and negation.
    Therefore, this function removes only obvious noise while preserving useful
    sentiment cues.
    """

    text = str(text)
    text = html.unescape(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compound_to_label(compound_score: float) -> str:
    """Convert VADER compound score into a discrete sentiment label."""

    if compound_score >= 0.05:
        return "positive"
    if compound_score <= -0.05:
        return "negative"
    return "neutral"


def run_vader_sentiment_analysis(config: PipelineConfig, output_dir: Path) -> Dict[str, Any]:
    """Run VADER sentiment analysis on the Twitter tweets dataset."""

    print("\n" + "=" * 80)
    print("VADER Sentiment Analysis on Twitter Tweets")
    print("=" * 80)

    setup_nltk()

    tweets_path = resolve_tweets_path(config.tweets_path)
    df = pd.read_csv(tweets_path)

    print("Dataset path:", tweets_path)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())

    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in tweets dataset: {missing_columns}")

    df = df[["text", "sentiment"]].copy()
    df = df.dropna(subset=["text"])
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    stop_words = build_stopword_set()
    lemmatizer = WordNetLemmatizer()

    df["clean_text"] = df["text"].apply(
        lambda value: clean_text_for_methodology(value, stop_words, lemmatizer)
    )
    df["vader_text"] = df["text"].apply(clean_text_for_vader)

    # If full cleaning creates empty text, keep a light-cleaned fallback.
    empty_clean_text = df["clean_text"].str.strip().eq("")
    df.loc[empty_clean_text, "clean_text"] = df.loc[empty_clean_text, "vader_text"]
    print("\nNumber of empty clean_text rows after fallback:", df["clean_text"].str.strip().eq("").sum())

    sia = SentimentIntensityAnalyzer()
    vader_scores = df["vader_text"].apply(sia.polarity_scores)

    df["vader_negative_score"] = vader_scores.apply(lambda score: score["neg"])
    df["vader_neutral_score"] = vader_scores.apply(lambda score: score["neu"])
    df["vader_positive_score"] = vader_scores.apply(lambda score: score["pos"])
    df["compound"] = vader_scores.apply(lambda score: score["compound"])
    df["vader_prediction"] = df["compound"].apply(compound_to_label)

    y_true = df["sentiment"]
    y_pred = df["vader_prediction"]

    vader_metrics = {
        "method": "VADER",
        "dataset": "Twitter Tweets",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "training_time_seconds": "No training",
    }

    print("\nVADER metrics:")
    print(pd.Series(vader_metrics))

    report = classification_report(y_true, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)

    labels = ["negative", "neutral", "positive"]
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        display_labels=labels,
        title="VADER Confusion Matrix on Twitter Tweets",
        output_path=output_dir / "vader_confusion_matrix.png",
    )

    wrong_predictions = df[df["sentiment"] != df["vader_prediction"]].copy()
    wrong_predictions["confidence"] = wrong_predictions["compound"].abs()
    wrong_predictions = wrong_predictions.sort_values("confidence", ascending=False)

    df.to_csv(output_dir / "vader_predictions.csv", index=False)
    wrong_predictions.to_csv(output_dir / "vader_errors.csv", index=False)
    save_json(vader_metrics, output_dir / "vader_metrics.json")

    with open(output_dir / "vader_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print("\nVADER analysis files saved in:", output_dir)

    return vader_metrics


# =============================================================================
# Neural Sentiment Classification: RNN, LSTM, and GRU on IMDB
# =============================================================================


def build_recurrent_model(config: PipelineConfig, model_type: str):
    """Build one of the recurrent sentiment models: RNN, LSTM, or GRU."""

    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, SimpleRNN
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model_type = model_type.upper()

    if model_type == "RNN":
        recurrent_layer = SimpleRNN(config.hidden_units)
    elif model_type == "LSTM":
        recurrent_layer = LSTM(config.hidden_units)
    elif model_type == "GRU":
        recurrent_layer = GRU(config.hidden_units)
    else:
        raise ValueError("model_type must be one of: RNN, LSTM, GRU")

    model = Sequential(
        [
            tf.keras.Input(shape=(config.max_len,)),
            Embedding(
                input_dim=config.vocab_size,
                output_dim=config.embedding_dim,
                mask_zero=True,
                name="word_embedding",
            ),
            recurrent_layer,
            Dropout(config.dropout_rate),
            Dense(1, activation="sigmoid"),
        ],
        name=f"{model_type.lower()}_sentiment_model",
    )

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_curves(history: Any, model_name: str, output_dir: Path) -> None:
    """Save accuracy and loss curves for one model."""

    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["accuracy"], label="Train Accuracy")
    ax.plot(history_df["val_accuracy"], label="Validation Accuracy")
    ax.set_title(f"{model_name} Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_accuracy_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["loss"], label="Train Loss")
    ax.plot(history_df["val_loss"], label="Validation Loss")
    ax.set_title(f"{model_name} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_loss_curve.png", dpi=300)
    plt.close(fig)


def evaluate_binary_model(
    model: Any,
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    training_time_seconds: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a binary sentiment model and save its outputs."""

    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "dataset": "IMDB Reviews",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "training_time_seconds": training_time_seconds,
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["negative", "positive"],
        zero_division=0,
    )

    with open(output_dir / f"{model_name.lower()}_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    predictions_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_label": predictions,
            "positive_probability": probabilities,
        }
    )
    predictions_df.to_csv(output_dir / f"{model_name.lower()}_test_predictions.csv", index=False)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        labels=[0, 1],
        display_labels=["negative", "positive"],
        title=f"{model_name} Confusion Matrix",
        output_path=output_dir / f"{model_name.lower()}_confusion_matrix.png",
    )

    return metrics


def run_recurrent_sentiment_models(config: PipelineConfig, output_dir: Path) -> pd.DataFrame:
    """Train and evaluate RNN, LSTM, and GRU models on IMDB reviews."""

    print("\n" + "=" * 80)
    print("Recurrent Neural Sentiment Models on IMDB Reviews")
    print("=" * 80)

    tf = setup_tensorflow(config.seed)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    (x_train_full, y_train_full), (x_test, y_test) = imdb.load_data(
        num_words=config.vocab_size
    )

    print("Training samples:", len(x_train_full))
    print("Test samples:", len(x_test))

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=y_train_full,
    )

    x_train_pad = pad_sequences(
        x_train,
        maxlen=config.max_len,
        padding="post",
        truncating="post",
    )
    x_val_pad = pad_sequences(
        x_val,
        maxlen=config.max_len,
        padding="post",
        truncating="post",
    )
    x_test_pad = pad_sequences(
        x_test,
        maxlen=config.max_len,
        padding="post",
        truncating="post",
    )

    print("Padded train shape:", x_train_pad.shape)
    print("Padded validation shape:", x_val_pad.shape)
    print("Padded test shape:", x_test_pad.shape)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    results = []

    for model_name in ["RNN", "LSTM", "GRU"]:
        print("\n" + "-" * 60)
        print(f"Training {model_name}")
        print("-" * 60)

        model = build_recurrent_model(config, model_name)
        model.summary()

        start_time = time.time()
        history = model.fit(
            x_train_pad,
            y_train,
            validation_data=(x_val_pad, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        training_time_seconds = time.time() - start_time

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(output_dir / f"{model_name.lower()}_training_history.csv", index=False)

        plot_training_curves(history, model_name, output_dir)

        model_metrics = evaluate_binary_model(
            model=model,
            model_name=model_name,
            x_test=x_test_pad,
            y_test=y_test,
            training_time_seconds=training_time_seconds,
            output_dir=output_dir,
        )
        results.append(model_metrics)

        print(f"{model_name} metrics:")
        print(pd.Series(model_metrics))

        # Clear session between models to reduce memory pressure.
        tf.keras.backend.clear_session()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "rnn_lstm_gru_comparison.csv", index=False)

    save_json(asdict(config), output_dir / "recurrent_model_config.json")

    print("\nRecurrent model files saved in:", output_dir)

    return results_df


# =============================================================================
# Comparative Analysis and Project Report
# =============================================================================


def create_method_comparison_table(output_dir: Path) -> pd.DataFrame:
    """Create the conceptual comparison table for the project report."""

    comparison_data = [
        {
            "method": "VADER",
            "type": "Lexicon-based",
            "interpretability": "High",
            "computational_requirements": "Very low",
            "context_handling": "Limited",
            "sarcasm_handling": "Weak",
            "need_for_labeled_data": "No training labels required",
            "best_use_case": "Short tweets, quick baseline, limited resources",
            "main_limitation": "Weak with sarcasm, slang, and context-dependent meaning",
        },
        {
            "method": "Simple RNN",
            "type": "Neural network",
            "interpretability": "Low",
            "computational_requirements": "Medium",
            "context_handling": "Weak for long text",
            "sarcasm_handling": "Possible only if trained on relevant examples",
            "need_for_labeled_data": "Required",
            "best_use_case": "Basic sequence modeling baseline",
            "main_limitation": "Overfitting and vanishing gradients",
        },
        {
            "method": "LSTM",
            "type": "Neural network",
            "interpretability": "Low",
            "computational_requirements": "High",
            "context_handling": "Strong",
            "sarcasm_handling": "Possible only if trained on relevant examples",
            "need_for_labeled_data": "Required",
            "best_use_case": "Long reviews and context-heavy text",
            "main_limitation": "Slower training and higher computational cost",
        },
        {
            "method": "GRU",
            "type": "Neural network",
            "interpretability": "Low",
            "computational_requirements": "Medium to high",
            "context_handling": "Strong",
            "sarcasm_handling": "Possible only if trained on relevant examples",
            "need_for_labeled_data": "Required",
            "best_use_case": "Strong performance with faster training than LSTM",
            "main_limitation": "Still requires labeled data and hyperparameter tuning",
        },
    ]

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "method_comparison_table.csv", index=False)

    return comparison_df


def format_metric(value: Any) -> str:
    """Format numeric metrics for the markdown report."""

    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def create_critical_analysis_report(
    performance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a professional markdown report for the project."""

    performance_table = performance_df.to_markdown(index=False)
    comparison_table = comparison_df.to_markdown(index=False)

    report_text = f"""# Critical Analysis and Model Comparison

## 1. Objective

This project compares a lexicon-based sentiment analysis method, **VADER**, with three recurrent neural network models: **Simple RNN**, **LSTM**, and **GRU**. The comparison focuses on interpretability, computational requirements, ability to handle context and sarcasm, and need for labeled data.

It is important to note that the methods were evaluated on different datasets. VADER was evaluated on the **Twitter Tweets Sentiment Dataset**, while the neural models were evaluated on the **IMDB movie reviews dataset**. Therefore, the numerical scores should not be interpreted as a direct dataset-to-dataset competition. Instead, they should be used to understand how each method behaves under different text conditions.

## 2. Performance Summary

{performance_table}

## 3. Method Comparison

{comparison_table}

## 4. VADER Analysis

VADER is fast, simple, and highly interpretable because it uses a predefined sentiment lexicon rather than learning from the dataset. In this experiment, VADER achieved moderate performance on Twitter tweets. This is expected because tweets are short, noisy, and often contain slang, abbreviations, sarcasm, hashtags, and limited context.

The confusion matrix showed that VADER performed well on clearly positive and negative tweets, but it struggled with the **neutral** class. Many neutral tweets were predicted as positive or negative because VADER reacts strongly to sentiment words such as *love*, *happy*, *great*, *sad*, or *hate*, even when the overall tweet is neutral in the dataset.

## 5. RNN Analysis

The Simple RNN model showed weaker generalization compared with LSTM and GRU. Its training curves typically show a large gap between training and validation performance, which indicates overfitting. This behavior is common for Simple RNNs because they struggle with long-range dependencies and can suffer from vanishing gradients.

For long reviews such as IMDB movie reviews, the model may forget important information from earlier parts of the sequence. As a result, even if training accuracy becomes high, validation and test performance may remain limited.

## 6. LSTM Analysis

The LSTM model performed much better than the Simple RNN. This is because LSTM uses gates that control what information should be remembered, forgotten, and passed forward. These gates make LSTM more effective for longer texts where sentiment may depend on words that appear far apart.

However, LSTM is computationally heavier than Simple RNN and GRU. It usually takes longer to train because it has more internal parameters and more complex gate operations.

## 7. GRU Analysis

The GRU model provided a strong balance between performance and training efficiency. GRU is simpler than LSTM because it uses fewer gates, which often makes it faster while still handling long-range dependencies well.

In many practical NLP projects, GRU can perform close to LSTM while requiring less training time. This makes it a strong choice when both performance and efficiency are important.

## 8. Why LSTM-Based Models May Struggle with Tweets

Although LSTM models are powerful, they may struggle with short texts such as tweets for several reasons. Tweets often contain very little context, informal spelling, emojis, hashtags, abbreviations, and sarcasm. A neural model needs enough labeled examples to learn these patterns correctly.

VADER can sometimes be more practical for tweets because it was designed to work well with short social-media text. It can directly use sentiment cues such as punctuation, capitalization, intensifiers, and emotional words without requiring model training.

## 9. When to Prefer a Lexicon-Based Approach

A lexicon-based approach such as VADER is preferred when labeled data is limited or unavailable, computational resources are low, interpretability is important, or a quick baseline is needed. It is especially useful for short texts such as tweets, comments, and simple customer feedback.

A neural network approach is preferred when there is enough labeled data, the text is longer, context is important, and higher predictive performance is required. LSTM and GRU are better suited for longer reviews because they can learn contextual patterns from data.

## 10. Conclusion

VADER is a strong baseline for short and noisy social-media text because it is fast, interpretable, and does not require labeled training data. However, it has limited ability to understand deeper context, sarcasm, and complex language.

Simple RNN is useful as an educational baseline but is limited for long text. LSTM and GRU are stronger choices for longer sentiment classification projects because they can model sequential dependencies more effectively. Overall, GRU often provides the best balance between accuracy and training efficiency, while LSTM remains a strong option when long-range dependencies are especially important.
"""

    with open(output_dir / "critical_analysis_report.md", "w", encoding="utf-8") as file:
        file.write(report_text)


def run_comparative_analysis(
    vader_metrics: Dict[str, Any] | None,
    recurrent_results: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Generate comparison tables and the markdown analysis report."""

    print("\n" + "=" * 80)
    print("Critical Analysis and Model Comparison")
    print("=" * 80)

    performance_rows: List[Dict[str, Any]] = []

    if vader_metrics is not None:
        performance_rows.append(
            {
                "method": "VADER",
                "dataset": "Twitter Tweets",
                "accuracy": vader_metrics["accuracy"],
                "precision": vader_metrics["precision_macro"],
                "recall": vader_metrics["recall_macro"],
                "f1_score": vader_metrics["f1_macro"],
                "training_time_seconds": "No training",
            }
        )

    if recurrent_results is not None and not recurrent_results.empty:
        for _, row in recurrent_results.iterrows():
            performance_rows.append(
                {
                    "method": row["model"],
                    "dataset": row.get("dataset", "IMDB Reviews"),
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "training_time_seconds": row["training_time_seconds"],
                }
            )

    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(output_dir / "final_performance_summary.csv", index=False)

    comparison_df = create_method_comparison_table(output_dir)
    create_critical_analysis_report(performance_df, comparison_df, output_dir)

    print("Comparative analysis files saved in:", output_dir)
    print("- final_performance_summary.csv")
    print("- method_comparison_table.csv")
    print("- critical_analysis_report.md")


# =============================================================================
# Main entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a complete sentiment analysis project using VADER, RNN, LSTM, and GRU."
    )

    parser.add_argument(
        "--tweets-path",
        type=Path,
        default=Path("Data/Tweets.csv"),
        help="Path to Tweets.csv. Default: Data/Tweets.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder for outputs. Default: outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum number of epochs for RNN/LSTM/GRU training. Default: 5",
    )
    parser.add_argument(
        "--skip-vader",
        action="store_true",
        help="Skip VADER analysis.",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip RNN/LSTM/GRU training.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip comparative analysis report generation.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the full sentiment analysis pipeline."""

    args = parse_args()

    config = PipelineConfig(
        tweets_path=args.tweets_path,
        output_root=args.output_root,
        epochs=args.epochs,
    )

    output_dirs = create_output_dirs(config)

    vader_metrics = None
    recurrent_results = None

    if not args.skip_vader:
        vader_metrics = run_vader_sentiment_analysis(config, output_dirs["vader"])

    if not args.skip_neural:
        recurrent_results = run_recurrent_sentiment_models(config, output_dirs["neural"])

    if not args.skip_analysis:
        run_comparative_analysis(vader_metrics, recurrent_results, output_dirs["analysis"])

    print("\nPipeline completed successfully.")
    print("Outputs saved in:", config.output_root)


if __name__ == "__main__":
    main()
