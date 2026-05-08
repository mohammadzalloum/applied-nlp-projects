# Sentiment Analysis & Classification Evaluation

A complete sentiment analysis project that compares a **lexicon-based approach** using VADER with **recurrent neural network models** using Simple RNN, LSTM, and GRU.

The project evaluates two different sentiment analysis workflows:

1. **VADER Sentiment Analysis** on Twitter tweets.
2. **Recurrent Neural Sentiment Models** on IMDB movie reviews.

It also generates evaluation metrics, confusion matrices, training curves, prediction files, and a professional critical analysis report.

---

## Project Overview

This project demonstrates two major approaches to sentiment analysis:

- **Rule-based / lexicon-based sentiment analysis** using VADER.
- **Deep learning sequence models** using TensorFlow/Keras recurrent architectures.

The goal is not only to compare accuracy, but also to analyze the practical differences between the methods in terms of:

- Interpretability
- Training cost
- Context handling
- Performance
- Overfitting behavior
- Suitability for short vs. long text

> **Important:** VADER is evaluated on Twitter tweets, while the neural models are evaluated on IMDB reviews. Therefore, the numerical results should not be treated as a direct dataset-to-dataset competition.

---

## Features

- Loads and analyzes the Twitter sentiment dataset from `Data/Tweets.csv`.
- Cleans tweet text using URL removal, mention removal, hashtag cleanup, stopword handling, and lemmatization.
- Uses VADER compound scores to classify tweets into negative, neutral, and positive classes.
- Trains and evaluates three recurrent neural models on IMDB reviews:
  - Simple RNN
  - LSTM
  - GRU
- Uses padding and truncation for fixed-length sequence input.
- Supports GPU training with TensorFlow when CUDA is available.
- Generates:
  - Classification reports
  - Confusion matrices
  - Training accuracy curves
  - Training loss curves
  - Final comparison tables
  - Critical analysis report

---

## Project Structure

```text
project/
├── Data/
│   └── Tweets.csv
├── outputs/
│   ├── vader_analysis/
│   ├── recurrent_models/
│   └── comparative_analysis/
├── sentiment_analysis_project.py
├── requirements.txt
├── setup.sh
├── README.md
└── .gitignore
```

---

## Main Files

| File | Description |
|---|---|
| `sentiment_analysis_project.py` | Main Python script for the complete pipeline. |
| `Data/Tweets.csv` | Twitter sentiment dataset used by VADER. |
| `requirements.txt` | Python dependencies. |
| `setup.sh` | Optional environment setup script. |
| `outputs/` | Generated results, plots, reports, and tables. |
| `critical_analysis_report.md` | Final written analysis report. |

---

## Methods Used

### 1. VADER Sentiment Analysis

VADER is a lexicon-based sentiment analysis tool that uses predefined sentiment scores for words and applies rule-based adjustments for punctuation, intensifiers, capitalization, and negation.

In this project, VADER is used on short Twitter posts because it is fast, interpretable, and suitable for social-media-style text.

### 2. Simple RNN

Simple RNN is used as a basic recurrent neural network baseline. It can process sequences, but it often struggles with long-term dependencies and may overfit quickly.

### 3. LSTM

LSTM improves on Simple RNN by using gates that help the model remember or forget information across long sequences. This makes it more suitable for longer reviews.

### 4. GRU

GRU is similar to LSTM but uses a simpler gating mechanism. It often provides a strong balance between performance and training speed.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you want TensorFlow with GPU support on Linux, you can install:

```bash
pip install "tensorflow[and-cuda]"
```

---

## Optional GPU Setup

The project can run on CPU, but the recurrent neural models train faster on GPU.

Check whether TensorFlow can detect your GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected GPU output:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If TensorFlow cannot find CUDA libraries inside the virtual environment, run:

```bash
export LD_LIBRARY_PATH="$(find "$VIRTUAL_ENV/lib" -path "*/site-packages/nvidia/*/lib" -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PATH="$(find "$VIRTUAL_ENV/lib" -path "*/site-packages/nvidia/cuda_nvcc/bin" -type d -print -quit):$PATH"
```

Then test again:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Dataset Setup

Place the Twitter dataset in:

```text
Data/Tweets.csv
```

The expected file should contain at least these columns:

```text
text
sentiment
```

The IMDB dataset is downloaded automatically through Keras when the neural model section runs.

---

## How to Run

Run the complete pipeline:

```bash
python sentiment_analysis_project.py
```

Run with a custom tweet dataset path:

```bash
python sentiment_analysis_project.py --tweets-path Data/Tweets.csv
```

Run with a custom number of epochs:

```bash
python sentiment_analysis_project.py --epochs 5
```

Skip neural model training:

```bash
python sentiment_analysis_project.py --skip-neural
```

Skip VADER analysis:

```bash
python sentiment_analysis_project.py --skip-vader
```

Skip final comparative analysis:

```bash
python sentiment_analysis_project.py --skip-analysis
```

---

## Outputs

After running the project, results are saved inside the `outputs/` folder.

### VADER outputs

```text
outputs/vader_analysis/
├── vader_metrics.json
├── vader_classification_report.txt
├── vader_confusion_matrix.png
├── vader_predictions.csv
└── vader_errors.csv
```

### Recurrent model outputs

```text
outputs/recurrent_models/
├── rnn_lstm_gru_comparison.csv
├── recurrent_model_config.json
├── rnn_training_history.csv
├── lstm_training_history.csv
├── gru_training_history.csv
├── rnn_classification_report.txt
├── lstm_classification_report.txt
├── gru_classification_report.txt
├── rnn_confusion_matrix.png
├── lstm_confusion_matrix.png
├── gru_confusion_matrix.png
├── rnn_accuracy_curve.png
├── lstm_accuracy_curve.png
├── gru_accuracy_curve.png
├── rnn_loss_curve.png
├── lstm_loss_curve.png
└── gru_loss_curve.png
```

### Comparative analysis outputs

```text
outputs/comparative_analysis/
├── final_performance_summary.csv
├── method_comparison_table.csv
└── critical_analysis_report.md
```

---

## Example Results

The exact results may change slightly depending on hardware, TensorFlow version, random seed behavior, and GPU/CPU execution.

| Method | Dataset | Accuracy | Precision | Recall | F1-score | Training Time |
|---|---|---:|---:|---:|---:|---:|
| VADER | Twitter Tweets | 0.6314 | 0.6536 | 0.6461 | 0.6294 | No training |
| Simple RNN | IMDB Reviews | 0.7622 | 0.7879 | 0.7175 | 0.7510 | 32.72s |
| LSTM | IMDB Reviews | 0.8348 | 0.8673 | 0.7905 | 0.8271 | 19.04s |
| GRU | IMDB Reviews | 0.8392 | 0.8307 | 0.8522 | 0.8413 | 16.81s |

In this run, **GRU achieved the strongest overall neural model performance**, with the highest F1-score and the fastest training time among LSTM and GRU.

---

## Key Findings

- VADER is fast, simple, and interpretable, but it struggles with neutral text, sarcasm, slang, and deeper context.
- Simple RNN is useful as a baseline, but it can overfit quickly and struggles with long-range dependencies.
- LSTM performs much better than Simple RNN because it can retain information over longer sequences.
- GRU provides the best balance between accuracy, F1-score, and training efficiency in this experiment.
- Neural models are better suited for longer reviews, while VADER is more practical for short social media text when labeled data or compute resources are limited.

---

## Reproducibility

The project uses a fixed random seed:

```python
seed = 42
```

Neural model settings:

```text
Vocabulary size: 10,000
Max sequence length: 200
Embedding dimension: 128
Hidden units: 64
Dropout rate: 0.3
Batch size: 64
Epochs: 5
Learning rate: 0.001
Validation size: 0.2
```

---

## Notes

- The VADER and neural models are evaluated on different datasets.
- GPU training is optional, but recommended for recurrent neural models.
- Generated prediction files can be large and are usually ignored in Git.
- Confusion matrices and training curves are useful for understanding model behavior beyond accuracy.

---

## Recommended GitHub Files

Recommended files to commit:

```text
sentiment_analysis_project.py
README.md
requirements.txt
setup.sh
.gitignore
Data/Tweets.csv
outputs/comparative_analysis/critical_analysis_report.md
outputs/comparative_analysis/final_performance_summary.csv
outputs/comparative_analysis/method_comparison_table.csv
outputs/recurrent_models/*_confusion_matrix.png
outputs/recurrent_models/*_accuracy_curve.png
outputs/recurrent_models/*_loss_curve.png
outputs/vader_analysis/vader_confusion_matrix.png
outputs/vader_analysis/vader_metrics.json
```

Recommended files to ignore:

```text
.venv/
.vscode/
__pycache__/
*.pyc
.ipynb_checkpoints/
outputs/**/vader_predictions.csv
outputs/**/vader_errors.csv
outputs/**/*_test_predictions.csv
```

---

## Author

Mohammad Zalloum

---

## License

This project is intended for educational purposes.
