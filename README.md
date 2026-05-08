# Applied NLP Projects

A collection of applied Natural Language Processing (NLP) projects covering text preprocessing, Word2Vec, named entity recognition, POS tagging, rule-based entity extraction, sentiment analysis, recurrent neural networks, model evaluation, and NLP visualization.
---

## Projects

### 1. Book Reviews NLP & Word2Vec

**Folder:** `01-book-reviews-nlp-word2vec`

This project focuses on text preprocessing and Word2Vec modeling using book review data.

**Main topics:**

- Text preprocessing
- Tokenization
- Stopword removal
- Text cleaning
- Word2Vec training
- Similar word exploration
- Basic NLP analysis

**Goal:**  
Prepare raw review text for NLP modeling and train a Word2Vec model to explore semantic relationships between words.

---

### 2. Named Entity Recognition and POS Analysis

**Folder:** `02-Named Entity Recognition (NER) on Sample Text`

This project applies a complete NLP pipeline using spaCy on a multi-domain text corpus.

**Main tasks:**

- Named Entity Recognition (NER)
- Entity extraction into CSV and JSON files
- Entity visualization using displaCy
- POS tagging
- POS tag frequency analysis
- POS distribution bar chart
- Grammar pattern extraction
- Rule-based entity extraction using spaCy EntityRuler

**Custom EntityRuler label:**

```text
TECHNOLOGY
```
---

3. Sentiment Analysis & Classification Evaluation

Folder: 03-Sentiment Analysis & Classification Evaluation( Vader Vs RNNGRU)

This project compares rule-based sentiment analysis with recurrent neural network models using two different sentiment datasets.

The project includes a VADER sentiment analysis workflow for Twitter tweets and neural sentiment classification models on IMDB movie reviews using Simple RNN, LSTM, and GRU.

Main topics:

- Sentiment analysis
- Text preprocessing
- VADER rule-based sentiment classification
- IMDB review classification
- Simple RNN modeling
- LSTM modeling
- GRU modeling
- Classification metrics
- Confusion matrix analysis
- Accuracy and loss curve visualization
- Model comparison and critical analysis

Models used:

- VADER
- Simple RNN
- LSTM
- GRU

Generated outputs:

- VADER predictions and metrics
- Classification reports
- Confusion matrices
- Training history files
- Accuracy and loss curves
- Final performance comparison table
- Critical analysis report

Goal:
Compare a lexicon-based sentiment analysis method with neural sequence models and evaluate their strengths, weaknesses, performance, interpretability, training cost, and ability to handle contextual sentiment.