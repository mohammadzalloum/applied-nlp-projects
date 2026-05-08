# Sentiment Analysis Comparative Report

## Executive Summary

This project compares two sentiment-analysis approaches: a lexicon-based method using **VADER** on Twitter tweets and neural sequence models using **Simple RNN, LSTM, and GRU** on the IMDB movie reviews dataset. The results show a clear difference between fast rule-based analysis and trained deep-learning models.

VADER achieved moderate performance on short, noisy Twitter text, with **63.14% accuracy** and **0.6294 macro F1-score**. This is reasonable because VADER does not learn from the dataset and must classify three sentiment classes: negative, neutral, and positive. Its weakest area was the neutral class, where many neutral tweets were incorrectly predicted as positive.

For the IMDB dataset, the neural models performed better because the task is binary and the reviews contain more context. **LSTM achieved the best overall balance**, with **84.21% accuracy** and a positive-class F1-score of **0.8401**. **GRU was the fastest model**, training in **11.41 seconds**, and had the highest positive precision (**0.8820**), but it missed more positive reviews than LSTM. **Simple RNN performed as a weaker baseline**, reaching **76.66% accuracy**, but showing clear signs of overfitting.

The main conclusion is that **VADER is useful as a fast, interpretable baseline for short social-media text**, while **LSTM and GRU are more suitable for longer text where context matters**. Among the neural models, LSTM is the strongest overall choice in this experiment, while GRU offers the best speed-efficiency tradeoff.

---

## Datasets and Experimental Setup

Two datasets were used because the project evaluates different types of sentiment-analysis methods.

### Twitter Tweets Dataset

The Twitter dataset contains short and noisy social-media text. After removing one missing text row, the final dataset contained **27,480 tweets**. The class distribution was not perfectly balanced: neutral was the largest class with **11,117 samples**, followed by positive with **8,582 samples**, and negative with **7,781 samples**.

VADER was applied to this dataset because it is designed for short text and social-media-style sentiment cues such as emotional words, punctuation, capitalization, and intensifiers.

### IMDB Movie Reviews Dataset

The IMDB dataset was used for the recurrent neural models. It contains binary sentiment labels:

- `0` = negative
- `1` = positive

The dataset was split into training, validation, and test sets. The training set contained **20,000 reviews**, the validation set contained **5,000 reviews**, and the test set contained **25,000 reviews**. Each review was padded or truncated to a fixed length of 200 tokens before training.

The model architecture was kept consistent across RNN, LSTM, and GRU to make the comparison fair:

- Embedding dimension: 128
- Hidden units: 64
- Dropout: 0.3
- Optimizer: Adam
- Loss function: binary cross-entropy
- Early stopping and learning-rate reduction were used to reduce overfitting
- Training used GPU acceleration

---

## Performance Summary

| Method | Dataset | Accuracy | Main F1-score | Training Time | Main Observation |
|---|---:|---:|---:|---:|---|
| VADER | Twitter Tweets | 0.6314 | 0.6294 macro F1 | No training | Fast and interpretable, but weak on neutral tweets |
| Simple RNN | IMDB Reviews | 0.7666 | 0.7342 positive F1 | 33.70 sec | Improved over VADER numerically, but overfits and misses positives |
| LSTM | IMDB Reviews | 0.8421 | 0.8401 positive F1 | 23.83 sec | Best overall balance and generalization |
| GRU | IMDB Reviews | 0.8225 | 0.8075 positive F1 | 11.41 sec | Fastest model and highest positive precision, but lower positive recall |

Important note: VADER and the neural models should not be compared as a direct accuracy competition because they were evaluated on different datasets with different label structures. VADER was evaluated on a three-class Twitter task, while RNN, LSTM, and GRU were evaluated on a binary IMDB task.

---

## VADER Analysis on Twitter Tweets

VADER achieved the following results:

- Accuracy: **0.6314**
- Macro precision: **0.6536**
- Macro recall: **0.6461**
- Macro F1-score: **0.6294**
- Weighted F1-score: **0.6230**

### Confusion Matrix Interpretation

| True Label | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---:|---:|---:|
| Negative | 4,647 | 1,419 | 1,715 |
| Neutral | 1,661 | 5,241 | 4,215 |
| Positive | 343 | 776 | 7,463 |

The confusion matrix shows that VADER performed well on clearly positive tweets. It correctly classified **7,463 positive tweets**, giving the positive class high recall. However, the positive precision was lower because VADER also predicted many neutral tweets as positive.

The most important weakness was the neutral class. Only **5,241 neutral tweets** were correctly classified, while **4,215 neutral tweets** were predicted as positive. This means VADER is sensitive to positive words even when the overall tweet is not necessarily positive.

For example, a neutral tweet may contain words like “good,” “love,” “fun,” or “happy” in a casual or contextual way. Since VADER depends on a sentiment lexicon, it may classify such tweets as positive even when the dataset label is neutral.

### Why VADER Failed in Some Cases

VADER’s errors are expected for several reasons:

1. **Neutral tweets are hard to detect.**  
   Neutral tweets may contain emotional words but still not express a clear opinion.

2. **Tweets are short and lack context.**  
   A tweet may be a reply to another message, so the sentiment cannot always be understood from the tweet alone.

3. **Sarcasm is difficult for lexicon-based methods.**  
   A sentence such as “Great, my phone died again” contains a positive word but expresses negative sentiment.

4. **Slang, spelling variations, and informal writing are common.**  
   Tweets often include misspellings, abbreviations, hashtags, and expressive forms such as “soooo,” which may not always be interpreted correctly.

Overall, VADER is a strong lightweight baseline, but its performance is limited by the fact that it does not learn from the dataset.

---

## Simple RNN Analysis on IMDB Reviews

The Simple RNN model achieved:

- Accuracy: **0.7666**
- Positive precision: **0.8525**
- Positive recall: **0.6448**
- Positive F1-score: **0.7342**
- Training time: **33.70 seconds**

### Confusion Matrix Interpretation

| True Label | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative | 11,105 | 1,395 |
| Positive | 4,440 | 8,060 |

The RNN correctly classified many negative reviews, but it missed a large number of positive reviews. Specifically, **4,440 positive reviews** were classified as negative. This explains why the positive recall was only **0.6448**.

The model was conservative when predicting positive sentiment: when it predicted positive, it was often correct, but it failed to identify many truly positive reviews.

### Training Curve Interpretation

The RNN training curves show a classic overfitting pattern:

- Training accuracy continued increasing strongly.
- Validation accuracy improved early, then stopped improving and started weakening.
- Training loss decreased sharply.
- Validation loss increased after the early epochs.

This means the RNN learned the training data too closely but did not generalize as well to unseen reviews. This behavior is common for Simple RNNs because they struggle with long-range dependencies and vanishing gradients.

The Simple RNN is useful as a baseline, but it is not the best architecture for long movie reviews.

---

## LSTM Analysis on IMDB Reviews

The LSTM model achieved:

- Accuracy: **0.8421**
- Positive precision: **0.8510**
- Positive recall: **0.8294**
- Positive F1-score: **0.8401**
- Training time: **23.83 seconds**

### Confusion Matrix Interpretation

| True Label | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative | 10,684 | 1,816 |
| Positive | 2,132 | 10,368 |

The LSTM had the best balance between positive and negative classification. It correctly classified **10,684 negative reviews** and **10,368 positive reviews**. Compared with RNN, it reduced the number of missed positive reviews significantly.

The LSTM’s positive recall was **0.8294**, much higher than the RNN’s positive recall of **0.6448**. This indicates that LSTM handled sentiment patterns in longer reviews more effectively.

### Training Curve Interpretation

The LSTM curves show stronger and more stable learning than the Simple RNN:

- Validation accuracy reached around **0.85**.
- Training accuracy continued increasing, but early stopping prevented excessive overfitting.
- Validation loss was lowest around the second epoch, which is why the best weights were restored from that point.

This behavior suggests that LSTM learned useful sequence patterns while still needing regularization to avoid overfitting after several epochs.

### Why LSTM Performed Best Overall

LSTM is designed to handle long-range dependencies using memory gates. This is important for movie reviews because the final sentiment may depend on information spread across many words or sentences.

For example, a review might say:

> “The movie started slowly and some scenes were boring, but the ending was powerful and emotional.”

A simple model may focus only on negative words like “slowly” and “boring,” while LSTM is better at using the broader sequence to understand the final sentiment.

In this experiment, LSTM provided the best overall balance between accuracy, recall, F1-score, and generalization.

---

## GRU Analysis on IMDB Reviews

The GRU model achieved:

- Accuracy: **0.8225**
- Positive precision: **0.8820**
- Positive recall: **0.7446**
- Positive F1-score: **0.8075**
- Training time: **11.41 seconds**

### Confusion Matrix Interpretation

| True Label | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative | 11,255 | 1,245 |
| Positive | 3,192 | 9,308 |

The GRU achieved the highest positive precision among the neural models. When it predicted a review as positive, it was usually correct. It also produced fewer false positives than LSTM and RNN.

However, the GRU missed more positive reviews than LSTM. It classified **3,192 positive reviews** as negative, which reduced positive recall to **0.7446**.

### Training Curve Interpretation

The GRU trained very quickly and stopped early. The validation accuracy improved rapidly, but the model did not continue training long enough to match LSTM’s balance. Its validation loss remained close across early epochs, and early stopping restored the best available weights.

The GRU is therefore the most efficient model in this experiment, but not the strongest overall model.

### Main Strength of GRU

GRU is simpler than LSTM and uses fewer parameters in the recurrent layer. This often makes it faster while keeping competitive performance. In this project, GRU trained in **11.41 seconds**, less than half the LSTM training time.

The best use case for GRU is when training speed matters and slightly lower recall is acceptable.

---

## Critical Comparison

### Interpretability

VADER is the easiest method to interpret because it uses predefined sentiment scores. If it predicts a tweet as positive, the reason is usually linked to known positive words in the text.

RNN, LSTM, and GRU are harder to interpret because they learn internal numerical representations. Their decisions are based on learned weights and hidden states, which are not directly understandable by humans.

### Computational Requirements

VADER is very lightweight. It does not require training and can run quickly on a CPU.

The neural models require more computation because they must learn embeddings and recurrent patterns from data. However, GPU acceleration made training much faster in this project.

### Handling Context

VADER has limited context understanding. It can handle some simple negation and intensifiers, but it cannot deeply understand long-distance meaning.

Simple RNN can process sequences, but it struggles with long-range dependencies. LSTM and GRU are much better at using context across longer texts.

### Handling Sarcasm

None of the models truly “understand” sarcasm by default. VADER is especially weak with sarcasm because it depends heavily on surface-level sentiment words.

LSTM and GRU can learn sarcasm patterns only if trained on enough examples that contain sarcasm. Without sarcasm-specific training data, they can still fail.

### Need for Labeled Data

VADER does not need labeled data for training, which makes it useful when labels are unavailable.

RNN, LSTM, and GRU require labeled data. Their performance depends heavily on the quality, size, and relevance of the training dataset.

---

## Why LSTM-Based Models May Struggle with Tweets

LSTM models are powerful for longer text, but they may struggle with tweets for several reasons:

1. **Tweets are very short.**  
   LSTM benefits from sequence context, but short tweets often do not provide enough context.

2. **Tweets are noisy.**  
   They often include slang, emojis, abbreviations, hashtags, spelling mistakes, and informal grammar.

3. **Sentiment may depend on external context.**  
   A tweet may be a reply or part of a conversation, and the model may not see the missing context.

4. **Sarcasm is common.**  
   Tweets often express sarcasm in ways that require world knowledge or conversational context.

5. **A large Twitter-specific training set is needed.**  
   LSTM would need many labeled tweets to learn informal social-media patterns effectively.

VADER can sometimes work well on tweets because it was designed to respond to short-text sentiment signals such as punctuation, capitalization, intensifiers, and emotional vocabulary.

---

## When to Prefer a Lexicon-Based Approach

A lexicon-based method such as VADER is preferred when:

- There is little or no labeled data.
- A fast baseline is needed.
- Computational resources are limited.
- Interpretability is important.
- The text is short, such as tweets, comments, or quick feedback.
- The goal is rapid analysis rather than maximum predictive performance.

A neural network approach is preferred when:

- There is enough labeled data.
- The text is longer and context matters.
- Higher performance is required.
- Training time and computational resources are available.
- The model needs to learn domain-specific language patterns.

---

## Model Selection Recommendation

Based on the results, the best model depends on the project goal:

| Goal | Recommended Method | Reason |
|---|---|---|
| Fast baseline on tweets | VADER | No training, easy to interpret |
| Best overall neural performance | LSTM | Highest accuracy and most balanced behavior |
| Fast neural model | GRU | Shortest training time and strong precision |
| Educational baseline | Simple RNN | Useful for comparison, but weaker than LSTM/GRU |

For this experiment, **LSTM is the best overall model** because it achieved the most balanced test performance. **GRU is the best efficiency-focused model** because it trained fastest while still performing well. **VADER remains valuable as a fast baseline**, especially for short social-media text.

---

## Limitations

This comparison has several limitations:

1. **Different datasets were used.**  
   VADER was evaluated on Twitter tweets, while the neural models were evaluated on IMDB reviews. Therefore, the results should not be interpreted as a direct competition.

2. **Different label structures were used.**  
   Twitter sentiment has three classes, while IMDB has two classes. Neutral sentiment is especially difficult and lowers VADER’s overall performance.

3. **The neural models used simple architectures.**  
   More advanced models such as Bidirectional LSTM, attention-based models, or transformers could improve performance.

4. **Training results may vary slightly.**  
   Deep-learning results can vary due to initialization, GPU operations, and training randomness.

5. **Sarcasm was not explicitly modeled.**  
   None of the methods used sarcasm-specific training data or features.

---

## Recommendations for Improvement

Several improvements could strengthen the project:

1. **Use early stopping consistently based on validation loss.**  
   This already helped reduce overfitting, especially for RNN and LSTM.

2. **Tune thresholds for neural models.**  
   The default 0.5 threshold may not be optimal. GRU, for example, had high precision but lower recall, so threshold tuning could improve balance.

3. **Try Bidirectional LSTM or Bidirectional GRU.**  
   These models can use both past and future context in a sequence.

4. **Add stronger regularization.**  
   Recurrent dropout, L2 regularization, or smaller embedding dimensions could reduce overfitting.

5. **Evaluate all models on the same dataset if direct comparison is required.**  
   For example, train neural models on the Twitter dataset or evaluate VADER on a binary subset.

6. **Use transformer-based models for a stronger benchmark.**  
   Models such as BERT or DistilBERT would likely improve context understanding, especially for longer reviews.

---

## Conclusion

This project demonstrates the tradeoff between lexicon-based sentiment analysis and neural sequence models. VADER is fast, simple, interpretable, and useful for short social-media text, but it struggles with neutral sentiment, sarcasm, slang, and context-dependent meaning.

Simple RNN provides a basic neural baseline, but its performance is limited by overfitting and difficulty handling long-range dependencies. LSTM performed best overall because it captured longer sequence patterns more effectively and produced balanced results across positive and negative reviews. GRU was the fastest neural model and achieved strong precision, but it missed more positive reviews than LSTM.

Overall, the best method depends on the use case. For quick and interpretable tweet analysis, VADER is a practical choice. For longer review-based sentiment classification, LSTM is the strongest model in this experiment, while GRU is a good alternative when training speed is a priority.
