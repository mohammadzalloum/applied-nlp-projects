# Book Reviews NLP Preprocessing and Word2Vec

A reproducible NLP project for cleaning, exploring, and modeling book review text using Python.

The project is divided into two main tasks:

1. **Text Preprocessing and Exploratory Data Analysis (EDA)**
2. **Word2Vec Training, Evaluation, Visualization, and Document Embeddings**

The final notebook is organized for GitHub submission and can regenerate all project outputs from the raw dataset.

---

## Project Overview

This project uses a book reviews dataset that contains book metadata, genres, authors, ratings, reviewer information, and raw review text. The main goal is to transform noisy review text into clean NLP-ready text, analyze the dataset visually, and train a Word2Vec model that learns contextual relationships between words used in book reviews.

The workflow includes:

- Cleaning raw review text.
- Removing URLs, HTML tags, special characters, noisy tokens, and stopwords.
- Tokenizing and lemmatizing reviews.
- Creating a cleaned review column.
- Analyzing rating distribution, review length, and word frequencies.
- Comparing original vs. cleaned review examples.
- Training a Word2Vec model using Gensim.
- Evaluating embeddings with word similarity queries.
- Visualizing word vectors using PCA and t-SNE.
- Creating document-level embeddings by averaging word vectors.

---

## Dataset

The dataset used in this project was provided as part of a **Sprints.ai course assignment**.

The raw dataset is **not included** in this repository because it may be course-provided material and may have sharing restrictions.

To reproduce the project, place the dataset file in the project root with this exact name:

```text
all_data.csv
```

Main columns used in the notebook:

| Column | Description |
|---|---|
| `book_title` | Book title |
| `Book_series` | Book series name, if available |
| `book_rating` | Book rating |
| `book_author` | Author name |
| `genre` | Book genres |
| `reviewer_name` | Reviewer name |
| `review` | Raw review text |
| `ID` | Review ID |

Main NLP column:

```text
review
```

Main rating column:

```text
book_rating
```

---

## Repository Structure

Recommended GitHub structure:

```text
.
├── book_reviews_nlp_word2vec.ipynb
├── README.md
├── requirements.txt
├── setup.sh
├── .gitignore
└── outputs/
    ├── original_vs_cleaned_examples.csv
    ├── word_frequency_before_preprocessing.csv
    ├── word_frequency_after_preprocessing.csv
    ├── plots/
    │   ├── eda_plots_report.pdf
    │   ├── rating_distribution_histogram.png
    │   ├── rating_distribution_pie.png
    │   ├── review_length_by_rating_boxplot.png
    │   ├── rating_vs_review_length_scatter.png
    │   ├── word_frequency_before_preprocessing.png
    │   ├── word_frequency_after_preprocessing.png
    │   └── wordcloud_after_preprocessing.png
    └── word2vec/
        ├── word_similarity_results.csv
        ├── word_pair_similarity.csv
        ├── word2vec_pca_top_words.png
        ├── word2vec_tsne_top_words.png
        └── word2vec_training_summary.md
```

Files that are usually **not committed** because they are raw, private, or large:

```text
all_data.csv
main.ipynb
hw.docx
outputs/preprocessed_book_reviews.csv
outputs/word2vec/text_review_word2vec.model
outputs/word2vec/document_embeddings.csv
```

`main.ipynb` can be kept locally as a draft or working notebook. The clean notebook for GitHub is:

```text
book_reviews_nlp_word2vec.ipynb
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script if available:

```bash
bash setup.sh
```

### 4. Add the dataset

Place the course-provided dataset in the project root:

```text
all_data.csv
```

### 5. Run the notebook

```bash
jupyter notebook book_reviews_nlp_word2vec.ipynb
```

Run all cells from top to bottom. Generated files will be saved inside:

```text
outputs/
```

---

## Required Libraries

The project uses:

- `pandas`
- `numpy`
- `matplotlib`
- `nltk`
- `wordcloud`
- `gensim`
- `scikit-learn`
- `scipy`
- `jupyter`
- `ipykernel`

NLTK resources are downloaded inside the notebook.

---

# Task 1: Text Preprocessing and EDA

## Text Preprocessing

The raw review text is cleaned using the following steps:

1. Convert text to string.
2. Remove `(less)` text commonly found in scraped review data.
3. Remove URLs.
4. Remove HTML tags.
5. Convert text to lowercase.
6. Tokenize text using NLTK.
7. Keep alphabetic tokens only.
8. Remove English stopwords and custom extra stopwords.
9. Lemmatize tokens.
10. Save cleaned text in `cleaned_review`.
11. Save tokenized text for Word2Vec training.

The cleaned dataset is generated as:

```text
outputs/preprocessed_book_reviews.csv
```

This file can be regenerated by running the notebook and is usually not committed if it is large.

---

## Custom Stopwords

Additional stopwords were used to remove high-frequency words that were not useful for the analysis:

```python
{
    "br", "amp", "quot",
    "n't", "'s", "'m", "'re", "'ve", "'ll",
    "would", "could", "also",
    "one", "get", "thing", "even", "make", "really", "know", "much",
    "way", "people", "want", "think", "well",
    "review", "see",
    "u", "ca", "de", "byte", "ymmv"
}
```

This improved the cleaned word frequency results by removing noisy and overly general tokens while keeping meaningful book-review words.

---

## EDA Outputs

| Output | Description |
|---|---|
| `outputs/plots/rating_distribution_histogram.png` | Histogram of book ratings |
| `outputs/plots/rating_distribution_pie.png` | Pie chart of rating groups |
| `outputs/plots/review_length_by_rating_boxplot.png` | Review length by rating group |
| `outputs/plots/rating_vs_review_length_scatter.png` | Relationship between rating and review word count |
| `outputs/word_frequency_before_preprocessing.csv` | Top words before cleaning |
| `outputs/word_frequency_after_preprocessing.csv` | Top words after cleaning |
| `outputs/original_vs_cleaned_examples.csv` | Side-by-side original and cleaned review examples |
| `outputs/plots/eda_plots_report.pdf` | Combined EDA report |

---

## Key EDA Findings

### Rating Distribution

Most book ratings are concentrated in the higher rating ranges:

| Rating Group | Percentage |
|---|---:|
| `<= 3.5` | 6.3% |
| `3.51 - 4.00` | 35.6% |
| `4.01 - 4.50` | 54.3% |
| `4.51 - 5.00` | 3.8% |

This shows that the dataset is skewed toward highly rated books.

### Review Length vs. Rating

Review word count varies widely across all rating groups. The boxplot and scatter plot do not show a strong visual relationship between book rating and review length. Some reviews are very long, but long reviews appear across different rating ranges.

### Word Frequency Before Preprocessing

Before preprocessing, the most frequent words were mostly stopwords:

| Word | Frequency |
|---|---:|
| `the` | 280,487 |
| `and` | 204,638 |
| `to` | 158,894 |
| `a` | 157,720 |
| `i` | 157,246 |

This confirms that raw text contains many uninformative high-frequency words.

### Word Frequency After Preprocessing

After preprocessing, the most frequent words became more meaningful:

| Word | Frequency |
|---|---:|
| `book` | 58,331 |
| `story` | 28,069 |
| `read` | 21,773 |
| `like` | 21,388 |
| `character` | 20,345 |
| `love` | 18,763 |
| `series` | 9,702 |
| `author` | 8,464 |
| `novel` | 7,871 |
| `star` | 7,649 |

These words better represent book content, reader experience, sentiment, and review context.

---

# Task 2: Word2Vec Training

## Word2Vec Input

The cleaned reviews are converted into token lists, for example:

```python
["book", "story", "character", "love", "author", "novel"]
```

The Word2Vec model is trained on a list of tokenized reviews.

---

## Training Summary

| Item | Value |
|---|---:|
| Reviews used for training | 20,395 |
| Total tokens | 2,840,613 |
| Unique tokens before filtering | 78,873 |
| Final Word2Vec vocabulary size | 24,233 |

---

## Model Parameters

| Parameter | Value |
|---|---|
| `vector_size` | 100 |
| `window` | 5 |
| `min_count` | 5 |
| `sg` | 1, Skip-gram |
| `negative` | 10 |
| `sample` | 1e-3 |
| `epochs` | 20 |

Skip-gram was used because it is effective for learning contextual word relationships, especially when important terms may not always be extremely frequent.

The trained model is generated as:

```text
outputs/word2vec/text_review_word2vec.model
```

This file can be regenerated and may be excluded from GitHub if it is large.

---

## Word Similarity Evaluation

The model was evaluated using similarity queries and pair similarities.

Example pair similarities:

| Word 1 | Word 2 | Similarity |
|---|---|---:|
| `book` | `story` | 0.7287 |
| `book` | `novel` | 0.7664 |
| `love` | `loved` | 0.7353 |
| `character` | `story` | 0.6730 |
| `author` | `book` | 0.6980 |
| `series` | `book` | 0.7702 |
| `star` | `good` | 0.4353 |

These results show that the model learned meaningful relationships between book-review terms.

### Interpretation

- `book`, `novel`, `story`, `read`, and `series` appear close together because they are central to book review language.
- `love` and `loved` are close because they share both meaning and review context.
- `series` and `book` are highly related because many reviews discuss books as part of a series.
- `star` and `good` are related, but less strongly, because `star` is often used in rating contexts while `good` is used in opinion contexts.

---

## PCA and t-SNE Visualizations

The project includes two vector visualizations:

```text
outputs/word2vec/word2vec_pca_top_words.png
outputs/word2vec/word2vec_tsne_top_words.png
```

### PCA Interpretation

The PCA plot shows broad groupings of related terms:

- Reading and book terms: `book`, `read`, `reading`, `author`, `novel`, `writing`, `series`
- Story elements: `story`, `plot`, `character`, `romance`
- People and relationship terms: `man`, `woman`, `girl`, `child`, `family`, `relationship`
- Sentiment and opinion terms: `good`, `great`, `liked`, `loved`, `enjoyed`

PCA is useful for a quick overview, but it compresses 100-dimensional vectors into only 2 dimensions, so some relationships may overlap.

### t-SNE Interpretation

The t-SNE plot provides clearer semantic separation:

- `book`, `read`, `reading`, `novel`, `author`, and `writing` appear close together.
- `story`, `character`, `plot`, and `interesting` appear in a related area.
- `love`, `loved`, `liked`, and `enjoyed` appear close together.
- `man`, `woman`, and `girl` form a people-related cluster.
- `felt`, `feel`, and `feeling` appear close together as emotion-related words.

This suggests that the Word2Vec model learned useful contextual relationships from the cleaned reviews.

---

## Document Embeddings

Document embeddings were generated by averaging word vectors for each review.

Generated file:

```text
outputs/word2vec/document_embeddings.csv
```

Each review is represented by a 100-dimensional vector. These embeddings can be used later for:

- Review clustering.
- Rating or genre classification.
- Recommendation systems.
- Similar review search.
- Search and retrieval tasks.

This file may be large, so it can be excluded from GitHub and regenerated by running the notebook.

---

## How to Run

1. Add the course-provided dataset to the project root as:

```text
all_data.csv
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the final notebook:

```bash
jupyter notebook book_reviews_nlp_word2vec.ipynb
```

4. Run all cells from top to bottom.

5. Check generated files inside:

```text
outputs/
```

---

## Main Deliverables

### Task 1

- Cleaned dataset with `cleaned_review`.
- Tokenized reviews.
- Rating distribution visualizations.
- Review length analysis.
- Word frequency before and after preprocessing.
- Original vs. cleaned review examples.

### Task 2

- Trained Word2Vec model.
- Word similarity results.
- Word pair similarity results.
- PCA visualization.
- t-SNE visualization.
- Document embeddings.

---

## Results Summary

The preprocessing pipeline successfully removed noisy text patterns, stopwords, URLs, HTML tags, and uninformative high-frequency words. After preprocessing, the most frequent words became more meaningful and related to book reviews, such as `book`, `story`, `read`, `character`, `love`, `author`, `series`, and `novel`.

The Word2Vec model learned meaningful contextual relationships. Words related to books, reading, story structure, characters, emotions, and ratings appeared close together in similarity queries and visualizations.

Overall, the project successfully prepares raw book review text for NLP analysis and trains useful word embeddings for future machine learning tasks.

---

## Future Improvements

Future work could include:

- Using POS-aware lemmatization for better normalization.
- Comparing Skip-gram and CBOW Word2Vec models.
- Trying different vector sizes, window sizes, and minimum frequency thresholds.
- Using FastText to better handle misspelled and rare words.
- Clustering document embeddings to discover review groups.
- Training a classifier to predict rating groups or genres.
- Building a review similarity search system.

---

## Notes for GitHub Submission

- The raw dataset is not included because it is course-provided material.
- `main.ipynb` is treated as a local draft notebook and is not required for GitHub.
- Large generated files can be excluded and regenerated by running the notebook.
- The final notebook to review is `book_reviews_nlp_word2vec.ipynb`.

---

## Author

**Mohammad Zalloum**  
AI Engineer