
# Word2Vec Training Summary

## Dataset
- Reviews used for training: 20395
- Total tokens: 2840613
- Unique tokens before min_count filtering: 78873
- Word2Vec vocabulary size after min_count filtering: 24233

## Model Parameters
- vector_size: 100
- window: 5
- min_count: 5
- sg: 1  # Skip-gram
- negative: 10
- sample: 1e-3
- epochs: 20

## Main Outputs
- outputs/preprocessed_book_reviews.csv
- outputs/plots/eda_plots_report.pdf
- outputs/word2vec/text_review_word2vec.model
- outputs/word2vec/word_similarity_results.csv
- outputs/word2vec/word_pair_similarity.csv
- outputs/word2vec/word2vec_pca_top_words.png
- outputs/word2vec/word2vec_tsne_top_words.png
- outputs/word2vec/document_embeddings.csv

## Interpretation
The model was trained on cleaned and tokenized book reviews. Similarity queries and vector visualizations show that related words such as book, story, novel, character, love, author, and series appear close together in the learned vector space.
