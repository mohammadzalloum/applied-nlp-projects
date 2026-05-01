# NLP Entity and POS Analysis Pipeline

A complete NLP project that applies **Named Entity Recognition (NER)**, **Part-of-Speech (POS) Tagging**, and a **rule-based EntityRuler fallback** using spaCy.

The project analyzes a small multi-domain text corpus covering technology, sports, politics, and science/literature. It converts raw text into structured outputs such as extracted named entities, POS-tagged tokens, grammar patterns, visualizations, and Markdown reports.

---

## Project Overview

This project includes three main tasks:

### Task 1: Named Entity Recognition

Uses spaCy's pre-trained `en_core_web_sm` model to extract named entities such as:

- `PERSON`
- `ORG`
- `DATE`
- `GPE`
- `MONEY`
- `FAC`
- `LOC`
- `WORK_OF_ART`

The NER output is saved as CSV and JSON files, and the extracted entities are visualized in an HTML file using displaCy.

### Task 2: POS Tagging and Grammar Pattern Analysis

Uses spaCy to extract token-level linguistic information, including:

- token text
- lemma
- POS tag
- detailed tag
- dependency label
- head token
- stop-word status

The task also generates a POS frequency table, a POS distribution chart, highlighted POS visualization, subject-verb patterns, and verb-object patterns.

### Task 3: Rule-Based EntityRuler Fallback

Extends the pre-trained NER model with a custom rule-based label:

```text
TECHNOLOGY
```

This allows the pipeline to detect domain-specific technology terms such as:

- Nexus X
- AI-powered smartphone
- Digital Wellness
- machine learning platform
- natural language processing
- cloud computing
- neural network
- AI assistant

This demonstrates how rule-based NLP can improve a pre-trained model without collecting training data or retraining the model.

---

## Key Results

### NER Results

The NER pipeline extracted **41 named entities** from the sample corpus.

| Entity Label | Count |
|---|---:|
| GPE | 11 |
| ORG | 8 |
| DATE | 8 |
| PERSON | 7 |
| NORP | 2 |
| WORK_OF_ART | 2 |
| MONEY | 1 |
| FAC | 1 |
| LOC | 1 |

### POS Tagging Results

Most frequent POS tags:

| POS Tag | Count |
|---|---:|
| PROPN | 62 |
| PUNCT | 39 |
| NOUN | 39 |
| VERB | 30 |
| ADP | 27 |
| DET | 22 |
| ADJ | 11 |
| NUM | 9 |
| PART | 6 |
| AUX | 6 |

### EntityRuler Results

The custom EntityRuler detected **9 TECHNOLOGY entities** in the test text.

---

## Project Structure

```text
.
├── nlp_entity_pos_pipeline.py
├── setup.sh
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    ├── annotated_ner.html
    ├── ner_entities.csv
    ├── ner_entities.json
    ├── required_ner_entities.csv
    ├── required_ner_entities.json
    ├── ner_summary_table.csv
    ├── ner_report.md
    ├── pos_tags.csv
    ├── pos_tags.json
    ├── pos_summary.csv
    ├── category_pos_summary.csv
    ├── grammar_patterns.csv
    ├── pos_distribution.png
    ├── pos_highlight.html
    ├── pos_insights_report.md
    ├── entity_ruler_entities.csv
    ├── entity_ruler_entities.json
    ├── technology_entities_only.csv
    ├── entity_ruler_comparison.csv
    ├── entity_ruler_technology.html
    └── task3_entity_ruler_report.md
```

---

## Setup

The easiest way to set up the project is to use the provided setup script.

Make the script executable:

```bash
chmod +x setup.sh
```

Run the setup script:

```bash
./setup.sh
```

The script will:

1. Check that `python3` is installed.
2. Create a `.venv` virtual environment if it does not already exist.
3. Activate the virtual environment.
4. Upgrade `pip`, `setuptools`, and `wheel`.
5. Install dependencies from `requirements.txt`.
6. Download the spaCy English model `en_core_web_sm`.
7. Verify that the spaCy model loads correctly.

---

## Manual Installation

If you prefer to set up the project manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## How to Run

After setup, run the full NLP pipeline:

```bash
python nlp_entity_pos_pipeline.py
```

The script creates an `outputs/` folder and saves all generated files there.

---

## Main Output Files

### NER Outputs

| File | Description |
|---|---|
| `outputs/ner_entities.csv` | All extracted named entities |
| `outputs/ner_entities.json` | JSON version of all named entities |
| `outputs/required_ner_entities.csv` | Only required labels: PERSON, ORG, DATE, GPE |
| `outputs/annotated_ner.html` | HTML visualization of named entities |
| `outputs/ner_report.md` | Summary report for Task 1 |

### POS Outputs

| File | Description |
|---|---|
| `outputs/pos_tags.csv` | Token-level POS tagging dataset |
| `outputs/pos_tags.json` | JSON version of POS-tagged tokens |
| `outputs/pos_summary.csv` | POS tag frequency summary |
| `outputs/pos_distribution.png` | Bar chart of POS tag frequencies |
| `outputs/pos_highlight.html` | HTML visualization of highlighted POS tags |
| `outputs/grammar_patterns.csv` | Extracted subject-verb and verb-object patterns |
| `outputs/pos_insights_report.md` | Summary report for Task 2 |

### EntityRuler Outputs

| File | Description |
|---|---|
| `outputs/entity_ruler_entities.csv` | All entities extracted from the EntityRuler test text |
| `outputs/entity_ruler_entities.json` | JSON version of EntityRuler results |
| `outputs/technology_entities_only.csv` | Only TECHNOLOGY entities |
| `outputs/entity_ruler_comparison.csv` | Comparison before and after EntityRuler |
| `outputs/entity_ruler_technology.html` | HTML visualization of custom TECHNOLOGY entities |
| `outputs/task3_entity_ruler_report.md` | Summary report for Task 3 |

---

## Example Insights

The corpus contains many proper nouns because it includes people, cities, organizations, products, and events. This explains why `PROPN` is the most frequent POS tag.

The NER model identifies many real-world entities such as people, organizations, dates, and locations. Some predictions may be imperfect because the project uses a small pre-trained model without fine-tuning.

The EntityRuler improves the pipeline by detecting technology-specific terms that the default NER model may miss. This is useful when working with domain-specific text.

---

## Technologies Used

- Python
- spaCy
- pandas
- matplotlib
- displaCy

---

## Notes

- The project uses a small sample corpus for demonstration and learning.
- The model is not fine-tuned.
- Rule-based matching is used only for custom technology terms.
- Output files are included to make the project easy to review on GitHub.

---

## Author

Mohammad Zalloum
