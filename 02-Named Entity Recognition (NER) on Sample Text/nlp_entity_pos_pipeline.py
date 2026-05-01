"""NLP Entity and POS Analysis Pipeline.

This script solves three NLP tasks in one clean, reusable pipeline:

1. Named Entity Recognition (NER) using spaCy.
2. POS tagging, POS distribution analysis, and grammar-pattern extraction.
3. Optional rule-based TECHNOLOGY entity extraction using spaCy EntityRuler.

Run:
    python nlp_entity_pos_pipeline.py

Before running for the first time:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc


MODEL_NAME = "en_core_web_sm"
DEFAULT_OUTPUT_DIR = "outputs"
REQUIRED_NER_LABELS = {"PERSON", "ORG", "DATE", "GPE"}
CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@dataclass(frozen=True)
class TextSample:
    """A labeled text sample used by the pipeline."""

    text_id: int
    category: str
    text: str


SAMPLE_CORPUS: list[TextSample] = [
    TextSample(
        text_id=1,
        category="Tech News",
        text="""
        TechCorp, a leading tech company based in New York, announced on Monday
        that CEO Jane Doe plans to introduce a new AI-powered smartphone.
        The device, named 'Nexus X,' will debut in Paris next month. Meanwhile,
        critics like Dr. Mark Smith argue that the launch could impact
        environmental policies. The company also partnered with HealthOrg, a
        nonprofit, to promote digital wellness. Apple declined to comment on
        the news. The event will coincide with the Global Tech Summit 2024.
        """,
    ),
    TextSample(
        text_id=2,
        category="Sports News",
        text="""
        Lionel Messi, the Argentine footballer, signed a $20 million contract
        with Miami FC on July 15, 2023. The deal was announced during a press
        conference at Hard Rock Stadium in Florida. Fans from across South
        America flooded social media to celebrate the move.
        """,
    ),
    TextSample(
        text_id=3,
        category="Politics",
        text="""
        President John Harper met with German Chancellor Angela Weber in Berlin
        last Friday to discuss NATO policies. The United Nations will host a
        climate summit in Geneva, Switzerland, in December 2025. Critics warn
        that the new tax law (HB 1420) might face delays in Congress.
        """,
    ),
    TextSample(
        text_id=4,
        category="Science/Literature",
        text="""
        Marie Curie, born in Warsaw in 1867, discovered radium and won the
        Nobel Prize in Chemistry in 1911. In "The Great Gatsby," Jay Gatsby
        hosts lavish parties in West Egg, New York, reflecting the excesses of
        the Jazz Age. A recent study in Nature Journal links sleep deprivation
        to decreased cognitive performance.
        """,
    ),
]


TECHNOLOGY_PATTERNS = [
    {"label": "TECHNOLOGY", "pattern": "Nexus X"},
    {"label": "TECHNOLOGY", "pattern": "AI-powered smartphone"},
    {"label": "TECHNOLOGY", "pattern": "smartphone"},
    {"label": "TECHNOLOGY", "pattern": "machine learning platform"},
    {"label": "TECHNOLOGY", "pattern": "cloud computing"},
    {"label": "TECHNOLOGY", "pattern": "digital wellness"},
    {"label": "TECHNOLOGY", "pattern": "natural language processing"},
    {"label": "TECHNOLOGY", "pattern": "computer vision"},
    {"label": "TECHNOLOGY", "pattern": "AI assistant"},
    {"label": "TECHNOLOGY", "pattern": "neural network"},
]


TECHNOLOGY_TEST_TEXT = """
TechCorp announced that Nexus X is an AI-powered smartphone designed to improve
Digital Wellness. The company said the device uses a machine learning platform
and natural language processing to support users. Experts believe cloud
computing and neural network technology will make the AI assistant faster.
Apple and Samsung are expected to respond with similar smartphone products next year.
"""


def configure_logging() -> None:
    """Configure readable logging for command-line runs."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def load_spacy_model(model_name: str = MODEL_NAME) -> Language:
    """Load a spaCy model with a helpful error message if it is missing."""

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{model_name}'. Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc


def clean_text(text: str) -> str:
    """Lightly clean text while preserving capitalization and useful punctuation.

    For NER, capitalization and punctuation are useful signals. Therefore, this
    function only normalizes whitespace and removes unusual control characters.
    It intentionally does not lowercase the text.
    """

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text


def prepare_corpus(samples: Iterable[TextSample]) -> list[TextSample]:
    """Return cleaned corpus samples."""

    return [
        TextSample(
            text_id=sample.text_id,
            category=sample.category,
            text=clean_text(sample.text),
        )
        for sample in samples
    ]


def process_corpus(nlp: Language, samples: list[TextSample]) -> list[tuple[TextSample, Doc]]:
    """Run spaCy over all samples efficiently with nlp.pipe."""

    docs = list(nlp.pipe(sample.text for sample in samples))
    return list(zip(samples, docs))


def save_dataframe(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    """Save a DataFrame as both CSV and JSON records."""

    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)

    logging.info("Saved %s and %s", csv_path, json_path)


def extract_ner_entities(processed_docs: list[tuple[TextSample, Doc]]) -> pd.DataFrame:
    """Extract named entities from processed spaCy docs."""

    rows: list[dict[str, object]] = []

    for sample, doc in processed_docs:
        for ent in doc.ents:
            rows.append(
                {
                    "text_id": sample.text_id,
                    "category": sample.category,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    return pd.DataFrame(rows)


def render_ner_html(processed_docs: list[tuple[TextSample, Doc]], output_dir: Path) -> None:
    """Create a displaCy HTML file for NER visualization."""

    docs = [doc for _, doc in processed_docs]
    html_output = displacy.render(docs, style="ent", page=True, jupyter=False)
    output_path = output_dir / "annotated_ner.html"
    output_path.write_text(html_output, encoding="utf-8")
    logging.info("Saved %s", output_path)


def create_ner_report(
    ner_df: pd.DataFrame,
    required_ner_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a concise Markdown report for Task 1."""

    label_counts = ner_df["entity_label"].value_counts().to_string()
    required_label_counts = required_ner_df["entity_label"].value_counts().to_string()

    report = f"""# Task 1: Named Entity Recognition Report

## Objective

Apply Named Entity Recognition (NER) to a small multi-domain text corpus using
spaCy's pre-trained `{MODEL_NAME}` model.

## Method

The text was lightly cleaned by normalizing whitespace. Capitalization was
preserved because it helps NER models identify names, places, and organizations.

## Results

Total extracted entities: **{len(ner_df)}**

Entity label distribution:

```text
{label_counts}
```

Required label distribution (`PERSON`, `ORG`, `DATE`, `GPE`):

```text
{required_label_counts}
```

## Notes

NER converts unstructured text into structured records such as people,
organizations, dates, and geopolitical entities. Some model predictions may be
imperfect because this task uses a small pre-trained model without fine-tuning.

## Output Files

- `ner_entities.csv`
- `ner_entities.json`
- `required_ner_entities.csv`
- `required_ner_entities.json`
- `ner_summary_table.csv`
- `annotated_ner.html`
"""

    output_path = output_dir / "ner_report.md"
    output_path.write_text(report, encoding="utf-8")
    logging.info("Saved %s", output_path)


def run_ner_task(processed_docs: list[tuple[TextSample, Doc]], output_dir: Path) -> None:
    """Run Task 1 outputs."""

    ner_df = extract_ner_entities(processed_docs)
    required_ner_df = ner_df[ner_df["entity_label"].isin(REQUIRED_NER_LABELS)].copy()

    save_dataframe(ner_df, output_dir, "ner_entities")
    save_dataframe(required_ner_df, output_dir, "required_ner_entities")

    summary_df = (
        ner_df.groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
    )
    summary_df.to_csv(output_dir / "ner_summary_table.csv", index=False)

    render_ner_html(processed_docs, output_dir)
    create_ner_report(ner_df, required_ner_df, output_dir)


def extract_pos_tags(processed_docs: list[tuple[TextSample, Doc]]) -> pd.DataFrame:
    """Extract token-level POS information."""

    rows: list[dict[str, object]] = []

    for sample, doc in processed_docs:
        for sentence_id, sentence in enumerate(doc.sents, start=1):
            for token in sentence:
                if token.is_space:
                    continue

                rows.append(
                    {
                        "text_id": sample.text_id,
                        "category": sample.category,
                        "sentence_id": sentence_id,
                        "sentence": sentence.text,
                        "token": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "dependency": token.dep_,
                        "head_token": token.head.text,
                        "is_alpha": token.is_alpha,
                        "is_stop": token.is_stop,
                    }
                )

    return pd.DataFrame(rows)


def plot_pos_distribution(pos_df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save a POS tag frequency bar chart."""

    pos_counts = pos_df["pos"].value_counts()

    plt.figure(figsize=(12, 7))
    pos_counts.plot(kind="bar")
    plt.title("POS Tag Frequency Distribution")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / "pos_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info("Saved %s", output_path)


def extract_subject_verb_patterns(processed_docs: list[tuple[TextSample, Doc]]) -> pd.DataFrame:
    """Extract subject-verb dependency patterns."""

    rows: list[dict[str, object]] = []

    for sample, doc in processed_docs:
        for sentence in doc.sents:
            for token in sentence:
                if token.dep_ in {"nsubj", "nsubjpass"} and token.head.pos_ == "VERB":
                    rows.append(
                        {
                            "text_id": sample.text_id,
                            "category": sample.category,
                            "pattern_type": "subject_verb",
                            "subject": token.text,
                            "verb": token.head.text,
                            "object": None,
                            "sentence": sentence.text,
                        }
                    )

    return pd.DataFrame(rows)


def extract_verb_object_patterns(processed_docs: list[tuple[TextSample, Doc]]) -> pd.DataFrame:
    """Extract verb-object dependency patterns."""

    rows: list[dict[str, object]] = []

    for sample, doc in processed_docs:
        for sentence in doc.sents:
            for token in sentence:
                if token.dep_ in {"dobj", "obj", "attr"} and token.head.pos_ == "VERB":
                    rows.append(
                        {
                            "text_id": sample.text_id,
                            "category": sample.category,
                            "pattern_type": "verb_object",
                            "subject": None,
                            "verb": token.head.text,
                            "object": token.text,
                            "sentence": sentence.text,
                        }
                    )

    return pd.DataFrame(rows)


def highlight_pos_tags(doc: Doc) -> str:
    """Return HTML with selected POS tags highlighted."""

    style_map = {
        "NOUN": "background-color:#dbeafe; padding:3px; border-radius:4px;",
        "PROPN": "background-color:#dbeafe; padding:3px; border-radius:4px;",
        "VERB": "background-color:#fee2e2; padding:3px; border-radius:4px;",
        "ADJ": "background-color:#dcfce7; padding:3px; border-radius:4px;",
        "ADV": "background-color:#f3e8ff; padding:3px; border-radius:4px;",
    }

    parts: list[str] = []
    for token in doc:
        safe_token = html.escape(token.text)
        style = style_map.get(token.pos_)

        if style:
            parts.append(f"<span style='{style}'>{safe_token}<sub>{token.pos_}</sub></span>")
        else:
            parts.append(safe_token)

        parts.append(token.whitespace_)

    return "".join(parts)


def render_pos_html(processed_docs: list[tuple[TextSample, Doc]], output_dir: Path) -> None:
    """Create an HTML file that highlights selected POS tags."""

    content = """
<html>
<head>
    <meta charset="UTF-8">
    <title>POS Tag Highlight Visualization</title>
</head>
<body style="font-family: Arial; line-height: 2; padding: 20px;">
<h1>POS Tag Highlight Visualization</h1>
<p><b>Legend:</b></p>
<ul>
    <li><span style='background-color:#dbeafe; padding:3px; border-radius:4px;'>NOUN / PROPN</span> = nouns and proper nouns</li>
    <li><span style='background-color:#fee2e2; padding:3px; border-radius:4px;'>VERB</span> = verbs</li>
    <li><span style='background-color:#dcfce7; padding:3px; border-radius:4px;'>ADJ</span> = adjectives</li>
    <li><span style='background-color:#f3e8ff; padding:3px; border-radius:4px;'>ADV</span> = adverbs</li>
</ul>
"""

    for sample, doc in processed_docs:
        content += f"<h2>{html.escape(sample.category)}</h2>"
        content += f"<p>{highlight_pos_tags(doc)}</p><hr>"

    content += "\n</body>\n</html>\n"

    output_path = output_dir / "pos_highlight.html"
    output_path.write_text(content, encoding="utf-8")
    logging.info("Saved %s", output_path)


def create_pos_report(
    pos_df: pd.DataFrame,
    grammar_patterns_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a concise Markdown report for Task 2."""

    pos_counts = pos_df["pos"].value_counts()
    top_pos_text = pos_counts.head(10).to_string()

    subject_verb_df = grammar_patterns_df[grammar_patterns_df["pattern_type"] == "subject_verb"]
    verb_object_df = grammar_patterns_df[grammar_patterns_df["pattern_type"] == "verb_object"]

    top_subject_verb = (
        subject_verb_df[["subject", "verb"]].value_counts().head(10).to_string()
        if not subject_verb_df.empty
        else "No subject-verb patterns found."
    )
    top_verb_object = (
        verb_object_df[["verb", "object"]].value_counts().head(10).to_string()
        if not verb_object_df.empty
        else "No verb-object patterns found."
    )

    report = f"""# Task 2: POS Tagging Insights Report

## Objective

Apply POS tagging to the same corpus used in Task 1. POS tagging identifies the
grammatical role of each token, such as nouns, verbs, adjectives, and adverbs.

## Method

The text was processed with spaCy's `{MODEL_NAME}` model. The pipeline extracted
sentence boundaries, tokens, lemmas, POS tags, detailed tags, dependency labels,
and head tokens.

## Main POS Tag Distribution

```text
{top_pos_text}
```

## Common Subject-Verb Patterns

```text
{top_subject_verb}
```

## Common Verb-Object Patterns

```text
{top_verb_object}
```

## Key Insights

1. `PROPN` is frequent because the corpus contains many named people, places,
   organizations, products, and events.
2. `NOUN` is frequent because news text often discusses objects, concepts,
   institutions, laws, and events.
3. `VERB` tokens represent the main actions in the text, such as announcing,
   signing, meeting, discovering, and hosting.
4. Dependency patterns help move from token-level tags to sentence-level meaning.

## Output Files

- `pos_tags.csv`
- `pos_tags.json`
- `pos_summary.csv`
- `category_pos_summary.csv`
- `grammar_patterns.csv`
- `pos_distribution.png`
- `pos_highlight.html`
"""

    output_path = output_dir / "pos_insights_report.md"
    output_path.write_text(report, encoding="utf-8")
    logging.info("Saved %s", output_path)


def run_pos_task(processed_docs: list[tuple[TextSample, Doc]], output_dir: Path) -> None:
    """Run Task 2 outputs."""

    pos_df = extract_pos_tags(processed_docs)
    save_dataframe(pos_df, output_dir, "pos_tags")

    pos_summary_df = pos_df["pos"].value_counts().reset_index()
    pos_summary_df.columns = ["pos", "count"]
    pos_summary_df.to_csv(output_dir / "pos_summary.csv", index=False)

    category_pos_summary_df = (
        pos_df.groupby(["category", "pos"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
    )
    category_pos_summary_df.to_csv(output_dir / "category_pos_summary.csv", index=False)

    subject_verb_df = extract_subject_verb_patterns(processed_docs)
    verb_object_df = extract_verb_object_patterns(processed_docs)
    grammar_patterns_df = pd.concat([subject_verb_df, verb_object_df], ignore_index=True)
    grammar_patterns_df.to_csv(output_dir / "grammar_patterns.csv", index=False)

    plot_pos_distribution(pos_df, output_dir)
    render_pos_html(processed_docs, output_dir)
    create_pos_report(pos_df, grammar_patterns_df, output_dir)


def build_technology_nlp(model_name: str = MODEL_NAME) -> Language:
    """Create a spaCy pipeline with EntityRuler for TECHNOLOGY terms."""

    nlp = load_spacy_model(model_name)
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={
            "overwrite_ents": True,
            "phrase_matcher_attr": "LOWER",
        },
    )
    ruler.add_patterns(TECHNOLOGY_PATTERNS)
    return nlp


def extract_doc_entities(doc: Doc) -> pd.DataFrame:
    """Extract entities from a single spaCy document."""

    rows = [
        {
            "entity_text": ent.text,
            "entity_label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]
    return pd.DataFrame(rows)


def run_entity_ruler_task(model_name: str, output_dir: Path) -> None:
    """Run Task 3 outputs."""

    regular_nlp = load_spacy_model(model_name)
    technology_nlp = build_technology_nlp(model_name)

    cleaned_test_text = clean_text(TECHNOLOGY_TEST_TEXT)
    regular_doc = regular_nlp(cleaned_test_text)
    technology_doc = technology_nlp(cleaned_test_text)

    entity_ruler_df = extract_doc_entities(technology_doc)
    technology_only_df = entity_ruler_df[
        entity_ruler_df["entity_label"] == "TECHNOLOGY"
    ].copy()

    save_dataframe(entity_ruler_df, output_dir, "entity_ruler_entities")
    technology_only_df.to_csv(output_dir / "technology_entities_only.csv", index=False)

    comparison_rows = []
    for version, doc in [
        ("without_entity_ruler", regular_doc),
        ("with_entity_ruler", technology_doc),
    ]:
        for ent in doc.ents:
            comparison_rows.append(
                {
                    "version": version,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "entity_ruler_comparison.csv", index=False)

    html_output = displacy.render(technology_doc, style="ent", page=True, jupyter=False)
    (output_dir / "entity_ruler_technology.html").write_text(html_output, encoding="utf-8")

    create_entity_ruler_report(technology_only_df, output_dir)


def create_entity_ruler_report(technology_only_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a concise Markdown report for Task 3."""

    technology_terms = (
        technology_only_df["entity_text"].to_string(index=False)
        if not technology_only_df.empty
        else "No TECHNOLOGY entities found."
    )

    report = f"""# Task 3: Rule-Based Fallback Using spaCy EntityRuler

## Objective

Extend the pre-trained spaCy NER model with a custom rule-based entity label:
`TECHNOLOGY`.

## Why EntityRuler Was Used

Pre-trained NER models can miss domain-specific terms such as technology
products, AI systems, and technical concepts. EntityRuler allows these terms to
be added without collecting training data or retraining the model.

## Method

A new spaCy pipeline was created with an `entity_ruler` component placed before
the default `ner` component. The rule matcher uses lowercase matching to make
custom technology terms easier to detect.

## Results

Total TECHNOLOGY entities detected: **{len(technology_only_df)}**

```text
{technology_terms}
```

## Output Files

- `entity_ruler_entities.csv`
- `entity_ruler_entities.json`
- `technology_entities_only.csv`
- `entity_ruler_comparison.csv`
- `entity_ruler_technology.html`
"""

    output_path = output_dir / "task3_entity_ruler_report.md"
    output_path.write_text(report, encoding="utf-8")
    logging.info("Saved %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run NER, POS tagging, and EntityRuler analysis with spaCy."
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"spaCy model name. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-entity-ruler",
        action="store_true",
        help="Skip optional Task 3 EntityRuler outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete pipeline."""

    configure_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading spaCy model: %s", args.model)
    nlp = load_spacy_model(args.model)

    logging.info("Preparing corpus")
    samples = prepare_corpus(SAMPLE_CORPUS)
    processed_docs = process_corpus(nlp, samples)

    logging.info("Running Task 1: NER")
    run_ner_task(processed_docs, output_dir)

    logging.info("Running Task 2: POS tagging")
    run_pos_task(processed_docs, output_dir)

    if not args.skip_entity_ruler:
        logging.info("Running Task 3: EntityRuler")
        run_entity_ruler_task(args.model, output_dir)

    logging.info("Pipeline completed successfully. Outputs saved in: %s", output_dir)


if __name__ == "__main__":
    main()
