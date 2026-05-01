# Task 1: Named Entity Recognition Report

## Objective

Apply Named Entity Recognition (NER) to a small multi-domain text corpus using
spaCy's pre-trained `en_core_web_sm` model.

## Method

The text was lightly cleaned by normalizing whitespace. Capitalization was
preserved because it helps NER models identify names, places, and organizations.

## Results

Total extracted entities: **41**

Entity label distribution:

```text
entity_label
GPE            11
ORG             8
DATE            8
PERSON          7
NORP            2
WORK_OF_ART     2
MONEY           1
FAC             1
LOC             1
```

Required label distribution (`PERSON`, `ORG`, `DATE`, `GPE`):

```text
entity_label
GPE       11
ORG        8
DATE       8
PERSON     7
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
