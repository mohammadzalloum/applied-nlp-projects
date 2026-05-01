# Task 2: POS Tagging Insights Report

## Objective

Apply POS tagging to the same corpus used in Task 1. POS tagging identifies the
grammatical role of each token, such as nouns, verbs, adjectives, and adverbs.

## Method

The text was processed with spaCy's `en_core_web_sm` model. The pipeline extracted
sentence boundaries, tokens, lemmas, POS tags, detailed tags, dependency labels,
and head tokens.

## Main POS Tag Distribution

```text
pos
PROPN    62
PUNCT    39
NOUN     39
VERB     30
ADP      27
DET      22
ADJ      11
NUM       9
PART      6
AUX       6
```

## Common Subject-Verb Patterns

```text
subject  verb     
Doe      plans        1
device   debut        1
critics  argue        1
launch   impact       1
company  partnered    1
Apple    declined     1
event    coincide     1
Messi    signed       1
deal     announced    1
Fans     flooded      1
```

## Common Verb-Object Patterns

```text
verb        object    
introduce   smartphone    1
impact      policies      1
promote     wellness      1
signed      contract      1
flooded     media         1
celebrate   move          1
discuss     policies      1
host        summit        1
face        delays        1
discovered  radium        1
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
