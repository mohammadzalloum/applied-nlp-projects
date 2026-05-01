# Task 3: Rule-Based Fallback Using spaCy EntityRuler

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

Total TECHNOLOGY entities detected: **9**

```text
                    Nexus X
      AI-powered smartphone
           Digital Wellness
  machine learning platform
natural language processing
            cloud computing
             neural network
               AI assistant
                 smartphone
```

## Output Files

- `entity_ruler_entities.csv`
- `entity_ruler_entities.json`
- `technology_entities_only.csv`
- `entity_ruler_comparison.csv`
- `entity_ruler_technology.html`
