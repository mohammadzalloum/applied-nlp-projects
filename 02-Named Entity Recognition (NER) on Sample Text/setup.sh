#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "Starting setup in: $PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
else
  echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  python -m pip install -r requirements.txt
else
  echo "Error: requirements.txt was not found."
  exit 1
fi

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Verifying spaCy model installation..."
python - <<'PY'
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is based in New York.")
print("spaCy model loaded successfully.")
print("Sample entities:", [(ent.text, ent.label_) for ent in doc.ents])
PY

echo
echo "Setup complete."
echo "To activate the environment later, run:"
echo "source .venv/bin/activate"
echo
echo "To run the project:"
echo "python nlp_entity_pos_pipeline.py"
