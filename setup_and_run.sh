#!/bin/bash
# ─────────────────────────────────────────────────────────────
# RecipeLens — one-shot setup script
# Run from the folder where all the .py files live
# ─────────────────────────────────────────────────────────────
set -e

echo "📦 Installing dependencies..."
pip install pandas numpy scikit-learn fastapi uvicorn datasets huggingface-hub

echo ""
echo "📥 Downloading dataset from HuggingFace (39,400 recipes)..."
python3 - <<PYEOF
from datasets import load_dataset
print("Downloading datahiveai/recipes-with-nutrition ...")
ds = load_dataset("datahiveai/recipes-with-nutrition", split="train")
print(f"  {len(ds)} recipes downloaded")
ds.to_pandas().to_csv("recipes-with-nutrition.csv", index=False)
print("  Saved to recipes-with-nutrition.csv")
PYEOF

echo ""
echo "⚙️  Running preprocessing (builds ./artifacts/)..."
python3 preprocess.py --data recipes-with-nutrition.csv --out ./artifacts

echo ""
echo "🚀 Starting API server on http://localhost:8000"
echo "   API docs at  http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
