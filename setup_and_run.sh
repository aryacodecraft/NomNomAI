#!/bin/bash
# ─────────────────────────────────────────────────────────────
# RecipeLens — one-shot setup script
# Run from the folder where all the .py files live
# ─────────────────────────────────────────────────────────────
set -e

# Detect python command
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Python is not installed or not in PATH."
    exit 1
fi

echo "📦 Installing dependencies from requirements.txt..."
$PY_CMD -m pip install -r requirements.txt

echo ""
echo "📥 Downloading dataset from HuggingFace (39,400 recipes)..."
$PY_CMD - <<PYEOF
from datasets import load_dataset
print("Downloading datahiveai/recipes-with-nutrition ...")
ds = load_dataset("datahiveai/recipes-with-nutrition", split="train")
print(f"  {len(ds)} recipes downloaded")
ds.to_pandas().to_csv("recipes-with-nutrition.csv", index=False)
print("  Saved to recipes-with-nutrition.csv")
PYEOF

echo ""
echo "⚙️  Running preprocessing (builds ./artifacts/)..."
$PY_CMD preprocess.py --data recipes-with-nutrition.csv --out ./artifacts

echo ""
echo "🚀 Starting API server on http://localhost:8000"
echo "   App UI at    http://localhost:8000/"
echo "   API docs at  http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""
$PY_CMD -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
