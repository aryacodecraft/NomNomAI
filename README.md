# NomNomAI

A hybrid NLP + ML recipe recommendation engine built with FastAPI. Searches and recommends from **39,400+ recipes** using a combination of text similarity, cuisine matching, and nutritional profiling.

## How It Works

Recipes are embedded into three vector spaces during preprocessing:

| Embedding | Method | Purpose |
|-----------|--------|---------|
| **Text** | TF-IDF + TruncatedSVD (128-dim) | Captures recipe name, ingredients, cuisine, dish type |
| **Cuisine** | One-hot encoding | Exact cuisine matching |
| **Nutrition** | Z-scored macro values | Nutritional similarity |

Recommendations use a weighted hybrid score:

```
Score = 0.70 × NLP_similarity + 0.20 × Cuisine_similarity + 0.10 × Nutrition_similarity
```

## Features

- **Text Search** — natural language queries like *"healthy vegetarian pasta"*
- **Similar Recipes** — find recipes similar to a given one using hybrid scoring
- **Ingredient Search** — input what you have, get recipes you can make (exact or flexible matching)
- **Pantry Management** — save ingredients per session for quick reuse
- **Filtering** — diet labels, calorie range, meal type

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML:** scikit-learn (TF-IDF, TruncatedSVD), NumPy, Pandas
- **Dataset:** [datahiveai/recipes-with-nutrition](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition) (HuggingFace)
- **Frontend:** Single-file HTML served by the API

## Setup

### Prerequisites

- Python 3.10+

### Quick Start (Linux/macOS)

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

This installs dependencies, downloads the dataset, runs preprocessing, and starts the server.

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess dataset (downloads from HuggingFace, generates ./artifacts/)
python preprocess.py

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

The app will be available at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

## Project Structure

```
NomNomAI/
├── api.py              # FastAPI server — all routes and exception handlers
├── recommender.py      # Recommendation engine — embedding loading, search, filtering
├── preprocess.py       # Data pipeline — downloads dataset, builds embeddings
├── models.py           # Pydantic request/response schemas
├── pantry.py           # In-memory pantry storage per session
├── frontend.html       # Single-file frontend UI
├── setup_and_run.sh    # One-shot setup script
├── requirements.txt    # Python dependencies
├── artifacts/          # Generated at preprocessing (not committed)
│   ├── recipes_processed.csv
│   ├── recipe_embeddings.npy
│   ├── cuisine_embeddings.npy
│   ├── nutrition_embeddings.npy
│   └── label_encoders.pkl
└── load_recipes.ipynb  # Notebook for exploratory data loading
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend UI |
| `GET` | `/health` | System health check |
| `GET` | `/recipes` | List all recipe names |
| `GET` | `/recipes/{id}` | Get recipe details by ID |
| `GET` | `/filters` | Available diet, cuisine, meal type filters |
| `POST` | `/search` | Text-based recipe search with filters |
| `POST` | `/similar` | Find similar recipes (hybrid scoring) |
| `POST` | `/ingredients` | Search by available ingredients |
| `GET` | `/ingredient-list` | All known ingredient tokens |
| `GET` | `/pantry/{session_id}` | Get saved pantry items |
| `POST` | `/pantry` | Save pantry items for a session |