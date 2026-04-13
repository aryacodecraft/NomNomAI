"""
Recipe Recommendation API (FastAPI)
────────────────────────────────────
Endpoints:
  GET  /                      Health check
  GET  /recipes               All recipe names (for dropdown)
  GET  /recipes/{id}          Single recipe details
  GET  /filters               Available filter options
  POST /search                Text-based semantic search
  POST /similar               Find similar recipes

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
sys.path.insert(0, "/home/claude")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os

from recommender import RecipeRecommender

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Recipe Recommendation API",
    description="Hybrid NLP + ML recipe recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton recommender (loaded once at startup)
recommender: Optional[RecipeRecommender] = None


@app.on_event("startup")
def startup():
    global recommender
    recommender = RecipeRecommender()


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., example="healthy vegetarian pasta under 500 calories")
    diet_filters: list[str] = Field(default=[], example=["Vegetarian", "Gluten-Free"])
    calorie_min: float = Field(default=0, ge=0)
    calorie_max: float = Field(default=9999, ge=0)
    meal_type: Optional[str] = Field(default=None, example="dinner")
    top_k: int = Field(default=10, ge=1, le=50)


class SimilarRequest(BaseModel):
    recipe_id: int = Field(..., example=10)
    diet_filters: list[str] = Field(default=[], example=[])
    calorie_min: float = Field(default=0, ge=0)
    calorie_max: float = Field(default=9999, ge=0)
    top_k: int = Field(default=10, ge=1, le=50)
    weights: tuple[float, float, float] = Field(
        default=(0.70, 0.20, 0.10),
        description="(NLP weight, cuisine weight, nutrition weight)"
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    n = len(recommender.df) if recommender else 0
    return {"status": "ok", "n_recipes": n, "version": "1.0.0"}


@app.get("/recipes")
def list_recipes():
    """All recipe names for dropdown population."""
    return recommender.get_all_recipe_names()


@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int):
    """Single recipe details by ID."""
    recipe = recommender.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")
    return recipe


@app.get("/filters")
def get_filters():
    """Available filter options (diet labels, cuisine types, calorie range, etc.)"""
    return recommender.get_filter_options()


@app.post("/search")
def search_recipes(req: SearchRequest):
    """
    Semantic text search.
    Converts query to embedding, computes cosine similarity, applies filters.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = recommender.search_by_text(
        query=req.query,
        diet_selected=req.diet_filters,
        calorie_min=req.calorie_min,
        calorie_max=req.calorie_max,
        meal_type=req.meal_type,
        top_k=req.top_k,
    )
    return {
        "query": req.query,
        "n_results": len(results),
        "filters_applied": {
            "diet": req.diet_filters,
            "calorie_range": [req.calorie_min, req.calorie_max],
            "meal_type": req.meal_type,
        },
        "results": results,
    }


@app.post("/similar")
def similar_recipes(req: SimilarRequest):
    """
    Hybrid similarity search based on a selected recipe.
    Formula: 0.70 * NLP_sim + 0.20 * cuisine_sim + 0.10 * nutrition_sim
    """
    if not recommender.get_recipe(req.recipe_id):
        raise HTTPException(status_code=404, detail=f"Recipe {req.recipe_id} not found")

    results = recommender.search_by_recipe(
        recipe_id=req.recipe_id,
        diet_selected=req.diet_filters,
        calorie_min=req.calorie_min,
        calorie_max=req.calorie_max,
        top_k=req.top_k,
        weights=tuple(req.weights),
    )
    source = recommender.get_recipe(req.recipe_id)
    return {
        "source_recipe": source,
        "n_results": len(results),
        "weights": {"nlp": req.weights[0], "cuisine": req.weights[1], "nutrition": req.weights[2]},
        "results": results,
    }


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
