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


# ─────────────────────────────────────────────
# INGREDIENT-BASED ENDPOINT
# ─────────────────────────────────────────────

class IngredientsRequest(BaseModel):
    ingredients: list[str] = Field(..., example=["chicken", "garlic", "tomato", "onion"])
    mode: str = Field(default="flexible", description="'exact' = only use provided, 'flexible' = allow a few extras")
    max_extra: int = Field(default=5, description="Max extra ingredients allowed (flexible mode)")
    diet_filters: list[str] = Field(default=[])
    calorie_min: float = Field(default=0)
    calorie_max: float = Field(default=9999)
    top_k: int = Field(default=12)


@app.post("/ingredients")
def search_by_ingredients(req: IngredientsRequest):
    """
    Find recipes based on ingredients the user has.
    Returns recipes sorted by how well the user's ingredients cover the recipe.
    Each result shows matched ingredients, missing ingredients, and coverage score.
    """
    import ast, re

    # Normalise user input
    user_tokens = set()
    for ing in req.ingredients:
        for word in ing.lower().split():
            if len(word) > 2:
                user_tokens.add(word)

    if not user_tokens:
        raise HTTPException(status_code=400, detail="No valid ingredients provided")

    df = recommender.df.copy()

    # Apply diet + calorie filters first
    df = recommender.apply_filters(df, req.diet_filters, req.calorie_min, req.calorie_max)

    results = []
    for _, row in df.iterrows():
        recipe_tokens = set()
        if row.get('ingredients_clean'):
            for w in str(row['ingredients_clean']).split():
                if len(w) > 2:
                    recipe_tokens.add(w.lower())

        if not recipe_tokens:
            continue

        matched   = user_tokens & recipe_tokens
        missing   = recipe_tokens - user_tokens
        extra_cnt = len(missing)

        if req.mode == 'exact' and extra_cnt > 0:
            continue
        if req.mode == 'flexible' and extra_cnt > req.max_extra:
            continue

        coverage      = len(matched) / max(len(recipe_tokens), 1)
        user_coverage = len(matched) / max(len(user_tokens), 1)
        score = 0.6 * user_coverage + 0.4 * coverage

        # Get display ingredient lines
        try:
            ing_lines = ast.literal_eval(str(row['ingredient_lines'])) if isinstance(row['ingredient_lines'], str) else (row['ingredient_lines'] or [])
        except:
            ing_lines = []

        def safe_list(v):
            if isinstance(v, list): return v
            try: return ast.literal_eval(str(v))
            except: return []

        results.append({
            "recipe_id":            int(row['recipe_id']),
            "name":                 str(row['recipe_name']),
            "cuisine":              str(row['cuisine']),
            "dish_type":            str(row['dish']),
            "meal_type":            str(row['meal']),
            "calories_per_serving": float(row['calories_per_serving'] or 0),
            "protein_g":            float(row['protein_per_serving'] or 0),
            "fat_g":                float(row['fat_per_serving'] or 0),
            "carbs_g":              float(row['carbs_per_serving'] or 0),
            "diet_labels":          safe_list(row.get('diet_labels', [])),
            "health_labels":        safe_list(row.get('health_labels', [])),
            "matched_ingredients":  sorted(matched),
            "missing_ingredients":  sorted(missing)[:8],
            "extra_needed":         extra_cnt,
            "can_make_now":         extra_cnt == 0,
            "match_count":          len(matched),
            "total_recipe_ings":    len(recipe_tokens),
            "coverage_score":       round(coverage, 3),
            "user_coverage":        round(user_coverage, 3),
            "score":                round(score, 3),
            "ingredient_lines":     [str(l) for l in ing_lines[:12]],
        })

    results.sort(key=lambda x: (-x['can_make_now'], -x['score'], x['extra_needed']))
    results = results[:req.top_k]

    can_make_now = [r for r in results if r['can_make_now']]

    return {
        "user_ingredients": req.ingredients,
        "mode": req.mode,
        "n_results": len(results),
        "can_make_now": len(can_make_now),
        "results": results,
    }


@app.get("/ingredient-list")
def get_ingredient_list():
    """All unique ingredient tokens from the dataset, for autocomplete."""
    import ast, re
    tokens = set()
    for _, row in recommender.df.iterrows():
        if row.get('ingredients_clean'):
            for w in str(row['ingredients_clean']).split():
                if len(w) > 2:
                    tokens.add(w.lower())
        try:
            lines = ast.literal_eval(str(row['ingredient_lines'])) if isinstance(row['ingredient_lines'], str) else (row['ingredient_lines'] or [])
            for line in lines:
                line = str(line).lower()
                line = re.sub(r'^\d[\d/. ]*', '', line)
                line = re.sub(r'\b(cups?|tbsp?|tsp?|tablespoons?|teaspoons?|oz|g|ml|large|medium|small|fresh|dried|chopped|sliced|diced|minced)\b\s*', '', line)
                line = line.strip(' ,.-()').split(',')[0].strip()
                if line and 2 < len(line) < 40:
                    tokens.add(line)
        except:
            pass
    return {"ingredients": sorted(tokens), "count": len(tokens)}
