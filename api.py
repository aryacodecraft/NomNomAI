"""
Recipe Recommendation API (FastAPI)
────────────────────────────────────
Hardened version with strict schema validations and standard response wrappers.
"""

import sys
import os
import ast
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from recommender import RecipeRecommender
from models import SearchRequest, SimilarRequest, IngredientsRequest, PantrySyncRequest, StandardResponse
from pantry import get_pantry, set_pantry

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NomNomAI.API")

# Singleton recommender setup via lifespan
recommender: RecipeRecommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    logger.info("Starting Recipe Recommendation API Initialization...")
    recommender = RecipeRecommender()
    if not recommender.is_loaded:
        logger.warning("Recommender failed to load artifacts. API is degraded (503).")
    else:
        logger.info("API Startup Complete. Artifacts bound to memory successfully.")
    
    yield
    logger.info("Shutting down Recipe Recommendation API.")

app = FastAPI(
    title="Recipe Recommendation API",
    description="Engineered Hybrid NLP + ML recipe recommendation system",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── EXCEPTION HANDLERS ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Server Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "data": None, "message": "An internal server error occurred."}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "data": None, "message": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation Error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "data": None, "message": "Invalid input provided. Check parameters."}
    )

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _ensure_loaded():
    if not recommender or not getattr(recommender, "is_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine is offline due to missing artifacts."
        )

def success(data=None, message=None):
    return StandardResponse(status="success", data=data, message=message).model_dump(exclude_none=True)

# ─────────────────────────────────────────────
# UI & UTILITY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serves the frontend UI directly (Bypasses JSON Standardization)"""
    if not os.path.exists("frontend.html"):
        raise HTTPException(status_code=404, detail="frontend.html not found.")
    return FileResponse("frontend.html")

@app.get("/health", response_model=StandardResponse)
def health_check():
    """System health check endpoint."""
    if not recommender:
        return success(data={"status": "error", "recipes_loaded": 0}, message="Recommender null")
        
    status = "ok" if getattr(recommender, "is_loaded", False) else "error_missing_artifacts"
    n = len(recommender.df) if getattr(recommender, 'df', None) is not None else 0
    return success(data={"status": status, "recipes_loaded": n})

# ─────────────────────────────────────────────
# RECIPE ROUTES
# ─────────────────────────────────────────────

@app.get("/recipes", response_model=StandardResponse)
def list_recipes():
    _ensure_loaded()
    names = recommender.get_all_recipe_names()
    return success(data=names)

@app.get("/recipes/{recipe_id}", response_model=StandardResponse)
def get_recipe(recipe_id: int):
    _ensure_loaded()
    recipe = recommender.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")
    return success(data=recipe)

@app.get("/filters", response_model=StandardResponse)
def get_filters():
    _ensure_loaded()
    return success(data=recommender.get_filter_options())

@app.post("/search", response_model=StandardResponse)
def search_recipes(req: SearchRequest):
    _ensure_loaded()
    
    # Handled natively by Pydantic min_length, but double check.
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string is blank")

    results = recommender.search_by_text(
        query=req.query,
        diet_selected=req.diet_filters,
        calorie_min=req.calorie_min,
        calorie_max=req.calorie_max,
        meal_type=req.meal_type,
        top_k=req.top_k,
        offset=req.offset,
    )
    
    resp_data = {
        "query": req.query,
        "source": {
            "name": "Unknown"
        },
        "n_results": len(results),
        "filters_applied": {
            "diet": req.diet_filters,
            "calorie_range": [req.calorie_min, req.calorie_max],
            "meal_type": req.meal_type,
        },
        "results": results,
    }
    msg = "Results found." if results else "No recipes match your criteria."
    return success(data=resp_data, message=msg)

@app.post("/similar", response_model=StandardResponse)
def similar_recipes(req: SimilarRequest):
    _ensure_loaded()
    source = recommender.get_recipe(req.recipe_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source recipe {req.recipe_id} not found")

    results = recommender.search_by_recipe(
        recipe_id=req.recipe_id,
        diet_selected=req.diet_filters,
        calorie_min=req.calorie_min,
        calorie_max=req.calorie_max,
        top_k=req.top_k,
        offset=req.offset,
        weights=tuple(req.weights),
    )
    
    source_name = source.get("name") if source and "name" in source else source.get("recipe_name", "Unknown") if source else "Unknown"
    
    resp_data = {
        "name": source_name,
        "source_recipe": source,
        "source": {
            "name": source_name
        },
        "n_results": len(results),
        "weights": {"nlp": req.weights[0], "cuisine": req.weights[1], "nutrition": req.weights[2]},
        "results": results,
    }
    return success(data=resp_data)

# ─────────────────────────────────────────────
# INGREDIENT & PANTRY ROUTES
# ─────────────────────────────────────────────

@app.post("/ingredients", response_model=StandardResponse)
def search_by_ingredients(req: IngredientsRequest):
    _ensure_loaded()
    
    user_tokens = set()
    for ing in req.ingredients:
        for word in ing.lower().split():
            if len(word) > 2:
                user_tokens.add(word)

    if not user_tokens:
        raise HTTPException(status_code=400, detail="No usable ingredients parsed from input")

    df = recommender.df.copy()
    df = recommender.apply_filters(df, req.diet_filters, req.calorie_min, req.calorie_max)

    if df.empty:
         return success(data={"n_results": 0, "results": [], "can_make_now": 0}, message="No recipes match strict diet filters.")

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

        try:
            val = str(row.get('ingredient_lines', '[]'))
            ing_lines = ast.literal_eval(val) if isinstance(row.get('ingredient_lines'), str) else (row.get('ingredient_lines') or [])
        except Exception:
            ing_lines = []

        def safe_list(v):
            if isinstance(v, list): return v
            try: return ast.literal_eval(str(v))
            except Exception: return []

        results.append({
            "recipe_id":            int(row.get('recipe_id', -1)),
            "name":                 str(row.get('recipe_name', '')),
            "cuisine":              str(row.get('cuisine', '')),
            "dish_type":            str(row.get('dish', '')),
            "meal_type":            str(row.get('meal', '')),
            "url":                  str(row.get("url") or row.get("recipe_url") or row.get("source_url") or ""),
            "calories_per_serving": float(row.get('calories_per_serving') or 0),
            "protein_g":            float(row.get('protein_per_serving') or 0),
            "fat_g":                float(row.get('fat_per_serving') or 0),
            "carbs_g":              float(row.get('carbs_per_serving') or 0),
            "diet_labels":          safe_list(row.get('diet_labels', [])),
            "health_labels":        safe_list(row.get('health_labels', [])),
            "matched_ingredients":  sorted(list(matched)),
            "missing_ingredients":  sorted(list(missing))[:8],
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
    results = results[req.offset : req.offset + req.top_k]
    can_make_now = [r for r in results if r['can_make_now']]

    resp_data = {
        "user_ingredients": req.ingredients,
        "mode": req.mode,
        "n_results": len(results),
        "can_make_now": len(can_make_now),
        "results": results,
    }
    return success(data=resp_data)

@app.get("/ingredient-list", response_model=StandardResponse)
def get_ingredient_list():
    _ensure_loaded()
    import re
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
        except Exception:
            pass
    return success(data={"ingredients": sorted(list(tokens)), "count": len(tokens)})

@app.get("/pantry/{session_id}", response_model=StandardResponse)
def fetch_pantry_items(session_id: str):
    """Retrieve the stored pantry items for a user session."""
    items = get_pantry(session_id)
    return success(data={"session_id": session_id, "ingredients": items})

@app.post("/pantry", response_model=StandardResponse)
def sync_pantry_items(req: PantrySyncRequest):
    """Save the pantry items for a user session."""
    saved_items = set_pantry(req.session_id, req.ingredients)
    return success(data={"session_id": req.session_id, "ingredients": saved_items})

if __name__ == "__main__":
    import uvicorn
    # Support environment injection mapping default to 8000
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Initiating Uvicorn Server on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False)
