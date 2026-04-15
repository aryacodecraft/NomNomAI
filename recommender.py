"""
Recipe Recommendation Engine
─────────────────────────────
Hybrid similarity:
  Final Score = 0.70 × NLP_similarity
              + 0.20 × cuisine_similarity
              + 0.10 × nutrition_similarity
"""

import ast
import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("NomNomAI.Recommender")

ARTIFACTS_DIR = "./artifacts"   # ← relative path, works from any directory

class RecipeRecommender:

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.is_loaded = False
        self._load_all()

    def _load_all(self):
        try:
            req_files = [
                "recipes_processed.csv",
                "recipe_embeddings.npy",
                "nutrition_embeddings.npy",
                "cuisine_embeddings.npy",
                "label_encoders.pkl"
            ]
            
            for file in req_files:
                fpath = os.path.join(self.artifacts_dir, file)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing required artifact: {fpath}")

            logger.info("Loading recipe database...")
            self.df = pd.read_csv(os.path.join(self.artifacts_dir, "recipes_processed.csv"))
            
            logger.info("Parsing list columns...")
            for col in ['ingredient_lines','diet_labels','health_labels']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(self._safe_parse_list)

            logger.info("Loading embeddings...")
            self.text_emb = np.load(os.path.join(self.artifacts_dir, "recipe_embeddings.npy"))
            self.nutr_emb = np.load(os.path.join(self.artifacts_dir, "nutrition_embeddings.npy"))
            self.cuis_emb = np.load(os.path.join(self.artifacts_dir, "cuisine_embeddings.npy"))

            logger.info("Loading label encoders and pipeline...")
            with open(os.path.join(self.artifacts_dir, "label_encoders.pkl"), "rb") as f:
                bundle = pickle.load(f)
            self.pipeline        = bundle["pipeline"]
            self.meta            = bundle["meta"]
            self.cuisine_classes = bundle["cuisine_classes"]

            # servings column is 'servings_clean' in real dataset, 'yield' in demo
            self._srv_col = 'servings_clean' if 'servings_clean' in self.df.columns else 'yield'
            logger.info(f"✅ RecipeRecommender Loaded: {len(self.df):,} recipes, dim={self.text_emb.shape[1]}")
            self.is_loaded = True
        except FileNotFoundError as e:
            logger.error(f"Failed to load artifacts: {e}. Please run preprocess.py first.")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Unexpected error loading artifacts: {e}", exc_info=True)
            self.is_loaded = False

    @staticmethod
    def _safe_parse_list(val):
        if pd.isna(val) or val == "" or val == "nan": return []
        if isinstance(val, list): return val
        try:
            r = ast.literal_eval(str(val)); return r if isinstance(r, list) else [r]
        except Exception: 
            return [str(val)] if str(val) else []

    def apply_filters(self, df, diet_selected=None, calorie_min=0, calorie_max=9999, meal_type=None):
        if df.empty:
            return df
            
        mask = pd.Series([True]*len(df), index=df.index)
        for label in (diet_selected or []):
            lower_label = label.lower()
            col = f"diet_{lower_label.replace('-','_')}"
            hcol = f"health_{lower_label.replace('-','_')}"
            
            label_mask = pd.Series([False]*len(df), index=df.index)
            if col in df.columns:
                label_mask |= df[col].fillna(0).astype(bool)
            if hcol in df.columns:
                label_mask |= df[hcol].fillna(0).astype(bool)
                
            label_mask |= df["diet_labels"].apply(lambda x: lower_label in [str(i).lower() for i in (x or [])])
            label_mask |= df["health_labels"].apply(lambda x: lower_label in [str(i).lower() for i in (x or [])])
            
            mask &= label_mask
            
        mask &= (df["calories_per_serving"].fillna(0) >= calorie_min)
        mask &= (df["calories_per_serving"].fillna(0) <= calorie_max)
        if meal_type:
            col = f"meal_{meal_type.lower().replace('/','_')}"
            if col in df.columns: mask &= df[col].fillna(0).astype(bool)
        return df[mask].copy()

    def search_by_text(self, query, diet_selected=None, calorie_min=0,
                       calorie_max=9999, meal_type=None, top_k=20, offset=0):
        # Prevent completely empty query crashing the transform
        if not query or not query.strip():
            logger.warning("Search called with empty query")
            return []
            
        q_emb = self.pipeline.transform([query]).astype(np.float32)
        norm = np.linalg.norm(q_emb) 
        q_emb /= (norm if norm > 0 else 1)
        sims = (self.text_emb @ q_emb.T).ravel()
        
        filtered = self.apply_filters(self.df, diet_selected, calorie_min, calorie_max, meal_type)
        if len(filtered) == 0: 
            logger.info("Search by text: 0 results after filtering.")
            return []
            
        filtered = filtered.copy()
        filtered["similarity_score"] = sims[filtered.index]
        filtered = filtered.sort_values(by="similarity_score", ascending=False)
        top = filtered.iloc[offset : offset + top_k]
        return self._format_results(top)

    def search_by_recipe(self, recipe_id, diet_selected=None, calorie_min=0,
                         calorie_max=9999, top_k=20, offset=0, weights=(0.70, 0.20, 0.10)):
        idx_s = self.df[self.df["recipe_id"] == recipe_id].index
        if len(idx_s) == 0: 
            logger.warning(f"search_by_recipe: recipe_id {recipe_id} not found.")
            return []
            
        idx = idx_s[0]
        w_nlp, w_cuis, w_nutr = weights
        hybrid = (w_nlp  * (self.text_emb @ self.text_emb[idx]) +
                  w_cuis * (self.cuis_emb @ self.cuis_emb[idx]) +
                  w_nutr * (self.nutr_emb @ self.nutr_emb[idx]))
                  
        filtered = self.apply_filters(self.df, diet_selected, calorie_min, calorie_max)
        filtered = filtered[filtered["recipe_id"] != recipe_id].copy()
        if len(filtered) == 0: 
            return []
            
        filtered["similarity_score"] = hybrid[filtered.index]
        filtered["nlp_score"]        = (self.text_emb @ self.text_emb[idx])[filtered.index]
        filtered["cuisine_score"]    = (self.cuis_emb @ self.cuis_emb[idx])[filtered.index]
        filtered["nutrition_score"]  = (self.nutr_emb @ self.nutr_emb[idx])[filtered.index]
        
        filtered = filtered.sort_values(by="similarity_score", ascending=False)
        top = filtered.iloc[offset : offset + top_k]
        results = self._format_results(top)
        
        src = self.df[self.df["recipe_id"] == recipe_id].iloc[0]
        for r in results:
            r["explanation"] = self._explain(src, r)
        return results

    def _explain(self, src, tgt):
        reasons = []
        if src.get("cuisine") == tgt.get("cuisine"):
            reasons.append(f"both {tgt.get('cuisine', '')} cuisine")
        if src.get("dish", "") == tgt.get("dish", ""):
            reasons.append(f"same dish type ({tgt.get('dish', '')})")
            
        shared = set(src.get("diet_labels") or []) & set(tgt.get("diet_labels") or [])
        if shared: 
            reasons.append(f"shared labels: {', '.join(sorted(shared))}")
            
        sc, tc = src.get("calories_per_serving",0), tgt.get("calories_per_serving",0)
        if sc and tc and abs(sc-tc) <= sc*0.25:
            reasons.append(f"similar calories (~{int(tc)} kcal)")
            
        return ("Similar because: " + "; ".join(reasons)) if reasons else "Similar ingredient profile"

    def get_recipe(self, recipe_id):
        rows = self.df[self.df["recipe_id"] == recipe_id]
        return self._row_to_dict(rows.iloc[0]) if not rows.empty else None

    def get_all_recipe_names(self):
        return [{"recipe_id": int(r.recipe_id), "name": str(r.recipe_name)}
                for _, r in self.df.iterrows()]

    def get_filter_options(self):
        return {**self.meta}

    @staticmethod
    def _row_to_dict(row):
        def plist(v):
            if isinstance(v, list): return v
            try: return ast.literal_eval(str(v))
            except Exception: return []
            
        return {
            "recipe_id":            int(row.get("recipe_id", -1)),
            "name":                 str(row.get("recipe_name", "")),
            "ingredients_clean":    str(row.get("ingredients_clean", "")),
            "calories_per_serving": float(row.get("calories_per_serving", 0) or 0),
            "protein_g":            float(row.get("protein_per_serving", 0) or 0),
            "fat_g":                float(row.get("fat_per_serving", 0) or 0),
            "carbs_g":              float(row.get("carbs_per_serving", 0) or 0),
            "fiber_g":              float(row.get("fiber_per_serving", 0) or 0),
            "sugar_g":              float(row.get("sugar_per_serving", 0) or 0),
            "diet_labels":          plist(row.get("diet_labels", [])),
            "health_labels":        plist(row.get("health_labels", [])),
            "cuisine":              str(row.get("cuisine", "")),
            "dish_type":            str(row.get("dish", "")),
            "meal_type":            str(row.get("meal", "")),
            "image_url":            str(row.get("image_url", "") or ""),
            "similarity_score":     float(row.get("similarity_score", 0) or 0),
        }

    def _format_results(self, df_top):
        results = []
        for _, row in df_top.iterrows():
            d = self._row_to_dict(row)
            for k in ["nlp_score","cuisine_score","nutrition_score"]:
                if k in row: d[k] = float(row[k])
            results.append(d)
        return results
