"""
preprocess.py — corrected for actual datahiveai/recipes-with-nutrition schema
39,400 recipes from HuggingFace

Actual columns:
  recipe_name, source, url, servings (float64), calories (float64),
  total_weight_g, image_url, diet_labels, health_labels, cautions,
  cuisine_type, meal_type, dish_type, ingredient_lines, ingredients,
  total_nutrients, daily_values, digest

Run:
    python preprocess.py --data recipes-with-nutrition.csv --out ./artifacts
"""

import argparse, pandas as pd, numpy as np, json, ast, re, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='recipes-with-nutrition.csv')
parser.add_argument('--out',  default='./artifacts')
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ── CONFIG ──────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
VOCAB_SIZE    = 15_000
BATCH_SIZE    = 2048   # for sentence-transformers if you upgrade

UNIT_RE  = re.compile(
    r'\b(\d+[/\d.]*\s*(?:cups?|tbsp?|tsp?|tablespoons?|teaspoons?|oz|ounces?|'
    r'lb|lbs?|pounds?|g|grams?|kg|ml|liters?|l|pints?|quarts?|cloves?|slices?|'
    r'pieces?|sprigs?|bunches?|heads?|stalks?|sticks?|cans?|jars?|packages?|'
    r'small|medium|large|handful|pinch|dash|drop))\b', re.I)
NUM_RE   = re.compile(r'^\d+[\d./]*\s*')
STOPS    = {'and','or','the','of','in','with','to','a','an','for','from',
            'sifted','chopped','sliced','diced','minced','peeled','fresh',
            'dried','frozen','cooked','raw','canned','optional','divided',
            'plus','about','more','thinly','finely','roughly','just','into'}

# Diet labels present in this dataset
DIET_LABELS   = ['Balanced','High-Fiber','High-Protein','Low-Fat','Low-Sodium',
                 'Low-Sugar','Vegetarian','Vegan']
# Health / allergen labels
HEALTH_LABELS = ['Gluten-Free','Dairy-Free','Egg-Free','Peanut-Free','Tree-Nut-Free',
                 'Soy-Free','Fish-Free','Shellfish-Free','Wheat-Free','Sesame-Free']
MEAL_TYPES    = ['breakfast','lunch/dinner','lunch','dinner','brunch','snack','teatime']

# ── HELPERS ─────────────────────────────────────────────────────────────────
def safe_list(v):
    if isinstance(v, list):
        return v
    if pd.isna(v) or v == '':
        return []
    try:
        import ast
        r = ast.literal_eval(str(v))
        return r if isinstance(r, list) else [r]
    except:
        return [str(v)]

def clean_ing(s):
    s = re.sub(r'[,;()]', ' ', s.lower())
    s = UNIT_RE.sub(' ', s); s = NUM_RE.sub(' ', s); s = re.sub(r'\d+', '', s)
    return ' '.join(w for w in s.split() if w not in STOPS and len(w) > 1)

def parse_nutr(v, servings):
    """Extract macros from total_nutrients dict, divide by servings."""
    srv = max(float(servings) if pd.notna(servings) and servings else 1, 1)
    empty = dict(calories_per_serving=0., protein_per_serving=0., fat_per_serving=0.,
                 carbs_per_serving=0., fiber_per_serving=0., sugar_per_serving=0.)
    if not v or pd.isna(v): return empty
    try:
        if isinstance(v, str):
            d = json.loads(v)
        elif isinstance(v, dict):
            d = v
        else:
            return empty
        return dict(
            calories_per_serving = round(float(d.get('ENERC_KCAL',{}).get('quantity',0) or 0) / srv, 1),
            protein_per_serving  = round(float(d.get('PROCNT',    {}).get('quantity',0) or 0) / srv, 1),
            fat_per_serving      = round(float(d.get('FAT',       {}).get('quantity',0) or 0) / srv, 1),
            carbs_per_serving    = round(float(d.get('CHOCDF',    {}).get('quantity',0) or 0) / srv, 1),
            fiber_per_serving    = round(float(d.get('FIBTG',     {}).get('quantity',0) or 0) / srv, 1),
            sugar_per_serving    = round(float(d.get('SUGAR',     {}).get('quantity',0) or 0) / srv, 1),
        )
    except: return empty

def norm_rows(m):
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n==0] = 1; return m / n

# ── LOAD ─────────────────────────────────────────────────────────────────────
print(f"📥 Loading dataset...")

from datasets import load_dataset
dataset = load_dataset("datahiveai/recipes-with-nutrition")
df = dataset['train'].to_pandas()

print(f"   {len(df):,} rows  |  columns: {list(df.columns)}")

# Add a recipe_id if not present (dataset has no id column)
if 'recipe_id' not in df.columns:
    df.insert(0, 'recipe_id', range(len(df)))

# ── PARSE LIST COLUMNS ───────────────────────────────────────────────────────
print("🔧 Parsing list columns...")
# Only apply if values are strings
for col in ['cuisine_type','dish_type','meal_type','diet_labels','health_labels','cautions','ingredient_lines']:
    if col in df.columns:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(safe_list)

df['cuisine'] = df['cuisine_type'].apply(lambda x: x[0].lower().strip() if x else 'unknown')
df['dish']    = df['dish_type'].apply(lambda x: x[0].lower().strip() if x else 'unknown')
df['meal']    = df['meal_type'].apply(lambda x: x[0].lower().strip() if x else 'unknown')

# ── NUTRITION ─────────────────────────────────────────────────────────────────
print("🥦 Extracting per-serving nutrition...")

# Use top-level 'calories' and 'servings' columns for calorie-per-serving
# Use total_nutrients for macros
df['servings_clean'] = df['servings'].apply(lambda x: max(float(x) if pd.notna(x) and x else 1, 1))
df['calories_per_serving'] = (df['calories'].fillna(0) / df['servings_clean']).round(1)

# Macros from total_nutrients
nutr_rows = [parse_nutr(v, s) for v, s in zip(df['total_nutrients'], df['servings'])]
nutr_df = pd.DataFrame(nutr_rows)

# Direct assignment (NO CONCAT = NO BUGS)
for col in nutr_df.columns:
    df[col] = nutr_df[col].values

# Use top-level calories_per_serving if total_nutrients calories is 0
mask = (df['calories_per_serving'] == 0)
mask = mask.astype(bool)
df['calories_per_serving'] = np.where(
    df['calories_per_serving'] == 0,
    df['calories'] / df['servings_clean'],
    df['calories_per_serving']
)

df['protein_per_calorie'] = np.where(
    df['calories_per_serving'] > 0,
    df['protein_per_serving'] / df['calories_per_serving'], 0)

cal_min = df['calories_per_serving'].min()
cal_max = df['calories_per_serving'].max()
print(f"   Calorie range: {cal_min:.0f} – {cal_max:.0f} kcal/serving")
print(f"   Median calories/serving: {df['calories_per_serving'].median():.0f}")

# ── CLEAN INGREDIENTS ─────────────────────────────────────────────────────────
print("🌿 Cleaning ingredient text...")
df['ingredients_clean'] = df['ingredient_lines'].apply(
    lambda ings: ' '.join(clean_ing(i) for i in ings) if ings else '')

# ── BINARY LABEL COLUMNS ─────────────────────────────────────────────────────
print("🏷️  Building binary label columns...")
for label in DIET_LABELS:
    df[f'diet_{label.lower().replace("-","_")}'] = df['diet_labels'].apply(lambda x: int(label in x))
for label in HEALTH_LABELS:
    df[f'health_{label.lower().replace("-","_")}'] = df['health_labels'].apply(lambda x: int(label in x))
for meal in MEAL_TYPES:
    df[f'meal_{meal.replace("/","_")}'] = df['meal'].apply(lambda x: int(meal in x.lower()))

# ── RICH TEXT FOR EMBEDDING ───────────────────────────────────────────────────
print("📝 Building rich text...")
def rich_text(row):
    name = str(row.get('recipe_name','') or '').lower()
    return ' '.join([name, name, name,
                     row['ingredients_clean'], row['ingredients_clean'],
                     row['cuisine'], row['dish']])
df['rich_text'] = df.apply(rich_text, axis=1)

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
print("🤖 Generating TF-IDF + LSA embeddings...")
print("   (Swap for SentenceTransformer('all-MiniLM-L6-v2') if GPU available)")

n_comp = min(EMBEDDING_DIM, len(df)-1)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=VOCAB_SIZE, ngram_range=(1,2),
                               sublinear_tf=True, min_df=2)),
    ('lsa',   TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)),
])
texts = df['rich_text'].fillna('').tolist()
print(f"   Fitting on {len(texts):,} recipes...")
emb = norm_rows(pipe.fit_transform(texts).astype(np.float32))
print(f"   Text embedding matrix: {emb.shape}")

# Nutrition embedding (z-scored, L2-normed)
n_cols = ['calories_per_serving','protein_per_serving','fat_per_serving',
          'carbs_per_serving','fiber_per_serving']
nm = df[n_cols].fillna(0).values.astype(np.float32)
mu, sd = nm.mean(0), nm.std(0); sd[sd==0] = 1
nutr_emb = norm_rows((nm - mu) / sd)

# Cuisine one-hot embedding
cuis_emb = norm_rows(pd.get_dummies(df['cuisine']).values.astype(np.float32))

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("💾 Saving artifacts...")
save_cols = [
    'recipe_id','recipe_name','url','source','ingredients_clean','ingredient_lines',
    'cuisine','dish','meal','image_url',
    'calories_per_serving','protein_per_serving','fat_per_serving',
    'carbs_per_serving','fiber_per_serving','sugar_per_serving',
    'protein_per_calorie','servings_clean','rich_text',
    'diet_labels','health_labels',
] + [c for c in df.columns if c.startswith(('diet_','health_','meal_'))]

save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_csv(f'{args.out}/recipes_processed.csv', index=False)
np.save(f'{args.out}/recipe_embeddings.npy',    emb)
np.save(f'{args.out}/nutrition_embeddings.npy', nutr_emb)
np.save(f'{args.out}/cuisine_embeddings.npy',   cuis_emb)

cuisine_classes = list(pd.get_dummies(df['cuisine']).columns)
meta = dict(
    diet_labels    = DIET_LABELS,
    health_labels  = HEALTH_LABELS,
    meal_types     = MEAL_TYPES,
    cuisine_types  = sorted(df['cuisine'].unique().tolist()),
    dish_types     = sorted(df['dish'].unique().tolist()),
    calorie_range  = [float(df['calories_per_serving'].min()),
                      float(df['calories_per_serving'].max())],
    n_recipes      = len(df),
    embedding_dim  = emb.shape[1],
    nutrition_col_means = mu.tolist(),
    nutrition_col_stds  = sd.tolist(),
)
with open(f'{args.out}/label_encoders.pkl','wb') as f:
    pickle.dump({'meta':meta,'pipeline':pipe,'cuisine_classes':cuisine_classes}, f)

print()
print(f"✅ Done! {len(df):,} recipes processed")
print(f"   recipe_embeddings.npy  → {emb.shape}")
print(f"   recipes_processed.csv  → {len(df):,} rows, {len(save_cols)} columns")
print(f"   label_encoders.pkl     → fitted TF-IDF+LSA pipeline + metadata")
print()
print("Next step:  uvicorn api:app --host 0.0.0.0 --port 8000")
