# app.py
# Smart Product Recommendation Streamlit app
# Put all artifacts in ./data/ (unzip artifacts into the folder)

import os
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = "./data"

st.set_page_config(page_title="Amazon-like Recommender", layout="wide")

# ---------------------------
# Safe loader
# ---------------------------
@st.cache_resource
def load_artifacts(base_dir):
    expected = {
        "als": "als_model.joblib",
        "user_le": "user_le.joblib",
        "item_le": "item_le.joblib",
        "user_item_matrix": "user_item_matrix.joblib",
        "co_view": "co_view_top.json",
        "popular": "popular_items.joblib",
        "prod_meta": "prod_meta.csv",
        "cat_rel": "category_relationships_large.csv"
    }

    # Check which required files exist (prod_meta and als are required)
    missing = [v for k,v in expected.items() if k in ("als","user_le","item_le","user_item_matrix","prod_meta") and not os.path.exists(os.path.join(base_dir, v))]
    if missing:
        st.error(f"Missing required artifact files in {base_dir}: {missing}. Upload them (unzipped) and refresh.")
        return None

    # Load artifacts with try/except
    try:
        model = joblib.load(os.path.join(base_dir, expected["als"]))
        user_le = joblib.load(os.path.join(base_dir, expected["user_le"]))
        item_le = joblib.load(os.path.join(base_dir, expected["item_le"]))
        user_item_matrix = joblib.load(os.path.join(base_dir, expected["user_item_matrix"]))
    except Exception as e:
        st.error(f"Error loading core artifacts (als/user/item/matrix): {e}")
        return None

    # Optional artifacts
    co_view_top = {}
    if os.path.exists(os.path.join(base_dir, expected["co_view"])):
        try:
            with open(os.path.join(base_dir, expected["co_view"]), "r") as f:
                co_view_top = json.load(f)
        except Exception:
            co_view_top = {}

    popular_items = []
    if os.path.exists(os.path.join(base_dir, expected["popular"])):
        try:
            popular_items = joblib.load(os.path.join(base_dir, expected["popular"]))
        except Exception:
            popular_items = []

    # prod_meta
    try:
        prod_meta_df = pd.read_csv(os.path.join(base_dir, expected["prod_meta"]))
    except Exception as e:
        st.error(f"Error reading prod_meta.csv: {e}")
        return None

    # Build fast lookup structures: map item_code -> meta and item_id -> meta
    prod_meta_index = {}        # keyed by item_code (string)
    prod_meta_by_itemid = {}    # keyed by item_id

    if "item_code" in prod_meta_df.columns:
        for _, r in prod_meta_df.iterrows():
            if pd.notna(r["item_code"]):
                prod_meta_index[str(int(r["item_code"]))] = {
                    "item_id": str(r.get("item_id","")),
                    "title": r.get("title",""),
                    "category_id": r.get("category_id",""),
                    "brand": r.get("brand",""),
                    "price": r.get("price","")
                }
    if "item_id" in prod_meta_df.columns:
        for _, r in prod_meta_df.iterrows():
            prod_meta_by_itemid[str(r["item_id"])] = {
                "title": r.get("title",""),
                "category_id": r.get("category_id",""),
                "brand": r.get("brand",""),
                "price": r.get("price","")
            }

    # category relationships (optional)
    cat_rel_map = {}
    cat_rel_path = os.path.join(base_dir, expected["cat_rel"])
    if os.path.exists(cat_rel_path):
        try:
            cr = pd.read_csv(cat_rel_path)
            for _, row in cr.iterrows():
                mc = str(row.get("main_category",""))
                rc = str(row.get("related_category",""))
                cat_rel_map.setdefault(mc, set()).add(rc)
        except Exception:
            cat_rel_map = {}

    return {
        "model": model,
        "user_le": user_le,
        "item_le": item_le,
        "user_item_matrix": user_item_matrix,
        "co_view_top": co_view_top,
        "popular_items": popular_items,
        "prod_meta_index": prod_meta_index,
        "prod_meta_by_itemid": prod_meta_by_itemid,
        "cat_rel_map": cat_rel_map
    }

art = load_artifacts(BASE_DIR)
if art is None:
    st.stop()

model = art["model"]
user_le = art["user_le"]
item_le = art["item_le"]
user_item_matrix = art["user_item_matrix"]
co_view_top = art["co_view_top"]
popular_items = art["popular_items"]
prod_meta_index = art["prod_meta_index"]
prod_meta_by_itemid = art["prod_meta_by_itemid"]
cat_rel_map = art["cat_rel_map"]

# ---------------------------
# Helpers
# ---------------------------
def get_meta_by_item_id(item_id):
    """Return metadata dict for item_id."""
    # try via item_le -> code -> prod_meta_index
    try:
        code = int(item_le.transform([item_id])[0])
        meta = prod_meta_index.get(str(code))
        if meta:
            return meta
    except Exception:
        pass
    # fallback by item_id string
    return prod_meta_by_itemid.get(str(item_id), {"title":"", "category_id":"", "brand":"", "price":""})

def als_recommend(user_id, N):
    if user_id not in user_le.classes_:
        return []
    try:
        u_idx = int(user_le.transform([user_id])[0])
        recs = model.recommend(u_idx, user_item_matrix, N=N)
        item_codes = [int(x[0]) for x in recs]
        return [item_le.inverse_transform([c])[0] for c in item_codes]
    except Exception:
        return []

def co_view_recommend(item_id, N):
    try:
        code = int(item_le.transform([item_id])[0])
        related_codes = co_view_top.get(str(code), [])[:N]
        return [item_le.inverse_transform([int(c)])[0] for c in related_codes]
    except Exception:
        return []

def filter_by_categories(candidate_item_ids, allowed_categories):
    if allowed_categories is None:
        return candidate_item_ids
    out = []
    for it in candidate_item_ids:
        meta = get_meta_by_item_id(it)
        if meta.get("category_id") in allowed_categories:
            out.append(it)
    return out

def show_item_info(items):
    rows = []
    for it in items:
        meta = get_meta_by_item_id(it)
        rows.append({
            "item_id": it,
            "title": meta.get("title",""),
            "brand": meta.get("brand",""),
            "category": meta.get("category_id",""),
            "price": meta.get("price","")
        })
    return pd.DataFrame(rows)

# ---------------------------
# UI
# ---------------------------
st.title("Amazon-like Product Recommender (Demo)")
st.markdown("Enter a user id and a current product id (from the catalog) to see recommendations.")

col1, col2 = st.columns([2,1])
with col1:
    user_id_input = st.text_input("User ID (e.g., u1) — leave blank for anonymous", "")
    item_id_input = st.text_input("Current Item ID (e.g., p1000)", "")

with col2:
    N = st.slider("Number of recommendations", 1, 20, 6)
    show_images = st.checkbox("Show images (if prod_meta has image_url)", value=False)

if st.button("Get Recommendations"):
    # Determine allowed categories (same category + related if available)
    allowed_categories = None
    if item_id_input:
        meta = get_meta_by_item_id(item_id_input)
        curr_cat = meta.get("category_id")
        if curr_cat:
            allowed_categories = {curr_cat}
            if curr_cat in cat_rel_map:
                allowed_categories.update(cat_rel_map[curr_cat])

    # Generate candidate lists (get extra so filtering still yields enough)
    als_cands = als_recommend(user_id_input, N*4)
    co_cands = co_view_recommend(item_id_input, N*4)
    pop_cands = [item_le.inverse_transform([i])[0] for i in popular_items[:N*6]] if popular_items else []

    # Filter by categories if available
    als_f = filter_by_categories(als_cands, allowed_categories)
    co_f = filter_by_categories(co_cands, allowed_categories)
    pop_f = filter_by_categories(pop_cands, allowed_categories)

    # Merge in priority: ALS > Co-view > Popular
    final = []
    for group in (als_f, co_f, pop_f):
        for it in group:
            if it not in final:
                final.append(it)
            if len(final) >= N:
                break
        if len(final) >= N:
            break

    # If still short, add unfiltered popular fallback
    if len(final) < N:
        for it in pop_cands:
            if it not in final:
                final.append(it)
            if len(final) >= N:
                break

    # Display
    st.subheader("Top Recommendations")
    df_show = show_item_info(final)
    st.table(df_show)

    st.subheader("Why these?")
    for it in final:
        reason = "Popular"
        if it in als_f:
            reason = "Personalized (ALS)"
        elif it in co_f:
            reason = "Co-view"
        st.write(f"- {it} — {reason}")
