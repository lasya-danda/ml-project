# app_streamlit.py
import os
import io
import uuid
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca.joblib")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.json")
MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.joblib")

# --- Streamlit page config ---
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation — Streamlit Demo")

# ---- helpers ----
def load_models():
    scaler = kmeans = pca = None
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(KMEANS_PATH):
            kmeans = joblib.load(KMEANS_PATH)
        if os.path.exists(PCA_PATH):
            pca = joblib.load(PCA_PATH)
    except Exception as e:
        st.warning(f"Failed to load some models: {e}")
    return scaler, kmeans, pca

def save_models(scaler, kmeans, pca):
    try:
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(kmeans, KMEANS_PATH)
        joblib.dump(pca, PCA_PATH)
    except Exception as e:
        st.warning(f"Failed to save models: {e}")

def save_feature_order(order):
    try:
        with open(FEATURE_ORDER_PATH, "w") as f:
            json.dump(order, f)
    except Exception as e:
        st.warning(f"Could not save feature order: {e}")

def load_feature_order():
    if os.path.exists(FEATURE_ORDER_PATH):
        try:
            with open(FEATURE_ORDER_PATH, "r") as f:
                fo = json.load(f)
                if isinstance(fo, list) and all(isinstance(x, str) for x in fo):
                    return fo
        except Exception:
            pass
    # fallback defaults (edit to suit your dataset)
    return ["Income", "Recency", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "MntWines", "MntMeatProducts"]

def save_medians(medians):
    try:
        joblib.dump(medians, MEDIANS_PATH)
    except Exception as e:
        st.warning(f"Could not save medians: {e}")

def load_medians():
    if os.path.exists(MEDIANS_PATH):
        try:
            return joblib.load(MEDIANS_PATH)
        except Exception:
            pass
    return None

def _clean_numeric_cols_from_df(df):
    """Return numeric columns list excluding ID-like and known clustering cols."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if not c.lower().startswith("id")]
    for c in ("Cluster", "PCA1", "PCA2"):
        if c in num_cols:
            num_cols.remove(c)
    return num_cols

# Robust cluster function: drops previous cluster columns if present
def cluster_dataframe(df, default_k=3, use_saved_if_possible=True):
    df = df.copy()

    # Remove previously-added clustering columns if present (helps when user re-uploads clustered CSV)
    to_drop = [c for c in ("Cluster", "PCA1", "PCA2") if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        st.info(f"Removed previously present columns: {', '.join(to_drop)}")

    # select numeric columns excluding id-like columns
    numeric_cols = _clean_numeric_cols_from_df(df)
    # try encoding Gender if present and numeric columns < 2
    if len(numeric_cols) < 2 and "Gender" in df.columns:
        df["Gender_enc"] = df["Gender"].astype("category").cat.codes
        if "Gender_enc" not in numeric_cols:
            numeric_cols.append("Gender_enc")

    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric features for clustering. Encode categorical columns or provide more numeric fields.")

    # Prepare X with consistent numeric columns
    X = df[numeric_cols].copy()
    X = X.fillna(X.median())

    scaler, kmeans, pca = (None, None, None)
    if use_saved_if_possible:
        scaler, kmeans, pca = load_models()

    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    if kmeans is None:
        kmeans = KMeans(n_clusters=default_k, random_state=42, n_init=10).fit(X_scaled)
    labels = kmeans.predict(X_scaled)

    if pca is None:
        pca = PCA(n_components=2, random_state=42).fit(X_scaled)
    pca_coords = pca.transform(X_scaled)

    out = df.copy()
    out["Cluster"] = labels
    out["PCA1"] = pca_coords[:, 0]
    out["PCA2"] = pca_coords[:, 1]

    # save models and metadata
    save_models(scaler, kmeans, pca)
    save_feature_order(numeric_cols)
    save_medians(X.median().to_dict())

    # save CSV
    filename = f"clustered_{uuid.uuid4().hex[:8]}.csv"
    out_path = os.path.join(UPLOADS_DIR, filename)
    out.to_csv(out_path, index=False)

    counts = out["Cluster"].value_counts().to_dict()
    profiles = out.groupby("Cluster")[numeric_cols].mean(numeric_only=True).round(3).to_dict(orient="index")

    return out, {"counts": counts, "profiles": profiles}, out_path, numeric_cols

def predict_single_row(row_dict, feature_order=None):
    scaler, kmeans, pca = load_models()
    if scaler is None or kmeans is None or pca is None:
        raise RuntimeError("Saved models not found. Upload and cluster a CSV first.")
    if feature_order is None:
        feature_order = load_feature_order()
    medians = load_medians()
    # build row in order
    row = []
    for f in feature_order:
        if f in row_dict:
            row.append(row_dict[f])
        else:
            # fallback to median if available, else 0
            if medians and f in medians:
                row.append(medians[f])
            else:
                row.append(0.0)
    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    label = int(kmeans.predict(X_scaled)[0])
    coords = pca.transform(X_scaled)[0]
    return label, coords

# ---- UI ----
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
use_existing = st.sidebar.checkbox("Use existing clustered CSV from uploads", value=False)
chosen_file = None
if use_existing:
    files = sorted(os.listdir(UPLOADS_DIR))
    chosen_file = st.sidebar.selectbox("Choose uploaded file", [""] + files)

# top layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload & Cluster")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"CSV loaded — shape {df.shape}")
            k_default = st.number_input("Choose k (clusters) for KMeans", min_value=2, max_value=12, value=3, step=1)
            if st.button("Run Clustering"):
                with st.spinner("Clustering..."):
                    clustered_df, summary, out_path, feature_cols = cluster_dataframe(df, default_k=int(k_default))
                    st.success("Clustering finished and models saved")
                    st.write("Download file saved at:", out_path)
                    st.subheader("Cluster sample (first 20 rows)")
                    st.dataframe(clustered_df.head(20))
                    st.session_state["last_clustered_file"] = out_path
                    st.session_state["last_feature_cols"] = feature_cols
        except Exception as e:
            st.error(f"Failed to read or cluster CSV: {e}")

    elif use_existing and chosen_file:
        fp = os.path.join(UPLOADS_DIR, chosen_file)
        if os.path.exists(fp):
            clustered_df = pd.read_csv(fp)
            st.success(f"Loaded {chosen_file} — shape {clustered_df.shape}")
            st.dataframe(clustered_df.head(20))
            st.session_state["last_clustered_file"] = fp
            st.session_state["last_feature_cols"] = [c for c in clustered_df.columns if c not in ("Cluster", "PCA1", "PCA2")]
        else:
            st.error("Selected file not found.")
    else:
        st.info("Upload a CSV to run clustering, or choose an existing clustered file from uploads.")

with col2:
    st.subheader("Cluster summary")
    if "last_clustered_file" in st.session_state:
        last_df = pd.read_csv(st.session_state["last_clustered_file"])
        counts = last_df["Cluster"].value_counts().sort_index().to_dict()
        st.metric("Clusters", len(counts))
        for k, v in counts.items():
            st.write(f"Cluster {k}: {v} customers")
    else:
        st.write("No clustered data loaded yet.")

st.markdown("---")

# plots & profiles
if "last_clustered_file" in st.session_state:
    df_plot = pd.read_csv(st.session_state["last_clustered_file"])

    # PCA scatter
    st.subheader("PCA Scatter")
    try:
        fig = px.scatter(
            df_plot,
            x="PCA1",
            y="PCA2",
            color=df_plot["Cluster"].astype(str),
            hover_data=df_plot.columns.tolist(),
            width=900,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write("Plotly failed, falling back to static plot.")
        st.pyplot()

    # Cluster Profiles (numeric means only)
    st.subheader("Cluster Profiles (numeric means)")
    # select numeric columns only and remove clustering/PCA columns
    num_cols = df_plot.select_dtypes(include=["number"]).columns.tolist()
    for c in ("Cluster", "PCA1", "PCA2"):
        if c in num_cols:
            num_cols.remove(c)

    if len(num_cols) == 0:
        st.write("No numeric columns available to compute profile means.")
    else:
        profile = df_plot.groupby("Cluster")[num_cols].mean(numeric_only=True).round(3)
        st.dataframe(profile)

    # Optional: show most common categorical values per cluster
    cat_cols = df_plot.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.subheader("Top categorical values per cluster")
        try:
            modes = df_plot.groupby("Cluster")[cat_cols].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else ""
            )
            st.dataframe(modes.fillna(""))
        except Exception:
            st.write("Could not compute categorical modes — some categorical columns may be complex.")

st.markdown("---")

# single prediction form
st.subheader("Predict Single Customer")
feature_order = load_feature_order()
st.info(f"Predict form uses features: {', '.join(feature_order)} (you can update feature_order.json after training)")

with st.form("predict_form"):
    input_vals = {}
    for f in feature_order:
        default_val = 0.0
        if "income" in f.lower():
            default_val = 50000.0
        if "recency" in f.lower():
            default_val = 30.0
        input_vals[f] = st.number_input(f, value=float(default_val))
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        label, coords = predict_single_row(input_vals, feature_order)
        st.success(f"Predicted cluster: {label}")
        st.write("PCA coordinates:", np.round(coords, 3).tolist())
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure you have run clustering (or have saved models) first.")

st.markdown("---")
st.subheader("Downloads")
if "last_clustered_file" in st.session_state:
    fp = st.session_state["last_clustered_file"]
    with open(fp, "rb") as f:
        st.download_button("Download clustered CSV", data=f, file_name=os.path.basename(fp))
else:
    st.info("No clustered CSV available yet. Upload & cluster first.")
    