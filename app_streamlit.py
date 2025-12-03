# app_streamlit.py
# Enhanced Streamlit app: uses ALL rows/columns by default, feature selector, target lists, suggestions
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
from sklearn.metrics import silhouette_score

# Paths
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

# Streamlit config
st.set_page_config(page_title="Customer Segmentation & Targeting", layout="wide")
st.title("Customer Segmentation & Targeting — Streamlit")

# ----------------- Helpers -----------------
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
        if scaler is not None:
            joblib.dump(scaler, SCALER_PATH)
        if kmeans is not None:
            joblib.dump(kmeans, KMEANS_PATH)
        if pca is not None:
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
    return None

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

def safe_numeric_df(df):
    """Return DataFrame where numeric-like columns are converted to numeric when possible."""
    df2 = df.copy()
    for c in df2.columns:
        # try convert strings that look numeric (remove commas)
        if df2[c].dtype == object:
            try:
                df2[c] = pd.to_numeric(df2[c].astype(str).str.replace(",", ""), errors="ignore")
            except Exception:
                pass
    return df2

def derive_metrics(df):
    """Create useful derived columns: TotalSpend (sum of Mnt*), TotalPurchases, CustomerValue."""
    d = df.copy()
    # total spend: sum columns that contain 'Mnt' or 'Amount' or 'Spend' (case-insensitive)
    spend_cols = [c for c in d.columns if ("mnt" in c.lower()) or ("amount" in c.lower()) or ("spend" in c.lower())]
    if not spend_cols:
        # fallback: common column names (adjust if needed)
        for name in ("MntWines","MntMeatProducts","MntFishProducts","MntFruits","MntGoldProds","MntSweetProducts"):
            if name in d.columns and name not in spend_cols:
                spend_cols.append(name)
    if spend_cols:
        d["TotalSpend"] = d[spend_cols].sum(axis=1)
    else:
        d["TotalSpend"] = 0.0

    # total purchases: sum of numeric purchase count columns
    purchase_cols = [c for c in d.columns if ("num" in c.lower()) or ("purch" in c.lower())]
    d["TotalPurchases"] = d[purchase_cols].sum(axis=1) if purchase_cols else 0

    # CustomerValue: normalized (Income + TotalSpend*weight + TotalPurchases*weight)
    inc = d["Income"] if "Income" in d.columns else pd.Series(0, index=d.index)
    # Use robust scaling for combination
    # Avoid divide by zero
    inc_norm = (inc.fillna(0) - inc.fillna(0).min()) / (inc.fillna(0).max() - inc.fillna(0).min() + 1e-9)
    spend = d["TotalSpend"].fillna(0)
    spend_norm = (spend - spend.min()) / (spend.max() - spend.min() + 1e-9)
    purch = d["TotalPurchases"].fillna(0)
    purch_norm = (purch - purch.min()) / (purch.max() - purch.min() + 1e-9)
    # weights: spend most important, then income, then purchases
    d["CustomerValue"] = (0.5 * spend_norm) + (0.35 * inc_norm) + (0.15 * purch_norm)
    return d, spend_cols, purchase_cols

# ----------------- UI Inputs -----------------
st.sidebar.header("Data & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (full dataset)", type=["csv"])
use_existing = st.sidebar.checkbox("Use existing clustered CSV from uploads", value=False)
chosen_file = None
if use_existing:
    files = sorted(os.listdir(UPLOADS_DIR))
    chosen_file = st.sidebar.selectbox("Choose uploaded file", [""] + files)

# clustering options
algo = st.sidebar.selectbox("Clustering algorithm", ["KMeans", "Agglomerative"], index=0)
k_min, k_max = st.sidebar.slider("k scan range (for diagnostics)", 2, 12, (2, 8))
run_scan = st.sidebar.checkbox("Show elbow & silhouette diagnostics", value=False)
max_preview = st.sidebar.number_input("Max preview rows", min_value=10, max_value=5000, value=200, step=10)

# main layout
col_left, col_right = st.columns([2, 1])

# ----------------- Upload / prepare data -----------------
with col_left:
    st.header("Upload / Prepare Data")
    df = None
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"CSV loaded — shape: {df_raw.shape}")
            df_raw = safe_numeric_df(df_raw)
            df, spend_cols, purchase_cols = derive_metrics(df_raw)
            st.write(f"Derived metrics: TotalSpend (from {len(spend_cols)} cols), TotalPurchases (from {len(purchase_cols)} cols)")
            st.session_state["last_uploaded_raw"] = True
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None
    elif use_existing and chosen_file:
        fp = os.path.join(UPLOADS_DIR, chosen_file)
        if os.path.exists(fp):
            df_loaded = pd.read_csv(fp)
            st.success(f"Loaded clustered CSV — shape {df_loaded.shape}")
            df_loaded = safe_numeric_df(df_loaded)
            df, spend_cols, purchase_cols = derive_metrics(df_loaded)
            st.session_state["last_clustered_file"] = fp
        else:
            st.error("Selected file not found.")
            df = None
    else:
        st.info("Upload a CSV or choose an existing clustered file.")

    if df is not None:
        # feature selector: default to all numeric columns except id-like and Cluster/PCA
        all_num = df.select_dtypes(include=["number"]).columns.tolist()
        all_num = [c for c in all_num if not c.lower().startswith("id") and c not in ("Cluster","PCA1","PCA2")]
        st.markdown("**Feature selector (used for clustering). By default all numeric features are selected.**")
        feature_choices = st.multiselect("Select features to include in clustering", options=all_num, default=all_num)

        # run diagnostics (elbow/silhouette)
        if run_scan and feature_choices:
            st.subheader("Diagnostics: Elbow & Silhouette")
            X_scan = df[feature_choices].fillna(df[feature_choices].median())
            scaler_local = StandardScaler().fit(X_scan)
            Xs = scaler_local.transform(X_scan)
            inertias = []
            sil_scores = []
            K_range = list(range(k_min, k_max + 1))
            for k in K_range:
                if algo == "KMeans":
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
                    labels = km.labels_
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(Xs, labels) if len(set(labels)) > 1 else float("nan"))
                else:
                    from sklearn.cluster import AgglomerativeClustering
                    cl = AgglomerativeClustering(n_clusters=k).fit(Xs)
                    labels = cl.labels_
                    inertias.append(np.nan)
                    sil_scores.append(silhouette_score(Xs, labels) if len(set(labels)) > 1 else float("nan"))

            c1, c2 = st.columns(2)
            with c1:
                st.line_chart({"k": K_range, "inertia": inertias})
            with c2:
                st.line_chart({"k": K_range, "silhouette": sil_scores})
            st.info("Pick k where inertia has elbow & silhouette is relatively high.")

        # choose k and run clustering
        k_choice = st.number_input("Choose k for final clustering", min_value=2, max_value=50, value=3, step=1)
        run_cluster = st.button("Run clustering on selected features")

        if run_cluster:
            if not feature_choices or len(feature_choices) < 2:
                st.error("Select at least 2 numerical features for clustering.")
            else:
                with st.spinner("Clustering..."):
                    # prepare X
                    X = df[feature_choices].fillna(df[feature_choices].median())
                    scaler_local = StandardScaler().fit(X)
                    Xs = scaler_local.transform(X)
                    if algo == "KMeans":
                        model = KMeans(n_clusters=int(k_choice), random_state=42, n_init=10)
                        labels = model.fit_predict(Xs)
                        inertia_val = model.inertia_
                    else:
                        from sklearn.cluster import AgglomerativeClustering
                        model = AgglomerativeClustering(n_clusters=int(k_choice))
                        labels = model.fit_predict(Xs)
                        inertia_val = None

                    pca_local = PCA(n_components=2, random_state=42).fit(Xs)
                    pca_coords = pca_local.transform(Xs)

                    out = df.copy()
                    out["Cluster"] = labels
                    out["PCA1"] = pca_coords[:, 0]
                    out["PCA2"] = pca_coords[:, 1]

                    # save models & metadata (for KMeans only save kmeans; always save scaler+pca)
                    if algo == "KMeans":
                        save_models(scaler_local, model, pca_local)
                    else:
                        save_models(scaler_local, None, pca_local)
                    save_feature_order(feature_choices)
                    save_medians(X.median().to_dict())

                    # save CSV
                    fname = f"clustered_{uuid.uuid4().hex[:8]}.csv"
                    out_path = os.path.join(UPLOADS_DIR, fname)
                    out.to_csv(out_path, index=False)
                    st.success(f"Clustering finished and saved to {out_path}")
                    st.session_state["last_clustered_file"] = out_path
                    st.session_state["last_feature_cols"] = feature_choices
                    st.session_state["last_algo"] = algo

                    # show counts and centers
                    counts = out["Cluster"].value_counts().sort_index()
                    st.write("Cluster counts:")
                    st.bar_chart(counts)

                    if counts.max() / counts.sum() > 0.85:
                        st.warning("One cluster contains >85% of samples — consider adjusting features or k.")

                    if algo == "KMeans" and hasattr(model, "cluster_centers_"):
                        centers_scaled = model.cluster_centers_
                        centers_orig = scaler_local.inverse_transform(centers_scaled)
                        centers_df = pd.DataFrame(centers_orig, columns=feature_choices)
                        centers_df.index.name = "Cluster"
                        st.subheader("Cluster centers (original scale)")
                        st.dataframe(centers_df.round(3))

                    # show preview table
                    st.subheader("Clustered data preview")
                    preview_n = int(min(max_preview, len(out)))
                    st.dataframe(out.head(preview_n))

# ----------------- Right column: summaries, plots, targeting -----------------
with col_right:
    st.header("Cluster Visualization & Targeting")
    if "last_clustered_file" in st.session_state:
        fp = st.session_state["last_clustered_file"]
        df_plot = pd.read_csv(fp)
        df_plot = safe_numeric_df(df_plot)
        # PCA scatter
        st.subheader("PCA Scatter (2D)")
        try:
            fig = px.scatter(df_plot, x="PCA1", y="PCA2", color=df_plot["Cluster"].astype(str),
                             hover_data=df_plot.columns.tolist(), width=700, height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Plotly scatter failed to render.")

        # Profile numeric
        st.subheader("Cluster profiles (numeric means)")
        num_cols = df_plot.select_dtypes(include=["number"]).columns.tolist()
        for c in ("Cluster","PCA1","PCA2"):
            if c in num_cols:
                num_cols.remove(c)
        if num_cols:
            profile = df_plot.groupby("Cluster")[num_cols].mean(numeric_only=True).round(3)
            st.dataframe(profile)
        else:
            st.write("No numeric columns for profile.")

        # categorical modes
        cat_cols = df_plot.select_dtypes(include=["object","category"]).columns.tolist()
        if cat_cols:
            try:
                modes = df_plot.groupby("Cluster")[cat_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "")
                st.subheader("Top categorical values per cluster")
                st.dataframe(modes.fillna(""))
            except Exception:
                st.write("Could not compute categorical modes.")

        # Marketing suggestions per cluster (simple rules)
        st.subheader("Marketing suggestions (auto)")
        suggestions = {}
        # compute quick metrics per cluster
        agg = df_plot.groupby("Cluster").agg({
            "CustomerValue":"mean" if "CustomerValue" in df_plot.columns else (lambda x: (x.fillna(0)).mean()),
            "TotalSpend":"mean" if "TotalSpend" in df_plot.columns else (lambda x: (x.fillna(0)).mean()),
            "Income":"mean" if "Income" in df_plot.columns else (lambda x: (x.fillna(0)).mean()),
            "TotalPurchases":"mean" if "TotalPurchases" in df_plot.columns else (lambda x: (x.fillna(0)).mean())
        })
        for cl in agg.index:
            # fallback safe reading
            cv = float(agg.loc[cl].get("CustomerValue", 0.0))
            ts = float(agg.loc[cl].get("TotalSpend", 0.0))
            inc = float(agg.loc[cl].get("Income", 0.0))
            tp = float(agg.loc[cl].get("TotalPurchases", 0.0))

            if cv > 0.6 or (ts > np.percentile(df_plot["TotalSpend"].fillna(0), 75)):
                sug = "VIP / Upsell: high spenders — target with premium bundles, loyalty perks."
            elif tp > np.percentile(df_plot["TotalPurchases"].fillna(0), 75) and cv > 0.3:
                sug = "Frequent buyers: cross-sell related products, increase AOV."
            elif ts < np.percentile(df_plot["TotalSpend"].fillna(0), 30) and inc < np.percentile(df_plot["Income"].fillna(0), 50):
                sug = "Price-sensitive: target with discounts, coupons, bundle offers."
            else:
                sug = "Retention / Engagement: personalized offers, email campaigns, re-engagement flows."

            suggestions[int(cl)] = sug
            st.write(f"Cluster {cl}: {sug}")

        st.markdown("---")
        # Target list generation
        st.subheader("Generate Target List")
        clusters_available = sorted(df_plot["Cluster"].unique().tolist())
        chosen_clusters = st.multiselect("Choose cluster(s) to target", options=clusters_available, default=clusters_available[:1])
        rank_by = st.selectbox("Rank targets by", ["CustomerValue","TotalSpend","TotalPurchases"], index=0)
        top_n = st.number_input("Top N to export per selected cluster (0 = all)", min_value=0, max_value=100000, value=200, step=10)
        generate = st.button("Generate target CSV")

        if generate:
            if not chosen_clusters:
                st.error("Pick at least one cluster.")
            else:
                targets = df_plot[df_plot["Cluster"].isin(chosen_clusters)].copy()
                if rank_by not in targets.columns:
                    st.warning(f"{rank_by} not found in data, computing CustomerValue/TotalSpend if possible.")
                    targets, _, _ = derive_metrics(targets)
                targets = targets.sort_values(by=rank_by, ascending=False)
                if top_n > 0:
                    targets_out = targets.groupby("Cluster").head(top_n).reset_index(drop=True)
                else:
                    targets_out = targets.reset_index(drop=True)

                st.success(f"Prepared {len(targets_out)} target rows (clusters: {chosen_clusters})")
                st.dataframe(targets_out.head(200))

                # download button
                csv_bytes = targets_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download target CSV", data=csv_bytes, file_name=f"targets_{uuid.uuid4().hex[:6]}.csv")

    else:
        st.info("No clustered file available. Run clustering (left) or choose an existing clustered CSV.")
