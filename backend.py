# backend/app.py
import os
import uuid
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
MODEL_DIR = os.path.join(APP_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca.joblib")

app = Flask(__name__)
CORS(app)

def safe_load_models():
    scaler = kmeans = pca = None
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(KMEANS_PATH):
            kmeans = joblib.load(KMEANS_PATH)
        if os.path.exists(PCA_PATH):
            pca = joblib.load(PCA_PATH)
    except Exception as e:
        app.logger.warning(f"Failed to load saved models: {e}")
    return scaler, kmeans, pca

def cluster_dataframe(df):
    """
    Given a pandas DataFrame, run preprocessing, scaling, clustering, PCA.
    Uses saved models if present; otherwise fits new ones and saves them.
    Returns (clustered_df, summary_dict, out_path)
    """
    scaler, kmeans, pca = safe_load_models()

    # select numeric columns (exclude ID-like)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.lower().startswith("id")]

    # fallback: try to encode Gender if present
    if len(numeric_cols) < 2 and "Gender" in df.columns:
        df = df.copy()
        df["Gender_enc"] = df["Gender"].astype("category").cat.codes
        numeric_cols.append("Gender_enc")

    if len(numeric_cols) < 2:
        raise ValueError("Not enough numeric features found for clustering. Provide a CSV with at least 2 numeric features or include a 'Gender' column to be encoded.")

    X = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fit scaler if missing
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Fit kmeans if missing (default k=3)
    if kmeans is None:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

    labels = kmeans.predict(X_scaled)

    # Fit PCA if missing
    if pca is None:
        pca = PCA(n_components=2, random_state=42).fit(X_scaled)
    pca_coords = pca.transform(X_scaled)

    out = df.copy()
    out["Cluster"] = labels
    out["PCA1"] = pca_coords[:, 0]
    out["PCA2"] = pca_coords[:, 1]

    # Summaries
    counts = out["Cluster"].value_counts().to_dict()
    profiles = out.groupby("Cluster")[numeric_cols].mean().round(3).to_dict(orient="index")

    # Save clustered CSV with unique name
    out_filename = f"clustered_{uuid.uuid4().hex[:8]}.csv"
    out_path = os.path.join(UPLOAD_DIR, out_filename)
    out.to_csv(out_path, index=False)

    # Save models for future requests
    try:
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(kmeans, KMEANS_PATH)
        joblib.dump(pca, PCA_PATH)
    except Exception as e:
        app.logger.warning(f"Failed to save models: {e}")

    return out, {"counts": counts, "profiles": profiles}, out_path

@app.route("/", methods=["GET"])
def index():
    return (
        "<h3>Customer Segmentation API is running.</h3>"
        "<p>Use POST /upload to upload a CSV file and GET /download/&lt;filename&gt; to download results.</p>"
    )

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no filename"}), 400

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        file.save(temp_path)
    except Exception as e:
        return jsonify({"error": f"failed to save upload: {str(e)}"}), 500

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        return jsonify({"error": f"could not read csv: {str(e)}"}), 400

    try:
        clustered_df, summary, out_path = cluster_dataframe(df)
    except Exception as e:
        app.logger.exception("Clustering failed")
        return jsonify({"error": f"clustering error: {str(e)}"}), 500

    # Return sample rows and summary and the download filename
    sample = clustered_df.head(200).to_dict(orient="records")
    download_filename = os.path.basename(out_path)
    return jsonify({"data": sample, "summary": summary, "download": download_filename}), 200

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    safe_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(safe_path):
        return jsonify({"error": "file not found"}), 404
    # return CSV file as attachment
    return send_file(safe_path, as_attachment=True)

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
