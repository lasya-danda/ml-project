// frontend/src/App.jsx
import React, { useState, useRef } from "react";
import axios from "axios";
import { Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

export default function App() {
  const fileRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [clusters, setClusters] = useState([]);
  const [summary, setSummary] = useState(null);
  const [downloadFilename, setDownloadFilename] = useState(null);
  const [error, setError] = useState(null);

  async function handleUpload() {
    const file = fileRef.current.files[0];
    if (!file) return alert("Choose a CSV file first");
    setLoading(true);
    setError(null);
    setDownloadFilename(null);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await axios.post("/upload", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });

      const payload = res.data;
      setClusters(payload.data || []);
      setSummary(payload.summary || null);
      setDownloadFilename(payload.download || null);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  function buildChartData() {
    if (!clusters || clusters.length === 0) return { datasets: [] };
    const groups = {};
    clusters.forEach((r) => {
      const k = r.Cluster;
      if (!groups[k]) groups[k] = [];
      groups[k].push({ x: +r.PCA1, y: +r.PCA2 });
    });
    const datasets = Object.keys(groups).map((k) => ({
      label: `Cluster ${k}`,
      data: groups[k],
      pointRadius: 4,
    }));
    return { datasets };
  }

  function handleDownload() {
    if (!downloadFilename) return alert("No processed file available. Upload first.");
    // navigate to download endpoint
    window.location.href = `/download/${downloadFilename}`;
  }

  return (
    <div style={{ padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h1>Customer Segmentation Dashboard</h1>

      <div style={{ marginBottom: 12 }}>
        <input ref={fileRef} type="file" accept=".csv" />
        <button onClick={handleUpload} disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? "Running..." : "Upload & Cluster"}
        </button>
        <button onClick={handleDownload} style={{ marginLeft: 8 }}>
          Download Clustered CSV
        </button>
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      {summary && (
        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          {Object.entries(summary.counts || {}).map(([k, v]) => (
            <div key={k} style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
              <div style={{ fontSize: 12, color: "#555" }}>Cluster {k}</div>
              <div style={{ fontSize: 20, fontWeight: "bold" }}>{v}</div>
              <div style={{ fontSize: 12, color: "#777" }}>customers</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ height: 420, border: "1px solid #eee", padding: 10 }}>
        <h3>PCA Scatter (clusters)</h3>
        <div style={{ height: 340 }}>
          <Scatter data={buildChartData()} options={{ maintainAspectRatio: false }} />
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Cluster Table (sample)</h3>
        <div style={{ maxHeight: 240, overflow: "auto", border: "1px solid #eee", padding: 8 }}>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                {clusters.length > 0 &&
                  Object.keys(clusters[0]).map((c) => (
                    <th key={c} style={{ textAlign: "left", paddingRight: 8 }}>{c}</th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {clusters.slice(0, 100).map((r, i) => (
                <tr key={i}>
                  {Object.values(r).map((v, j) => (
                    <td key={j} style={{ paddingRight: 8 }}>{String(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
