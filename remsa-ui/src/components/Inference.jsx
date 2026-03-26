import { useState, useRef, useEffect } from "react";
import { fromArrayBuffer } from "geotiff";
import "./Inference.css";

export default function Inference() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [task, setTask] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [maskOpacity, setMaskOpacity] = useState(0.75);
  const [classResult, setClassResult] = useState(null);
  const inputCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch("http://localhost:8350/inference/status");
      const data = await res.json();
      setTask(data.task);
      setJobId(data.job_id);
    } catch (err) {
      console.error("Failed to fetch inference status", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    window.addEventListener("deployment-changed", fetchStatus);
    return () => window.removeEventListener("deployment-changed", fetchStatus);
  }, []);

  useEffect(() => {
    if (task !== "segmentation" || !maskUrl) return;
    const inputCanvas = inputCanvasRef.current;
    const outputCanvas = outputCanvasRef.current;
    if (!inputCanvas || !outputCanvas) return;

    outputCanvas.width = inputCanvas.width;
    outputCanvas.height = inputCanvas.height;
    const ctx = outputCanvas.getContext("2d");
    ctx.drawImage(inputCanvas, 0, 0);
  }, [maskUrl, task]);

  function stretch(value, min, max) {
    if (max === min) return 0;
    const x = ((value - min) / (max - min)) * 255;
    return Math.max(0, Math.min(255, x));
  }

  async function renderGeoTIFF(f) {
    const buffer = await f.arrayBuffer();
    const tiff = await fromArrayBuffer(buffer);
    const image = await tiff.getImage();

    const width = image.getWidth();
    const height = image.getHeight();
    const totalPixels = width * height;

    const rasters = await image.readRasters({ interleave: true });
    const numBands = rasters.length / totalPixels;

    const canvas = inputCanvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    let rIndex = 0, gIndex = 1, bIndex = 2;
    if (numBands === 13) { rIndex = 3; gIndex = 2; bIndex = 1; }
    else if (numBands === 6) { rIndex = 2; gIndex = 1; bIndex = 0; }
    else if (numBands === 3) { rIndex = 0; gIndex = 1; bIndex = 2; }

    let rMin = Infinity, rMax = -Infinity;
    let gMin = Infinity, gMax = -Infinity;
    let bMin = Infinity, bMax = -Infinity;

    for (let i = 0; i < totalPixels; i++) {
      const r = rasters[i * numBands + rIndex];
      const g = rasters[i * numBands + gIndex];
      const b = rasters[i * numBands + bIndex];
      if (r < rMin) rMin = r; if (r > rMax) rMax = r;
      if (g < gMin) gMin = g; if (g > gMax) gMax = g;
      if (b < bMin) bMin = b; if (b > bMax) bMax = b;
    }

    for (let i = 0; i < totalPixels; i++) {
      imageData.data[i * 4]     = stretch(rasters[i * numBands + rIndex], rMin, rMax);
      imageData.data[i * 4 + 1] = stretch(rasters[i * numBands + gIndex], gMin, gMax);
      imageData.data[i * 4 + 2] = stretch(rasters[i * numBands + bIndex], bMin, bMax);
      imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
  }

  async function handleFile(f) {
    setError(null);
    setMaskUrl(null);
    setClassResult(null);

    if (!f.name.endsWith(".tif") && !f.name.endsWith(".tiff")) {
      setError("Invalid file type.");
      return;
    }

    setFile(f);
    await renderGeoTIFF(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }

  async function runPredict() {
    if (!file || !task) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8350/inference/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error(await res.text());

      if (task === "segmentation") {
        const blob = await res.blob();
        setMaskUrl(URL.createObjectURL(blob));
      } else {
        const data = await res.json();
        setClassResult({
          predictedIndex: data.predicted_index,
          predictedLabel: data.predicted_label,
          classes: [...data.classes].sort((a, b) => b.probability - a.probability)
        });
      }
    } catch (err) {
      setError(err?.message || "Inference failed");
    }

    setLoading(false);
  }

  function clear() {
    setFile(null);
    setMaskUrl(null);
    setClassResult(null);
    setError(null);
    const c = inputCanvasRef.current;
    if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
  }

  return (
    <div className="inference-container">
      <div className="inference-header">
        <div className="status-top-bar">
          <div className="title-area">
            <h2 className="panel-title">Inference</h2>
            {task && (
              <div className={`task-indicator-pill ${task}`}>
                <span className="dot"></span>
                {task === "segmentation" ? "Segmentation" : "Classification"}
              </div>
            )}
          </div>

          <div className="active-model-card">
            <div className="model-info-label">Active Deploy</div>
            <div className="model-info-value">
              {jobId ? (
                <>
                  <span className="job-id-tag">ID: {jobId.split("-")[0]}</span>
                </>
              ) : (
                <span className="no-model">Waiting for deployment...</span>
              )}
            </div>
          </div>
        </div>
      </div>
      <div className="inference-split">
        {/* LEFT: UPLOAD & PREVIEW */}
        <div className="inf-left">
          <div
            className={`dropzone ${dragging ? "dragging" : ""} ${!task ? "disabled" : ""}`}
            onDragOver={(e) => { e.preventDefault(); if (task) setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".tif,.tiff"
              hidden
              id="inf-file-input"
              onChange={(e) => handleFile(e.target.files?.[0])}
              disabled={loading || !task}
            />

            {!file ? (
              <label htmlFor="inf-file-input" className="dropzone-label">
                Drag & drop a GeoTIFF here
                <span className="hint">or click to browse</span>
              </label>
            ) : (
              <>
                <canvas ref={inputCanvasRef} className="tiff-canvas" />
                {!loading && (
                  <button className="remove-btn-inline" onClick={clear}>
                    Remove File
                  </button>
                )}
              </>
            )}
          </div>

          {error && <div className="error">{error}</div>}

          <button
            className="run-btn"
            disabled={!file || loading || !task}
            onClick={runPredict}
          >
            {loading ? "Running inference…" : `Run Inference`}
          </button>
        </div>

        <div className={`inf-right ${task === "classification" ? "compact-mode" : ""}`}>

          {(!maskUrl && !classResult) && (
            <div className={`result-placeholder ${task === "classification" ? "small" : ""}`}>
              {!task ? "Deploy a model first." : !file ? "Upload a file to see results." : loading ? "Running…" : "No result yet."}
            </div>
          )}

          {/* Classification Results */}
          {task === "classification" && classResult && (
             <div className="result-compact">
              <div className="prediction-main">
                <div className="pred-label">{classResult.predictedLabel}</div>
              </div>
              <div className="class-list">
                {classResult.classes.map(c => (
                  <div key={c.index} className={"class-row " + (c.index === classResult.predictedIndex ? "top" : "")}>
                    <div className="class-name">{c.label}</div>
                    <div className="class-bar-wrapper">
                      <div className="class-bar" style={{ width: `${c.probability * 100}%` }} />
                    </div>
                    <div className="class-prob">{(c.probability * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Segmentation Results */}
          {task === "segmentation" && maskUrl && (
             <div className="segmentation-result-wrapper">
              <div className="image-wrapper">
                <canvas ref={outputCanvasRef} className="tiff-canvas" />
                <img src={maskUrl} alt="mask" className="mask-overlay" style={{ opacity: maskOpacity }} />
              </div>
              <div className="opacity-control">
                <label>Mask opacity</label>
                <input type="range" min="0" max="1" step="0.01" value={maskOpacity} onChange={e => setMaskOpacity(Number(e.target.value))} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}