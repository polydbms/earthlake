import { useState, useRef } from "react"
import { fromArrayBuffer } from "geotiff"
import "./Classification.css"

export default function Classification() {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [deployStatus, setDeployStatus] = useState(null)

  const previewCanvasRef = useRef(null)

  function stretch(value, min, max) {
    if (max === min) return 0
    return ((value - min) / (max - min)) * 255
  }
  async function deployModel() {
    setDeployStatus("Deploying…")

    try {
      const res = await fetch("http://localhost:8350/deploy/51b88894-do-not-delete/classification", {
        method: "POST",
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text)
      }

      const data = await res.json()
      console.log("Deploy response:", data)

      setDeployStatus("Model deployed")
    } catch (err) {
      console.error(err)
      setDeployStatus("Deploy failed")
    }
  }

  async function renderGeoTIFFPreview(f) {
  const buffer = await f.arrayBuffer()
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()

  const width = image.getWidth()
  const height = image.getHeight()
  const totalPixels = width * height

  const rasters = await image.readRasters({ interleave: true })
  const numBands = rasters.length / totalPixels

  const canvas = previewCanvasRef.current
  const ctx = canvas.getContext("2d")

  canvas.width = width
  canvas.height = height

  const imageData = ctx.createImageData(width, height)

  // ---- RGB index selection ----
  let rIndex = 0
  let gIndex = 1
  let bIndex = 2

  if (numBands === 13) {
    // Sentinel-2 full stack
    rIndex = 3
    gIndex = 2
    bIndex = 1
  }
  else if (numBands === 6) {
    // Reduced Sentinel-2 stack (B02,B03,B04,...)
    rIndex = 2
    gIndex = 1
    bIndex = 0
  }
  else if (numBands === 4 || numBands === 3) {
    rIndex = 0
    gIndex = 1
    bIndex = 2
  }

  // ---- Min/Max detection ----
  let rMin = Infinity, rMax = -Infinity
  let gMin = Infinity, gMax = -Infinity
  let bMin = Infinity, bMax = -Infinity

  for (let i = 0; i < totalPixels; i++) {
    const r = rasters[i * numBands + rIndex]
    const g = rasters[i * numBands + gIndex]
    const b = rasters[i * numBands + bIndex]

    if (r < rMin) rMin = r
    if (r > rMax) rMax = r
    if (g < gMin) gMin = g
    if (g > gMax) gMax = g
    if (b < bMin) bMin = b
    if (b > bMax) bMax = b
  }

  // ---- Stretch + render ----
  for (let i = 0; i < totalPixels; i++) {
    const r = rasters[i * numBands + rIndex]
    const g = rasters[i * numBands + gIndex]
    const b = rasters[i * numBands + bIndex]

    imageData.data[i * 4]     = stretch(r, rMin, rMax)
    imageData.data[i * 4 + 1] = stretch(g, gMin, gMax)
    imageData.data[i * 4 + 2] = stretch(b, bMin, bMax)
    imageData.data[i * 4 + 3] = 255
  }

  ctx.putImageData(imageData, 0, 0)
}

  async function validateAndSetFile(f) {
    setError(null)
    setResult(null)

    if (f.name.endsWith(".tif") || f.name.endsWith(".tiff")) {
      setFile(f)
      await renderGeoTIFFPreview(f)
    } else {
      setError("Invalid file type. Please upload a GeoTIFF (.tif or .tiff).")
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) validateAndSetFile(f)
  }

  function handleSelect(e) {
    const f = e.target.files?.[0]
    if (f) validateAndSetFile(f)
  }

  function clearFile() {
    setFile(null)
    setResult(null)
    setError(null)

    const canvas = previewCanvasRef.current
    if (canvas) {
      const ctx = canvas.getContext("2d")
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  async function runInference() {
  if (!file) return

  setIsLoading(true)
  setError(null)
  setResult(null)

  const formData = new FormData()
  formData.append("file", file)

  try {
    const res = await fetch("http://localhost:8350/predict", {
      method: "POST",
      body: formData,
    })

    if (!res.ok) {
      const text = await res.text()
      console.error(text)
      throw new Error(text)
    }

    const data = await res.json()
    console.log("Backend response:", data)

    setResult({
      predictedIndex: data.predicted_index,
      predictedLabel: data.predicted_label,
      classes: [...data.classes].sort(
        (a, b) => b.probability - a.probability
      )
    })

  } catch (err) {
  console.error("FETCH ERROR:", err)
  setError(err?.message ?? "Inference failed")
}

  setIsLoading(false)
}

  return (
    <div className="classification-container">
      <h2>Classification</h2>
      <p className="subtitle">Upload a GeoTIFF and run inference.</p>

      <div className="classification-split">

        {/* LEFT — Preview */}
        <div className="class-left">
          <button
            className="deploy-btn"
            onClick={deployModel}
            disabled={isLoading}
          >
            Deploy hardcoded model
          </button>

          {deployStatus && (
            <div className="deploy-status">
              {deployStatus}
            </div>
          )}
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".tif,.tiff"
              hidden
              id="file-input"
              onChange={handleSelect}
              disabled={isLoading}
            />

            {!file ? (
              <label htmlFor="file-input" className="dropzone-label">
                Drag & drop a GeoTIFF here
                <span className="hint">or click to browse</span>
              </label>
            ) : (
              <>
                <canvas ref={previewCanvasRef} className="tiff-canvas" />

                {!isLoading && (
                  <button
                    className="remove-btn-inline"
                    onClick={clearFile}
                  >
                    Remove
                  </button>
                )}
              </>
            )}
          </div>

          {error && <div className="error">{error}</div>}

          <button
            className="run-btn"
            onClick={runInference}
            disabled={!file || isLoading}
          >
            {isLoading ? "Running inference…" : "Run inference"}
          </button>
        </div>

        {/* RIGHT — Result only */}
        <div className="class-right compact">
          {!result ? (
            <div className="result-placeholder small">
              {!file
                ? "Upload a file."
                : isLoading
                ? "Running…"
                : "No result yet."}
            </div>
          ) : (
            <div className="result-compact">
              <div className="prediction-main">
                <div className="pred-label">
                  {result.predictedLabel}
                </div>
              </div>

              <div className="class-list">
                {result.classes.map(c => (
                  <div
                    key={c.index}
                    className={
                      "class-row " +
                      (c.index === result.predictedIndex ? "top" : "")
                    }
                  >
                    <div className="class-name">{c.label}</div>

                    <div className="class-bar-wrapper">
                      <div
                        className="class-bar"
                        style={{ width: `${c.probability * 100}%` }}
                      />
                    </div>

                    <div className="class-prob">
                      {(c.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}