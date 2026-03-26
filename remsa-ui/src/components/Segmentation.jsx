import { useState, useRef, useEffect } from "react"
import { fromArrayBuffer } from "geotiff"
import "./Segmentation.css"

export default function Segmentation() {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [maskUrl, setMaskUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [maskOpacity, setMaskOpacity] = useState(0.5)
  const [deployStatus, setDeployStatus] = useState(null)
  const [isDeployed, setIsDeployed] = useState(false)

  const inputCanvasRef = useRef(null)
  const outputCanvasRef = useRef(null)

  useEffect(() => {
    if (!maskUrl) return

    const inputCanvas = inputCanvasRef.current
    const outputCanvas = outputCanvasRef.current

    if (!inputCanvas || !outputCanvas) return

    outputCanvas.width = inputCanvas.width
    outputCanvas.height = inputCanvas.height

    const ctx = outputCanvas.getContext("2d")
    ctx.drawImage(inputCanvas, 0, 0)

  }, [maskUrl])

  async function deploySegmentationModel() {
    setDeployStatus("Deploying…")

    try {
      const res = await fetch(
        "http://localhost:8350/deploy/44b0522b-do-not-delete/segmentation",
        { method: "POST" }
      )

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text)
      }

      await res.json()

      setDeployStatus("Model deployed")
      setIsDeployed(true)

    } catch (err) {
      console.error(err)
      setDeployStatus("Deploy failed")
    }
  }

  async function renderGeoTIFF(f) {
    const buffer = await f.arrayBuffer()
    const tiff = await fromArrayBuffer(buffer)
    const image = await tiff.getImage()

    const width = image.getWidth()
    const height = image.getHeight()
    const totalPixels = width * height

    const rasters = await image.readRasters({ interleave: true })
    const numBands = rasters.length / totalPixels

    const canvas = inputCanvasRef.current
    const ctx = canvas.getContext("2d")

    canvas.width = width
    canvas.height = height

    const imageData = ctx.createImageData(width, height)

    let rIndex = 0
    let gIndex = 1
    let bIndex = 2

    if (numBands === 13) {
      // Full Sentinel-2
      rIndex = 3  // B04
      gIndex = 2  // B03
      bIndex = 1  // B02
    }
    else if (numBands === 6) {
      // Sen1Floods11 S2L1CHand
      rIndex = 2
      gIndex = 1
      bIndex = 0
    }
    else if (numBands === 3) {
      rIndex = 0
      gIndex = 1
      bIndex = 2
    }

    const rVals = new Float32Array(totalPixels)
    const gVals = new Float32Array(totalPixels)
    const bVals = new Float32Array(totalPixels)

    for (let i = 0; i < totalPixels; i++) {
      rVals[i] = rasters[i * numBands + rIndex]
      gVals[i] = rasters[i * numBands + gIndex]
      bVals[i] = rasters[i * numBands + bIndex]
    }

    //  stretch
    function getPercentileBounds(arr) {
      const sorted = Array.from(arr).sort((a, b) => a - b)
      const low = sorted[Math.floor(sorted.length * 0.02)]
      const high = sorted[Math.floor(sorted.length * 0.98)]
      return [low, high]
    }

    const [rMin, rMax] = getPercentileBounds(rVals)
    const [gMin, gMax] = getPercentileBounds(gVals)
    const [bMin, bMax] = getPercentileBounds(bVals)

    function stretch(v, min, max) {
      if (max === min) return 0
      const x = ((v - min) / (max - min)) * 255
      return Math.max(0, Math.min(255, x))
    }

    for (let i = 0; i < totalPixels; i++) {
      imageData.data[i * 4]     = stretch(rVals[i], rMin, rMax)
      imageData.data[i * 4 + 1] = stretch(gVals[i], gMin, gMax)
      imageData.data[i * 4 + 2] = stretch(bVals[i], bMin, bMax)
      imageData.data[i * 4 + 3] = 255
    }

    ctx.putImageData(imageData, 0, 0)
  }

  async function handleFile(f) {
    setError(null)
    setMaskUrl(null)

    if (!f.name.endsWith(".tif") && !f.name.endsWith(".tiff")) {
      setError("Invalid file type.")
      return
    }

    setFile(f)
    await renderGeoTIFF(f)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }

  function handleSelect(e) {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  async function runSegmentation() {
  if (!file) return

  setLoading(true)

  const formData = new FormData()
  formData.append("file", file)

  try {
    const res = await fetch(
      "http://localhost:8350/predict",
      {
        method: "POST",
        body: formData
      }
    )

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text)
    }

    const blob = await res.blob()
    const url = URL.createObjectURL(blob)

    setMaskUrl(url)

  } catch (err) {
    console.error(err)
  }

  setLoading(false)
}

  function clear() {
    setFile(null)
    setMaskUrl(null)
    setError(null)

    ;[inputCanvasRef.current, outputCanvasRef.current].forEach(c => {
      if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height)
    })
  }

  return (
    <div className="segmentation-container">
      <h2>Segmentation</h2>
      <p className="subtitle">Upload a GeoTIFF and run segmentation.</p>

      <div className="segmentation-split">
        <div className="seg-left">
          <button
            className="deploy-btn"
            onClick={deploySegmentationModel}
            disabled={loading}
          >
            Deploy hardcoded segmentation model
          </button>

          {deployStatus && (
            <div className="deploy-status">
              {deployStatus}
            </div>
          )}
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".tif,.tiff"
              hidden
              id="seg-file-input"
              onChange={handleSelect}
              disabled={loading}
            />

            {!file ? (
              <label htmlFor="seg-file-input" className="dropzone-label">
                Drag & drop image here
                <span className="hint">or click to browse</span>
              </label>
            ) : (
              <>
                <canvas ref={inputCanvasRef} className="tiff-canvas" />
                {!loading && (
                  <button className="remove-btn-inline" onClick={clear}>
                    Remove
                  </button>
                )}
              </>
            )}
          </div>

          {error && <div className="error">{error}</div>}

          <button
            className="run-btn"
            disabled={!file || loading}
            onClick={runSegmentation}
          >
            {loading ? "Running segmentation…" : "Run segmentation"}
          </button>
        </div>

        <div className="seg-right">
          {!maskUrl ? (
            <div className="result-placeholder">
              {!file
                ? "Upload a file."
                : loading
                ? "Running…"
                : "No result yet."}
            </div>
          ) : (
            <>
              <div className="image-wrapper">
                <canvas ref={outputCanvasRef} className="tiff-canvas" />
                <img
                  src={maskUrl}
                  alt="mask"
                  className="mask-overlay"
                  style={{ opacity: maskOpacity }}
                />
              </div>

              <div className="opacity-control">
                <label>Mask opacity</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={maskOpacity}
                  onChange={(e) =>
                    setMaskOpacity(Number(e.target.value))
                  }
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}