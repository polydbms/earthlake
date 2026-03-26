import { useState } from "react";
import "./PaperExtractor.css";

export default function PaperExtractor() {
  const [file, setFile] = useState(null);
  const [extractResult, setExtractResult] = useState(null);
  const [rawResult, setRawResult] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [cleanJson, setCleanJson] = useState("");

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setExtractResult(null);
    setRawResult("");
    setError("");

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("http://localhost:8100/extract", {
        method: "POST",
        body: form
      });

      if (!res.ok) throw new Error(`Server responded with ${res.status}`);

      const json = await res.json();
      setExtractResult(json);
      setRawResult(JSON.stringify(json, null, 2));
    } catch (err) {
      console.error(err);
      setError("Upload or extraction failed.");
    }

    setUploading(false);
  }

  function handleCreateJson() {
    if (!extractResult) return;

    const clean = stripConfidence(extractResult);
    setCleanJson(JSON.stringify(clean, null, 2));
  }

  function handleFieldChange(fieldName, newValue) {
    setExtractResult(prev => {
      if (!prev) return prev;

      try {
        const parsed = JSON.parse(newValue);
        return { ...prev, [fieldName]: parsed };
      } catch (err) {
      }

      const current = prev[fieldName];

      if (current && typeof current === "object" && "value" in current) {
        return {
          ...prev,
          [fieldName]: { ...current, value: newValue }
        };
      }

      return { ...prev, [fieldName]: newValue };
    });
  }

  return (
    <div className="extract-container">
      <h2>PDF Extraction</h2>

      <div className="extract-upload-row">
        <input
          type="file"
          accept="application/pdf"
          onChange={e => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? "Extracting..." : "Upload & Extract"}
        </button>
      </div>

      {error && <div className="extract-error">{error}</div>}

      {!extractResult && !error && (
        <div className="extract-empty-hint">
          Upload a PDF describing a remote sensing foundation model.
        </div>
      )}

      {extractResult && (
        <>
          <h3 className="extract-subtitle">Extracted Metadata</h3>
          <ExtractionTable data={extractResult} onChange={handleFieldChange} />
        </>
      )}
      <button
        className="create-json-btn"
        onClick={handleCreateJson}
      >
        Create JSON
      </button>

      {cleanJson && (
        <textarea
          className="clean-json-textarea"
          readOnly
          value={cleanJson}
          ref={el => {
            if (el) {
              el.style.height = "auto";
              el.style.height = `${el.scrollHeight}px`;
            }
          }}
        />
      )}
    </div>
  );
}

function ExtractionTable({ data, onChange }) {
  if (!data) return null;

  const entries = Object.entries(data);

  return (
    <table className="extract-table">
      <thead>
        <tr>
          <th className="col-field">Field</th>
          <th className="col-value">Value</th>
          <th className="col-conf">Confidence</th>
        </tr>
      </thead>

      <tbody>
        {entries.map(([field, rawValue]) => {
          const { displayValue, editable, confidence } =
            normalizeField(field, rawValue);

          const isMultiline = displayValue.includes("\n");

          return (
            <tr key={field}>
              <td className="field-name">{field}</td>

              <td className="field-editor">
                {editable ? (
                  isMultiline ? (
                    <textarea
                      className="field-textarea"
                      value={displayValue}
                      onChange={e => onChange(field, e.target.value)}
                    />
                  ) : (
                    <input
                      className="field-input"
                      value={displayValue}
                      onChange={e => onChange(field, e.target.value)}
                    />
                  )
                ) : (
                  <pre className="field-readonly">{displayValue}</pre>
                )}
              </td>

              <td className="field-confidence">
                <ConfidenceBadge value={confidence} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}


function normalizeField(fieldName, raw) {
  if (raw && typeof raw === "object" && "value" in raw) {
    const v = raw.value;

    if (Array.isArray(v)) {
      const joined = v
        .map(item => (item && typeof item === "object" ? item.value : item))
        .join("; ");

      const confs = v
        .map(item => item?.confidence)
        .filter(num => typeof num === "number");

      const meanConf =
        confs.length > 0
          ? confs.reduce((a, b) => a + b, 0) / confs.length
          : null;

      return {
        displayValue: joined,
        editable: true,
        confidence: meanConf
      };
    }

    if (typeof v === "string" || typeof v === "number" || v === null) {
      return {
        displayValue: v == null ? "" : String(v),
        editable: true,
        confidence: raw.confidence ?? null
      };
    }

    return {
      displayValue: JSON.stringify(v, null, 2),
      editable: true,
      confidence: raw.confidence ?? null
    };
  }

  if (Array.isArray(raw)) {
    return {
      displayValue: JSON.stringify(raw, null, 2),
      editable: true,
      confidence: null
    };
  }

  // fallback
  return {
    displayValue:
      raw == null
        ? ""
        : typeof raw === "string"
        ? raw
        : JSON.stringify(raw, null, 2),
    editable: true,
    confidence: null
  };
}

function ConfidenceBadge({ value }) {
  if (typeof value !== "number") {
    return <span className="conf-badge conf-none">—</span>;
  }

  let level = "low";
  if (value >= 0.9) level = "high";
  else if (value >= 0.7) level = "med";
  else if (value >= 0.4) level = "midlow";

  return (
    <span className={`conf-badge conf-${level}`}>
      {value.toFixed(3)}
    </span>
  );
}

function stripConfidence(value) {
  if (Array.isArray(value)) {
    return value.map(stripConfidence);
  }

  if (value && typeof value === "object") {
    if ("value" in value) {
      return stripConfidence(value.value);
    }

    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = stripConfidence(v);
    }
    return out;
  }

  return value;
}