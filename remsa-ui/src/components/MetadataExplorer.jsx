import { useState, useEffect, useRef } from "react";
import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import "./MetadataExplorer.css";

export default function MetadataExplorer() {
  const [sql, setSql] = useState(
    //"SELECT * FROM models_top_level_schema LIMIT 5"
  "SELECT * FROM foundation_models LIMIT 5"
  );
  const [sqlResult, setSqlResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState("table");
  const textareaRef = useRef(null);

  async function executeSQL(queryOverride) {
    const query = queryOverride ?? sql;

    setLoading(true);
    setSqlResult("");

    try {
      const res = await fetch("http://localhost:8150/sql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      const json = await res.json();
      setSqlResult(JSON.stringify(json, null, 2));
    } catch {
      setSqlResult("Query failed.");
    }

    setLoading(false);
  }

  function JsonPretty({ jsonText }) {
    let data;

    try {
      data = JSON.parse(jsonText);
    } catch {
      return <div className="json-error">Invalid JSON</div>;
    }

    return (
      <div className="json-pretty-container">
        <JsonView data={data} style={defaultStyles} collapsed={1} />
      </div>
    );
  }

  useEffect(() => {
    if (!textareaRef.current) return;

    const el = textareaRef.current;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [sql]);

  // listen for explore-model events from chat
  useEffect(() => {
    function handleModelEvent(e) {
      const models = Array.isArray(e.detail) ? e.detail : [e.detail];
      console.log("Received explore-model event:", models);

      //const q = `SELECT model_name, short_description, paper_link, repository, backbone, masking_strategy, domain_knowledge, supported_sensors, modalities, spectral_alignment, temporal_alignment, spatial_resolution, temporal_resolution, bands FROM models_top_level_schema WHERE model_id = '${modelId}'`;
      const list = models.map(m => `'${m}'`).join(",");

      const q = `SELECT model_name, short_description, paper_link, repository,
       backbone, masking_strategy, domain_knowledge,
       supported_sensors, modalities, spectral_alignment,
       temporal_alignment, spatial_resolution,
       temporal_resolution, bands
FROM foundation_models
WHERE model_id IN (${list})`;
      setSql(q);
      executeSQL(q);
    }

    window.addEventListener("explore-model", handleModelEvent);
    return () => window.removeEventListener("explore-model", handleModelEvent);
  }, []);

  return (
    <div className="metadata-container" id="metadata-panel">
      <h2>Model Database Explorer</h2>

      <div className="view-toggle">
        <label>
          <input
            type="radio"
            name="viewMode"
            value="table"
            checked={viewMode === "table"}
            onChange={() => setViewMode("table")}
          />
          Table View
        </label>

        <label>
          <input
            type="radio"
            name="viewMode"
            value="json"
            checked={viewMode === "json"}
            onChange={() => setViewMode("json")}
          />
          JSON
        </label>
      </div>

      <textarea
        ref={textareaRef}
        className="sql-textarea"
        value={sql}
        onChange={e => {
          setSql(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${e.target.scrollHeight}px`;
        }}
      />

      <button className="run-button" onClick={() => executeSQL()} disabled={loading}>
        {loading ? "Running..." : "Run SQL"}
      </button>

      <div className="result-container">
        {viewMode === "json" ? (
          <JsonPretty jsonText={sqlResult} />
        ) : (
          <JsonTable data={sqlResult} />
        )}
      </div>
    </div>
  );
}

function JsonTable({ data }) {
  if (!data) return null;

  let parsed;
  try {
    parsed = JSON.parse(data);
  } catch {
    return <pre className="raw-result">{data}</pre>;
  }

  const rows = Array.isArray(parsed) ? parsed : [parsed];
  if (rows.length === 0) return <div className="no-results">No results.</div>;

  const keys = Array.from(
    new Set(rows.flatMap(obj => Object.keys(obj)))
  );

  return (
    <table className="meta-table multi-col-table">
      <thead>
        <tr>
          <th className="meta-key-cell"></th>
          {rows.map((r, i) => (
            <th key={i} className="meta-col-header">
              {r.model_name || r.model_id || `Model ${i + 1}`}
            </th>
          ))}
        </tr>
      </thead>

      <tbody>
        {keys.map(key => (
          <tr key={key} className="meta-row">
            <td className="meta-key-cell">{key}</td>

            {rows.map((row, i) => (
              <td key={i} className="meta-value-cell">
                {formatCell(key, row[key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatCell(key, value) {
  if (value === null || value === undefined) return "";
  if (key === "repository" || key === "paper_link" || key === "weights") {
    return (
      <a href={value} target="_blank" rel="noopener noreferrer">
        {value}
      </a>
    );
  }

  if (key === "pretraining_phases" || key === "benchmarks") {
    try {
      const parsed = JSON.parse(value);
      if (typeof parsed === "object") {
        return <JsonTextarea value={parsed} />;
      }
    } catch {}
  }

  if (typeof value === "string" && value.startsWith("[") && value.endsWith("]")) {
    try {
      const arr = JSON.parse(value);
      return arr.join(", ");
    } catch {}
  }
  return String(value);
}

function JsonTextarea({ value }) {
  const [expanded, setExpanded] = useState(false);

  const text =
    typeof value === "string"
      ? value
      : JSON.stringify(value, null, 2);

  return (
    <textarea
      className={`json-textarea ${expanded ? "expanded" : ""}`}
      readOnly
      value={text}
      rows={expanded ? 12 : 2}
      onClick={() => setExpanded(true)}
      onBlur={() => setExpanded(false)}
    />
  );
}