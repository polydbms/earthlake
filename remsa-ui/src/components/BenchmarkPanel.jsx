import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import "./BenchmarkPanel.css";

const API = "http://localhost:8300/benchmark";

const BAR_COLORS = [
  "#3B82F6", // blue
  "#10B981", // emerald
  "#F59E0B", // amber
  "#EF4444", // red
  "#8B5CF6", // violet
  "#06B6D4", // cyan
  "#F97316", // orange
  "#6366F1"  // indigo
];

function formatDuration(seconds) {
  if (seconds == null) return "\u2014";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

function timeAgo(date) {
  if (!date) return "";
  const diff = Math.max(0, Math.floor((Date.now() - new Date(date).getTime()) / 1000));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function StatusBadge({ status }) {
  return <span className={`status-badge ${status}`}>{status.charAt(0).toUpperCase() + status.slice(1)}</span>;
}

function ModeBadge({ mode }) {
  const label =
    mode === "linear_probe"
      ? "Linear Probe"
      : mode === "finetune"
      ? "Fine-Tune"
      : mode;

  return (
    <span className={`mode-badge ${mode}`}>
      {label}
    </span>
  );
}

// --- PLOTTING COMPONENTS ---

function MetricChart({ metricKey, label, runs, colorMap }) {

  const data = runs.map((r) => {
    const val =
      r.metrics?.[metricKey] ??
      r.metrics?.[`val_${metricKey}`] ??
      r[metricKey];

    if (val == null) return null;

    return {
      name: r.model_variant,
      value: parseFloat(val.toFixed(3)),
      color: colorMap[r.job_id]
    };
  }).filter(Boolean);

  if (data.length === 0) return null

  return (
    <div className="compare-chart-card">
      <p className="compare-chart-title">{label}</p>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 40 }}>
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11 }}
            angle={-35}
            textAnchor="end"
            interval={0}
          />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v) => [v, label]} />

          <Bar dataKey="value" radius={[4,4,0,0]}>
            {data.map((entry,i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

    </div>
  )
}

function MetricPlotSingle({ checkpoints, metricKey, title, color }) {
  const validData = checkpoints.filter(c => c[metricKey] != null && !isNaN(c[metricKey]));
  if (!validData.length) return null;

  const epochs = checkpoints.map(c => c.epoch);
  const xMin = Math.min(...epochs);
  const xMax = Math.max(...epochs);
  const xRange = Math.max(1, xMax - xMin);

  const width = 300;
  const height = 150;
  const margin = { top: 10, right: 10, bottom: 35, left: 45 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const values = validData.map(d => parseFloat(d[metricKey]));
  const max = Math.max(...values);
  const min = Math.min(...values);
  const padding = (max - min) * 0.1 || 0.01;
  const yMin = min - padding;
  const yMax = max + padding;

  const scaleX = epoch => margin.left + ((epoch - xMin) / xRange) * innerWidth;
  const scaleY = v => margin.top + innerHeight - ((v - yMin) / (yMax - yMin)) * innerHeight;

  const linePoints = validData.map(d => `${scaleX(d.epoch)},${scaleY(parseFloat(d[metricKey]))}`).join(" ");
  const yStep = (yMax - yMin) / 3;

  return (
    <div className="plot-card">
      <div className="plot-header">
        <span className="plot-title">{title}</span>
        <span className="plot-value" style={{ color }}>
          {values[values.length - 1] < 1 ? values[values.length - 1].toFixed(3) : values[values.length - 1].toFixed(1)}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart">
        <text x={margin.left + innerWidth / 2} y={height - 4} textAnchor="middle" fontSize="10" fill="#9ca3af" fontWeight="600" letterSpacing="0.05em" textTransform="uppercase">
          Epochs
        </text>
        {[0, 1, 2, 3].map(i => {
          const val = yMin + yStep * i;
          const y = scaleY(val);
          return (
            <g key={i}>
              <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} stroke="#f1f5f9" strokeWidth="1.5" />
              <text x={margin.left - 8} y={y + 3} textAnchor="end" fontSize="10" fill="#9ca3af">{val.toFixed(2)}</text>
            </g>
          );
        })}
        <line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} stroke="#cbd5e1" strokeWidth="1.5" />
        {checkpoints.map((c, i) => {
          const x = scaleX(c.epoch);
          const y = height - margin.bottom;
          if (checkpoints.length > 10 && i % Math.ceil(checkpoints.length / 5) !== 0 && i !== checkpoints.length - 1) return null;
          return (
            <g key={c.epoch}>
              <line x1={x} x2={x} y1={y} y2={y + 5} stroke="#cbd5e1" strokeWidth="1" />
              <text x={x} y={y + 16} textAnchor="middle" fontSize="10" fill="#9ca3af">{c.epoch}</text>
            </g>
          );
        })}
        <polyline fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={linePoints} />
      </svg>
    </div>
  );
}

function MetricPlotLoss({ checkpoints, color }) {
  const epochs = checkpoints.map(c => c.epoch);
  const xMin = Math.min(...epochs);
  const xMax = Math.max(...epochs);
  const xRange = Math.max(1, xMax - xMin);

  const valData = checkpoints.filter(c => c.val_loss != null && !isNaN(c.val_loss));
  const trainData = checkpoints.filter(c => c.train_loss != null && !isNaN(c.train_loss));

  const hasTrain = trainData.length > 0;
  if (!valData.length) return null;

  const width = 300;
  const height = 150;
  const margin = { top: 10, right: 10, bottom: 35, left: 45 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const allVals = [...valData.map(d => parseFloat(d.val_loss)), ...trainData.map(d => parseFloat(d.train_loss))];
  const max = Math.max(...allVals);
  const min = Math.min(...allVals);
  const padding = (max - min) * 0.1 || 0.01;
  const yMin = min - padding;
  const yMax = max + padding;

  const scaleX = epoch => margin.left + ((epoch - xMin) / xRange) * innerWidth;
  const scaleY = v => margin.top + innerHeight - ((v - yMin) / (yMax - yMin)) * innerHeight;

  const valPoints = valData.map(d => `${scaleX(d.epoch)},${scaleY(parseFloat(d.val_loss))}`).join(" ");
  const trainPoints = trainData.map(d => `${scaleX(d.epoch)},${scaleY(parseFloat(d.train_loss))}`).join(" ");

  return (
    <div className="plot-card">
      <div className="plot-header">
        <span className="plot-title">Loss</span>
        <div className="plot-legend">
          {hasTrain && (
            <div className="legend-row">
              <span
                className="legend-dot"
                style={{ backgroundColor: "#f97316" }}
              />
              <span className="legend-label">Train</span>
              <span
                className="plot-value train-number"
                style={{ color: "#f97316" }}
              >
                {parseFloat(trainData[trainData.length - 1].train_loss).toFixed(3)}
              </span>
            </div>
          )}

          <div className="legend-row">
            <span
              className="legend-dot"
              style={{ backgroundColor: color }}
            />
            <span className="legend-label">Val</span>
            <span
              className="plot-value val-number"
              style={{ color }}
            >
              {parseFloat(valData[valData.length - 1].val_loss).toFixed(3)}
            </span>
          </div>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart">
         <text x={margin.left + innerWidth / 2} y={height - 4} textAnchor="middle" fontSize="10" fill="#9ca3af" fontWeight="600" letterSpacing="0.05em" textTransform="uppercase">
          Epochs
        </text>
        {[0, 1, 2, 3].map(i => {
          const valTick = yMin + ((yMax - yMin) / 3) * i;
          return (
            <g key={i}>
              <line x1={margin.left} x2={width - margin.right} y1={scaleY(valTick)} y2={scaleY(valTick)} stroke="#f1f5f9" strokeWidth="1.5" />
              <text x={margin.left - 8} y={scaleY(valTick) + 3} textAnchor="end" fontSize="10" fill="#9ca3af">{valTick.toFixed(2)}</text>
            </g>
          );
        })}
        {checkpoints.map((c, i) => {
          const x = scaleX(c.epoch);
          const y = height - margin.bottom;
          if (checkpoints.length > 10 && i % Math.ceil(checkpoints.length / 5) !== 0 && i !== checkpoints.length - 1) return null;
          return (
            <g key={c.epoch}>
              <line x1={x} x2={x} y1={y} y2={y + 5} stroke="#cbd5e1" strokeWidth="1" />
              <text x={x} y={y + 16} textAnchor="middle" fontSize="10" fill="#9ca3af">{c.epoch}</text>
            </g>
          );
        })}
        {hasTrain && <polyline fill="none" stroke="#f97316" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={trainPoints} />}
        <polyline fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={valPoints} />
      </svg>
    </div>
  );
}


export default function BenchmarkPanel() {
const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [task, setTask] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [mode, setMode] = useState("test");
  const [batchSize, setBatchSize] = useState("8");
  const [maxEpochs, setMaxEpochs] = useState("10");
  const [learningRate, setLearningRate] = useState("1e-4");
  const [numWorkers, setNumWorkers] = useState("4");
  const [checkpointPath, setCheckpointPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [selectedJobs, setSelectedJobs] = useState(new Set());
  const [detailedMetrics, setDetailedMetrics] = useState({});
  const [lastRefresh, setLastRefresh] = useState(null);
  const [activeTab] = useState("run");
  const pollingRef = useRef(null);
  const availableTasks = useMemo(() => {
    const tasks = new Set(datasets.map(d => d.task).filter(Boolean));
    return Array.from(tasks).sort();
  }, [datasets]);
  const filteredDatasets = useMemo(() => {
    if (!task) return datasets;
    return datasets.filter(d => d.task === task);
  }, [datasets, task]);

  useEffect(() => {
    if (filteredDatasets.length > 0) {
      const currentStillValid = filteredDatasets.some(d => d.name === selectedDataset);
      if (!currentStillValid) {
        setSelectedDataset(filteredDatasets[0].name);
      }
    } else {
      setSelectedDataset("");
    }
  }, [task, filteredDatasets, selectedDataset]);

  const selectedRuns = jobs.filter(j => selectedJobs.has(j.job_id));

  const colorMap = {};
  jobs.forEach((j, i) => {
    colorMap[j.job_id] = BAR_COLORS[i % BAR_COLORS.length];
  });

  useEffect(() => {
    function handleBenchmarkEvent(e) {
      const modelsFromChat = e.detail || [];

      setSelectedModels(prev => {
        const prevSet = new Set(prev);
        const chatSet = new Set(modelsFromChat);

        const next = new Set([...prevSet].filter(m => chatSet.has(m)));

        modelsFromChat.forEach(m => next.add(m));

        return Array.from(next);
      });
    }

    window.addEventListener("run-benchmark", handleBenchmarkEvent);

    return () => {
      window.removeEventListener("run-benchmark", handleBenchmarkEvent);
    };
  }, []);

  useEffect(() => {
    fetch(`${API}/models`).then(r => r.json()).then(data => {
      setModels(data);
    }).catch(console.error);

    fetch(`${API}/datasets`).then(r => r.json()).then(data => {
      setDatasets(data);
      if (data.length > 0) setSelectedDataset(data[0].name || "");
    }).catch(console.error);

    fetchJobs();
  }, []);

  useEffect(() => {
    const hasActive = jobs.some(j => j.status === "queued" || j.status === "running");
    if (hasActive) {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(fetchJobs, 5000);
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [jobs]);

  const fetchJobs = useCallback(() => {
    fetch(`${API}/jobs`).then(r => r.json()).then(data => {
      setJobs(data);
      setLastRefresh(new Date());
    }).catch(console.error);
  }, []);

  const datasetInfo = datasets.find(d => d.name === selectedDataset);

  function handleAddModel(e) {
    const val = e.target.value;
    if (val && !selectedModels.includes(val)) {
      setSelectedModels([...selectedModels, val]);
    }
    e.target.value = "";
  }

  function removeModel(modelToRemove) {
    setSelectedModels(selectedModels.filter(m => m !== modelToRemove));
  }

  function toggleJob(jobId) {
    setSelectedJobs(prev => {
      const next = new Set(prev)
      const selecting = !next.has(jobId)

      if (selecting) next.add(jobId)
      else next.delete(jobId)

      if (selecting && !detailedMetrics[jobId]) {
        fetch(`${API}/${jobId}/metrics`)
          .then(r => r.json())
          .then(data => {
            setDetailedMetrics(current => ({
              ...current,
              [jobId]: data
            }))
          })
          .catch(err => console.error("Failed to fetch job details:", err))
      }
      return next
    })
  }


  async function deployCheckpoint(job) {
    try {
      const dataset = datasets.find(d => d.name === job.dataset)
      const taskType = dataset?.task || "segmentation"

      const res = await fetch(
        `http://localhost:8350/deploy/${job.job_id}/${taskType}`,
        { method: "POST" }
      )
      if (!res.ok) throw new Error(await res.text())
      window.dispatchEvent(new Event("deployment-changed"));
    } catch (err) {
      console.error("Deployment failed:", err)
      alert("Deployment failed. Check console.");
    }
  }

  function handleFineTune(job) {
    setSelectedModels([job.model_variant])
    setSelectedDataset(job.dataset)
    setMode("finetune")
    setBatchSize("8")
    setMaxEpochs("40")
    setLearningRate("1e-5")
    setNumWorkers("4")

    const el = document.getElementById("benchmark-panel")
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  async function handleSubmit() {
    if (selectedModels.length === 0) {
      alert("Please add at least one model to the queue.");
      return;
    }

    setSubmitting(true);

    for (const modelVariant of selectedModels) {
      try {
        const body = {
          model_variant: modelVariant,
          dataset: selectedDataset,
          mode,
          batch_size: parseInt(batchSize, 10),
          max_epochs: parseInt(maxEpochs, 10),
          learning_rate: parseFloat(learningRate),
          num_workers: parseInt(numWorkers, 10),
        };
        if (checkpointPath.trim()) body.checkpoint_path = checkpointPath.trim();

        const res = await fetch(`${API}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`Server responded with ${res.status} for ${modelVariant}`);
      } catch (err) {
        console.error(`Failed to submit benchmark for ${modelVariant}:`, err);
      }
    }

    await fetchJobs();
    setSubmitting(false);
  }

  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, []);


  return (
    <div id="benchmark-panel">
      <div className="benchmark-header">
        <h2 className="panel-title">Benchmark</h2>
        <div className="benchmark-tabs">
        </div>
      </div>
      {activeTab === "run" && <div className="benchmark-layout-v2">

        <div className="top-config">

          {/* LEFT: Models Queue & Dataset */}
          <div className="config-left">

            <div className="form-group">
              <label>Add Models to Queue</label>
              <select onChange={handleAddModel} value="">
                <option value="" disabled>-- Select a model to add --</option>
                {models.map(m => {
                  const val = m.variant || m.name;
                  return (
                    <option key={val} value={val} disabled={selectedModels.includes(val)}>
                      {m.display_name || m.name || m.variant} {selectedModels.includes(val) ? "(Added)" : ""}
                    </option>
                  )
                })}
              </select>
            </div>

            {/* Render the Queue */}
            <div className="model-queue">
              {selectedModels.length === 0 && (
                <div className="queue-empty">No models selected</div>
              )}
              {selectedModels.map(val => {
                const info = models.find(m => (m.variant || m.name) === val);
                const displayName = info?.display_name || info?.name || val;
                return (
                  <div key={val} className="queue-item">
                    <div className="queue-item-info">
                      <span className="queue-item-name">{displayName}</span>
                      {info?.num_params && <span className="queue-item-meta">{info.num_params}</span>}
                    </div>
                    <button className="queue-remove-btn" onClick={() => removeModel(val)}>✕</button>
                  </div>
                )
              })}
            </div>

            {/* TASK SELECTOR */}
            <div className="form-group" style={{ marginTop: "auto" }}>
              <label>Task Filter</label>
              <select value={task} onChange={e => setTask(e.target.value)}>
                <option value="">All Tasks</option>
                {availableTasks.map(t => (
                  <option key={t} value={t}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* DATASET SELECTOR */}
            <div className="form-group">
              <label>Dataset</label>
              <select value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}>
                {filteredDatasets.length === 0 ? (
                  <option value="" disabled>No datasets found</option>
                ) : (
                  filteredDatasets.map(d => (
                    <option key={d.name} value={d.name}>{d.display_name || d.name}</option>
                  ))
                )}
              </select>

              {/* Dataset Metadata Badges */}
              {datasetInfo && (
                <div className="dataset-tags">
                  {datasetInfo.task && <span className="dataset-tag">{datasetInfo.task}</span>}
                  {datasetInfo.size && <span className="dataset-tag">{datasetInfo.size} samples</span>}
                  {datasetInfo.modality && <span className="dataset-tag">{datasetInfo.modality}</span>}
                  {datasetInfo.resolution && <span className="dataset-tag">{datasetInfo.resolution}</span>}
                </div>
              )}
            </div>

          </div>

          {/* RIGHT: Advanced Settings & Run */}
          <div className="config-right">
             <div className="form-group">
                <label>Mode</label>
                <div className="mode-toggle">
                  <button
                    type="button"
                    className={mode === "linear_probe" ? "active" : ""}
                    onClick={() => setMode("linear_probe")}
                  >
                    Linear Probe
                  </button>

                  <button
                    type="button"
                    className={mode === "finetune" ? "active" : ""}
                    onClick={() => setMode("finetune")}
                  >
                    Fine-Tune
                  </button>
                </div>
              </div>
              <div className="advanced-row">
                <div className="form-group">
                  <label>Batch Size</label>
                  <input type="number" value={batchSize} onChange={e => setBatchSize(e.target.value)} />
                </div>
                <div className="form-group">
                  <label>Max Epochs</label>
                  <input type="number" value={maxEpochs} onChange={e => setMaxEpochs(e.target.value)} />
                </div>
              </div>
              <div className="advanced-row">
                <div className="form-group">
                  <label>Learning Rate</label>
                  <input type="text" value={learningRate} onChange={e => setLearningRate(e.target.value)} />
                </div>
                <div className="form-group">
                  <label>Num Workers</label>
                  <input type="number" value={numWorkers} onChange={e => setNumWorkers(e.target.value)} />
                </div>
              </div>
              <div className="form-group">
                <label>Checkpoint Path (optional)</label>
                <input type="text" value={checkpointPath} onChange={e => setCheckpointPath(e.target.value)} placeholder="/path/to/checkpoint.ckpt" />
              </div>

            <button
              className="run-btn"
              onClick={handleSubmit}
              disabled={submitting || selectedModels.length === 0 || !selectedDataset}
              style={{ marginTop: "auto" }}
            >
              {submitting ? "Submitting Queue..." : `▶ Run ${selectedModels.length} Job${selectedModels.length !== 1 ? 's' : ''}`}
            </button>
          </div>
        </div>

        {/* JOB TABLE */}
        <div className="results-section section-divider">
          <div className="section-header">
            <h3>Jobs</h3>
            <div className="refresh-info">
              {lastRefresh && `Updated ${timeAgo(lastRefresh)}`}
              <button className="refresh-btn" onClick={fetchJobs}>&#8635;</button>
            </div>
          </div>

          {jobs.length === 0 ? (
            <div className="bm-empty-state">No benchmark jobs yet. Configure and run your first benchmark.</div>
          ) : (
            <>
              <table className="job-table">
                <thead>
                  <tr>
                    <th></th>
                    <th>Status</th>
                    <th>Mode</th>
                    <th>Model</th>
                    <th>Dataset</th>
                    <th>mIoU</th>
                    <th>Accuracy</th>
                    <th>Duration</th>
                    <th>Action</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(j => {
                    const miou = j.miou ?? j.metrics?.miou ?? j.metrics?.val_miou;
                    const acc = j.accuracy ?? j.metrics?.accuracy ?? j.metrics?.val_accuracy;
                    const loss = j.loss ?? j.metrics?.loss ?? j.metrics?.val_loss;

                    const isCancellable = ["queued", "pending", "running"].includes(j.status);
                    const isDeletable = ["completed", "failed", "cancelled"].includes(j.status);

                    return (
                      <tr
                        key={j.job_id}
                        className={selectedJobs.has(j.job_id) ? "selected" : ""}
                        onClick={() => toggleJob(j.job_id)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            checked={selectedJobs.has(j.job_id)}
                            onClick={(e) => e.stopPropagation()}
                            onChange={() => toggleJob(j.job_id)}
                          />
                        </td>
                        <td>
                          <StatusBadge status={j.status} />
                        </td>
                        <td><ModeBadge mode={j.mode} /></td>
                        <td>{j.model_variant}</td>
                        <td>{j.dataset}</td>
                        <td>{miou != null ? miou.toFixed(3) : "—"}</td>
                        <td>{acc != null ? acc.toFixed(3) : "—"}</td>
                        <td>{formatDuration(j.duration_seconds)}</td>

                        {/* COLUMN 1: Positive Actions */}
                        <td className="action-cell">
                          {j.has_checkpoint && (
                            <button
                              className="action-mini-btn primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                deployCheckpoint(j);
                              }}
                            >
                              Deploy
                            </button>
                          )}
                          {(j.mode === "test" || j.mode === "linear_probe") && (
                            <button
                              className="action-mini-btn primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleFineTune(j);
                              }}
                            >
                              Fine-Tune
                            </button>
                          )}
                        </td>

                        {/* COLUMN 2: Destructive Action (Cancel / Delete) */}
                        <td className="action-cell" style={{ textAlign: "right", width: "1%", whiteSpace: "nowrap" }}>
                          {isCancellable && (
                            <button
                              className="action-mini-btn secondary cancel-btn icon-btn"
                              title="Cancel Job"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (window.confirm("Are you sure you want to cancel this job?")) {
                                  fetch(`${API}/jobs/${j.job_id}/cancel`, { method: "POST" })
                                    .then(() => fetchJobs());
                                }
                              }}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                              </svg>
                            </button>
                          )}

                          {isDeletable && (
                            <button
                              className="action-mini-btn secondary delete-btn icon-btn"
                              title="Delete Job"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (window.confirm("Permanently delete this job and its files?")) {
                                  fetch(`${API}/jobs/${j.job_id}`, { method: "DELETE" })
                                    .then(() => fetchJobs());
                                }
                              }}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                              </svg>
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>

              {/* CHARTS CONTAINER */}
              {selectedRuns.length > 0 && (
                <div className="charts-and-compare">

                  <div className="charts-left">
                    {selectedRuns.map(run => {

                      if (run.status === "failed") {
                        return (
                          <div key={run.job_id} className="model-chart-row">
                            <h4 className="model-chart-title">
                              {run.model_variant}
                              <span className="model-chart-subtitle">({run.dataset}) - <span style={{color: "#dc2626"}}>Failed</span></span>
                            </h4>
                            <div className="error-box">
                              <strong>Error</strong>
                              <pre className="error-text">
                                {run.error || "Unknown error occurred. Check server logs."}
                              </pre>
                            </div>
                          </div>
                        );
                      }

                      const checkpoints = detailedMetrics[run.job_id]?.checkpoints || [];
                      if (checkpoints.length === 0) return null;

                      return (
                        <div key={run.job_id} className="model-chart-row">
                          <h4 className="model-chart-title">
                            {run.model_variant}
                            <span className="model-chart-subtitle">({run.dataset})</span>
                          </h4>

                          <div className="metrics-plot-grid">
                            <MetricPlotLoss
                              checkpoints={checkpoints}
                              color={colorMap[run.job_id]}
                            />

                            {checkpoints[0]?.val_miou !== undefined && (
                              <MetricPlotSingle
                                checkpoints={checkpoints}
                                metricKey="val_miou"
                                title="Val mIoU"
                                color={colorMap[run.job_id]}
                              />
                            )}

                            {checkpoints[0]?.val_accuracy !== undefined && (
                              <MetricPlotSingle
                                checkpoints={checkpoints}
                                metricKey="val_accuracy"
                                title="Val Accuracy"
                                color={colorMap[run.job_id]}
                              />
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <div className="charts-right">

                    <MetricChart
                      metricKey="miou"
                      label="mIoU"
                      runs={selectedRuns}
                      colorMap={colorMap}
                    />

                    <MetricChart
                      metricKey="accuracy"
                      label="Accuracy"
                      runs={selectedRuns}
                      colorMap={colorMap}
                    />

                    <MetricChart
                      metricKey="loss"
                      label="Loss"
                      runs={selectedRuns}
                      colorMap={colorMap}
                    />

                  </div>

                </div>
              )}
            </>
          )}
        </div>
      </div>}
    </div>
  );
}