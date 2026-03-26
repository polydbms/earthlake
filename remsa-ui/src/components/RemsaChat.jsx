import { useState, useRef, useEffect, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import "./RemsaChat.css";

export default function RemsaChat() {

  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [scores, setScores] = useState([]);
  const [selectedModels, setSelectedModels] = useState(new Set());
  const [selectedBenchmarks, setSelectedBenchmarks] = useState(new Set());

  const benchmarkEligible = useMemo(
    () => new Set(scores.filter(s => s.benchmark_eligible).map(s => s.model_id)),
    [scores]
  );

  const textareaRef = useRef(null);
  const chatWindowRef = useRef(null);
  const lastMessageRef = useRef(null);

  function toggleModel(modelId) {

    setSelectedModels(prev => {

      const next = new Set(prev);

      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }

      const selected = Array.from(next);

      window.dispatchEvent(
        new CustomEvent("explore-model", { detail: selected })
      );

      return next;
    });

  }

  function toggleBenchmark(modelId) {

    setSelectedBenchmarks(prev => {

      const next = new Set(prev);

      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }

      const selected = Array.from(next);

      window.dispatchEvent(
        new CustomEvent("run-benchmark", { detail: selected })
      );

      return next;

    });

  }

  useEffect(() => {

    if (!chatWindowRef.current || !lastMessageRef.current) return;

    const container = chatWindowRef.current;
    const target = lastMessageRef.current;

    const relativeTop = target.offsetTop - container.offsetTop;

    container.scrollTo({
      top: relativeTop,
      behavior: "smooth"
    });

  }, [messages]);

  async function sendQuery(e) {

    if (e) e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: "user", text: input };

    setMessages(m => [...m, userMsg]);
    setInput("");
    setScores([]);
    setLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {

      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.text })
      });

      const data = await res.json();

      const agentText =
        data.response ?? data.message ?? JSON.stringify(data, null, 2);

      setStatus(data.status);
      setScores(data.scores ?? []);

      setMessages(m => [...m, { role: "agent", text: agentText }]);

    } catch {

      setMessages(m => [...m, { role: "agent", text: "Connection error." }]);
      setStatus("error");

    }

    setLoading(false);

  }

  return (
    <div className="chat-ui">

      <div className="chat-window" ref={chatWindowRef}>

        {messages.length === 0 && (
          <div className="empty-state">
            <div className="example-queries">
              <div className="example">
                "I want a model for flood mapping using Sentinel-1 SAR data with 30m resolution"
              </div>
              <div className="example">
                "Recommend a foundation model for land cover classification"
              </div>
              <div className="example">
                "I need fast inference for urban expansion detection in near real-time on Jetson Nano using optical satellite imagery."
              </div>
            </div>
          </div>
        )}

        {messages.map((m, i) => {

          const isLast = i === messages.length - 1;

          return (
            <div
              key={i}
              className={`msg ${m.role}`}
              ref={isLast ? lastMessageRef : null}
            >

              {m.role === "user" ? (

                <div className="user-bubble">{m.text}</div>

              ) : (

                <div className="agent-message">

                  <ReactMarkdown
                    components={{
                      p: ({ node, ...props }) => {

                        if (status !== "done") {
                          return <p {...props} />;
                        }

                        const modelId = String(props.children[1]).trim();
                        const active = selectedModels.has(modelId);
                        const benchmarkActive = selectedBenchmarks.has(modelId);

                        return (
                          <div className="model-header">

                            <p {...props} />

                            <a
                              href="#metadata-panel"
                              className={`explore-btn ${active ? "active" : ""}`}
                              onClick={(e) => {
                                e.preventDefault();
                                toggleModel(modelId);
                              }}
                            >
                              🔎 Explore details
                            </a>

                            {true && (
                              <a
                                href="#"
                                className={`benchmark-btn ${benchmarkActive ? "active" : ""}`}
                                onClick={(e) => {
                                  e.preventDefault();
                                  toggleBenchmark(modelId);
                                }}
                              >
                                🚀 Run Benchmark
                              </a>
                            )}

                          </div>
                        );

                      }
                    }}
                  >
                    {m.text}
                  </ReactMarkdown>

                </div>

              )}

            </div>
          );

        })}

      </div>

      <form className="chat-input" onSubmit={sendQuery}>

        <textarea
          ref={textareaRef}
          className="chat-textarea"
          value={input}
          placeholder="Ask me about foundation models for your Earth observation task…"
          rows={1}
          onChange={e => {
            setInput(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${e.target.scrollHeight}px`;
          }}
        />

        <button type="submit" disabled={loading}>
          {loading ? "…" : "Send"}
        </button>

      </form>

    </div>
  );

}