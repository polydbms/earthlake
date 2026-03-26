import './App.css'

import RemsaChat from "./components/RemsaChat.jsx"
import MetadataExplorer from "./components/MetadataExplorer.jsx";
import PaperExtractor from "./components/PaperExtractor.jsx"
import BenchmarkPanel from "./components/BenchmarkPanel.jsx"
import Header from "./components/Header.jsx"
import InferencePanel from "./components/Inference.jsx";

function App() {
  return (
    <>
      <Header></Header>
      <div className="layout-container">
        <div id="chat-panel" className="panel">
          <RemsaChat ></RemsaChat>
        </div>
        <div id="metadata-panel" className="panel">
          <MetadataExplorer ></MetadataExplorer>
        </div>
        <div id="paper-panel" className="panel">
          <PaperExtractor ></PaperExtractor>
        </div>
        <div id="benchmark-panel" className="panel">
          <BenchmarkPanel />
        </div>
        <div id="inference-panel" className="panel">
          <InferencePanel />
        </div>
      </div>
    </>
)
}

export default App