import "./Header.css";

export default function Header() {
  return (
    <header className="app-header">
    <div className="header-left">
      <h1>Earth Lake</h1>
      <span className="tagline">Earth Observation Foundation Model Management</span>
    </div>

    <nav className="nav-links">
      <a href="#chat-panel">REMSA Chat</a>
      <a href="#metadata-panel">Explore</a>
      <a href="#paper-panel">Contribute</a>
      <a href="#benchmark-panel">Benchmark</a>
      <a href="#inference-panel">Inference</a>
    </nav>

    <div className="header-right"></div>
  </header>
  );
}