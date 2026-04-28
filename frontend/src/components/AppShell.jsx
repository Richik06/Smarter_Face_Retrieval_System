import { NavLink, useLocation } from "react-router-dom";

const steps = [
  { path: "/upload", label: "Upload Batch", index: "01" },
  { path: "/query", label: "Query Face", index: "02" },
  { path: "/results", label: "Results", index: "03" }
];

export default function AppShell({ children }) {
  const location = useLocation();

  return (
    <div className="app-shell">
      <div className="app-shell__backdrop" aria-hidden="true" />
      <header className="topbar">
        <div>
          <p className="eyebrow">Face Retrieval System</p>
          <h1 className="topbar__title">Find every photo of one person from an event set.</h1>
        </div>
        <div className="status-chip">
          <span className="status-chip__dot" />
          <span>FastAPI + Cloudinary + React</span>
        </div>
      </header>

      <nav className="step-nav" aria-label="Workflow">
        {steps.map((step) => {
          const isActive = location.pathname === step.path;
          return (
            <NavLink
              key={step.path}
              to={step.path}
              className={`step-pill ${isActive ? "step-pill--active" : ""}`}
            >
              <span className="step-pill__index">{step.index}</span>
              <span>{step.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <main className="page-shell">{children}</main>
    </div>
  );
}
