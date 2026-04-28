import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import AppShell from "./components/AppShell";

const UploadPage = lazy(() => import("./pages/UploadPage"));
const QueryPage = lazy(() => import("./pages/QueryPage"));
const ResultsPage = lazy(() => import("./pages/ResultsPage"));

function RouteFallback() {
  return (
    <div className="page-card page-card--center">
      <div className="spinner" aria-hidden="true" />
      <p className="muted-text">Loading the workspace...</p>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/query" element={<QueryPage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </Suspense>
      </AppShell>
    </BrowserRouter>
  );
}
