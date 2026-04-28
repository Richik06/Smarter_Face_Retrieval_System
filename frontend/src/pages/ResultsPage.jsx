import { useDeferredValue, useState } from "react";
import { Link } from "react-router-dom";

import EmptyState from "../components/EmptyState";
import ResultCard from "../components/ResultCard";
import StatusBanner from "../components/StatusBanner";
import { useSessionState } from "../hooks/useSessionState";
import { downloadAllMatches } from "../services/api";
import { buildFullImageUrl } from "../services/cloudinary";
import { downloadUrlAsFile, saveBlob } from "../utils/download";
import { inferFileNameFromUrl } from "../utils/files";

const initialSession = {
  eventName: "",
  eventId: "",
  uploadUrls: [],
  processSummary: null,
  queryUrl: "",
  searchResult: null
};

export default function ResultsPage() {
  const [session] = useSessionState("face-retrieval-session", initialSession);
  const [downloadError, setDownloadError] = useState("");
  const [isDownloadingAll, setIsDownloadingAll] = useState(false);
  const deferredMatches = useDeferredValue(session.searchResult?.matched_images || []);

  if (!session.searchResult) {
    return (
      <EmptyState
        title="No search results yet"
        message="Run a query after indexing an event to see matched images here."
        actionLabel="Go to query"
        actionTo="/query"
      />
    );
  }

  const hasMatches = deferredMatches.length > 0;

  async function handleSingleDownload(url, index) {
    setDownloadError("");
    try {
      await downloadUrlAsFile(buildFullImageUrl(url), inferFileNameFromUrl(url, `match-${index + 1}`));
    } catch (error) {
      setDownloadError(error.message || "Could not download image.");
    }
  }

  async function handleDownloadAll() {
    if (!hasMatches) {
      return;
    }

    setIsDownloadingAll(true);
    setDownloadError("");

    try {
      const blob = await downloadAllMatches({
        eventId: session.eventId,
        imageUrls: deferredMatches
      });
      saveBlob(blob, `${session.eventId}-matches.zip`);
    } catch (error) {
      setDownloadError(error.message || "Could not download ZIP archive.");
    } finally {
      setIsDownloadingAll(false);
    }
  }

  return (
    <div className="stack fade-in">
      <section className="page-card page-card--hero">
        <div className="card-heading">
          <div>
            <p className="eyebrow">Step 3</p>
            <h2>{hasMatches ? "Matches found" : "No matches found"}</h2>
          </div>
          <span className="metric-chip">
            Similarity {session.searchResult.similarity.toFixed(3)}
          </span>
        </div>

        <StatusBanner
          tone={hasMatches ? "success" : "info"}
          title={hasMatches ? "Gallery ready" : "Search completed"}
          message={session.searchResult.message}
        />
        <StatusBanner tone="error" title="Download issue" message={downloadError} />

        <div className="summary-grid">
          <div className="summary-card">
            <span className="summary-card__label">Event</span>
            <strong>{session.eventName || session.eventId}</strong>
          </div>
          <div className="summary-card">
            <span className="summary-card__label">Cluster</span>
            <strong>
              {session.searchResult.matched_cluster_id ?? "No cluster"}
            </strong>
          </div>
          <div className="summary-card">
            <span className="summary-card__label">Images</span>
            <strong>{deferredMatches.length}</strong>
          </div>
        </div>

        <div className="actions-row">
          <div className="button-row">
            <Link className="button button--ghost" to="/query">
              Run another query
            </Link>
            <Link className="button button--ghost" to="/upload">
              Start a new event
            </Link>
          </div>
          <button
            type="button"
            className="button button--primary"
            disabled={!hasMatches || isDownloadingAll}
            onClick={handleDownloadAll}
          >
            {isDownloadingAll ? "Preparing ZIP..." : "Download All"}
          </button>
        </div>
      </section>

      {session.queryUrl ? (
        <section className="panel">
          <div className="panel__header">
            <h3>Query reference</h3>
          </div>
          <div className="query-preview">
            <img src={buildFullImageUrl(session.queryUrl)} alt="Query face" />
          </div>
        </section>
      ) : null}

      {hasMatches ? (
        <section className="result-grid">
          {deferredMatches.map((url, index) => (
            <ResultCard
              key={`${url}-${index}`}
              url={url}
              index={index}
              onDownload={handleSingleDownload}
            />
          ))}
        </section>
      ) : (
        <section className="page-card page-card--center">
          <div className="empty-state__orb" aria-hidden="true" />
          <h3>No gallery to show</h3>
          <p className="muted-text">
            Try a clearer face crop or index more event images to improve recall.
          </p>
        </section>
      )}
    </div>
  );
}
