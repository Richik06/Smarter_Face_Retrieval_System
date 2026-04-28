import { startTransition, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import EmptyState from "../components/EmptyState";
import FilePreviewGrid from "../components/FilePreviewGrid";
import StatusBanner from "../components/StatusBanner";
import UploadDropzone from "../components/UploadDropzone";
import { useSessionState } from "../hooks/useSessionState";
import { getCloudinarySignature, searchFaceUrl } from "../services/api";
import { uploadImageToCloudinary } from "../services/cloudinary";
import { buildFileId } from "../utils/files";

const initialSession = {
  eventName: "",
  eventId: "",
  uploadUrls: [],
  processSummary: null,
  queryUrl: "",
  searchResult: null
};

export default function QueryPage() {
  const navigate = useNavigate();
  const [session, setSession] = useSessionState("face-retrieval-session", initialSession);
  const [item, setItem] = useState(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (item?.previewUrl) {
        URL.revokeObjectURL(item.previewUrl);
      }
    };
  }, [item]);

  if (!session.eventId || !session.uploadUrls?.length) {
    return (
      <EmptyState
        title="No indexed event yet"
        message="Upload and process an event batch first, then come back with a single face image to search."
        actionLabel="Go to upload"
        actionTo="/upload"
      />
    );
  }

  function handleFileSelection(files) {
    const [file] = files;
    if (!file) {
      return;
    }

    if (item?.previewUrl) {
      URL.revokeObjectURL(item.previewUrl);
    }

    setItem({
      id: buildFileId(file),
      file,
      previewUrl: URL.createObjectURL(file)
    });
    setProgress(0);
    setError("");
  }

  async function handleFindMatches() {
    if (!item) {
      return;
    }

    setStage("uploading");
    setError("");

    try {
      const signature = await getCloudinarySignature(session.eventId, "queries");
      const uploadResult = await uploadImageToCloudinary({
        file: item.file,
        signature,
        onProgress: setProgress
      });

      setStage("searching");
      const searchResult = await searchFaceUrl({
        eventId: session.eventId,
        imageUrl: uploadResult.secure_url
      });

      setSession({
        ...session,
        queryUrl: uploadResult.secure_url,
        searchResult
      });

      setStage("done");
      startTransition(() => {
        navigate("/results");
      });
    } catch (searchError) {
      setStage("error");
      setError(searchError.message || "Search failed.");
    }
  }

  return (
    <div className="page-grid fade-in">
      <section className="page-card page-card--hero">
        <div className="card-heading">
          <div>
            <p className="eyebrow">Step 2</p>
            <h2>Upload a single face image</h2>
          </div>
          <span className="metric-chip">{session.eventId}</span>
        </div>

        <p className="muted-text">
          Use a clear face photo for the best match quality. The backend will compare the face
          embedding against every clustered person in the indexed event.
        </p>

        <UploadDropzone
          title="Drop one query image"
          description="Portraits, selfies, cropped faces, or a sharp frame from the same event."
          onFilesSelected={handleFileSelection}
          disabled={stage === "uploading" || stage === "searching"}
        />

        <StatusBanner
          tone="error"
          title="Search issue"
          message={error}
        />

        <div className="actions-row">
          <p className="muted-text">
            {stage === "uploading"
              ? `Uploading query... ${progress}%`
              : stage === "searching"
                ? "Finding matches..."
                : "We will upload the query image, then search the existing event clusters."}
          </p>
          <button
            type="button"
            className="button button--primary"
            disabled={!item || stage === "uploading" || stage === "searching"}
            onClick={handleFindMatches}
          >
            {stage === "uploading"
              ? "Uploading..."
              : stage === "searching"
                ? "Finding..."
                : "Find Matches"}
          </button>
        </div>
      </section>

      <div className="stack">
        {item ? <FilePreviewGrid items={[item]} /> : null}

        <section className="panel">
          <div className="panel__header">
            <h3>Current event</h3>
          </div>
          <p className="muted-text">
            <strong>{session.eventName || session.eventId}</strong>
            <br />
            {session.processSummary?.num_people_detected ?? 0} people clustered from{" "}
            {session.uploadUrls.length} uploaded images.
          </p>
        </section>
      </div>
    </div>
  );
}
