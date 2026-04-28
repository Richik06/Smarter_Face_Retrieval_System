import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import FilePreviewGrid from "../components/FilePreviewGrid";
import ProgressList from "../components/ProgressList";
import StatusBanner from "../components/StatusBanner";
import UploadDropzone from "../components/UploadDropzone";
import { useSessionState } from "../hooks/useSessionState";
import { getCloudinarySignature, processEventUrls } from "../services/api";
import { uploadImageToCloudinary } from "../services/cloudinary";
import { buildFileId, slugifyEventId } from "../utils/files";

const initialSession = {
  eventName: "",
  eventId: "",
  uploadUrls: [],
  processSummary: null,
  queryUrl: "",
  searchResult: null
};

export default function UploadPage() {
  const navigate = useNavigate();
  const [session, setSession] = useSessionState("face-retrieval-session", initialSession);
  const [eventName, setEventName] = useState(session.eventName || "");
  const [eventId, setEventId] = useState(session.eventId || slugifyEventId(session.eventName || ""));
  const [items, setItems] = useState([]);
  const itemsRef = useRef(items);
  const [stage, setStage] = useState("idle");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  useEffect(() => {
    return () => {
      itemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    };
  }, []);

  const summary = session.processSummary;
  const overallProgress = useMemo(() => {
    if (!items.length) {
      return 0;
    }
    return Math.round(
      items.reduce((sum, item) => sum + item.progress, 0) / items.length
    );
  }, [items]);

  const progressItems = items.map((item) => ({
    ...item,
    statusLabel:
      item.status === "done"
        ? "Uploaded"
        : item.status === "error"
          ? "Failed"
          : item.status === "uploading"
            ? "Uploading..."
            : "Queued"
  }));

  function updateEventName(value) {
    const previousSuggested = slugifyEventId(eventName || "");
    const nextSuggested = slugifyEventId(value || "");
    setEventName(value);
    setEventId((current) => (!current || current === previousSuggested ? nextSuggested : current));
  }

  function addFiles(newFiles) {
    const nextItems = newFiles.map((file) => ({
      id: buildFileId(file),
      file,
      previewUrl: URL.createObjectURL(file),
      progress: 0,
      status: "pending",
      uploadedUrl: ""
    }));

    setItems((current) => {
      const known = new Set(current.map((item) => item.id));
      return [...current, ...nextItems.filter((item) => !known.has(item.id))];
    });
  }

  function removeFile(itemId) {
    setItems((current) => {
      const item = current.find((entry) => entry.id === itemId);
      if (item) {
        URL.revokeObjectURL(item.previewUrl);
      }
      return current.filter((entry) => entry.id !== itemId);
    });
  }

  async function handleUploadAndProcess() {
    const normalizedEventId = slugifyEventId(eventId || eventName);
    if (!normalizedEventId || !items.length) {
      return;
    }

    setError("");
    setInfo("");
    setStage("uploading");

    try {
      const signature = await getCloudinarySignature(normalizedEventId, "events");
      const uploadedUrls = [];

      for (const item of items) {
        setItems((current) =>
          current.map((entry) =>
            entry.id === item.id ? { ...entry, status: "uploading", progress: 5 } : entry
          )
        );

        const uploadResult = await uploadImageToCloudinary({
          file: item.file,
          signature,
          onProgress: (progress) => {
            setItems((current) =>
              current.map((entry) =>
                entry.id === item.id ? { ...entry, status: "uploading", progress } : entry
              )
            );
          }
        });

        uploadedUrls.push(uploadResult.secure_url);
        setItems((current) =>
          current.map((entry) =>
            entry.id === item.id
              ? {
                  ...entry,
                  status: "done",
                  progress: 100,
                  uploadedUrl: uploadResult.secure_url
                }
              : entry
          )
        );
      }

      setStage("processing");
      const processSummary = await processEventUrls({
        eventId: normalizedEventId,
        imageUrls: uploadedUrls
      });

      setSession({
        eventName,
        eventId: normalizedEventId,
        uploadUrls: uploadedUrls,
        processSummary,
        queryUrl: "",
        searchResult: null
      });
      setInfo("Batch uploaded and indexed successfully.");
      setStage("done");

      startTransition(() => {
        navigate("/query");
      });
    } catch (uploadError) {
      setStage("error");
      setItems((current) =>
        current.map((entry) =>
          entry.status === "uploading" ? { ...entry, status: "error" } : entry
        )
      );
      setError(uploadError.message || "Something went wrong during upload.");
    }
  }

  return (
    <div className="page-grid fade-in">
      <section className="page-card page-card--hero">
        <div className="card-heading">
          <div>
            <p className="eyebrow">Step 1</p>
            <h2>Upload an event batch</h2>
          </div>
          <span className="metric-chip">{items.length} selected</span>
        </div>

        <div className="field-grid">
          <label className="field">
            <span className="field__label">Event name</span>
            <input
              className="input"
              type="text"
              placeholder="Annual meetup, wedding, launch night..."
              value={eventName}
              onChange={(event) => updateEventName(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field__label">Event ID</span>
            <input
              className="input input--mono"
              type="text"
              value={eventId}
              onChange={(event) => setEventId(slugifyEventId(event.target.value))}
            />
          </label>
        </div>

        <UploadDropzone
          multiple
          disabled={stage === "uploading" || stage === "processing"}
          title="Drop event photos here"
          description="Upload group shots, event albums, or any image set you want indexed."
          onFilesSelected={addFiles}
        />

        <StatusBanner
          tone="error"
          title="Upload blocked"
          message={error}
        />
        <StatusBanner
          tone="success"
          title="Ready for query"
          message={info}
        />

        <div className="actions-row">
          <div>
            <p className="muted-text">
              {stage === "uploading"
                ? `Uploading to Cloudinary... ${overallProgress}%`
                : stage === "processing"
                  ? "Processing embeddings and clusters..."
                  : "Your images will upload first, then the backend will index them."}
            </p>
          </div>
          <button
            type="button"
            className="button button--primary"
            disabled={!items.length || !eventId || stage === "uploading" || stage === "processing"}
            onClick={handleUploadAndProcess}
          >
            {stage === "uploading"
              ? "Uploading..."
              : stage === "processing"
                ? "Processing..."
                : "Upload & Index Event"}
          </button>
        </div>
      </section>

      <div className="stack">
        <FilePreviewGrid items={items} onRemove={removeFile} />
        <ProgressList items={progressItems} label="Upload progress" />

        {summary ? (
          <section className="panel">
            <div className="panel__header">
              <h3>Latest indexed event</h3>
              <span className="metric-chip">{summary.num_people_detected} people</span>
            </div>
            <p className="muted-text">
              Event <span className="inline-code">{summary.event_id}</span> is ready for querying.
            </p>
          </section>
        ) : null}
      </div>
    </div>
  );
}
