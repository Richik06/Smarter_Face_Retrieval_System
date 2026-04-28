import { buildGalleryImageUrl } from "../services/cloudinary";

export default function ResultCard({ url, index, onDownload }) {
  const previewUrl = buildGalleryImageUrl(url);

  return (
    <article className="result-card">
      <img
        className="result-card__image"
        src={previewUrl}
        alt={`Matched result ${index + 1}`}
        loading="lazy"
      />
      <div className="result-card__footer">
        <div>
          <p className="result-card__title">Match {String(index + 1).padStart(2, "0")}</p>
          <p className="result-card__meta">Optimized preview</p>
        </div>
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => onDownload(url, index)}
        >
          Download
        </button>
      </div>
    </article>
  );
}
