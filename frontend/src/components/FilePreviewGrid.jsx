import { formatBytes } from "../utils/files";

export default function FilePreviewGrid({ items, onRemove }) {
  if (!items.length) {
    return null;
  }

  return (
    <div className="preview-grid">
      {items.map((item) => (
        <article key={item.id} className="preview-card">
          <img
            className="preview-card__image"
            src={item.previewUrl}
            alt={item.file.name}
            loading="lazy"
          />
          <div className="preview-card__body">
            <div>
              <p className="preview-card__title">{item.file.name}</p>
              <p className="preview-card__meta">{formatBytes(item.file.size)}</p>
            </div>
            {onRemove ? (
              <button
                type="button"
                className="button button--ghost button--small"
                onClick={() => onRemove(item.id)}
              >
                Remove
              </button>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  );
}
