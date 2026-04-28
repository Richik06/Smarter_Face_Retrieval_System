export default function StatusBanner({ tone = "info", title, message }) {
  if (!message) {
    return null;
  }

  return (
    <div className={`status-banner status-banner--${tone}`}>
      <div>
        {title ? <p className="status-banner__title">{title}</p> : null}
        <p className="status-banner__message">{message}</p>
      </div>
    </div>
  );
}
