export default function ProgressList({ items, label = "Upload queue" }) {
  if (!items.length) {
    return null;
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h3>{label}</h3>
      </div>
      <div className="progress-list">
        {items.map((item) => (
          <div key={item.id} className="progress-row">
            <div className="progress-row__top">
              <span className="progress-row__name">{item.file.name}</span>
              <span className="progress-row__status">{item.statusLabel}</span>
            </div>
            <div className="progress-bar" aria-hidden="true">
              <span style={{ width: `${item.progress}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
