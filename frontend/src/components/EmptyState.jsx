import { Link } from "react-router-dom";

export default function EmptyState({ title, message, actionLabel, actionTo }) {
  return (
    <div className="page-card page-card--center">
      <div className="empty-state__orb" aria-hidden="true" />
      <h2>{title}</h2>
      <p className="muted-text">{message}</p>
      {actionLabel && actionTo ? (
        <Link className="button button--primary" to={actionTo}>
          {actionLabel}
        </Link>
      ) : null}
    </div>
  );
}
