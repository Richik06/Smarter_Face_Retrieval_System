"""
SearchService
─────────────
Given a query embedding, finds the best-matching cluster in a stored event.

Two backends are supported:
  1. FAISS (fast approximate nearest-neighbour, default when USE_FAISS=True)
  2. NumPy brute-force cosine similarity (exact, always available)

Strategy
────────
• We compare the query against every cluster's *centroid*.
• If the best cosine similarity is below SIMILARITY_THRESHOLD, we return no
  match (the person is likely not present in the event).
• As a tie-breaking refinement we can also compute similarity against all
  individual member embeddings within the top-k candidates and pick the
  cluster whose *maximum member similarity* is highest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings
from utils.logger import setup_logger
from utils.storage import load_embeddings, load_clusters, faiss_index_path

logger = setup_logger(__name__)


# ── FAISS helper

def _build_faiss_index(vectors: np.ndarray):
    """Build an inner-product (= cosine when normalised) FAISS index."""
    try:
        import faiss  # optional dependency
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # Inner Product
        index.add(vectors.astype(np.float32))
        return index
    except ImportError:
        logger.warning("FAISS not installed; falling back to NumPy similarity search.")
        return None


def _save_faiss_index(index, path) -> None:
    try:
        import faiss
        faiss.write_index(index, str(path))
    except Exception as exc:
        logger.warning("Could not save FAISS index: %s", exc)


def _load_faiss_index(path):
    try:
        import faiss
        if path.exists():
            return faiss.read_index(str(path))
    except Exception as exc:
        logger.warning("Could not load FAISS index: %s", exc)
    return None


# ── Main service 

class SearchService:

    @classmethod
    def search(
        cls,
        query_embedding: np.ndarray,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        Find the best-matching cluster for *query_embedding* in *event_id*.

        Returns
        -------
        dict with keys:
          matched_cluster_id : int | None
          similarity          : float
          matched_images      : List[str]
          message             : str
        """
        clusters = load_clusters(event_id)
        if clusters is None or not clusters.get("clusters"):
            return cls._no_match("No cluster data found for this event.")

        cluster_map: Dict[str, Any] = clusters["clusters"]
        if not cluster_map:
            return cls._no_match("Event has no clusters yet.")

        # Build centroid matrix  (C, dim)
        cluster_ids = list(cluster_map.keys())
        centroids = np.array(
            [cluster_map[cid]["centroid"] for cid in cluster_ids],
            dtype=np.float32,
        )

        # ── Similarity search ────────────────────────────────────────────────
        best_idx, best_score = cls._find_best_match(
            query_embedding, centroids, event_id
        )

        if best_score < settings.SIMILARITY_THRESHOLD:
            logger.info(
                "Best similarity %.4f below threshold %.4f – no match.",
                best_score,
                settings.SIMILARITY_THRESHOLD,
            )
            return cls._no_match(
                f"No match found (best similarity {best_score:.4f} < "
                f"threshold {settings.SIMILARITY_THRESHOLD})."
            )

        matched_cid = cluster_ids[best_idx]
        matched_cluster = cluster_map[matched_cid]

        logger.info(
            "Matched cluster %s (similarity=%.4f) in event '%s'.",
            matched_cid,
            best_score,
            event_id,
        )
        return {
            "matched_cluster_id": int(matched_cluster["cluster_id"]),
            "similarity": round(float(best_score), 6),
            "matched_images": matched_cluster["image_paths"],
            "message": "Match found.",
        }

    # ── Similarity backends
    @classmethod
    def _find_best_match(
        cls,
        query: np.ndarray,
        centroids: np.ndarray,
        event_id: str,
    ) -> Tuple[int, float]:
        """Return (index_into_centroids, cosine_similarity)."""

        if settings.USE_FAISS:
            result = cls._faiss_search(query, centroids, event_id)
            if result is not None:
                return result

        return cls._numpy_search(query, centroids)

    @classmethod
    def _numpy_search(
        cls, query: np.ndarray, centroids: np.ndarray
    ) -> Tuple[int, float]:
        """Exact cosine similarity (dot product, since vectors are L2-normalised)."""
        scores = centroids @ query                          # (C,)
        best_idx = int(np.argmax(scores))
        return best_idx, float(scores[best_idx])

    @classmethod
    def _faiss_search(
        cls,
        query: np.ndarray,
        centroids: np.ndarray,
        event_id: str,
    ) -> Optional[Tuple[int, float]]:
        """
        Attempt a FAISS-accelerated search.
        Rebuilds or loads the index as needed.
        Returns None if FAISS is unavailable.
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            return None

        idx_path = faiss_index_path(event_id)
        index = _load_faiss_index(idx_path)

        if index is None or index.ntotal != len(centroids):
            logger.debug("Rebuilding FAISS index for event '%s'.", event_id)
            index = _build_faiss_index(centroids)
            if index is None:
                return None
            _save_faiss_index(index, idx_path)

        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q, 1)
        best_idx = int(indices[0][0])
        best_score = float(scores[0][0])
        return best_idx, best_score

    # ── Helpers 
    @staticmethod
    def _no_match(message: str) -> Dict[str, Any]:
        return {
            "matched_cluster_id": None,
            "similarity": 0.0,
            "matched_images": [],
            "message": message,
        }

    # ── FAISS index management 

    @classmethod
    def build_and_save_faiss_index(
        cls, event_id: str, centroids: np.ndarray
    ) -> None:
        """Pre-build the FAISS index after processing an event."""
        if not settings.USE_FAISS:
            return
        try:
            import faiss  # noqa: F401
        except ImportError:
            return

        index = _build_faiss_index(centroids)
        if index is not None:
            _save_faiss_index(index, faiss_index_path(event_id))
            logger.info("FAISS index built for event '%s' (%d vectors).", event_id, len(centroids))