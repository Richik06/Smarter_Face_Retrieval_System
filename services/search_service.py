"""
SearchService
─────────────
Two-stage face retrieval:

  Stage 1 — FAISS (GPU or CPU):
    Searches cluster CENTROIDS for top-K candidates.
    Uses faiss_engine which auto-selects GPU/CPU based on config.

  Stage 2 — Exact NumPy refinement:
    For each top-K candidate, computes cosine similarity against every
    individual member embedding stored in embeddings.npy.
    Returns the cluster whose BEST MEMBER embedding is closest to the query.
    This is more accurate than centroid-only matching.

Why two stages?
───────────────
A cluster centroid can drift when a person appears in many group photos at
different angles. Stage 2 catches cases where the centroid moved away from
the query even though at least one member embedding is a great match.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings
from utils.faiss_engine import (
    faiss_available,
    build_index,
    save_index,
    load_index,
    search_index,
    get_faiss_info,
)
from utils.logger import setup_logger
from utils.storage import (
    load_embeddings,
    load_clusters,
    load_meta,
    faiss_index_path,
)

logger = setup_logger(__name__)

# Number of centroid candidates fetched in Stage 1
_TOP_K = 5


class SearchService:

    # ── Public search ─────────────────────────────────────────────────────────

    @classmethod
    def search(
        cls,
        query_embedding: np.ndarray,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        Full two-stage search.

        Returns
        -------
        {
          matched_cluster_id : int | None,
          similarity         : float,
          matched_images     : List[str],
          search_device      : str,   ← "gpu" | "cpu" | "numpy"
          message            : str,
        }
        """
        clusters = load_clusters(event_id)
        if clusters is None or not clusters.get("clusters"):
            return cls._no_match("No cluster data found for this event.")

        cluster_map: Dict[str, Any] = clusters["clusters"]
        if not cluster_map:
            return cls._no_match("Event has no clusters yet.")

        cluster_ids = list(cluster_map.keys())
        centroids = np.array(
            [cluster_map[cid]["centroid"] for cid in cluster_ids],
            dtype=np.float32,
        )

        # ── Stage 1: fast centroid search ────────────────────────────────────
        k = min(_TOP_K, len(cluster_ids))
        top_indices, top_scores, search_device = cls._stage1_search(
            query_embedding, centroids, event_id, k=k
        )

        if not top_indices:
            return cls._no_match("Search returned no candidates.")

        # Early exit if best centroid score is far below threshold
        if top_scores[0] < settings.SIMILARITY_THRESHOLD * 0.5:
            return cls._no_match(
                f"No match found (best centroid similarity "
                f"{top_scores[0]:.4f} well below threshold)."
            )

        # ── Stage 2: exact member-level refinement ────────────────────────────
        best_cluster_id, best_score = cls._stage2_refine(
            query_embedding, top_indices, cluster_ids, cluster_map, event_id
        )

        if best_score < settings.SIMILARITY_THRESHOLD:
            return cls._no_match(
                f"No match found (best member similarity {best_score:.4f} "
                f"< threshold {settings.SIMILARITY_THRESHOLD})."
            )

        matched = cluster_map[best_cluster_id]
        logger.info(
            "Matched cluster %s (similarity=%.4f, device=%s) in event '%s'.",
            best_cluster_id, best_score, search_device, event_id,
        )
        return {
            "matched_cluster_id": int(matched["cluster_id"]),
            "similarity": round(float(best_score), 6),
            "matched_images": matched["image_paths"],
            "search_device": search_device,
            "message": "Match found.",
        }

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    @classmethod
    def _stage1_search(
        cls,
        query: np.ndarray,
        centroids: np.ndarray,
        event_id: str,
        k: int = 5,
    ) -> Tuple[List[int], List[float], str]:
        """
        Returns (top_indices, top_scores, device_used).
        device_used is "gpu", "cpu", or "numpy".
        """
        if settings.USE_FAISS and faiss_available():
            indices, scores = cls._faiss_search(query, centroids, event_id, k=k)
            if indices:
                from utils.faiss_engine import get_device
                return indices, scores, get_device()

        # NumPy fallback
        indices, scores = cls._numpy_topk(query, centroids, k=k)
        return indices, scores, "numpy"

    @classmethod
    def _faiss_search(
        cls,
        query: np.ndarray,
        centroids: np.ndarray,
        event_id: str,
        k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """Load or rebuild the FAISS index and search."""
        idx_path = faiss_index_path(event_id)
        index = load_index(idx_path)

        # Rebuild if missing or stale (different number of clusters)
        if index is None or index.ntotal != len(centroids):
            logger.debug("Rebuilding FAISS index for event '%s'.", event_id)
            index = build_index(centroids)
            if index is None:
                return [], []
            save_index(index, idx_path)

        return search_index(index, query, k=k)

    @staticmethod
    def _numpy_topk(
        query: np.ndarray,
        centroids: np.ndarray,
        k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """Pure NumPy exact top-k (fallback when FAISS unavailable)."""
        scores = centroids @ query
        k = min(k, len(scores))
        top_idx = np.argsort(scores)[::-1][:k]
        return list(top_idx.tolist()), list(scores[top_idx].tolist())

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    @classmethod
    def _stage2_refine(
        cls,
        query: np.ndarray,
        candidate_indices: List[int],
        cluster_ids: List[str],
        cluster_map: Dict[str, Any],
        event_id: str,
    ) -> Tuple[str, float]:
        """
        For each candidate cluster compare query against every member embedding.
        Returns (best_cluster_id_str, best_cosine_similarity).
        Falls back to centroid similarity when embeddings.npy is unavailable.
        """
        all_embeddings = load_embeddings(event_id)
        all_meta = load_meta(event_id)

        best_cid = cluster_ids[candidate_indices[0]]
        best_score = -1.0

        for idx in candidate_indices:
            cid = cluster_ids[idx]
            cluster = cluster_map[cid]
            cluster_id_int = cluster["cluster_id"]

            if all_embeddings is not None and all_meta is not None:
                # Gather row indices belonging to this cluster
                member_rows = [
                    i for i, m in enumerate(all_meta)
                    if m.get("cluster_id") == cluster_id_int
                ]
                if member_rows:
                    member_embs = all_embeddings[member_rows]     # (k, dim)
                    score = float(np.max(member_embs @ query))    # best member sim
                else:
                    centroid = np.array(cluster["centroid"], dtype=np.float32)
                    score = float(np.dot(centroid, query))
            else:
                # No embeddings on disk — use centroid only
                centroid = np.array(cluster["centroid"], dtype=np.float32)
                score = float(np.dot(centroid, query))

            if score > best_score:
                best_score = score
                best_cid = cid

        return best_cid, best_score

    # ── Index management ──────────────────────────────────────────────────────

    @classmethod
    def build_and_save_faiss_index(
        cls, event_id: str, centroids: np.ndarray
    ) -> None:
        """
        Pre-build and persist FAISS index after event processing.
        Called automatically by EventService.
        """
        if not settings.USE_FAISS or not faiss_available():
            logger.info(
                "FAISS skipped for '%s' (USE_FAISS=%s, available=%s).",
                event_id, settings.USE_FAISS, faiss_available(),
            )
            return

        if len(centroids) == 0:
            logger.warning("No centroids to index for event '%s'.", event_id)
            return

        index = build_index(centroids)
        if index is not None:
            saved = save_index(index, faiss_index_path(event_id))
            if saved:
                logger.info(
                    "FAISS index saved for event '%s' (%d centroids).",
                    event_id, len(centroids),
                )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @classmethod
    def faiss_info(cls) -> dict:
        """Return FAISS device/version info for health/debug endpoints."""
        return get_faiss_info()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _no_match(message: str) -> Dict[str, Any]:
        return {
            "matched_cluster_id": None,
            "similarity": 0.0,
            "matched_images": [],
            "search_device": "none",
            "message": message,
        }