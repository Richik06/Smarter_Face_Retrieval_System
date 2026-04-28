"""
ClusteringService
─────────────────
Groups face embeddings into per-person clusters using DBSCAN.

Why DBSCAN?
• No need to specify the number of people in advance.
• Naturally handles noise (cluster_id == -1).
• Works well with cosine distance when embeddings are L2-normalised.

Cluster manifest schema (stored in clusters.json)
──────────────────────────────────────────────────
{
  "event_id": "...",
  "num_clusters": N,
  "clusters": {
    "<cluster_id>": {
      "cluster_id": int,
      "centroid": [float, ...],          ← mean of all member embeddings
      "num_faces": int,
      "num_images": int,
      "image_paths": [str, ...]          ← deduplicated
    }
  }
}
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ClusteringService:
    """Stateless clustering helpers."""

    # ── Core clustering

    @classmethod
    def cluster_embeddings(
        cls,
        embeddings: np.ndarray,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run DBSCAN on *embeddings* (shape: N × dim, L2-normalised float32).

        Returns
        -------
        labels : np.ndarray  shape (N,)  int
            Cluster label per embedding.  -1 means noise.
        """
        if len(embeddings) == 0:
            return np.array([], dtype=int)

        _eps = eps if eps is not None else settings.DBSCAN_EPS
        _min_samples = min_samples if min_samples is not None else settings.DBSCAN_MIN_SAMPLES

        logger.info(
            "Running DBSCAN on %d embeddings  (eps=%.3f, min_samples=%d, metric=%s).",
            len(embeddings),
            _eps,
            _min_samples,
            settings.DBSCAN_METRIC,
        )

        db = DBSCAN(
            eps=_eps,
            min_samples=_min_samples,
            metric=settings.DBSCAN_METRIC,
            algorithm="brute",   # required for cosine metric
            n_jobs=-1,
        )
        labels: np.ndarray = db.fit_predict(embeddings)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        logger.info(
            "DBSCAN result: %d cluster(s), %d noise point(s).",
            n_clusters,
            n_noise,
        )
        return labels

    # ── Manifest building 
    @classmethod
    def build_cluster_manifest(
        cls,
        event_id: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        meta: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble the full cluster manifest from embeddings, DBSCAN labels,
        and per-embedding metadata.

        Parameters
        ----------
        event_id  : str
        embeddings: np.ndarray  (N, dim)
        labels    : np.ndarray  (N,)  – DBSCAN output
        meta      : list of dicts with at least "image_path" per row

        Returns
        -------
        manifest dict (matches the clusters.json schema above)
        """
        # Group row indices by cluster label
        cluster_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_indices[int(label)].append(idx)

        clusters_out: Dict[str, Any] = {}

        for label, indices in cluster_indices.items():
            if label == -1:
                # Noise points: assign each as its own singleton cluster
                # so no image is ever lost.
                for solo_idx in indices:
                    solo_label = cls._next_noise_id(clusters_out)
                    clusters_out[str(solo_label)] = cls._make_cluster_entry(
                        cluster_id=solo_label,
                        indices=[solo_idx],
                        embeddings=embeddings,
                        meta=meta,
                    )
            else:
                clusters_out[str(label)] = cls._make_cluster_entry(
                    cluster_id=label,
                    indices=indices,
                    embeddings=embeddings,
                    meta=meta,
                )

        manifest = {
            "event_id": event_id,
            "num_clusters": len(clusters_out),
            "clusters": clusters_out,
        }
        logger.info(
            "Built manifest for event '%s': %d clusters.",
            event_id,
            len(clusters_out),
        )
        return manifest

    # ── Helpers

    @staticmethod
    def _make_cluster_entry(
        cluster_id: int,
        indices: List[int],
        embeddings: np.ndarray,
        meta: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cluster_embeddings = embeddings[indices]                        # (k, dim)
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)       # L2-norm

        image_paths = list(
            dict.fromkeys(meta[i]["image_path"] for i in indices)      # deduplicated, order-preserving
        )
        return {
            "cluster_id": cluster_id,
            "centroid": centroid.tolist(),
            "num_faces": len(indices),
            "num_images": len(image_paths),
            "image_paths": image_paths,
        }

    @staticmethod
    def _next_noise_id(existing: Dict[str, Any]) -> int:
        """Generate a new negative-style ID for noise-point singleton clusters."""
        existing_ids = [int(k) for k in existing]
        # Start at 10000 to avoid colliding with DBSCAN labels
        base = 10000
        candidate = base
        while candidate in existing_ids:
            candidate += 1
        return candidate

    # ── Re-clustering 

    @classmethod
    def recluster(
        cls,
        event_id: str,
        embeddings: np.ndarray,
        meta: List[Dict[str, Any]],
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Re-run clustering from scratch (e.g. after new images are added).
        Returns the updated manifest.
        """
        labels = cls.cluster_embeddings(embeddings, eps=eps, min_samples=min_samples)
        # Update cluster_id in meta in-place
        for i, label in enumerate(labels):
            meta[i]["cluster_id"] = int(label)
        return cls.build_cluster_manifest(event_id, embeddings, labels, meta)