"""
EventService
────────────
Orchestrates the full event-processing pipeline:

  1. Accept uploaded images
  2. Detect faces & generate embeddings  (EmbeddingService)
  3. Persist embeddings + metadata        (storage utils)
  4. Run DBSCAN clustering               (ClusteringService)
  5. Build and persist cluster manifest   (storage utils)
  6. Pre-build FAISS index               (SearchService)

Also exposes re-clustering for when new images are added to an existing event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import UploadFile

from config import settings
from services.clustering_service import ClusteringService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from utils.image_utils import decode_upload, save_image, is_valid_image
from utils.logger import setup_logger
from utils.storage import (
    ensure_event_dirs,
    images_dir,
    load_embeddings,
    load_meta,
    save_embeddings,
    save_clusters,
    save_meta,
    load_clusters,
)

logger = setup_logger(__name__)


class EventService:

    # ── Main pipeline 
    @classmethod
    def process_event(
        cls,
        event_id: str,
        images: List[UploadFile],
        recluster_eps: float | None = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: images → embeddings → clusters → manifest.

        Parameters
        ----------
        event_id        : unique event identifier
        images          : list of uploaded image files
        recluster_eps   : optional override for DBSCAN eps (for re-clustering)

        Returns
        -------
        Response payload dict
        """
        ensure_event_dirs(event_id)
        img_dir = images_dir(event_id)

        # ── Check for existing data (incremental add) 
        existing_embeddings = load_embeddings(event_id)
        existing_meta = load_meta(event_id)

        all_embeddings: List[np.ndarray] = (
            list(existing_embeddings) if existing_embeddings is not None else []
        )
        all_meta: List[Dict[str, Any]] = existing_meta if existing_meta is not None else []

        new_faces_total = 0
        images_with_no_face: List[str] = []

        # ── Process each uploaded image
        for upload in images:
            original_filename = upload.filename or "unknown.jpg"
            logger.info("Processing image: %s", original_filename)

            try:
                img = decode_upload(upload)
            except Exception as exc:
                logger.warning("Could not decode '%s': %s", original_filename, exc)
                continue

            if not is_valid_image(img):
                logger.warning("Invalid image skipped: %s", original_filename)
                continue

            # Save the source image
            saved_path = save_image(img, img_dir, filename=original_filename)
            rel_path = str(saved_path)   # store absolute path for portability

            # Detect faces & embed
            detections = EmbeddingService.get_embeddings_from_image(img)

            if not detections:
                logger.info("No face detected in '%s'.", original_filename)
                images_with_no_face.append(original_filename)
                continue

            for face_idx, (embedding, facial_area) in enumerate(detections):
                # Duplicate-detection: skip if this embedding is extremely close
                # to one already stored (same image uploaded twice).
                if cls._is_duplicate(embedding, all_embeddings):
                    logger.debug("Duplicate face skipped in '%s'.", original_filename)
                    continue

                all_embeddings.append(embedding)
                all_meta.append(
                    {
                        "image_path": rel_path,
                        "face_index": face_idx,
                        "facial_area": facial_area,
                        "cluster_id": None,       # filled in after clustering
                    }
                )
                new_faces_total += 1

        if not all_embeddings:
            logger.warning("No faces found across all images for event '%s'.", event_id)
            return {
                "event_id": event_id,
                "num_people_detected": 0,
                "clusters": [],
                "warning": "No faces detected in any of the provided images.",
                "images_with_no_face": images_with_no_face,
            }

        # ── Persist embeddings & meta 
        emb_array = np.array(all_embeddings, dtype=np.float32)   # (N, dim)
        save_embeddings(event_id, emb_array)
        # Temporarily save meta without cluster_ids
        save_meta(event_id, all_meta)

        # ── Cluster
        labels = ClusteringService.cluster_embeddings(
            emb_array, eps=recluster_eps
        )

        # Write cluster_id back into meta
        for i, label in enumerate(labels):
            all_meta[i]["cluster_id"] = int(label)
        save_meta(event_id, all_meta)

        # ── Build manifest
        manifest = ClusteringService.build_cluster_manifest(
            event_id, emb_array, labels, all_meta
        )
        save_clusters(event_id, manifest)

        # ── Pre-build FAISS index
        centroids = np.array(
            [c["centroid"] for c in manifest["clusters"].values()],
            dtype=np.float32,
        )
        SearchService.build_and_save_faiss_index(event_id, centroids)

        # ── Build response
        clusters_summary = [
            {
                "cluster_id": c["cluster_id"],
                "num_images": c["num_images"],
                "image_paths": c["image_paths"],
            }
            for c in manifest["clusters"].values()
        ]

        response = {
            "event_id": event_id,
            "num_people_detected": manifest["num_clusters"],
            "clusters": clusters_summary,
        }
        if images_with_no_face:
            response["images_with_no_face"] = images_with_no_face

        logger.info(
            "Event '%s' processed: %d new face(s), %d total cluster(s).",
            event_id,
            new_faces_total,
            manifest["num_clusters"],
        )
        return response

    # ── Re-clustering
    @classmethod
    def recluster_event(
        cls,
        event_id: str,
        eps: float | None = None,
        min_samples: int | None = None,
    ) -> Dict[str, Any]:
        """
        Re-run clustering on already-stored embeddings.
        Useful after tuning eps or adding more images.
        """
        emb_array = load_embeddings(event_id)
        meta = load_meta(event_id)

        if emb_array is None or meta is None:
            raise ValueError(f"No processed data found for event '{event_id}'.")

        manifest = ClusteringService.recluster(
            event_id, emb_array, meta, eps=eps, min_samples=min_samples
        )
        save_meta(event_id, meta)
        save_clusters(event_id, manifest)

        # Rebuild FAISS index
        centroids = np.array(
            [c["centroid"] for c in manifest["clusters"].values()],
            dtype=np.float32,
        )
        SearchService.build_and_save_faiss_index(event_id, centroids)

        clusters_summary = [
            {
                "cluster_id": c["cluster_id"],
                "num_images": c["num_images"],
                "image_paths": c["image_paths"],
            }
            for c in manifest["clusters"].values()
        ]
        return {
            "event_id": event_id,
            "num_people_detected": manifest["num_clusters"],
            "clusters": clusters_summary,
            "message": "Re-clustering completed.",
        }

    # ── Helpers 

    @staticmethod
    def _is_duplicate(
        new_emb: np.ndarray,
        existing: List[np.ndarray],
        threshold: float = 0.98,
    ) -> bool:
        """
        Return True when *new_emb* is suspiciously close to any existing
        embedding (cosine similarity > threshold).
        Avoids storing the same face twice when the same image is uploaded again.
        """
        if not existing:
            return False
        stack = np.array(existing, dtype=np.float32)   # (N, dim)
        sims = stack @ new_emb                          # (N,)  dot product = cosine
        return bool(np.max(sims) > threshold)