"""
Storage helpers – all disk I/O for event data lives here.

Directory layout
────────────────
event_data/
  <event_id>/
    images/          ← saved source images
    embeddings.npy   ← (N, dim) float32 array of all face embeddings
    meta.json        ← list of {image_path, face_index, cluster_id} per embedding
    clusters.json    ← final cluster manifest
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Path helpers

def event_dir(event_id: str) -> Path:
    return settings.EVENT_DATA_DIR / event_id


def images_dir(event_id: str) -> Path:
    return event_dir(event_id) / "images"


def embeddings_path(event_id: str) -> Path:
    return event_dir(event_id) / "embeddings.npy"


def meta_path(event_id: str) -> Path:
    return event_dir(event_id) / "meta.json"


def clusters_path(event_id: str) -> Path:
    return event_dir(event_id) / "clusters.json"


def faiss_index_path(event_id: str) -> Path:
    return event_dir(event_id) / "faiss.index"


def ensure_event_dirs(event_id: str) -> None:
    images_dir(event_id).mkdir(parents=True, exist_ok=True)


# ── Embeddings 

def save_embeddings(event_id: str, embeddings: np.ndarray) -> None:
    """Persist a (N, dim) float32 array to disk."""
    ensure_event_dirs(event_id)
    np.save(str(embeddings_path(event_id)), embeddings.astype(np.float32))
    logger.debug("Saved %d embeddings for event '%s'.", len(embeddings), event_id)


def load_embeddings(event_id: str) -> Optional[np.ndarray]:
    """Load embeddings array, or None if not found."""
    p = embeddings_path(event_id)
    if not p.exists():
        return None
    return np.load(str(p)).astype(np.float32)


# ── Per-face metadata 

def save_meta(event_id: str, meta: List[Dict[str, Any]]) -> None:
    """
    meta is a list of dicts, one per embedding row:
      {"image_path": str, "face_index": int, "cluster_id": int | None}
    """
    ensure_event_dirs(event_id)
    with open(meta_path(event_id), "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(event_id: str) -> Optional[List[Dict[str, Any]]]:
    p = meta_path(event_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── Cluster manifest

def save_clusters(event_id: str, clusters: Dict[str, Any]) -> None:
    ensure_event_dirs(event_id)
    with open(clusters_path(event_id), "w") as f:
        json.dump(clusters, f, indent=2)
    logger.debug("Saved cluster manifest for event '%s'.", event_id)


def load_clusters(event_id: str) -> Optional[Dict[str, Any]]:
    p = clusters_path(event_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── Existence checks

def event_exists(event_id: str) -> bool:
    return event_dir(event_id).exists()


def clusters_exist(event_id: str) -> bool:
    return clusters_path(event_id).exists()