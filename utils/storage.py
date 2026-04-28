"""
Storage helpers - all disk I/O for event data lives here.

Directory layout
event_data/
  <event_id>/
    images/
    embeddings.npy
    meta.json
    clusters.json
    asset_urls.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


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


def asset_urls_path(event_id: str) -> Path:
    return event_dir(event_id) / "asset_urls.json"


def ensure_event_dirs(event_id: str) -> None:
    images_dir(event_id).mkdir(parents=True, exist_ok=True)


def save_embeddings(event_id: str, embeddings: np.ndarray) -> None:
    ensure_event_dirs(event_id)
    np.save(str(embeddings_path(event_id)), embeddings.astype(np.float32))
    logger.debug("Saved %d embeddings for event '%s'.", len(embeddings), event_id)


def load_embeddings(event_id: str) -> Optional[np.ndarray]:
    path = embeddings_path(event_id)
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.float32)


def save_meta(event_id: str, meta: List[Dict[str, Any]]) -> None:
    ensure_event_dirs(event_id)
    with open(meta_path(event_id), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def load_meta(event_id: str) -> Optional[List[Dict[str, Any]]]:
    path = meta_path(event_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_clusters(event_id: str, clusters: Dict[str, Any]) -> None:
    ensure_event_dirs(event_id)
    with open(clusters_path(event_id), "w", encoding="utf-8") as file:
        json.dump(clusters, file, indent=2)
    logger.debug("Saved cluster manifest for event '%s'.", event_id)


def load_clusters(event_id: str) -> Optional[Dict[str, Any]]:
    path = clusters_path(event_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_asset_urls(event_id: str, asset_urls: Dict[str, str]) -> None:
    ensure_event_dirs(event_id)
    with open(asset_urls_path(event_id), "w", encoding="utf-8") as file:
        json.dump(asset_urls, file, indent=2)


def load_asset_urls(event_id: str) -> Optional[Dict[str, str]]:
    path = asset_urls_path(event_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def event_exists(event_id: str) -> bool:
    return event_dir(event_id).exists()


def clusters_exist(event_id: str) -> bool:
    return clusters_path(event_id).exists()
