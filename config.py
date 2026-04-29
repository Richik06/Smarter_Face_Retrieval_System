"""
Centralised configuration loaded from environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Storage ───────────────────────────────────────────────────────────────
    EVENT_DATA_DIR: Path = Path("event_data")

    # ── Face detection ────────────────────────────────────────────────────────
    DETECTOR_BACKEND: str = "retinaface"

    # ── Face embedding ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "Facenet512"
    EMBEDDING_DIM: int = 512

    # ── DBSCAN clustering ─────────────────────────────────────────────────────
    DBSCAN_EPS: float = 0.35
    DBSCAN_MIN_SAMPLES: int = 1
    DBSCAN_METRIC: str = "cosine"

    # ── Similarity search ─────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.55

    # ── FAISS ─────────────────────────────────────────────────────────────────
    USE_FAISS: bool = True
    # Windows: always use "cpu" — faiss-gpu not available on Windows PyPI
    # Linux:   use "auto" to auto-detect GPU
    FAISS_DEVICE: str = "cpu"
    FAISS_GPU_ID: int = 0
    FAISS_IVF_THRESHOLD: int = 100
    FAISS_NLIST: int = 10
    FAISS_NPROBE: int = 10

    # ── Google Drive ──────────────────────────────────────────────────────────
    GOOGLE_DRIVE_API_KEY: str = ""
    DRIVE_TEMP_DIR: Path = Path("event_data/_drive_tmp")
    DRIVE_MAX_IMAGES: int = 0

    # ── Cloudinary (optional — for cloud image hosting) ───────────────────────
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_UPLOAD_FOLDER: str = "face-retrieval"

    # ── Misc ──────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure root storage directory exists at import time
settings.EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)